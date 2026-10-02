"""Track the football from the holder's hands through its flight and draw the path."""
import sys
from pathlib import Path

import cv2
import numpy as np

from field import field_coords, to_field, to_frame
from kick import find_kick
from motion import camera_motion, find_candidates, nearest, overlay_mask, read_frames, specks_now, to_gray

MATCH_RADIUS = 25  # how far (pixels) a candidate can be from the prediction
MAX_MISSES = 3     # frames a track can go unseen before it ends
CURVE_DEGREE = 3   # the flight path is fitted as a cubic curve over time
SEARCH = 30        # how far (pixels) from the predicted spot to look when extending
GAP_SEARCH = 60    # how far (pixels) from the curve to look for the ball in empty frames
FILL_ROUNDS = 3    # rounds of fit-the-curve-then-search
KICK_SLACK = 5     # frames the kick may come before the camera starts following


# ---------- Step 1: find the ball's main flight track ----------

def camera_move(point, M):
    """Apply one frame's camera motion to an (x, y) point."""
    x, y = point
    return (M[0, 0] * x + M[0, 1] * y + M[0, 2], M[1, 0] * x + M[1, 1] * y + M[1, 2])


def main_track(grays, motions, ignore, kick):
    """Link moving specks into tracks; return the one that behaves most like a ball."""
    active, finished = [], []
    for i in range(kick + 1, len(grays)):
        M = motions[i]
        cands = find_candidates(grays[i - 1], grays[i], M, ignore)
        for t in sorted(active, key=lambda t: -len(t["points"])):  # long tracks pick first
            moved = camera_move(t["pos"], M)                        # follow the camera
            guess = (moved[0] + t["vel"][0], moved[1] + t["vel"][1])
            dists = [np.hypot(c[0] - guess[0], c[1] - guess[1]) for c in cands]
            if dists and min(dists) < MATCH_RADIUS:
                c = cands.pop(int(np.argmin(dists)))
                t["vel"] = (c[0] - moved[0], c[1] - moved[1])
                t["pos"], t["misses"] = c, 0
                t["points"][i] = c
                t["net"] = (t["net"][0] + t["vel"][0], t["net"][1] + t["vel"][1])
            else:
                t["pos"], t["misses"] = guess, t["misses"] + 1
        finished += [t for t in active if t["misses"] > MAX_MISSES]
        active = [t for t in active if t["misses"] <= MAX_MISSES]
        active += [{"pos": c, "vel": (0, 0), "misses": 0, "points": {i: c}, "net": (0, 0)} for c in cands]
    if not finished + active:
        raise SystemExit("No moving objects found after the kick")
    # The ball: a track that lasts long AND travels far against the background.
    # Jittery noise cancels itself out; people move slowly; a ball keeps going.
    ball = max(finished + active, key=lambda t: len(t["points"]) * np.hypot(*t["net"]))
    return {i: np.array(p) for i, p in ball["points"].items()}


# ---------- Step 2: extend the track at both ends ----------

def extend(points, step, frames, grays, motions, C, holder, earliest):
    """Follow the ball past one end of its track (step=-1 backward, +1 forward).

    The next position is predicted by fitting a short arc (a parabola) through
    the ~10 sightings nearest in time and extending it one frame; then we look
    for something moving near that spot. Going backward we stop when the ball
    is back down at the holder: that frame is the kick.
    Returns the frame where the ball reached the holder (or None).
    """
    h, w = grays[0].shape
    i, misses = (min(points) if step < 0 else max(points)), 0
    holder_y = holder[1]
    while misses < MAX_MISSES:
        i += step
        if i < earliest or i >= len(frames) - 1:
            break
        recent = sorted(points, key=lambda k: abs(k - i))[:10]
        t = np.array(recent, float)
        xy = np.array([to_field(C, k, points[k]) for k in recent])
        degree = 2 if len(recent) >= 5 else 1
        guess = to_frame(C, i, [np.polyval(np.polyfit(t, xy[:, d], degree), i) for d in (0, 1)])
        if not (0 <= guess[0] < w and 0 <= guess[1] < h):
            break  # the ball has left the picture
        found = nearest(specks_now(frames, grays, motions, i, guess, SEARCH), guess)
        if found is None:
            misses += 1
            continue
        points[i], misses = found, 0
        if step < 0 and to_field(C, i, found)[1] > holder_y - 20:
            return i  # back down at the holder's level: this is the kick
    return None


# ---------- Step 3: pin each sighting onto the ball, fill gaps ----------

def refine(points, frames, grays, motions):
    """Move each sighting onto the ball's position in that exact frame."""
    for i, p in list(points.items()):
        found = nearest(specks_now(frames, grays, motions, i, p, 30, exact=True), p)
        if found is not None:
            points[i] = found
    return points


def fill_gaps(points, path, frames, grays, motions, C):
    """Look for the ball near the fitted curve in frames where it wasn't seen.

    Only after the first real sighting: before that the curve passes the
    players, and their movement would be mistaken for the ball.
    """
    added, first = 0, min(points)
    for i, p in path.items():
        if i in points or i < first:
            continue
        guess = to_frame(C, i, p)
        found = nearest(specks_now(frames, grays, motions, i, guess, GAP_SEARCH), guess)
        if found is not None:
            points[i], added = found, added + 1
    return added


# ---------- Step 4: fit one smooth curve from the holder through the sightings ----------

def fit_path(points, holder, kick, C):
    """Return ({frame: field position} starting at the holder, frames of the kept sightings).

    Seen from the kick frame's fixed view, a kick is a smooth arc, so x and y
    are each fitted as a cubic of time. The holder gets a heavy weight so the
    curve starts there; sightings far from the curve are dropped and it is refit.
    Finally the curve is nudged onto every kept sighting so the line sits on the ball.
    """
    seen = sorted(points)
    t = np.array([kick] + seen, float)
    xy = np.array([holder] + [to_field(C, i, points[i]) for i in seen])
    weight = np.r_[20.0, np.ones(len(seen))]
    keep = np.ones(len(t), bool)
    for _ in range(3):
        cx = np.polyfit(t[keep], xy[keep, 0], CURVE_DEGREE, w=weight[keep])
        cy = np.polyfit(t[keep], xy[keep, 1], CURVE_DEGREE, w=weight[keep])
        off = np.hypot(np.polyval(cx, t) - xy[:, 0], np.polyval(cy, t) - xy[:, 1])
        keep = off < max(3 * np.median(off[1:]), 10)
        keep[0] = True
    every = np.arange(kick, int(t[keep].max()) + 1)
    curve = np.c_[np.polyval(cx, every), np.polyval(cy, every)]
    # How far each kept sighting sits from the curve; spread those gaps over
    # the frames in between and add them back, so the line passes through the ball.
    kt, kxy = t[keep], xy[keep]
    gap = kxy - np.c_[np.polyval(cx, kt), np.polyval(cy, kt)]
    gap[0] = 0  # the holder stays put
    offset = np.c_[np.interp(every, kt, gap[:, 0]), np.interp(every, kt, gap[:, 1])]
    return dict(zip(every.tolist(), curve + offset)), [int(i) for i in kt[1:]]


def launch_time(points, holder, C, earliest):
    """Estimate the kick frame from how fast the ball was going when first seen.

    The ball slows down on screen as it flies away, so it covered the distance
    from the holder at least this fast: distance / early speed = frames of flight.
    """
    first = sorted(points)[:6]
    xy = [to_field(C, i, points[i]) for i in first]
    speed = np.hypot(*(xy[-1] - xy[0])) / max(first[-1] - first[0], 1)
    frames_flying = np.hypot(*(xy[0] - np.asarray(holder))) / max(speed, 1e-6)
    return int(np.clip(round(first[0] - frames_flying), earliest, first[0] - 1))


def track(frames):
    grays = [to_gray(f) for f in frames]
    ignore = overlay_mask(grays)
    motions = [None] + [camera_motion(grays[i - 1], grays[i]) for i in range(1, len(grays))]
    camera_kick, holder = find_kick(grays, motions, ignore)
    earliest = max(camera_kick - KICK_SLACK, 1)  # the kick can't be much before the camera reacts
    C = field_coords(motions, camera_kick)
    points = main_track(grays, motions, ignore, camera_kick)
    seen = len(points)
    reached_holder = extend(points, -1, frames, grays, motions, C, holder, earliest)
    extend(points, +1, frames, grays, motions, C, holder, earliest)
    extended = len(points) - seen
    points = refine(points, frames, grays, motions)
    # The kick: when the ball was traced back to the holder, or else estimated from its speed
    kick = reached_holder if reached_holder else launch_time(points, holder, C, earliest)
    # Fit a curve, look for the ball near it in the empty frames, refit, repeat.
    # Each round the curve gets closer to the ball, so the next search finds more.
    filled = 0
    for _ in range(FILL_ROUNDS):
        path, kept = fit_path(points, holder, kick, C)
        filled += fill_gaps(points, path, frames, grays, motions, C)
    path, kept = fit_path(points, holder, kick, C)
    strays = len(points) - len(kept)
    print(f"  holder at ({holder[0]:.0f}, {holder[1]:.0f}); camera reacts at frame {camera_kick}, "
          f"ball kicked at about frame {kick}")
    print(f"  ball seen in {len(points)} frames ({min(points)}-{max(points)}): {seen} by the main tracker, "
          f"{extended} by extending, {filled} by searching along the curve ({strays} strays ignored)")
    sightings = {i: points[i] for i in kept}
    return {"path": path, "C": C, "sightings": sightings, "kick": kick, "holder": holder,
            "grays": grays, "motions": motions}


# ---------- Drawing ----------

def draw_trace(frames, path, C, out_path):
    """Write a video with the path drawn on; return the frame where the path ends."""
    h, w = frames[0].shape[:2]
    video = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    order = sorted(path)
    last = None
    for i, frame in enumerate(frames):
        out = frame.copy()
        trail = [to_frame(C, i, path[k]) for k in order if k <= i]  # field -> this frame
        if len(trail) > 1:
            cv2.polylines(out, [np.int32(trail)], False, (0, 0, 255), 4, cv2.LINE_AA)
        video.write(out)
        if i == order[-1]:
            last = out
    video.release()
    return last


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python src/track_ball.py <video_path>")
    clip = Path(sys.argv[1])
    print(clip.name)
    frames = read_frames(clip)
    result = track(frames)
    still = draw_trace(frames, result["path"], result["C"], Path("outputs") / f"{clip.stem}_trace.mp4")
    cv2.imwrite(str(Path("outputs") / f"{clip.stem}_trace.png"), still)
