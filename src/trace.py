"""Trace a field goal: find the ball with the trained detector and draw its flight.

    python src/trace.py data/raw/kick_05.mp4
    python src/trace.py data/raw/kick_05.mp4 --model models/ball/weights/best.pt
    python src/trace.py data/raw/kick_05.mp4 --verdict   # green / yellow / red: is it good?

Steps (each in its own file):
  motion.py     read frames, measure camera movement, find screen overlays
  kick.py       when the camera starts following the kick, where the holder is
  field.py      one coordinate system for the whole clip despite camera moves
  detect_ball.py  run the trained detector on three-frame motion images
  trajectory.py link detections into one smooth path from the holder
  draw.py       draw the tracer line
  verdict.py    --verdict: color the line by whether the kick looks good
"""
import argparse
import hashlib
import pickle
from pathlib import Path

import cv2
from ultralytics import YOLO

from detect_ball import detect_all
from draw import render
from field import field_coords
from kick import find_kick
from motion import camera_homography, camera_motion, nearest, overlay_mask, read_frames, specks_now, to_gray
from trajectory import FILL_RADIUS, drift_correction, fill_from_curve, fit_flight, grow, seed_track
from verdict import kick_distance, label, verdict_color, verdicts

KICK_SLACK = 5       # frames the kick may come before the camera starts following
SEED_CONF = 0.3      # detections at least this confident start the track
GROW_CONF = 0.05     # weaker detections are fine for extending a track we trust
FILL_ROUNDS = 5      # rounds of fit-the-physics-then-search
AIRBORNE_FRAMES = 8  # --start airborne: frames after the kick before the line appears (~1/4 s)


def cached_detections(clip, model, grays, motions, first):
    """Running the detector is the slow part, so save its results next to the outputs."""
    weights = Path(model.ckpt_path)
    tag = hashlib.md5(str(weights.resolve()).encode()).hexdigest()[:8]  # one cache per model
    cache = Path("outputs") / "cache" / f"{clip.stem}_{tag}.pkl"
    cache.parent.mkdir(parents=True, exist_ok=True)
    key = (str(weights), weights.stat().st_mtime, first, GROW_CONF)
    if cache.exists():
        saved_key, dets = pickle.loads(cache.read_bytes())
        if saved_key == key:
            return dets
    dets = detect_all(model, grays, motions, first=first, conf=GROW_CONF)
    cache.write_bytes(pickle.dumps((key, dets)))
    return dets


def trace(clip, model):
    frames = read_frames(clip)
    blurred = [to_gray(f) for f in frames]                         # for camera motion
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]  # for the detector
    ignore = overlay_mask(blurred)
    # Two camera-motion estimates: the simple one builds the detector's motion
    # images (it was trained on those); the exact one (homography) maps every
    # frame into one fixed view, which the physics fit needs.
    motions = [None] + [camera_motion(blurred[i - 1], blurred[i]) for i in range(1, len(frames))]
    exact = [None] + [camera_homography(blurred[i - 1], blurred[i], ignore) for i in range(1, len(frames))]
    camera_kick, holder = find_kick(blurred, motions, ignore)
    if camera_kick is None:
        print("  couldn't tell when the camera starts following the kick; searching the whole clip")
        camera_kick = 1
    earliest = max(camera_kick - KICK_SLACK, 1)
    C = field_coords(exact, camera_kick)

    dets = cached_detections(clip, model, grays, motions, earliest)
    dets = {i: [d for d in ds if not ignore[int(d[1]), int(d[0])]] for i, ds in dets.items()}
    strong = {i: [d for d in ds if d[2] >= SEED_CONF] for i, ds in dets.items()}

    pts = seed_track(strong, C)
    if len(pts) < 3:
        raise SystemExit("  no ball found: the detector didn't see a clear flight in this clip")
    seeded = len(pts)
    last = len(frames) - 2
    reached = grow(pts, dets, C, -1, earliest, last, holder)
    grow(pts, dets, C, +1, earliest, last)
    start = reached - 1 if reached else earliest
    size = frames[0].shape[1::-1]
    # Fit the physics curve, look for the ball near it in empty frames, refit, repeat.
    # Where the detector saw nothing, plain motion specks near the curve are the backup.
    backup = lambda i, guess: nearest(specks_now(frames, blurred, motions, i, guess, FILL_RADIUS), guess)
    for _ in range(FILL_ROUNDS):
        curve, kick, kept = fit_flight(pts, C, holder, start)
        if not fill_from_curve(pts, dets, C, curve, kick, last, size, backup):
            break
    curve, kick, kept = fit_flight(pts, C, holder, start)
    end = max(kept)
    shift = drift_correction({i: pts[i] for i in kept}, C, curve, kick)

    print(f"  detector hits in {sum(1 for d in dets.values() if d)} of {len(dets)} frames")
    print(f"  ball: {seeded} frames from the seed track, {len(pts)} after growing, "
          f"{len(kept)} fit the physics curve")
    where = "not found" if holder is None else f"({holder[0]:.0f}, {holder[1]:.0f})"
    print(f"  holder {where}; kick at frame {kick:.1f}")
    return {"frames": frames, "grays": grays, "motions": motions, "C": C, "kick": kick,
            "end": end, "curve": curve, "shift": shift, "sightings": {i: pts[i] for i in kept}, "detections": dets,
            "holder": holder, "start": start, "ignore": ignore}


def airborne_frame(r):
    """Frame where the line starts: a fixed moment after the kick.

    By then the ball is clear of the kicker and holder, so the line can't latch
    onto a foot or a back, and every kick starts at the same point in its flight.
    """
    return min(int(round(r["kick"])) + AIRBORNE_FRAMES, r["end"] - 1)


def playback_fps(clip, kept_frames):
    """Frames per second that keeps real-time speed after dropping duplicate frames."""
    cap = cv2.VideoCapture(str(clip))
    fps, total = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return fps * kept_frames / total if total else 30


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("clip")
    parser.add_argument("--model", default="models/ball/weights/best.pt")
    parser.add_argument("--style", default="line", choices=["line", "comet"],
                        help="line: the whole path; comet: a tail that follows the ball")
    parser.add_argument("--start", default="airborne", choices=["kick", "airborne"],
                        help="kick: line starts at the holder; airborne: once the ball is clearly in the air")
    parser.add_argument("--verdict", action="store_true",
                        help="color the line: green = good, yellow = unsure, red = no good")
    parser.add_argument("--distance", type=float,
                        help="kick distance in yards for --verdict (default: from data/clips.csv)")
    args = parser.parse_args()
    clip = Path(args.clip)
    print(clip.name)
    r = trace(clip, YOLO(args.model))
    suffix = "tracer" if args.style == "line" else args.style
    if args.start == "kick":
        suffix += "_from_kick"
    fps = playback_fps(clip, len(r["frames"]))
    colors = None
    if args.verdict:
        suffix += "_verdict"
        distance = args.distance or kick_distance(clip)
        if distance is None:
            print("  kick distance unknown (use --distance); the verdict will be less sure")
        chance = verdicts(r, fps, distance)
        colors = [verdict_color(p) for p in chance]
        print(f"  verdict: {label(chance[r['end']])} (chance it's good {chance[r['end']]:.2f})")
    out = Path("outputs") / f"{clip.stem}_{suffix}.mp4"
    start = airborne_frame(r) if args.start == "airborne" else None
    still = render(r["frames"], r["curve"], r["shift"], r["C"], r["kick"], r["end"], out,
                   fps, style=args.style, start=start, colors=colors)
    cv2.imwrite(str(out.with_suffix(".png")), still)
    print(f"  saved {out}")
