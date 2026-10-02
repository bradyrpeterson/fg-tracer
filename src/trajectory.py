"""Turn per-frame ball detections into one smooth flight path.

1. seed:   link detections frame to frame; keep the track that lasts long and
           travels far (noise jitters in place, people move slowly).
2. grow:   extend that track forward and backward one frame at a time,
           predicting with a short arc and taking the nearest detection.
           Going backward we stop when the ball is back at the holder.
3. fit:    fit the physics of a kick (a parabola seen through a camera) to the
           sightings; it also gives the exact kick time and launch spot.
4. fill:   look for the ball near the fitted flight where it wasn't found; refit.
5. drift:  a slow, smooth per-frame shift that keeps the line's head on the ball.
"""
import numpy as np
from scipy.interpolate import make_smoothing_spline
from scipy.optimize import least_squares

from field import to_field, to_frame

LINK_RADIUS = 40    # pixels: how far a detection may be from a track's prediction
SEARCH = 35         # pixels: how far from the predicted spot to look when growing
MAX_MISSES = 5      # frames without a detection before a track / growth stops
OUTLIER_PIXELS = 8   # sightings that jump this far from their neighbours are dropped
TAIL_GAP = 8         # frames: a gap this long near the end means the flight is over
MIN_DEPTH = 0.3      # the ball stays at least this fraction of its kick distance from the camera
PIXEL_NOISE = 3      # pixels: how far off a good detection usually is
HOLDER_TOLERANCE = 25  # pixels: how far the true launch spot may be from the holder estimate
FILL_RADIUS = 25    # pixels: how close to the fitted flight a detection must be to be added
LOOK_AHEAD = 15     # frames past the last sighting to keep looking
DRIFT_STIFFNESS = 20  # higher = the drift correction changes more slowly


def seed_track(dets, C):
    """Greedily link detections into tracks; return the most ball-like one as {frame: (x, y)}."""
    active, finished = [], []
    for i in sorted(dets):
        cands = list(dets[i])
        for t in sorted(active, key=lambda t: -len(t["pts"])):
            guess = predict(t["pts"], C, i)
            if not cands:
                t["misses"] += 1
                continue
            d = [np.hypot(x - guess[0], y - guess[1]) for x, y, _ in cands]
            k = int(np.argmin(d))
            if d[k] < LINK_RADIUS:
                x, y, c = cands.pop(k)
                t["pts"][i], t["conf"], t["misses"] = (x, y), t["conf"] + c, 0
            else:
                t["misses"] += 1
        finished += [t for t in active if t["misses"] > MAX_MISSES]
        active = [t for t in active if t["misses"] <= MAX_MISSES]
        active += [{"pts": {i: (x, y)}, "conf": c, "misses": 0} for x, y, c in cands]
    tracks = finished + active
    if not tracks:
        return {}
    # long, confident and travelling = ball
    return max(tracks, key=lambda t: t["conf"] * (1 + travel(t["pts"], C)))["pts"]


def travel(pts, C):
    """Distance (field units) between a track's first and last point."""
    a, b = min(pts), max(pts)
    return float(np.hypot(*(to_field(C, a, pts[a]) - to_field(C, b, pts[b]))))


def predict(pts, C, i):
    """Where the ball should be in frame i, from a short arc through the nearest points."""
    near = sorted(pts, key=lambda k: abs(k - i))[:8]
    xy = np.array([to_field(C, k, pts[k]) for k in near])
    if len(near) == 1:
        return to_frame(C, i, xy[0])
    degree = 2 if len(near) >= 5 else 1
    t = np.array(near, float)
    return to_frame(C, i, [np.polyval(np.polyfit(t, xy[:, d], degree), i) for d in (0, 1)])


def grow(pts, dets, C, step, first, last, holder=None):
    """Extend a track one frame at a time. Returns the kick frame if it reached the holder."""
    i, misses = (min(pts) if step < 0 else max(pts)), 0
    while misses < MAX_MISSES:
        i += step
        if i < first or i > last:
            break
        guess = predict(pts, C, i)
        near = [(x, y) for x, y, _ in dets.get(i, []) if np.hypot(x - guess[0], y - guess[1]) < SEARCH]
        if not near:
            misses += 1
            continue
        pts[i] = min(near, key=lambda p: np.hypot(p[0] - guess[0], p[1] - guess[1]))
        misses = 0
        if step < 0 and holder is not None and to_field(C, i, pts[i])[1] > holder[1] - 20:
            return i
    return None


def flight(p, tau):
    """Ball position (field coordinates) tau frames after the kick.

    A kicked ball moves at steady speed sideways and forward while gravity pulls
    it down, so in 3D its path is a parabola. Seen through a camera from one
    fixed spot, a parabola becomes a ratio of two quadratics in time, with the
    same bottom part for x and y (that part is how far the ball is from the camera):

        x(tau) = (x0 + a1*tau + a2*tau^2) / (1 + q1*tau + q2*tau^2)
        y(tau) = (y0 + b1*tau + b2*tau^2) / (1 + q1*tau + q2*tau^2)

    Only physically possible flights fit this, so the line can't jab or wiggle.
    p = (kick time, x0, y0, a1, a2, b1, b2, q1, q2)
    """
    _, x0, y0, a1, a2, b1, b2, q1, q2 = p
    depth = 1 + q1 * tau + q2 * tau ** 2
    return (x0 + a1 * tau + a2 * tau ** 2) / depth, (y0 + b1 * tau + b2 * tau ** 2) / depth


def fit_flight(pts, C, holder, earliest):
    """Fit the physics curve to the sightings; it also finds the kick time and launch spot.

    Errors are measured in each sighting's own frame pixels. A "soft" loss stops
    a few bad detections from pulling the curve; sightings still far off are
    dropped and the fit is redone. The holder is a strong hint (not a hard rule)
    for where the ball starts, because the holder estimate can be a little off.
    Returns (curve(t) -> field position, kick time, frames of kept sightings).
    """
    frames = np.array(sorted(i for i in pts if i > earliest))
    seen = np.array([pts[i] for i in frames], float)
    to_px = np.array([np.linalg.inv(C[i]) for i in frames])  # field -> each frame's pixels
    field = np.array([to_field(C, i, pts[i]) for i in frames])
    holder = None if holder is None else np.asarray(holder, float)
    keep = np.ones(len(frames), bool)

    def residuals(p):
        u, v = flight(p, frames[keep] - p[0])
        hom = np.einsum("nij,nj->ni", to_px[keep], np.c_[u, v, np.ones(len(u))])
        err = (hom[:, :2] / hom[:, 2:]) - seen[keep]
        # The bottom part of the formula is how far the ball is from the camera
        # (relative to the kick). It can't get near zero, or the curve would
        # shoot off to infinity and draw a straight streak across the picture.
        _, _, _, _, _, _, _, q1, q2 = p
        tau = np.linspace(0, frames.max() - p[0], 40)
        depth = 1 + q1 * tau + q2 * tau ** 2
        too_close = np.maximum(0, MIN_DEPTH - depth) * 1000
        if holder is None:
            return np.r_[err.ravel(), too_close]
        holder_err = (p[1:3] - holder) * (PIXEL_NOISE / HOLDER_TOLERANCE)
        return np.r_[err.ravel(), holder_err, too_close]

    p = initial_guess(frames, field, holder, earliest)
    for _ in range(3):
        lo = [earliest - 2] + [-np.inf] * 8
        hi = [frames[keep].min() - 0.5] + [np.inf] * 8
        p = least_squares(residuals, np.clip(p, lo, hi), bounds=(lo, hi),
                          loss="soft_l1", f_scale=PIXEL_NOISE).x
        u, v = flight(p, frames - p[0])
        hom = np.einsum("nij,nj->ni", to_px, np.c_[u, v, np.ones(len(u))])
        error = seen - hom[:, :2] / hom[:, 2:]
        # Slow camera drift is fine (drift_correction handles it); a sighting is
        # a mistake only if it jumps away from the slowly changing drift.
        slow = np.c_[[make_smoothing_spline(frames[keep], error[keep, d], lam=DRIFT_STIFFNESS)(frames)
                      for d in (0, 1)]].T
        off = np.hypot(*(error - slow).T)
        new_keep = off < max(OUTLIER_PIXELS, 3 * np.median(off[keep]))
        new_keep &= ~after_tail_gap(frames, new_keep)
        if (new_keep == keep).all():
            break
        keep = new_keep
    curve = lambda t: np.array(flight(p, t - p[0]), float)
    return curve, float(p[0]), [int(i) for i in frames[keep]]


def initial_guess(frames, field, holder, earliest):
    """A quick straight-line-algebra fit for each possible kick time; keep the best.

    Multiplying both sides of the flight formula by the bottom part makes it
    linear in the unknowns, which numpy can solve directly.
    """
    best, best_err = None, np.inf
    for t0 in np.arange(earliest - 2, frames.min() - 0.5, 0.5):
        tau = frames - t0
        u, v = field[:, 0], field[:, 1]
        zero = np.zeros_like(tau)
        one = np.ones_like(tau)
        # unknowns: x0, y0, a1, a2, b1, b2, q1, q2
        rows_u = np.c_[one, zero, tau, tau ** 2, zero, zero, -u * tau, -u * tau ** 2]
        rows_v = np.c_[zero, one, zero, zero, tau, tau ** 2, -v * tau, -v * tau ** 2]
        A, b = np.vstack([rows_u, rows_v]), np.r_[u, v]
        if holder is not None:  # a gentle hint that the flight starts at the holder
            hint = np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0]]) * 0.2
            A, b = np.vstack([A, hint]), np.r_[b, holder * 0.2]
        z = np.linalg.lstsq(A, b, rcond=None)[0]
        p = np.r_[t0, z]
        fu, fv = flight(p, tau)
        err = np.median(np.hypot(fu - u, fv - v))
        if err < best_err:
            best, best_err = p, err
    return best
def drift_correction(sightings, C, curve, kick):
    """A gentle per-frame shift that keeps the line's head on the ball.

    Tiny camera-motion errors add up over many frames, so the "fixed" view
    slowly slides and the physics curve drifts a few pixels off the ball. The
    drift changes slowly, so we fit a smoothing spline to it over time (zero at
    the kick) and shift the whole line by that amount in each frame. The line's
    shape still comes from physics, and the shift can only change gradually.
    Returns shift(frame) -> (dx, dy) in pixels.
    """
    frames = sorted(sightings)
    off = np.array([np.asarray(sightings[i]) - to_frame(C, i, curve(i)) for i in frames])
    t = np.r_[kick, frames]
    off = np.vstack([[0, 0], off])
    w = np.r_[20.0, np.ones(len(frames))]
    sx = make_smoothing_spline(t, off[:, 0], w=w, lam=DRIFT_STIFFNESS)
    sy = make_smoothing_spline(t, off[:, 1], w=w, lam=DRIFT_STIFFNESS)
    end = frames[-1]
    return lambda i: np.array([float(sx(min(i, end))), float(sy(min(i, end)))])


def after_tail_gap(frames, keep):
    """True for kept sightings that come after a long gap at the end of the flight.

    Once the ball is gone (into the net, out of the picture) the camera often
    cuts or zooms out, and stray movement found long after the last real
    sighting would drag the end of the line somewhere the ball never went.
    """
    kept = frames[keep]
    late = np.zeros(len(frames), bool)
    half = kept[len(kept) // 2]
    for a, b in zip(kept, kept[1:]):
        if a >= half and b - a > TAIL_GAP:
            late = frames >= b
            break
    return late


def fill_from_curve(pts, dets, C, curve, kick, last, size, backup=None):
    """Look for the ball near the fitted flight in frames where it wasn't found.

    Physics predicts well even across long gaps (e.g. when the ball crosses a
    dark part of the stadium), and a bit past the last sighting too. If the
    detector saw nothing there, backup(frame, guess) gets a try (trace.py uses
    plain motion specks); anything it finds must still fit the physics curve.
    Returns how many sightings were added.
    """
    w, h = size
    seen_until = max(pts)
    # Right after the kick the ball is next to the players, whose movement would
    # fool the backup, so it only helps after the detector has seen the ball itself.
    detector_first = min((i for i in pts if any(np.hypot(x - pts[i][0], y - pts[i][1]) < 1
                                                  for x, y, _ in dets.get(i, []))), default=seen_until)
    added = 0
    for i in range(int(np.ceil(kick)) + 1, min(last, seen_until + LOOK_AHEAD) + 1):
        if i in pts:
            continue
        gx, gy = to_frame(C, i, curve(i))
        if not (0 <= gx < w and 0 <= gy < h):
            continue
        near = [(x, y) for x, y, _ in dets.get(i, []) if np.hypot(x - gx, y - gy) < FILL_RADIUS]
        if near:
            pts[i] = min(near, key=lambda p: np.hypot(p[0] - gx, p[1] - gy))
            added += 1
        elif backup is not None and i > detector_first:
            found = backup(i, (gx, gy))
            if found is not None:
                pts[i] = (float(found[0]), float(found[1]))
                added += 1
    return added
