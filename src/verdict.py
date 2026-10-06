"""Is the kick good? A golf-tracer style verdict for every frame.

At frame i we only use what has been seen up to frame i (no peeking at the
rest of the flight), so the color is an honest live prediction:

1. Goal:    the posts found so far (uprights.py), in field coordinates.
2. Fit:     the physics curve through the sightings so far (trajectory.fit_flight).
3. Window:  the ball reaches the goal roughly distance / speed seconds after the
            kick. Its depth is hard to see from behind, so we try speeds between
            SPEEDS and read off where the curve is, relative to the posts, at each.
4. Trust:   if the prediction is still jumping around between frames, it isn't
            reliable yet; that spread becomes the uncertainty.
5. Chance:  the share of the window where the ball is between the posts, allowing
            for the uncertainty. 1 = good, 0.5 = unsure, 0 = no good.

Only left / right is judged. Whether the ball clears the crossbar can't be told
reliably from behind the kicker (a ball past the goal looks low on screen even
when it cleared), and kicks are rarely short.

Positions are measured in goal widths: 0 = left post, 1 = right post.
"""
import csv
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import norm

from field import to_field
from trajectory import fit_flight
from uprights import find_uprights

SPEEDS = (20, 15)            # yd/s toward the goal: fast and slow guess (drag slows long kicks)
UNKNOWN_DISTANCE = (25, 55)  # yards, if the kick distance isn't known
SAMPLES = 7                  # crossing times tried across the window
MIN_SIGHTINGS = 6            # sightings needed before the first prediction
TRUST_FRAMES = 10            # compare the prediction with the last this-many frames
BASE_SPREAD = 0.04           # goal widths: uncertainty even when predictions agree
SMOOTHING = 0.25             # 0..1: how quickly the chance follows a new prediction

GREEN, YELLOW, RED = (70, 220, 50), (0, 215, 255), (50, 50, 240)  # blue, green, red


def kick_distance(clip):
    """Distance (yards) from data/clips.csv, or None if the clip isn't listed."""
    table = Path("data/clips.csv")
    if not table.exists():
        return None
    for row in csv.DictReader(table.open()):
        if row["clip_id"] == Path(clip).stem and row["distance_yds"]:
            return float(row["distance_yds"])
    return None


def find_goal(frames, C, ignore):
    """For each frame, (left post x, right post x) in field coordinates (None if unseen)."""
    goal = []
    for i, frame in enumerate(frames):
        u = find_uprights(frame, ignore)
        foot = lambda p: None if p is None else float(to_field(C, i, (p[0] * p[3] + p[1], p[3]))[0])
        goal.append((foot(u["left"]), foot(u["right"])))
    return goal


def chance_inside(x, spread):
    """Chance a position known to +-spread is between 0 and 1."""
    return norm.cdf(x / spread) + norm.cdf((1 - x) / spread) - 1


def verdicts(r, fps, distance=None):
    """Chance the kick is good, for every frame (0.5 = no idea yet)."""
    C, n = r["C"], len(r["frames"])
    goal = find_goal(r["frames"], C, r["ignore"])
    near, far = (distance, distance) if distance else UNKNOWN_DISTANCE
    times = np.linspace(near / SPEEDS[0], far / SPEEDS[1], SAMPLES)  # seconds after the kick
    pts = r["sightings"]
    raw, history = [0.5] * n, []
    for i in range(n):
        past = {j: p for j, p in pts.items() if j <= i}
        posts = [g for g in goal[:i + 1] if g[0] is not None and g[1] is not None]
        if len(past) < MIN_SIGHTINGS or not posts or i > r["end"]:
            raw[i] = raw[i - 1] if i > r["end"] else 0.5
            continue
        left, right = np.median([g[0] for g in posts]), np.median([g[1] for g in posts])
        width = right - left
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                curve, kick, _ = fit_flight(past, C, r["holder"], r["start"])
        except ValueError:  # too few sightings left after dropping bad ones: no prediction yet
            raw[i] = 0.5
            continue
        x = np.array([(curve(kick + t * fps)[0] - left) / width for t in times])
        history.append(float(np.mean(x)))
        recent = history[-TRUST_FRAMES:]
        spread = BASE_SPREAD + (np.std(recent) if len(recent) >= 3 else 1.0)
        raw[i] = float(np.mean(chance_inside(x, spread)))
    # Smooth over time so the color doesn't flicker
    smooth = [raw[0]]
    for p in raw[1:]:
        smooth.append(smooth[-1] + SMOOTHING * (p - smooth[-1]))
    return smooth


def verdict_color(p):
    """Green when good (p near 1), yellow when unsure (0.5), red when no good (p near 0)."""
    if p >= 0.5:
        t, target = min(1.0, (p - 0.5) / 0.3), GREEN
    else:
        t, target = min(1.0, (0.5 - p) / 0.3), RED
    return tuple(float(a + t * (b - a)) for a, b in zip(YELLOW, target))


def label(p):
    return "GOOD" if p > 0.8 else "NO GOOD" if p < 0.2 else "UNSURE"
