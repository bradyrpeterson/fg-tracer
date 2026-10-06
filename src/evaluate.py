"""Measure how close the traced line is to the real ball.

labels/ground_truth.json holds ball positions from the old tracker, checked by eye:
    {"kick_05": {"95": [x, y], ...}, ...}
labels/hand_labels.json (label.py) holds hand-clicked ones: use --truth for those.
For each clip we trace the ball and report the distance (pixels) between the
line's head and the true ball in those frames.

    python src/evaluate.py kick_05 [--model models/ball/weights/best.pt]
    python src/evaluate.py kick_09 --truth labels/hand_labels.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from field import to_frame
from trace import trace


def score(clip, model, truth):
    r = trace(Path(f"data/raw/{clip}.mp4"), model)
    C, kick, end, curve = r["C"], r["kick"], r["end"], r["curve"]
    errors = []
    for k, (x, y) in truth.items():
        k = int(k)
        if kick <= k <= end:
            hx, hy = to_frame(C, k, curve(k)) + r["shift"](k)
            errors.append(np.hypot(hx - x, hy - y))
        else:
            errors.append(np.inf)  # the line didn't cover this frame at all
    # the network on its own: did it put a detection on the ball in that frame?
    found = [any(np.hypot(dx - x, dy - y) < 10 for dx, dy, _ in r["detections"].get(int(k), []))
             for k, (x, y) in truth.items()]
    e = np.array(errors)
    print(f"  {clip}: median {np.median(e):.1f}px, 90% within {np.percentile(e, 90):.1f}px, "
          f"{(e < 10).mean():.0%} of frames within 10px ({len(e)} checked); "
          f"network found the ball in {np.mean(found):.0%} of them")
    return e


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("clips", nargs="+")
    parser.add_argument("--model", default="models/ball/weights/best.pt")
    parser.add_argument("--truth", default="labels/ground_truth.json")
    args = parser.parse_args()
    truth = json.loads(Path(args.truth).read_text())
    model = YOLO(args.model)
    for clip in args.clips:
        entry = truth[clip]
        score(clip, model, entry.get("ball", entry))  # hand_labels.json keeps positions under "ball"
