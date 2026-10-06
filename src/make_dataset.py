"""Build training data for the ball detector from the trackers' own results.

Hand-labeling hundreds of frames is slow, so we "pseudo-label": run a tracker,
keep only the sightings that fit its smooth curve, and use those as labels.

  Round 1 (no model yet):  labels come from the classic tracker (track_ball.py).
  Round 2+ (--model):      labels come from the detector pipeline (trace.py),
                           which is cleaner. Places where the detector fired but
                           the ball wasn't ("hard negatives") are saved too, so
                           the next model learns not to repeat those mistakes.
  Hand labels (--labels):  for clips you clicked with label.py, those positions
                           are used instead of a tracker's. Needed for clips the
                           tracker gets wrong (wide still shots, new stadiums).
                           With --model too, the detector's mistakes become
                           hard negatives.

Each example is a 640x640 tile of the three-frame motion image (stack.py), in YOLO format:
    data/ball_dataset/<clip>/images/<name>.jpg
    data/ball_dataset/<clip>/labels/<name>.txt   ->  "0 x_center y_center width height" (0-1 scale)

    python src/make_dataset.py data/raw/kick_0*.mp4
    python src/make_dataset.py data/raw/kick_0*.mp4 --model models/ball/weights/best.pt
    python src/make_dataset.py data/raw/kick_09.mp4 --labels labels/hand_labels.json --model models/ball/weights/best.pt
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from motion import camera_motion, read_frames, to_gray
from stack import three_frame_stack

TILE = 640           # YOLO's normal input size, so the ball isn't shrunk
BOX = 16             # label box size in pixels around the ball's center
NOT_BALL = 30        # a detection this far (pixels) from the ball is a mistake
OUT = Path("data/ball_dataset")


def save_tile(folder, name, image, x0, y0, ball):
    """Save the tile whose top-left is (x0, y0); label the ball if it's inside."""
    (folder / "images").mkdir(parents=True, exist_ok=True)
    (folder / "labels").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / "images" / f"{name}.jpg"), image[y0:y0 + TILE, x0:x0 + TILE],
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    label = ""
    if ball is not None:
        bx, by = ball[0] - x0, ball[1] - y0
        if 0 <= bx < TILE and 0 <= by < TILE:
            label = f"0 {bx / TILE:.6f} {by / TILE:.6f} {BOX / TILE:.6f} {BOX / TILE:.6f}\n"
    (folder / "labels" / f"{name}.txt").write_text(label)


def corner_around(point, w, h, rng):
    """Top-left of a tile that contains point somewhere inside (not always centered)."""
    x0 = int(np.clip(point[0] - rng.integers(40, TILE - 40), 0, w - TILE))
    y0 = int(np.clip(point[1] - rng.integers(40, TILE - 40), 0, h - TILE))
    return x0, y0


def from_hand_labels(frames, ball, model=None):
    """Use ball positions clicked with label.py instead of a tracker's."""
    blurred = [to_gray(f) for f in frames]
    motions = [None] + [camera_motion(blurred[i - 1], blurred[i]) for i in range(1, len(frames))]
    sightings = {int(i): tuple(p) for i, p in ball.items()}
    detections = {}
    if model is not None:  # the detector's guesses around the flight, to find its mistakes
        from detect_ball import detect_all
        grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
        detections = detect_all(model, grays, motions, max(min(sightings) - 20, 1), max(sightings), conf=0.2)
    return {"sightings": sightings, "motions": motions, "detections": detections}


def examples_for_clip(path, rng, model=None, hand=None):
    frames = read_frames(path)
    if hand:
        result = from_hand_labels(frames, hand, model)
    elif model is None:
        from track_ball import track
        result = track(frames)
    else:
        from trace import trace
        result = trace(path, model)
    # confident detections far from the ball are mistakes worth learning from
    mistakes = {} if model is None else {
        i: [(x, y) for x, y, c in ds if c >= 0.2 and
            (i not in result["sightings"] or
             np.hypot(x - result["sightings"][i][0], y - result["sightings"][i][1]) > NOT_BALL)]
        for i, ds in result["detections"].items()}
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    h, w = grays[0].shape
    folder = OUT / path.stem
    shutil.rmtree(folder, ignore_errors=True)
    balls = negatives = 0
    for i, ball in result["sightings"].items():
        if not (0 <= ball[0] < w and 0 <= ball[1] < h):
            continue
        image = three_frame_stack(grays, result["motions"], i)
        for k in range(2):  # two tiles with the ball in different spots
            save_tile(folder, f"{path.stem}_{i:04d}_{k}", image, *corner_around(ball, w, h, rng), ball)
            balls += 1
        for k in range(20):  # one tile somewhere else, usually without the ball
            x0, y0 = rng.integers(0, w - TILE), rng.integers(0, h - TILE)
            if not (x0 - 30 < ball[0] < x0 + TILE + 30 and y0 - 30 < ball[1] < y0 + TILE + 30):
                save_tile(folder, f"{path.stem}_{i:04d}_bg", image, x0, y0, None)
                break
    for i, spots in mistakes.items():
        if not spots:
            continue
        image = three_frame_stack(grays, result["motions"], i)
        for k, spot in enumerate(spots[:2]):
            save_tile(folder, f"{path.stem}_{i:04d}_neg{k}", image, *corner_around(spot, w, h, rng),
                      result["sightings"].get(i))
            negatives += 1
    print(f"  saved {balls} ball tiles from {len(result['sightings'])} sightings "
          f"and {negatives} hard-negative tiles -> {folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("clips", nargs="+")
    parser.add_argument("--model", help="trained detector; omit for round 1")
    parser.add_argument("--labels", help="hand labels from label.py (labels/hand_labels.json)")
    args = parser.parse_args()
    hand = json.loads(Path(args.labels).read_text()) if args.labels else {}
    model = None
    if args.model:
        from ultralytics import YOLO
        model = YOLO(args.model)
    rng = np.random.default_rng(0)
    for arg in args.clips:
        path = Path(arg)
        ball = hand.get(path.stem, {}).get("ball")
        print(path.name, "(hand labels)" if ball else "")
        examples_for_clip(path, rng, model, ball)
