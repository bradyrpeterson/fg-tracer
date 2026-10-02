"""Run the trained ball detector over every frame of a clip.

Each frame's three-frame motion image (stack.py) is cut into overlapping
640x640 tiles at full resolution (so a 5-15 pixel ball is never shrunk),
the tiles go through YOLO in batches, and detections are mapped back to
full-frame coordinates.
"""
import numpy as np
import torch

from stack import three_frame_stack

TILE = 640
OVERLAP = 120
DEVICE = "mps" if torch.backends.mps.is_available() else 0 if torch.cuda.is_available() else "cpu"  # best available chip


def tile_starts(length):
    """Left/top edges of overlapping tiles that cover `length` pixels."""
    if length <= TILE:
        return [0]
    count = int(np.ceil((length - OVERLAP) / (TILE - OVERLAP)))
    return [round(k * (length - TILE) / (count - 1)) for k in range(count)]


def detect_all(model, grays, motions, first=1, last=None, conf=0.1):
    """Return {frame: [(x, y, confidence), ...]} for frames first..last."""
    h, w = grays[0].shape
    xs, ys = tile_starts(w), tile_starts(h)
    last = len(grays) - 2 if last is None else last
    found = {}
    for i in range(max(first, 1), last + 1):
        image = three_frame_stack(grays, motions, i)
        tiles = [image[y:y + TILE, x:x + TILE] for y in ys for x in xs]
        corners = [(x, y) for y in ys for x in xs]
        dets = []
        for (x0, y0), result in zip(corners, model.predict(tiles, imgsz=TILE, conf=conf, verbose=False, device=DEVICE)):
            for (x1, y1, x2, y2), c in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
                dets.append((x0 + (x1 + x2) / 2, y0 + (y1 + y2) / 2, c))
        found[i] = merge_duplicates(dets)
    return found


def merge_duplicates(dets, radius=12):
    """Overlapping tiles can see the same ball twice; keep the most confident one."""
    kept = []
    for x, y, c in sorted(dets, key=lambda d: -d[2]):
        if all(np.hypot(x - kx, y - ky) > radius for kx, ky, _ in kept):
            kept.append((x, y, c))
    return kept
