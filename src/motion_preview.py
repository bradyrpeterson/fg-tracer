"""Show where motion happens in a clip by stacking frame differences into one image."""
import sys
from pathlib import Path

import cv2
import numpy as np

if len(sys.argv) != 2:
    sys.exit("Usage: python src/motion_preview.py <video_path>")
video_path = sys.argv[1]
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    sys.exit(f"Error: could not open video '{video_path}'")

ok, first = cap.read()
prev = cv2.GaussianBlur(cv2.cvtColor(first, cv2.COLOR_BGR2GRAY), (5, 5), 0)
trail = np.zeros_like(prev)  # collects every pixel that ever moved
moving_pct = []              # % of the screen moving in each frame

while True:
    ok, frame = cap.read()
    if not ok:
        break
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    diff = cv2.absdiff(gray, prev)                              # how much each pixel changed
    _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)   # changed a lot -> white
    trail = cv2.max(trail, mask)
    moving_pct.append(100 * np.count_nonzero(mask) / mask.size)
    prev = gray
cap.release()

# Paint moving pixels red on top of the first frame
overlay = first.copy()
overlay[trail > 0] = (0, 0, 255)
out = Path("outputs") / f"{Path(video_path).stem}_motion.png"
cv2.imwrite(str(out), overlay)

# One number per ~10 frames: big values mean the whole camera moved
print(" ".join(f"{p:.0f}" for p in moving_pct[::10]))
print(f"Average moving: {np.mean(moving_pct):.1f}%   Max: {max(moving_pct):.1f}%   Saved: {out}")
