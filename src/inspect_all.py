"""Run inspect_video on every .mp4 in data/raw and print one line per clip."""
from pathlib import Path

from inspect_video import inspect

for path in sorted(Path("data/raw").glob("*.mp4")):
    w, h, fps, n, dur, out = inspect(path)
    # Flag clips that may be too low quality to track a ball well
    flags = []
    if fps < 24:
        flags.append("LOW FPS")
    if h < 720:
        flags.append("LOW RES")
    print(f"{path.name:<14} {w}x{h:<6} {fps:6.2f} fps {n:6} frames {dur:6.2f}s  {' '.join(flags)}")
