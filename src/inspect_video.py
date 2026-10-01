"""Print basic info about a video and save its first frame."""
import sys
from pathlib import Path

import cv2


def inspect(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"Error: could not open video '{video_path}'")

    # Read properties stored in the video file
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frames / fps if fps else 0

    # Grab the first frame and save it as a PNG
    ok, frame = cap.read()
    out_path = Path("outputs") / f"{Path(video_path).stem}_first_frame.png"
    if ok:
        cv2.imwrite(str(out_path), frame)
    cap.release()
    return width, height, fps, frames, duration, out_path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python src/inspect_video.py <video_path>")
    w, h, fps, n, dur, out = inspect(sys.argv[1])
    print(f"Resolution: {w}x{h}\nFPS: {fps:.2f}\nFrames: {n}\nDuration: {dur:.2f}s\nSaved: {out}")
