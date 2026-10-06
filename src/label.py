"""Label the ball by hand: click it in each frame.

    python src/label.py data/raw/kick_09.mp4
    python src/label.py data/raw/kick_09.mp4 --start 50    # begin at frame 50

Hand labels are the "right answers": make_dataset.py --labels trains on them and
evaluate.py --truth scores against them. They're saved after every click to
labels/hand_labels.json as {clip: {"kick": frame, "ball": {frame: [x, y]}}}.
Frame numbers skip duplicate frames (motion.read_frames), like every other script.

Controls (click the window once so it gets the keys):
  click        the ball's center -> saved, jumps to the next frame
  d / space    next frame (use when the ball can't be seen)
  a            previous frame
  f / b        10 frames forward / back
  x            remove this frame's label
  k            mark this frame as the kick (the moment the foot hits the ball)
  h            show / hide hints (earlier clicks and the cyan guess; off at start)
  q / esc      quit (everything is already saved)

The box in the corner magnifies the area under the mouse. Move the mouse into
the box and it freezes, so you can click the ball there for extra precision.
Green circle = this frame's label (seen when you go back). With hints on (h):
gray dots = the last few clicks, cyan circle = where the ball should be next.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from motion import read_frames

LABELS = Path("labels/hand_labels.json")
MAX_W, MAX_H = 1500, 850  # largest window size on screen
ZOOM, LOOK = 4, 40        # magnifier: 4x zoom of the 80x80 pixels around the mouse


def load():
    return json.loads(LABELS.read_text()) if LABELS.exists() else {}


def save(all_labels):
    # Write frames in number order so the file is easy to read. Sort a copy:
    # main() keeps adding clicks to the same "ball" dict, so it must not be replaced.
    tidy = {name: {**clip, "ball": dict(sorted(clip["ball"].items(), key=lambda kv: int(kv[0])))}
            for name, clip in all_labels.items()}
    LABELS.write_text(json.dumps(tidy, indent=1))


def hint(ball, i):
    """Where the ball should be in frame i, continuing the last two labels before it."""
    before = sorted(int(k) for k in ball if int(k) < i)[-2:]
    if len(before) < 2:
        return None
    (j1, j2), p1, p2 = before, np.array(ball[str(before[0])]), np.array(ball[str(before[1])])
    return p2 + (p2 - p1) * (i - j2) / (j2 - j1)


def draw(frame, i, ball, hints):
    """The full-size frame with this frame's label (and, with hints, earlier labels and a guess)."""
    image = frame.copy()
    for j in range(i - 6, i) if hints else []:
        if str(j) in ball:
            cv2.circle(image, tuple(int(v) for v in ball[str(j)]), 4, (160, 160, 160), -1)
    guess = hint(ball, i) if hints else None
    if guess is not None:
        cv2.circle(image, tuple(int(v) for v in guess), 14, (255, 255, 0), 1)
    if str(i) in ball:
        cv2.circle(image, tuple(int(v) for v in ball[str(i)]), 12, (0, 255, 0), 2)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("clip")
    parser.add_argument("--start", type=int, default=0, help="first frame to show")
    args = parser.parse_args()
    clip = Path(args.clip)
    print(f"reading {clip.name} ...")
    frames = read_frames(clip)
    h, w = frames[0].shape[:2]
    scale = min(MAX_W / w, MAX_H / h, 1.0)
    size = LOOK * 2 * ZOOM
    all_labels = load()
    mine = all_labels.setdefault(clip.stem, {"kick": None, "ball": {}})
    ball = mine["ball"]
    state = {"i": min(args.start, len(frames) - 1), "look": (w / 2, h / 2), "box": (0, 0), "hints": False}

    def in_box(x, y):
        bx, by = state["box"]
        return bx <= x < bx + size and by <= y < by + size

    def on_mouse(event, x, y, flags, _):
        if in_box(x, y):  # inside the magnifier: map back through the zoom
            bx, by = state["box"]
            point = (state["look"][0] + (x - bx - size / 2) / ZOOM, state["look"][1] + (y - by - size / 2) / ZOOM)
        else:
            point = (x / scale, y / scale)
            state["look"] = point
        if event == cv2.EVENT_LBUTTONDOWN:
            ball[str(state["i"])] = [round(point[0], 1), round(point[1], 1)]
            save(all_labels)
            state["i"] = min(state["i"] + 1, len(frames) - 1)

    cv2.namedWindow("label", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("label", on_mouse)
    while True:
        i = state["i"]
        full = draw(frames[i], i, ball, state["hints"])
        view = cv2.resize(full, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        # magnifier in the top corner away from the mouse
        lx, ly = state["look"]
        bx = 10 if lx * scale > view.shape[1] / 2 else view.shape[1] - size - 10
        state["box"] = (bx, 50)
        zoomed = cv2.resize(cv2.getRectSubPix(full, (2 * LOOK, 2 * LOOK), (lx, ly)), (size, size),
                            interpolation=cv2.INTER_NEAREST)
        cv2.line(zoomed, (size // 2, 0), (size // 2, size), (0, 0, 255), 1)
        cv2.line(zoomed, (0, size // 2), (size, size // 2), (0, 0, 255), 1)
        view[50:50 + size, bx:bx + size] = zoomed
        cv2.rectangle(view, (bx, 50), (bx + size, 50 + size), (255, 255, 255), 1)
        status = (f"{clip.stem}  frame {i}/{len(frames) - 1}  labeled {len(ball)}  kick {mine['kick']}"
                  "   click=ball  d/space=next  a=back  f/b=10  x=erase  k=kick  h=hints  q=quit")
        cv2.rectangle(view, (0, 0), (view.shape[1], 36), (0, 0, 0), -1)
        cv2.putText(view, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("label", view)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key in (ord("d"), ord(" ")):
            state["i"] = min(i + 1, len(frames) - 1)
        elif key == ord("a"):
            state["i"] = max(i - 1, 0)
        elif key == ord("f"):
            state["i"] = min(i + 10, len(frames) - 1)
        elif key == ord("b"):
            state["i"] = max(i - 10, 0)
        elif key == ord("x"):
            ball.pop(str(i), None)
            save(all_labels)
        elif key == ord("h"):
            state["hints"] = not state["hints"]
        elif key == ord("k"):
            mine["kick"] = i
            save(all_labels)
    cv2.destroyAllWindows()
    save(all_labels)
    print(f"{len(ball)} ball positions for {clip.stem} in {LABELS}; kick frame {mine['kick']}")


if __name__ == "__main__":
    main()
