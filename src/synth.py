"""Make synthetic training examples: a drawn football flying over real footage.

Real field-goal clips are scarce, but any football footage works as background.
For each example we take three consecutive real frames, draw a football in
each one where a real flying ball would be (moving steadily, with the real
camera motion between the frames), and build the same three-frame motion
image the detector sees (stack.py). The label is exact because we drew it.

    python src/synth.py --count 4000 data/raw/kick_01.mp4 data/external/*.webm ...

Writes YOLO-format tiles to data/ball_dataset/synthetic/.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from motion import alignment_error, camera_motion, invert, to_gray
from stack import three_frame_stack

TILE = 640
OUT = Path("data/ball_dataset/synthetic")
SUPER = 4  # draw at 4x size, then shrink, for smooth edges
MAX_MISALIGN = 9  # frames that don't line up better than this are skipped


def draw_ball(frame, center, length, angle, color, blur, streak):
    """Blend a football (an ellipse with shading, plus motion blur) into frame in place."""
    pad = int(length + streak + 6)
    size = 2 * pad * SUPER
    sprite = np.zeros((size, size, 3), np.float32)
    alpha = np.zeros((size, size), np.float32)
    c = (size // 2, size // 2)
    axes = (max(1, int(length * SUPER / 2)), max(1, int(length * 0.6 * SUPER / 2)))
    cv2.ellipse(alpha, c, axes, angle, 0, 360, 1.0, -1)
    cv2.ellipse(sprite, c, axes, angle, 0, 360, color, -1)
    # light from above: brighter top half, darker bottom
    yy = np.linspace(-1, 1, size)[:, None]
    sprite *= (1.15 - 0.3 * (yy + 1) / 2)[..., None]
    if streak > 0.5:  # motion blur along the direction of travel
        k = max(3, int(streak * SUPER) | 1)
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2, :] = 1 / k
        rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1)
        kernel = cv2.warpAffine(kernel, rot, (k, k))
        kernel /= kernel.sum()
        sprite = cv2.filter2D(sprite, -1, kernel)
        alpha = cv2.filter2D(alpha, -1, kernel)
    sprite = cv2.resize(sprite, (2 * pad, 2 * pad), interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(alpha, (2 * pad, 2 * pad), interpolation=cv2.INTER_AREA)
    if blur > 0:
        sprite = cv2.GaussianBlur(sprite, (0, 0), blur)
        alpha = cv2.GaussianBlur(alpha, (0, 0), blur)
    x0, y0 = int(round(center[0])) - pad, int(round(center[1])) - pad
    h, w = frame.shape[:2]
    fx0, fy0, fx1, fy1 = max(x0, 0), max(y0, 0), min(x0 + 2 * pad, w), min(y0 + 2 * pad, h)
    if fx1 <= fx0 or fy1 <= fy0:
        return
    s = sprite[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0]
    a = alpha[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0, None]
    region = frame[fy0:fy1, fx0:fx1].astype(np.float32)
    frame[fy0:fy1, fx0:fx1] = np.clip(region * (1 - a) + s * a, 0, 255).astype(np.uint8)


def random_ball(rng):
    """Size, color and blur of one ball, varied like real broadcasts."""
    hue = rng.uniform(4, 16)                    # brown-orange (OpenCV hue scale 0-180)
    sat = rng.uniform(70, 200)
    val = rng.uniform(45, 190)
    bgr = cv2.cvtColor(np.uint8([[[hue, sat, val]]]), cv2.COLOR_HSV2BGR)[0, 0].astype(float)
    return {"length": rng.uniform(4, 18), "color": tuple(bgr), "blur": rng.uniform(0.3, 1.3),
            "shutter": rng.uniform(0.0, 0.35)}


def project(M, p):
    """Apply a 2x3 camera motion to a point."""
    return np.array([M[0, 0] * p[0] + M[0, 1] * p[1] + M[0, 2], M[1, 0] * p[0] + M[1, 1] * p[1] + M[1, 2]])


def make_example(frames3, rng):
    """frames3: three consecutive BGR frames. Returns (motion image, ball xy, ball size) or None."""
    blurred = [to_gray(f) for f in frames3]
    motions = [None, camera_motion(blurred[0], blurred[1]), camera_motion(blurred[1], blurred[2])]
    if motions[1] is None or motions[2] is None:
        return None
    # skip scene cuts and whip-pans: the frames must line up well, as in real flights
    for k in (1, 2):
        if alignment_error(blurred[k - 1], blurred[k], np.vstack([motions[k], [0, 0, 1]])) > MAX_MISALIGN:
            return None
    h, w = frames3[1].shape[:2]
    ball = random_ball(rng)
    speed = rng.uniform(3, 60) if rng.random() < 0.9 else rng.uniform(1, 4)  # a few nearly still
    heading = np.deg2rad(rng.uniform(-170, -10) if rng.random() < 0.85 else rng.uniform(0, 360))
    v = speed * np.array([np.cos(heading), np.sin(heading)])
    bend = rng.normal(0, 0.08) * speed  # slight curve, like gravity pulling the arc
    p1 = np.array([rng.uniform(40, w - 40), rng.uniform(40, h - 40)])
    p0_here = p1 - v + np.array([0, -bend])          # where it was, seen in frame 1's view
    p2_here = p1 + v + np.array([0, bend])           # where it will be, in frame 1's view
    p0 = project(invert(motions[1]), p0_here)        # ...converted into frame 0's own pixels
    p2 = project(motions[2], p2_here)                # ...and frame 2's own pixels
    # Make sure the ball stands out from what's behind it (mostly): real balls
    # are usually visible, and an invisible label only teaches guessing.
    background = float(np.median(blurred[1][int(max(p1[1] - 15, 0)):int(p1[1] + 15),
                                              int(max(p1[0] - 15, 0)):int(p1[0] + 15)]))
    gray = float(np.dot(ball["color"], [0.114, 0.587, 0.299]))
    want = rng.uniform(25, 90) if rng.random() < 0.85 else rng.uniform(8, 25)
    if abs(gray - background) < want:
        target = background - want if background > 128 else background + want
        ball["color"] = tuple(np.clip(np.array(ball["color"]) * (max(target, 15) / max(gray, 1)), 0, 255))
    drawn = [f.copy() for f in frames3]
    angle = np.rad2deg(heading) + rng.normal(0, 25)  # a spiral isn't perfectly aligned
    streak = min(speed * ball["shutter"], 2 * ball["length"])  # real balls look like blobs, not long streaks
    for frame, p in zip(drawn, (p0, p1, p2)):
        draw_ball(frame, p, ball["length"], angle + rng.normal(0, 10), ball["color"], ball["blur"], streak)
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in drawn]
    image = three_frame_stack(grays, motions, 1)
    return image, p1, ball["length"] + streak


def read_some(path, rng, n_triplets, scale_to=1920):
    """Yield random triplets of consecutive frames, scaled to broadcast width.

    Reads the video straight through once (jumping around in long videos is
    very slow) and keeps a triplet at randomly chosen spots.
    """
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    starts = set(rng.choice(max(total - 3, 1), size=min(n_triplets, max(total - 3, 1)), replace=False).tolist())
    recent, i = [], 0
    while starts:
        ok, f = cap.read()
        if not ok:
            break
        recent = (recent + [f])[-3:]
        if i - 2 in starts and len(recent) == 3:
            starts.discard(i - 2)
            trio = recent
            if trio[0].shape[1] < scale_to and rng.random() < 0.7:
                size = (scale_to, round(trio[0].shape[0] * scale_to / trio[0].shape[1]))
                trio = [cv2.resize(t, size) for t in trio]
            if cv2.absdiff(trio[0], trio[1]).mean() > 0.5:  # skip duplicate frames
                yield trio
        i += 1
    cap.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="+")
    parser.add_argument("--count", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "labels").mkdir(parents=True, exist_ok=True)
    per_video = int(np.ceil(args.count / len(args.videos)))
    made = start = len(list((OUT / "images").glob("*.jpg")))  # add to what's already there
    for path in args.videos:
        for trio in read_some(path, rng, per_video):
            result = make_example(trio, rng)
            if result is None:
                continue
            image, (bx, by), size = result
            H, W = image.shape[:2]
            if W < TILE or H < TILE:
                continue
            x0 = int(np.clip(bx - rng.integers(40, TILE - 40), 0, W - TILE))
            y0 = int(np.clip(by - rng.integers(40, TILE - 40), 0, H - TILE))
            name = f"syn_{made:05d}"
            cv2.imwrite(str(OUT / "images" / f"{name}.jpg"), image[y0:y0 + TILE, x0:x0 + TILE],
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            box = max(10, min(size, 40)) / TILE
            cx, cy = (bx - x0) / TILE, (by - y0) / TILE
            label = f"0 {cx:.6f} {cy:.6f} {box:.6f} {box:.6f}\n" if 0 < cx < 1 and 0 < cy < 1 else ""
            (OUT / "labels" / f"{name}.txt").write_text(label)
            made += 1
        print(f"  {Path(path).name}: {made - start} new examples so far", flush=True)


if __name__ == "__main__":
    main()
