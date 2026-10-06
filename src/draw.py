"""Draw a smooth, broadcast-style tracer line onto the video."""
import cv2
import imageio_ffmpeg
import numpy as np

from field import to_frame

COLOR = (40, 60, 255)   # blue, green, red -> a bright red-orange
GLOW_WIDTH = 16
CORE_WIDTH = 5
SAMPLES_PER_FRAME = 4   # draw the curve between frames too, so it looks smooth
PRECISION = 4           # OpenCV draws at 1/16-pixel precision with shift=4
COMET_SECONDS = 0.5     # comet style: how much of the recent flight the tail shows
COMET_WIDTH = 9         # comet style: tail thickness at the ball (pixels)
COMET_FADE = 12         # comet style: frames to fade out after the flight ends
FADE_IN = 6             # frames for a late-starting line to fade in
BALL_DOT = 7            # comet style: radius of the glowing dot on the ball


def draw_line(image, points, color=COLOR):
    """Glow + core polyline with sub-pixel precision."""
    pts = np.round(np.asarray(points) * (1 << PRECISION)).astype(np.int32)
    glow = image.copy()
    cv2.polylines(glow, [pts], False, color, GLOW_WIDTH, cv2.LINE_AA, shift=PRECISION)
    cv2.addWeighted(glow, 0.35, image, 0.65, 0, dst=image)
    cv2.polylines(image, [pts], False, color, CORE_WIDTH, cv2.LINE_AA, shift=PRECISION)
    cv2.polylines(image, [pts], False, (200, 220, 255), 1, cv2.LINE_AA, shift=PRECISION)  # bright center


def draw_comet(image, points, fade=1.0, color=COLOR):
    """A tapered, fading tail ending in a glowing dot (points run from tail to ball).

    The tail is drawn into a brightness mask: thin and faint at the back, thick
    and solid at the ball. A blurred copy of the mask becomes the glow.
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), np.float32)
    pts = np.round(np.asarray(points) * (1 << PRECISION)).astype(np.int32)
    n = len(pts)
    for k in range(1, n):
        along = k / (n - 1)  # 0 at the tail, 1 at the ball
        width = max(1, int(round(1 + (COMET_WIDTH - 1) * along)))
        piece = np.zeros_like(mask)
        cv2.line(piece, tuple(pts[k - 1]), tuple(pts[k]), along ** 1.5, width, cv2.LINE_AA, shift=PRECISION)
        np.maximum(mask, piece, out=mask)
    head = tuple(pts[-1])
    cv2.circle(mask, head, BALL_DOT << PRECISION, 1.0, -1, cv2.LINE_AA, shift=PRECISION)
    glow = cv2.GaussianBlur(mask, (0, 0), 6) * 1.6
    core = np.clip(mask * fade, 0, 1)[..., None]
    halo = np.clip(glow * fade, 0, 1)[..., None]
    out = image.astype(np.float32)
    out = out * (1 - 0.5 * halo) + np.array(color, np.float32) * 0.5 * halo   # soft colored glow
    out = out * (1 - core) + np.array(color, np.float32) * core              # solid tail
    white = np.zeros((h, w), np.float32)
    cv2.circle(white, head, max(1, BALL_DOT - 3) << PRECISION, 1.0, -1, cv2.LINE_AA, shift=PRECISION)
    white = (white * fade)[..., None]
    out = out * (1 - white) + np.array((230, 240, 255), np.float32) * white   # hot center on the ball
    image[:] = np.clip(out, 0, 255).astype(np.uint8)


def render(frames, curve, shift, C, kick, end, out_path, fps, style="line", start=None, colors=None):
    """Write the clip with the tracer; return the frame where the ball's path ends.

    curve(t) gives the ball's field position at (possibly fractional) frame t;
    shift(i) is the small drift correction (pixels) for frame i.
    style "line" draws the whole path; "comet" draws only the last moment of
    flight, following the ball, and fades out after the flight ends.
    start: frame where the line begins (default: the kick). A later start
    hides the messy first frames next to the kicker; the line fades in there.
    colors: optional color for each frame (e.g. the green/yellow/red verdict).
    """
    start = kick if start is None else start
    h, w = frames[0].shape[:2]
    # H.264 at high quality: OpenCV's default codec turns thin lines into blocky dots
    video = imageio_ffmpeg.write_frames(str(out_path), (w, h), fps=fps, codec="libx264",
                                        quality=None, output_params=["-crf", "16", "-preset", "slow"],
                                        pix_fmt_in="bgr24", pix_fmt_out="yuv420p", macro_block_size=1)
    video.send(None)  # start the encoder
    last = frames[-1]
    for i, frame in enumerate(frames):
        out = frame.copy()
        color = COLOR if colors is None else colors[i]
        upto = min(i, end)
        if style == "comet":
            tail = int(round(COMET_SECONDS * fps))
            fade = 1 - (i - end) / COMET_FADE if i > end else 1.0
            if start > kick:  # late start: fade in too
                fade = min(fade, (i - start) / FADE_IN)
            first = max(start, upto - tail)
            if upto > start and fade > 0:
                ts = np.linspace(first, upto, int((upto - first) * SAMPLES_PER_FRAME) + 2)
                draw_comet(out, [to_frame(C, i, curve(t)) + shift(i) for t in ts], fade, color)
        elif upto > start:
            ts = np.linspace(start, upto, int((upto - start) * SAMPLES_PER_FRAME) + 1)
            fade_in = min(1.0, (i - start) / FADE_IN) if start > kick else 1.0
            drawn = out.copy()
            draw_line(drawn, [to_frame(C, i, curve(t)) + shift(i) for t in ts], color)
            cv2.addWeighted(drawn, fade_in, out, 1 - fade_in, 0, dst=out)
        video.send(np.ascontiguousarray(out))
        if i == end:
            last = out
    video.close()
    return last
