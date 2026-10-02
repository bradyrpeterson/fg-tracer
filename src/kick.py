"""Find roughly when the ball is kicked, and where the holder is."""
import cv2
import numpy as np

from motion import moving_mask

TILT_START = 1.5  # pixels/frame of upward camera tilt that means it's following the kick


def find_kick(grays, motions, ignore):
    """Return (frame where the camera starts following the kick, (x, y) of the holder).

    Either can be None if it can't be found (e.g. a camera behind the posts).

    The camera usually reacts within a few frames of the kick, but some
    operators start tilting early, so track_ball.py refines the exact moment.
    """
    # 1. The camera starts tilting up to follow the ball, so the picture slides down.
    tilt = [0] + [M[1, 2] for M in motions[1:]]
    start = next((i for i in range(1, len(tilt) - 5) if min(tilt[i:i + 5]) > TILT_START), None)
    if start is None:
        return None, None  # the camera never followed a kick upward (other angle or static camera)
    kick = max(start - 1, 1)

    # 2. Formation: the biggest cluster of motion during the snap and blocking
    blocking = cv2.dilate(moving_mask(grays, motions, ignore, kick - 15, kick - 3), np.ones((25, 25)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(blocking)
    if n < 2:
        return kick, None
    biggest = max(range(1, n), key=lambda k: stats[k, cv2.CC_STAT_AREA])
    fx, fy, fw, fh = stats[biggest, :4]
    middle = fx + fw / 2

    # 3. Holder: the kicker swinging at the holder makes motion just below
    #    the middle of the formation. Take the lowest such blob's bottom.
    swing = moving_mask(grays, motions, ignore, kick - 5, kick + 1)
    n, _, stats, centers = cv2.connectedComponentsWithStats(swing)
    blobs = [k for k in range(1, n)
             if stats[k, cv2.CC_STAT_AREA] >= 80
             and fy + fh * 0.5 < centers[k][1] < fy + fh * 1.25
             and abs(centers[k][0] - middle) < fw * 0.5]
    if not blobs:  # fall back to the bottom-middle of the formation
        return kick, (middle, fy + fh)
    low = max(blobs, key=lambda k: stats[k, 1] + stats[k, 3] - 0.5 * abs(centers[k][0] - middle))
    x, y, w, h = stats[low, :4]
    return kick, (centers[low][0], y + h)
