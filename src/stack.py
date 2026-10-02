"""Turn three neighboring frames into one image that shows motion as color.

The previous, current and next frames (lined up so the camera's movement is
cancelled) become the red, green and blue channels of one picture. Anything
that stays still looks gray; anything that moves shows up as colored fringes.
This is the main idea behind TrackNet, a ball tracker for tiny, fast balls.
"""
import numpy as np

from motion import invert, warp


def three_frame_stack(grays, motions, i):
    """Color image for frame i built from frames i-1, i, i+1 (camera motion removed)."""
    h, w = grays[i].shape
    i_prev, i_next = max(i - 1, 0), min(i + 1, len(grays) - 1)
    prev = grays[i_prev] if i_prev == i else warp(grays[i_prev], motions[i], (w, h))
    nxt = grays[i_next] if i_next == i else warp(grays[i_next], invert(motions[i_next]), (w, h))
    return np.dstack([nxt, grays[i], prev])  # OpenCV stores color as blue, green, red
