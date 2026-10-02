"""One shared coordinate system for a whole clip ("field coordinates").

The camera pans, tilts and zooms, so the same spot on the field lands on
different pixels in every frame. C[i] converts a point in frame i to where it
would be in one reference frame (the kick frame). Positions stored that way
stay stuck to the field while the camera moves, and can be converted back
into any frame for drawing.
"""
import numpy as np


def field_coords(motions, kick):
    """motions[i] maps frame i-1 to frame i (2x3 affine or 3x3 homography)."""
    to3 = lambda M: M if M.shape == (3, 3) else np.vstack([M, [0, 0, 1]])
    C = [None] * len(motions)
    C[kick] = np.eye(3)
    for i in range(kick + 1, len(motions)):
        C[i] = C[i - 1] @ np.linalg.inv(to3(motions[i]))
    for i in range(kick - 1, -1, -1):
        C[i] = C[i + 1] @ to3(motions[i + 1])
    return C


def to_field(C, i, p):
    x, y, s = C[i] @ [p[0], p[1], 1]
    return np.array([x / s, y / s])


def to_frame(C, i, p):
    x, y, s = np.linalg.solve(C[i], [p[0], p[1], 1])
    return np.array([x / s, y / s])


def zoom(C, i):
    """How many field units one pixel of frame i covers (smaller = zoomed in)."""
    return float(np.sqrt(abs(np.linalg.det(C[i][:2, :2]))))
