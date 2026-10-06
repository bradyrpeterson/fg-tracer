"""Find the goalposts (two yellow uprights and the crossbar) in a frame.

NFL uprights are bright yellow and thin. We keep only yellow pixels, join the
dashes a thin post breaks into, and look for straight lines: long vertical ones
are the posts, a horizontal one between their feet is the crossbar. The camera
often shows only part of the goal (the bar below the picture, a post off the
side), so each part is optional.

A post is stored as a line x = slope * y + x0 (it may lean a little), the bar
as y = slope * x + y0. White college posts aren't handled yet.
"""
import cv2
import numpy as np

MIN_POST = 0.12   # a post must be at least this fraction of the picture height
POST_GAP = 40     # pixels: post pieces closer than this (sideways) are the same post
MIN_WIDTH = 60    # pixels: the posts must be at least this far apart


def yellow_mask(frame, ignore=None):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = ((h >= 18) & (h <= 40) & (s > 90) & (v > 130)).astype(np.uint8) * 255
    if ignore is not None:
        mask[ignore > 0] = 0  # scoreboard text can be yellow too
    return mask


def fit_line(points, along):
    """Least-squares line through (x, y) points: x = f(y) if along == "y", else y = f(x)."""
    p = np.asarray(points, float)
    a, b = (p[:, 1], p[:, 0]) if along == "y" else (p[:, 0], p[:, 1])
    slope, offset = np.polyfit(a, b, 1) if np.ptp(a) > 0 else (0.0, b.mean())
    return float(slope), float(offset)


def find_uprights(frame, ignore=None):
    """Return {"left": post, "right": post, "bar": bar}; any part may be None.

    post = (slope, x0, top_y, bottom_y) meaning x = slope * y + x0 for top_y..bottom_y
    bar  = (slope, y0, left_x, right_x) meaning y = slope * x + y0
    """
    h, w = frame.shape[:2]
    mask = yellow_mask(frame, ignore)
    upright = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 1), np.uint8))  # join post dashes
    lines = cv2.HoughLinesP(upright, 1, np.pi / 180, 30, minLineLength=h * MIN_POST / 2, maxLineGap=25)
    flat = cv2.HoughLinesP(mask, 1, np.pi / 180, 30, minLineLength=40, maxLineGap=10)
    segs = [] if lines is None else lines.reshape(-1, 4).tolist()
    vertical = [s for s in segs if abs(s[2] - s[0]) < 0.1 * abs(s[3] - s[1])]
    horizontal = [] if flat is None else [l for l in flat.reshape(-1, 4).tolist() if abs(l[3] - l[1]) < 0.05 * abs(l[2] - l[0])]

    # Group vertical pieces into posts by their x position
    groups = []
    for s in sorted(vertical, key=lambda s: (s[0] + s[2]) / 2):
        x = (s[0] + s[2]) / 2
        if groups and x - groups[-1]["x"] < POST_GAP:
            groups[-1]["segs"].append(s)
        else:
            groups.append({"x": x, "segs": [s]})
    posts = []
    for g in groups:
        pts = [(s[0], s[1]) for s in g["segs"]] + [(s[2], s[3]) for s in g["segs"]]
        top, bottom = min(p[1] for p in pts), max(p[1] for p in pts)
        if bottom - top >= h * MIN_POST:
            posts.append((*fit_line(pts, "y"), top, bottom))

    # Crossbar: the longest horizontal yellow line
    bar = None
    if horizontal:
        s = max(horizontal, key=lambda s: abs(s[2] - s[0]))
        bar = (*fit_line([(s[0], s[1]), (s[2], s[3])], "x"), min(s[0], s[2]), max(s[0], s[2]))
    post_x = lambda p, y: p[0] * y + p[1]

    if bar is not None:
        # Posts stand on the bar's ends; the support under its middle doesn't count
        bar_y = lambda x: bar[0] * x + bar[1]
        posts = [p for p in posts if p[2] < bar_y(post_x(p, p[3])) - 10]
    # The two posts: the longest pair far enough apart
    posts.sort(key=lambda p: p[3] - p[2], reverse=True)
    pair = next(((a, b) for k, a in enumerate(posts) for b in posts[k + 1:]
                 if abs(post_x(a, a[3]) - post_x(b, b[3])) > MIN_WIDTH), None)
    if pair is None:
        left = right = None
        if posts and bar is not None:  # one post: which side of the bar is it on?
            middle = (bar[2] + bar[3]) / 2
            left, right = (posts[0], None) if post_x(posts[0], posts[0][3]) < middle else (None, posts[0])
        elif posts:
            left = posts[0]  # side unknown; call it left
        return {"left": left, "right": right, "bar": bar}
    left, right = sorted(pair, key=lambda p: post_x(p, p[3]))
    return {"left": left, "right": right, "bar": bar}


def draw_uprights(image, u, color=(255, 0, 255)):
    """Draw what find_uprights found (for checking by eye)."""
    for p in (u["left"], u["right"]):
        if p is not None:
            slope, x0, top, bottom = p
            cv2.line(image, (int(slope * top + x0), int(top)), (int(slope * bottom + x0), int(bottom)), color, 3)
    if u["bar"] is not None:
        slope, y0, x1, x2 = u["bar"]
        cv2.line(image, (int(x1), int(slope * x1 + y0)), (int(x2), int(slope * x2 + y0)), (255, 255, 0), 3)
