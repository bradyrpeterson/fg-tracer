"""Find small moving blobs in a video after cancelling out camera movement."""
import cv2
import numpy as np


def read_frames(path):
    """Read every frame, skipping near-duplicates left by screen recording."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Error: could not open video '{path}'")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frames and cv2.absdiff(frame, frames[-1]).mean() < 1:
            continue
        frames.append(frame)
    cap.release()
    return frames


def to_gray(frame):
    return cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)


def overlay_mask(grays):
    """Pixels that never change (scoreboard, logos) plus a thin border."""
    most_change = np.zeros_like(grays[0])
    for a, b in zip(grays, grays[1:]):
        most_change = cv2.max(most_change, cv2.absdiff(a, b))
    mask = np.where(most_change < 25, 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25)))  # fill gaps
    # Cover each overlay's whole bounding box (clock digits change, so edges are patchy)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if cv2.contourArea(c) > 500 and w < mask.shape[1] / 2 and h < mask.shape[0] / 2:
            mask[y:y + h, x:x + w] = 255  # skip the video's own border, which spans the frame
    mask = cv2.dilate(mask, np.ones((41, 41)))                          # add margin
    mask[:20, :] = mask[-20:, :] = mask[:, :20] = mask[:, -20:] = 255
    return mask


def camera_motion(prev, cur):
    """Estimate how the camera moved (shift, zoom, rotation) from prev to cur."""
    pts = cv2.goodFeaturesToTrack(prev, 400, 0.01, 20)
    new_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None)
    good = status.ravel() == 1
    M, _ = cv2.estimateAffinePartial2D(pts[good], new_pts[good])
    return M


def camera_homography(prev, cur, ignore=None):
    """Like camera_motion, but the exact model for a camera that pans, tilts and zooms.

    Returns a 3x3 matrix (a "homography"). It also captures the slight
    perspective change as the camera swings, which matters when every frame
    is mapped back to one fixed view. Many tracked points + MAGSAC (a robust
    voting method) keep moving players and fans from spoiling the estimate.

    During fast, blurry swings a homography can lock onto the wrong points, so
    we also compute the simpler shift+zoom+rotation estimate and keep whichever
    actually lines the two pictures up better.
    """
    mask = None if ignore is None else (ignore == 0).astype(np.uint8)
    pts = cv2.goodFeaturesToTrack(prev, 1500, 0.005, 10, mask=mask)
    new_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None, winSize=(31, 31), maxLevel=4)
    good = status.ravel() == 1
    H, _ = cv2.findHomography(pts[good], new_pts[good], cv2.USAC_MAGSAC, 3.0)
    A, _ = cv2.estimateAffinePartial2D(pts[good], new_pts[good])
    choices = [M for M in (H, None if A is None else np.vstack([A, [0, 0, 1]])) if M is not None]
    return min(choices, key=lambda M: alignment_error(prev, cur, M, mask))


def alignment_error(prev, cur, M, mask=None):
    """Average brightness difference after moving prev by M (lower = better lined up)."""
    small = 0.25
    S = np.diag([small, small, 1.0])
    M_small = S @ M @ np.linalg.inv(S)
    a = cv2.resize(prev, None, fx=small, fy=small)
    b = cv2.resize(cur, None, fx=small, fy=small)
    diff = cv2.absdiff(cv2.warpPerspective(a, M_small, b.shape[::-1]), b)
    if mask is not None:
        diff = diff[cv2.resize(mask, b.shape[::-1]) > 0]
    return float(diff.mean())


def warp(image, M, size):
    """Move an image by a camera motion (2x3 affine or 3x3 homography)."""
    if M.shape == (3, 3):
        return cv2.warpPerspective(image, M, size)
    return cv2.warpAffine(image, M, size)


def invert(M):
    """The opposite camera motion."""
    if M.shape == (3, 3):
        return np.linalg.inv(M)
    return cv2.invertAffineTransform(M)


def moving_mask(grays, motions, ignore, start, end, thresh=35, min_hits=2):
    """Pixels that moved (after camera correction) in at least min_hits frames of start..end."""
    h, w = grays[0].shape
    hits = np.zeros((h, w), np.uint8)
    for i in range(max(1, start), min(end, len(grays) - 1) + 1):
        diff = cv2.absdiff(grays[i], warp(grays[i - 1], motions[i], (w, h)))
        hits += (diff > thresh).astype(np.uint8)
    mask = np.where(hits >= min_hits, 255, 0).astype(np.uint8)
    mask[ignore > 0] = 0
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9)))


def find_candidates(prev, cur, M, ignore):
    """Line prev up with cur, subtract, and return centers of small blobs."""
    h, w = cur.shape
    diff = cv2.absdiff(cur, cv2.warpAffine(prev, M, (w, h)))
    _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    mask[ignore > 0] = 0
    # Ignore the edge strip the warp left empty (it looks like fake motion)
    covered = cv2.warpAffine(np.full((h, w), 255, np.uint8), M, (w, h))
    mask[cv2.erode(covered, np.ones((31, 31))) == 0] = 0
    mask[int(h * 0.75):, :] = 0  # broadcast graphics live at the bottom; the ball is up high
    n, _, stats, centers = cv2.connectedComponentsWithStats(mask)
    return [tuple(centers[k]) for k in range(1, n) if 8 <= stats[k, cv2.CC_STAT_AREA] <= 400]


def specks_now(frames, grays, motions, i, center, radius, exact=False):
    """Small moving things near center in frame i (grass-green and white ones skipped).

    Normal mode: pixels that changed since the previous frame. Good for finding
    the ball, but it shows a moving ball twice (where it was and where it is).
    exact=True: pixels that differ from BOTH the previous and the next frame,
    which is only where the ball is now. Good for pinning down its position.
    """
    h, w = grays[0].shape
    if not 0 < i < len(grays) - 1:
        return []
    x0, y0 = int(max(center[0] - radius, 0)), int(max(center[1] - radius, 0))
    x1, y1 = int(min(center[0] + radius, w)), int(min(center[1] + radius, h))
    if x1 <= x0 or y1 <= y0:
        return []
    before = warp(grays[i - 1], motions[i], (w, h))[y0:y1, x0:x1]
    after = warp(grays[i + 1], invert(motions[i + 1]), (w, h))[y0:y1, x0:x1]
    now = grays[i][y0:y1, x0:x1].astype(int)
    moved = np.abs(now - before) > 12
    if exact:
        moved &= np.abs(now - after) > 12
    hsv = cv2.cvtColor(frames[i][y0:y1, x0:x1], cv2.COLOR_BGR2HSV).astype(int)
    green = (hsv[..., 0] > 30) & (hsv[..., 0] < 90) & (hsv[..., 1] > 50)
    white = (hsv[..., 1] < 40) & (hsv[..., 2] > 170)
    n, _, stats, centers = cv2.connectedComponentsWithStats((moved & ~green & ~white).astype(np.uint8))
    return [centers[k] + (x0, y0) for k in range(1, n) if 3 <= stats[k, cv2.CC_STAT_AREA] <= 300]


def nearest(specks, point):
    return min(specks, key=lambda c: np.hypot(c[0] - point[0], c[1] - point[1]), default=None)
