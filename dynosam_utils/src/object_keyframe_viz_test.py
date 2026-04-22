import numpy as np
import cv2

np.random.seed(42)

# # -----------------------------
# # Style (paper-friendly)
# # -----------------------------
BG_COLOR = (255, 255, 255)   # white background
KF_COLOR = (60, 160, 60)     # green
CUR_COLOR = (60, 60, 200)    # blue/red
ELLIPSE_KF = (120, 200, 120)
ELLIPSE_CUR = (120, 120, 220)

TEXT_COLOR = (255, 255, 255)   # white text
TEXT_BG = (30, 30, 30)         # dark box behind text

FONT = cv2.FONT_HERSHEY_SIMPLEX

IMG_SIZE = 400

# -----------------------------
# Feature sampling
# -----------------------------
def sample_blob_points(center, radius, n=100):
    pts = []
    for _ in range(n):
        r = radius * np.sqrt(np.random.rand())
        theta = 2 * np.pi * np.random.rand()
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        pts.append([x, y])
    return np.array(pts, dtype=np.float32)

# -----------------------------
# Covariance-based metrics
# -----------------------------
def compute_covariance(pts):
    mean = np.mean(pts, axis=0)
    d = pts - mean
    cov = (d.T @ d) / len(pts)
    return cov, mean

def compute_coverage(pts_kf, pts_cur, chi2_threshold=5.991):
    mean_kf = np.mean(pts_kf, axis=0)
    d_kf = pts_kf - mean_kf
    cov_kf = (d_kf.T @ d_kf) / len(pts_kf)

    # regularization (important for near-degenerate cases)
    cov_kf += 1e-6 * np.eye(2)

    cov_inv = np.linalg.inv(cov_kf)

    # --- remove translation (important!)
    mean_cur = np.mean(pts_cur, axis=0)

    d = (pts_cur - mean_cur) - (pts_kf - mean_kf)

    # --- Mahalanobis distances
    d2 = np.sum(d @ cov_inv * d, axis=1)

    # --- threshold
    inside = np.sum(d2 < chi2_threshold)

    coverage = inside / len(d2)

    return coverage


def align_similarity(pts_kf, pts_cur):
    # translation
    mu_kf = np.mean(pts_kf, axis=0)
    mu_cur = np.mean(pts_cur, axis=0)

    K = pts_kf - mu_kf
    C = pts_cur - mu_cur

    # isotropic scale
    s_kf = np.sqrt(np.mean(np.sum(K**2, axis=1)))
    s_cur = np.sqrt(np.mean(np.sum(C**2, axis=1)))

    K = K * (s_cur / (s_kf + 1e-6))

    return K, C

def normalize_scale_only(pts_kf, pts_cur):
    """
    Removes only isotropic scale, keeps translation.
    """

    # compute scale from RMS spread
    def scale(pts):
        return np.sqrt(np.mean(np.sum(pts**2, axis=1)))

    s_kf = scale(pts_kf)
    s_cur = scale(pts_cur)

    pts_kf_norm = pts_kf / (s_kf + 1e-6)
    pts_cur_norm = pts_cur / (s_cur + 1e-6)

    return pts_kf_norm, pts_cur_norm

# between 0 and 1
def coverage_keyframe_support(pts_kf, pts_cur, radius=10.0):
    """
    Fraction of keyframe points that are still covered
    by current tracked points (after alignment).
    """

    if len(pts_kf) == 0 or len(pts_cur) == 0:
        return 0.0

    covered = 0

    for k in pts_kf:
        d = np.linalg.norm(pts_cur - k, axis=1)
        if np.min(d) < radius:
            covered += 1

    return covered / len(pts_kf)


def compute_metrics(pts_kf, pts_cur):
    cov_kf, mean_kf = compute_covariance(pts_kf)
    cov_cur, mean_cur = compute_covariance(pts_cur)

    eig_kf = np.linalg.eigvals(cov_kf)
    eig_cur = np.linalg.eigvals(cov_cur)

    eig_kf = np.sort(eig_kf)[::-1]
    eig_cur = np.sort(eig_cur)[::-1]

    # --- scale
    # beween 0 and 1 how close the scale is
    scale_ratio = np.sqrt(
        (eig_cur.sum()) / (eig_kf.sum() + 1e-6)
    )

    # --- shape
    # between 0 and 1 how close is the shape
    ratio_kf = eig_kf[1] / (eig_kf[0] + 1e-6)
    ratio_cur = eig_cur[1] / (eig_cur[0] + 1e-6)
    shape_score = min(ratio_cur / ratio_kf, ratio_kf / ratio_cur)

    coverage = coverage_keyframe_support(pts_kf, pts_cur)

    return scale_ratio, shape_score, coverage

# -----------------------------
# Visualization
# -----------------------------
def draw_scene(pts_kf, pts_cur, metrics, title):
    img = np.zeros((600, 600, 3), dtype=np.uint8)

    # draw keyframe
    for p in pts_kf:
        cv2.circle(img, tuple(p.astype(int)), 2, (0,255,0), -1)

    # draw current
    for p in pts_cur:
        cv2.circle(img, tuple(p.astype(int)), 2, (0,0,255), -1)

    scale, shape, coverage = metrics

    cv2.putText(img, f"{title}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.putText(img, f"scale: {scale:.2f}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.putText(img, f"shape: {shape:.2f}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.putText(img, f"coverage: {coverage:.2f}", (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    return img

# -----------------------------
# Drawing helpers
# -----------------------------
def draw_points(img, pts, color):
    for p in pts:
        cv2.circle(img, tuple(p.astype(int)), 2, color, -1, lineType=cv2.LINE_AA)

def draw_ellipse(img, cov, mean, color):
    eigvals, eigvecs = np.linalg.eig(cov)

    angle = np.degrees(np.arctan2(eigvecs[1,0], eigvecs[0,0]))
    axes = (
        int(np.sqrt(eigvals[0]) * 2),
        int(np.sqrt(eigvals[1]) * 2)
    )

    cv2.ellipse(img,
                tuple(mean.astype(int)),
                axes,
                angle,
                0, 360,
                color, 1, lineType=cv2.LINE_AA)

def draw_text_block(img, lines, origin=(10, 10)):
    x, y = origin
    line_height = 20
    padding = 5

    # compute box size
    width = 0
    for line in lines:
        (w, h), _ = cv2.getTextSize(line, FONT, 0.5, 1)
        width = max(width, w)

    height = line_height * len(lines)

    # draw filled rectangle
    cv2.rectangle(img,
                  (x - padding, y - padding),
                  (x + width + padding, y + height + padding),
                  TEXT_BG, -1)

    # draw text
    for i, line in enumerate(lines):
        cv2.putText(img,
                    line,
                    (x, y + (i+1)*line_height - 5),
                    FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)

def render_panel(title, pts_kf, pts_cur):
    canvas = np.full((IMG_SIZE, IMG_SIZE, 3), BG_COLOR, dtype=np.uint8)

    scale, shape, coverage = compute_metrics(pts_kf, pts_cur)

    # draw
    draw_points(canvas, pts_kf, KF_COLOR)
    draw_points(canvas, pts_cur, CUR_COLOR)

    cov_kf, mean_kf = compute_covariance(pts_kf)
    cov_cur, mean_cur = compute_covariance(pts_cur)

    draw_ellipse(canvas, cov_kf, mean_kf, ELLIPSE_KF)
    draw_ellipse(canvas, cov_cur, mean_cur, ELLIPSE_CUR)

    # text
    lines = [
        title,
        f"scale: {scale:.2f}",
        f"shape: {shape:.2f}",
        f"coverage: {coverage:.2f}"
    ]
    draw_text_block(canvas, lines)

    return canvas



# -----------------------------
# Test scenarios
# -----------------------------
def run_tests():
    center = np.array([IMG_SIZE/2, IMG_SIZE/2])
    pts_kf = sample_blob_points(center, 80, 120)

    tests = []

    # 1. Pure translation
    pts_trans = pts_kf + np.array([50, 0])
    tests.append(("Translation", pts_trans))

    # 2. Scale (toward camera)
    pts_scale = (pts_kf - center) * 1.5 + center
    tests.append(("Scale Up", pts_scale))

    # 3. Scale down
    pts_scale2 = (pts_kf - center) * 0.6 + center
    tests.append(("Scale Down", pts_scale2))

    # 4. Collapse (bad tracking)
    pts_collapse = pts_kf.copy()
    pts_collapse[:,1] = center[1]  # flatten to line
    tests.append(("Collapse", pts_collapse))

    # 5. Partial visibility
    mask = pts_kf[:,0] > center[0]
    pts_partial = pts_kf[mask]
    tests.append(("Partial", pts_partial))

    # 6. Random drift
    pts_drift = pts_kf + np.random.randn(*pts_kf.shape) * 30
    tests.append(("Drift", pts_drift))


    panels = []
    for name, pts_cur in tests:
        n = min(len(pts_kf), len(pts_cur))
        panels.append(render_panel(name, pts_kf[:n], pts_cur[:n]))

    row1 = np.hstack(panels[:3])
    row2 = np.hstack(panels[3:])

    grid = np.vstack([row1, row2])

    cv2.imshow("metrics_white_bg", grid)
    cv2.waitKey(0)

    # cv2.imwrite("metrics_visualization_white.png", grid)

if __name__ == "__main__":
    run_tests()

import numpy as np
import cv2
