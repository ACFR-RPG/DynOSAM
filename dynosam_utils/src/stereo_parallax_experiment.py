# #!/usr/bin/env python3
# """
# Synthetic rectified-stereo experiment.

# Projects 3D points into a left/right pinhole camera pair, back-projects the
# pixels into unit bearing rays, and plots disparity and the angle between rays
# against point depth.

# Dependencies:
#     python3 -m pip install numpy matplotlib

# Coordinate convention:
#     - Left camera frame is the world frame.
#     - Both cameras look along +Z and have the same orientation (rectified).
#     - Right camera centre is (+baseline, 0, 0) in the left-camera frame.
#     - T_left_right maps points from the right camera frame into the left frame.
# """

# import numpy as np
# import matplotlib.pyplot as plt


# # ----------------------------- Configuration -----------------------------

# FX = 500.0                 # focal length in pixels
# FY = 500.0
# CX = 320.0                 # principal point in pixels
# CY = 240.0
# BASELINE_M = 0.12          # distance between camera centres, metres
# DEPTHS_M = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0])
# OUTPUT_PLOT = "stereo_parallax_vs_depth.png"


# # ---------------------------- Camera functions ---------------------------

# K = np.array([
#     [FX, 0.0, CX],
#     [0.0, FY, CY],
#     [0.0, 0.0, 1.0],
# ], dtype=float)

# # Point coordinates in the left camera frame.
# # X and Y remain fixed while Z varies.
# POINT_X_M = 0.5
# POINT_Y_M = 0.2


# def project_point(point_camera: np.ndarray, K: np.ndarray) -> np.ndarray:
#     """Project a 3D point in camera coordinates to pixel (u, v)."""
#     point_camera = np.asarray(point_camera, dtype=float)
#     if point_camera.shape != (3,):
#         raise ValueError("point_camera must have shape (3,)")
#     if point_camera[2] <= 0.0:
#         raise ValueError("Point must be in front of the camera (Z > 0).")

#     pixel_h = K @ point_camera
#     return pixel_h[:2] / pixel_h[2]


# def backproject_pixel(pixel: np.ndarray, K: np.ndarray) -> np.ndarray:
#     """Back-project pixel (u, v) into a unit bearing ray in camera coordinates."""
#     pixel = np.asarray(pixel, dtype=float)
#     if pixel.shape != (2,):
#         raise ValueError("pixel must have shape (2,)")

#     pixel_h = np.array([pixel[0], pixel[1], 1.0], dtype=float)
#     ray = np.linalg.solve(K, pixel_h)
#     return ray / np.linalg.norm(ray)


# def angle_between_unit_rays(ray_a: np.ndarray, ray_b: np.ndarray) -> float:
#     """Return the angle between unit rays, in radians, robustly."""
#     cosine = np.clip(np.dot(ray_a, ray_b), -1.0, 1.0)
#     return float(np.arccos(cosine))


# # ------------------------------- Experiment ------------------------------

# def main() -> None:
#     # Rectified rig: R_left_right = I. The right camera centre is
#     # (+BASELINE_M, 0, 0) in the left frame.
#     R_left_right = np.eye(3)
#     t_left_right = np.array([BASELINE_M, 0.0, 0.0])

#     rows = []

#     for depth in DEPTHS_M:
#         # Test point is on the left camera's optical axis.
#         # point_left = np.array([120.0, 60.0, depth])
#         point_left = np.array([POINT_X_M, POINT_Y_M, depth])

#         # T_left_right maps right-frame points into the left frame:
#         # p_left = R_left_right @ p_right + t_left_right.
#         # Therefore p_right = R_left_right.T @ (p_left - t_left_right).
#         point_right = R_left_right.T @ (point_left - t_left_right)

#         # Project the same 3D point into each camera.
#         pixel_left = project_point(point_left, K)
#         pixel_right = project_point(point_right, K)

#         # Back-project both image points into unit bearing rays.
#         ray_left = backproject_pixel(pixel_left, K)
#         ray_right = backproject_pixel(pixel_right, K)

#         # Express the right ray direction in the left-camera frame.
#         # Translation is NOT applied to a direction vector.
#         ray_right_in_left = R_left_right @ ray_right
#         ray_right_in_left /= np.linalg.norm(ray_right_in_left)

#         disparity_px = float(pixel_left[0] - pixel_right[0])
#         parallax_rad = angle_between_unit_rays(ray_left, ray_right_in_left)

#         rows.append({
#             "depth_m": float(depth),
#             "u_left_px": float(pixel_left[0]),
#             "u_right_px": float(pixel_right[0]),
#             "disparity_px": disparity_px,
#             "parallax_deg": float(np.degrees(parallax_rad)),
#             "ray_left": ray_left,
#             "ray_right_in_left": ray_right_in_left,
#         })

#     # Print results in a readable table.
#     print(
#         f"{'Z (m)':>8} {'u_left (px)':>13} {'u_right (px)':>14} "
#         f"{'disparity (px)':>16} {'ray angle (deg)':>17}"
#     )
#     print("-" * 73)
#     for row in rows:
#         print(
#             f"{row['depth_m']:8.3f} "
#             f"{row['u_left_px']:13.3f} "
#             f"{row['u_right_px']:14.3f} "
#             f"{row['disparity_px']:16.3f} "
#             f"{row['parallax_deg']:17.6f}"
#         )

#     example = min(rows, key=lambda row: abs(row["depth_m"] - 5.0))
#     print(f"\nRay details at Z = {example['depth_m']:.3f} m:")
#     print(f"  left pixel:              ({example['u_left_px']:.3f}, {CY:.3f})")
#     print(f"  right pixel:             ({example['u_right_px']:.3f}, {CY:.3f})")
#     print(f"  left unit ray:           {example['ray_left']}")
#     print(f"  right unit ray in left:  {example['ray_right_in_left']}")
#     print(f"  disparity:               {example['disparity_px']:.6f} px")
#     print(f"  angle:                   {example['parallax_deg']:.6f} degrees")

#     # For this setup, these analytic expressions should match the numerical
#     # projection/back-projection results: disparity = fx * baseline / Z and
#     # angle = atan(baseline / Z).
#     for row in rows:
#         left_direction = point_left / np.linalg.norm(point_left)
#         right_direction = point_left - np.array([BASELINE_M, 0.0, 0.0])
#         right_direction /= np.linalg.norm(right_direction)

#         expected_angle = np.degrees(
#             np.arccos(np.clip(left_direction.dot(right_direction), -1.0, 1.0))
#         )

#         expected_angle = np.degrees(
#             np.arccos(np.clip(left_direction.dot(right_direction), -1.0, 1.0))
#         )
#         assert np.isclose(
#             np.degrees(parallax_rad),
#             expected_angle,
#             atol=1e-7,
#         )

#     depths = np.array([row["depth_m"] for row in rows])
#     disparities = np.array([row["disparity_px"] for row in rows])
#     angles = np.array([row["parallax_deg"] for row in rows])

#     # One figure with two vertically stacked plots, sharing the depth axis.
#     fig, (ax_disparity, ax_angle) = plt.subplots(
#         2, 1, figsize=(8.5, 7.0), sharex=True, constrained_layout=True
#     )

#     ax_disparity.plot(depths, disparities, marker="o")
#     ax_disparity.set_ylabel("Disparity (pixels)")
#     ax_disparity.set_title("Rectified stereo: disparity and ray angle vs depth")
#     ax_disparity.grid(True, which="both", alpha=0.35)

#     ax_angle.plot(depths, angles, marker="s")
#     ax_angle.set_xlabel("Point depth Z (m)")
#     ax_angle.set_ylabel("Angle between rays (degrees)")
#     ax_angle.grid(True, which="both", alpha=0.35)

#     # Log depth axis makes the full range of depths easier to inspect.
#     # ax_angle.set_xscale("log")

#     # fig.savefig(OUTPUT_PLOT, dpi=160)
#     # print(f"\nSaved plot to: {OUTPUT_PLOT}")
#     plt.show()


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3

"""
Rectified stereo simulation starting from matched image pixels.

The simulation:
  1. Generates 3D points at different depths and image locations.
  2. Projects each point into left and right rectified stereo images.
  3. Computes disparity from matched pixel coordinates.
  4. Estimates depth from Z = fx * baseline / disparity.
  5. Computes unit bearing vectors from the pixels.
  6. Computes the geometric stereo parallax angle.
  7. Checks the estimated depth and parallax against ground truth.

Dependencies:
    pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASELINE_M = 0.05       # Stereo baseline [m]

FX_PX = 500.0           # Horizontal focal length [pixels]
FY_PX = 500.0           # Vertical focal length [pixels]
CX_PX = 320.0           # Principal point x [pixels]
CY_PX = 240.0           # Principal point y [pixels]

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# Ground-truth depths [m].
DEPTHS_M = np.array([
    0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0
])

# Normalized horizontal and vertical positions relative to the left camera.
# These need not be zero: points can lie anywhere in the field of view.
# X/Z = 0.5 means the point is half a depth unit to the right of the optical
# axis; Y/Z = 0.2 means it is above/below the axis depending on convention.
NORMALIZED_X = 0.5
NORMALIZED_Y = 0.2


# ---------------------------------------------------------------------------
# Camera geometry
# ---------------------------------------------------------------------------

def project_point(point_camera, fx, fy, cx, cy):
    """Project a 3D point in a camera frame to pixel coordinates."""
    x, y, z = point_camera

    if z <= 0.0:
        raise ValueError("Point must be in front of the camera.")

    u = fx * x / z + cx
    v = fy * y / z + cy

    return np.array([u, v], dtype=float)


def backproject_pixel(pixel, fx, fy, cx, cy):
    """
    Convert a pixel into a unit bearing vector in the camera frame.

    Assumes an undistorted pinhole image.
    """
    u, v = pixel

    x = (u - cx) / fx
    y = (v - cy) / fy

    bearing = np.array([x, y, 1.0], dtype=float)
    return bearing / np.linalg.norm(bearing)


def compute_depth_from_disparity(u_left, u_right, fx, baseline):
    """Estimate optical-axis depth from rectified stereo disparity."""
    disparity = u_left - u_right

    if disparity <= 0.0:
        raise ValueError(
            f"Invalid disparity {disparity:.6f}; expected positive disparity."
        )

    return fx * baseline / disparity


def compute_parallax_angle_from_pixels(
    pixel_left,
    pixel_right,
    depth,
    fx,
    fy,
    cx,
    cy,
    baseline,
):
    """
    Compute the geometric angle between the rays from the two camera centres
    to the same triangulated point.

    Both ray directions are expressed in the left camera frame.

    Assumptions:
      - Rectified stereo: both cameras have identical orientation.
      - Right camera centre is [baseline, 0, 0] in the left camera frame.
      - The supplied depth is optical-axis depth in the left camera frame.
    """
    # Unit bearing directions from each pixel.
    bearing_left = backproject_pixel(pixel_left, fx, fy, cx, cy)
    bearing_right = backproject_pixel(pixel_right, fx, fy, cx, cy)

    # Reconstruct the 3D point in the left camera frame.
    # The bearing is unit length, so scale it to the known optical-axis depth.
    point_left = (depth / bearing_left[2]) * bearing_left

    # Right camera centre, expressed in the left camera frame.
    camera_right_in_left = np.array([baseline, 0.0, 0.0])

    # Direction from the right camera centre to the 3D point.
    direction_right = point_left - camera_right_in_left
    direction_right /= np.linalg.norm(direction_right)

    # Direction from the left camera centre to the point.
    direction_left = bearing_left

    cos_angle = np.clip(
        np.dot(direction_left, direction_right),
        -1.0,
        1.0,
    )
    print(f"cos_angle {cos_angle}")
    return np.arccos(cos_angle)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def main():
    results = []

    for depth_gt in DEPTHS_M:
        # Generate a 3D point in the left camera frame.
        # Its position is not restricted to the optical axis.
        point_left = np.array([
            NORMALIZED_X * depth_gt,
            NORMALIZED_Y * depth_gt,
            depth_gt,
        ])

        # Rectified stereo camera geometry:
        # the right camera centre is translated +baseline along X in the
        # left camera frame, with no relative rotation.
        camera_right_in_left = np.array([BASELINE_M, 0.0, 0.0])
        point_right = point_left - camera_right_in_left

        # Project the 3D point into both images.
        pixel_left = project_point(
            point_left, FX_PX, FY_PX, CX_PX, CY_PX
        )
        pixel_right = project_point(
            point_right, FX_PX, FY_PX, CX_PX, CY_PX
        )

        # Optionally skip points outside either image.
        in_left_image = (
            0.0 <= pixel_left[0] < IMAGE_WIDTH
            and 0.0 <= pixel_left[1] < IMAGE_HEIGHT
        )
        in_right_image = (
            0.0 <= pixel_right[0] < IMAGE_WIDTH
            and 0.0 <= pixel_right[1] < IMAGE_HEIGHT
        )

        if not (in_left_image and in_right_image):
            print(
                f"Skipping depth {depth_gt:.2f} m: "
                "point is outside the image."
            )
            continue

        # In a real system, these pixel coordinates would come from stereo
        # matching. Here they are perfect, noise-free synthetic matches.
        u_left, v_left = pixel_left
        u_right, v_right = pixel_right

        disparity = u_left - u_right


        # Estimate depth from the matched pixels.
        depth_est = compute_depth_from_disparity(
            u_left,
            u_right,
            FX_PX,
            BASELINE_M,
        )

        # Estimate the geometric stereo parallax angle from the pixels.
        parallax_rad = compute_parallax_angle_from_pixels(
            pixel_left,
            pixel_right,
            depth_est,
            FX_PX,
            FY_PX,
            CX_PX,
            CY_PX,
            BASELINE_M,
        )
        print(f"parallax_rad {parallax_rad}")
        sigma = 1.0 / (0.5 * (FX_PX+FY_PX))
        print(f"dispartiy {disparity}")
        print(f"depth_gt {depth_gt}")

        import math
        # print(math.cos(6.0 * sigma))

        parallax_deg = np.degrees(parallax_rad)

        # Ground-truth geometric parallax angle.
        ray_left_gt = point_left / np.linalg.norm(point_left)
        ray_right_gt = point_right / np.linalg.norm(point_right)

        cos_gt = np.clip(np.dot(ray_left_gt, ray_right_gt), -1.0, 1.0)
        parallax_gt_deg = np.degrees(np.arccos(cos_gt))

        results.append({
            "depth_gt_m": depth_gt,
            "depth_est_m": depth_est,
            "disparity_px": disparity,
            "u_left_px": u_left,
            "v_left_px": v_left,
            "u_right_px": u_right,
            "v_right_px": v_right,
            "parallax_deg": parallax_deg,
            "parallax_gt_deg": parallax_gt_deg,
        })

    if not results:
        raise RuntimeError("No points were inside both stereo images.")

    # -----------------------------------------------------------------------
    # Print results
    # -----------------------------------------------------------------------

    print("\nRectified stereo simulation starting from pixel matches")
    print(f"Baseline: {BASELINE_M:.3f} m")
    print(f"Intrinsics: fx={FX_PX:.1f}, fy={FY_PX:.1f}, "
          f"cx={CX_PX:.1f}, cy={CY_PX:.1f}")
    print(f"Normalized point position: "
          f"X/Z={NORMALIZED_X:.2f}, Y/Z={NORMALIZED_Y:.2f}\n")

    header = (
        f"{'Z_gt (m)':>9} "
        f"{'Z_est (m)':>10} "
        f"{'Disparity (px)':>15} "
        f"{'Parallax (deg)':>15} "
        f"{'GT angle (deg)':>15}"
    )

    print(header)
    print("-" * len(header))

    for row in results:
        print(
            f"{row['depth_gt_m']:9.3f} "
            f"{row['depth_est_m']:10.3f} "
            f"{row['disparity_px']:15.4f} "
            f"{row['parallax_deg']:15.6f} "
            f"{row['parallax_gt_deg']:15.6f}"
        )

    # -----------------------------------------------------------------------
    # Verify depth and parallax calculations.
    # -----------------------------------------------------------------------

    for row in results:
        assert np.isclose(
            row["depth_est_m"],
            row["depth_gt_m"],
            rtol=1e-9,
            atol=1e-9,
        ), "Estimated depth does not match ground truth."

        assert np.isclose(
            row["parallax_deg"],
            row["parallax_gt_deg"],
            rtol=1e-7,
            atol=1e-7,
        ), "Estimated parallax does not match ground truth."

    print("\nAll depth and parallax checks passed.")

    # -----------------------------------------------------------------------
    # Plot results
    # -----------------------------------------------------------------------

    depths_gt = np.array([r["depth_gt_m"] for r in results])
    depths_est = np.array([r["depth_est_m"] for r in results])
    disparities = np.array([r["disparity_px"] for r in results])
    parallax = np.array([r["parallax_deg"] for r in results])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].plot(depths_gt, depths_gt, "k--", label="Ground truth")
    axes[0].plot(depths_gt, depths_est, "o-", label="From disparity")
    axes[0].set_xlabel("Ground-truth depth (m)")
    axes[0].set_ylabel("Estimated depth (m)")
    axes[0].set_title("Depth estimation")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(depths_gt, disparities, "o-")
    axes[1].set_xlabel("Ground-truth depth (m)")
    axes[1].set_ylabel("Disparity (pixels)")
    axes[1].set_title("Stereo disparity")
    axes[1].grid(True)

    axes[2].plot(depths_gt, parallax, "o-")
    axes[2].set_xlabel("Ground-truth depth (m)")
    axes[2].set_ylabel("Parallax angle (degrees)")
    axes[2].set_title("Geometric stereo parallax")
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
