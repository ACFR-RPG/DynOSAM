import numpy as np
import matplotlib.pyplot as plt
import gtsam
from gtsam import symbol
from gtsam.noiseModel import Base as SharedNoiseModel, Diagonal, Gaussian
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Setup
np.random.seed(42)

def plot_covariance_ellipsoid(mean, cov, ax, n_std=1.0, color='r', alpha=0.3):
    """
    Plot a 3D covariance ellipsoid.

    Args:
        mean: 3D position (center of ellipsoid)
        cov: 3x3 covariance matrix
        ax: matplotlib 3D axis
        n_std: Number of standard deviations
        color: Color of the ellipsoid
        alpha: Transparency
    """
    # Generate points on a unit sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    sphere = np.stack((x, y, z), axis=-1)  # shape (100, 100, 3)

    # Eigen-decomposition of covariance
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Scale and rotate the sphere
    radii = n_std * np.sqrt(eigvals)
    T = eigvecs @ np.diag(radii)
    ellipsoid = sphere @ T.T + mean  # shape (100, 100, 3)

    ax.plot_surface(ellipsoid[:, :, 0],
                    ellipsoid[:, :, 1],
                    ellipsoid[:, :, 2],
                    rstride=4, cstride=4, color=color, alpha=alpha)


def pose_to_point_factor(
    model: SharedNoiseModel,
    pose_key: int,
    point_key: int,
    measured_point_local: gtsam.Point3
) -> gtsam.NonlinearFactor:
    """
    Create a custom Pose-to-Point factor.
    Args:
        model: Noise model for the factor.
        pose_key: Key for the pose variable.
        point_key: Key for the 3D point variable.
        measured_point_local: The observed 3D point in the local frame of the pose.

    Returns:
        CustomFactor: The created Pose-to-Point factor.
    """

    def error_func(this: gtsam.CustomFactor, v: gtsam.Values, H: list[np.ndarray]) -> np.ndarray:
        # Retrieve the pose and point from the values container
        pose = v.atPose3(this.keys()[0])  # Pose in world frame
        point = v.atPoint3(this.keys()[1])  # 3D point in world frame

        # Transform the point from the world frame to the pose's local frame
        Dpose = np.zeros((3, 6), order="F")  # Jacobian w.r.t. pose
        Dpoint = np.zeros((3, 3), order="F")  # Jacobian w.r.t. point
        local_point = pose.transformTo(point, Dpose, Dpoint)

        # If Jacobians are not requested, return just the error
        if H is None:
            return local_point - measured_point_local

        # Compute Jacobians if needed
        result = local_point - measured_point_local
        H[0] = Dpose  # Jacobian w.r.t. pose
        H[1] = Dpoint  # Jacobian w.r.t. point

        return result

    # Create and return the CustomFactor
    return gtsam.CustomFactor(model, gtsam.KeyVector([pose_key, point_key]), error_func)

def construct_3d_projection_covariance(K: gtsam.Cal3_S2, u: float, v: float, depth: float, sigma_u:float, sigma_v:float, sigma_d:float):
    fx = K.fx()
    fy = K.fy()
    cx = K.px()
    cy = K.py()

    sigma_u2 = math.pow(sigma_u, 2)
    sigma_v2 = math.pow(sigma_v, 2)
    sigma_d2 = math.pow(sigma_d, 2)

    fx2 = math.pow(fx, 2)
    fy2 = math.pow(fy, 2)

    # print(depth)
    depth2 = depth ** 2

    # what is this!!
    sigma_uv = sigma_u

    # main diagonal
    # sigma_x2 = ((sigma_u2 * sigma_d2) + (sigma_u2 * depth2) + math.pow((u - cx), 2)* sigma_d2)/fx2
    # sigma_y2 = ((sigma_v2 * sigma_d2) + (sigma_v2 * depth2) + math.pow((v - cx), 2)* sigma_d2)/fy2
    # sigma_z2 = depth2

    sigma_x2 = ((sigma_u2 + depth2)*(sigma_d2 + (u**2)) - ((u**2) * depth2) + ((cx**2) * sigma_d2))/(fx2)
    sigma_y2 = ((sigma_v2 + depth2)*(sigma_d2 + (v**2)) - ((v**2) * depth2) + ((cy**2) * sigma_d2))/(fy2)
    sigma_z2 = sigma_d2

    # sigma_x2 = (((u - cx)**2 * sigma_d2) + (depth2 + sigma_u2) + (sigma_u2 * sigma_d2)) / fx2
    # sigma_y2 = (((v - cy)**2 * sigma_d2) + (depth2 + sigma_v2) + (sigma_v2 * sigma_d2)) / fy2

    # sigma_xy = (((u - cx) * (v - cy) * sigma_d2) + (depth2 + sigma_d2) * sigma_uv) / (fx * fy)
    # sigma_xz = (sigma_d2 * (u - cx)) / fx
    # sigma_yz = (sigma_d2 * (v - cy)) / fy


    # off diagonal terms
    sigma_xz = sigma_d2 * (u - cx)/fx
    sigma_yz = (sigma_d2 * (v - cy))/fy
    sigma_xy = (sigma_d2 * (u - cx) * (v - cy))/(fx * fy)

    # print(u - cx)
    # print("sigma_xz", sigma_xz)
    # print("sigma_yz", sigma_yz)
    # print("sigma_xy", sigma_xy)



    # covariance_matrix = np.array([[sigma_z2, sigma_xz, sigma_yz],
    #                               [sigma_xz, sigma_x2, sigma_xy],
    #                               [sigma_yz, sigma_xy, sigma_y2]])

    # covariance_matrix = np.array([[sigma_x2, sigma_xy, sigma_xz],
    #                               [sigma_xy, sigma_y2, sigma_yz],
    #                               [sigma_xz, sigma_yz, sigma_z2]])

    # covariance_matrix = 0.5 * (covariance_matrix + covariance_matrix.T)
    # covariance_matrix += 1e-6 * np.eye(3)


    # Jacobian J of backprojection w.r.t. (u, v, d)
    J = np.array([
        [depth / fx,        0, (u - cx) / fx],
        [0,        depth / fy, (v - cy) / fy],
        [0,             0, 1]
    ])

    # Covariance in pixel-depth space
    Sigma_uvd = np.diag([sigma_u**2, sigma_v**2, sigma_d**2])

    print(Sigma_uvd)

    # Propagate to 3D covariance
    Sigma_3d = J @ Sigma_uvd @ J.T
    covariance_matrix = Sigma_3d

    # regularization = 1e-5
    # covariance_matrix += regularization * np.eye()

    # print("Cov", covariance_matrix)

    import sys

    print(covariance_matrix)
    # sys.exit(0)

    # covariance_matrix = np.array([[sigma_x2, 0, 0],
    #                               [0, sigma_y2, 0],
    #                               [0, 0, sigma_z2]])

    # return gtsam.Gaussian()
    return Gaussian.Covariance(covariance_matrix, False)


num_poses = 15
num_points = 50
# pixel_noise_sigma = 0.2
# point_noise_sigma = 0.01
pixel_noise_sigma = 2.0
point_noise_sigma = 0.4
# fx, fy, s, u0, v0 = 500, 500, 0, 320, 240
fx, fy, s, u0, v0 = 500, 500, 0, 320, 240


K = gtsam.Cal3_S2(fx, fy, s, u0, v0)
outlier_ratio = 0.4

# Ground truth trajectory (along x-axis)
poses_gt = [gtsam.Pose3(gtsam.Rot3(), np.array([i, i * 0.3, i*0.02])) for i in range(num_poses)]
# poses_gt = []
# for i in range(num_poses):
#     # Example: rotation increases with pose index
#     roll  = 0.01 * i
#     pitch = 0.005 * i
#     yaw   = 0.02 * i
#     rot = gtsam.Rot3.RzRyRx(roll, pitch, yaw)

#     # Translation
#     t = np.array([i, i * 0.3, i * 0.02])

#     poses_gt.append(gtsam.Pose3(rot, t))
initial_estimates = gtsam.Values()
initial_estimates_ptp = gtsam.Values()

# 3D points in front of camera
points_gt = [np.array([np.random.uniform(0, num_poses), np.random.uniform(-1, 1), np.random.uniform(3, 5)])
             for _ in range(num_points)]

k_huber = 0.01 #1.345
# Noise models
pixel_measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, pixel_noise_sigma)
robust_pixel_noise = gtsam.noiseModel.Robust.Create(
    gtsam.noiseModel.mEstimator.Huber.Create(k_huber),
    pixel_measurement_noise
)
ptp_measurement_noise = gtsam.noiseModel.Isotropic.Sigma(3, point_noise_sigma)
robust_point_noise = gtsam.noiseModel.Robust.Create(
    gtsam.noiseModel.mEstimator.Huber.Create(k_huber),
    ptp_measurement_noise
)

# Factor graphs
graph_proj = gtsam.NonlinearFactorGraph()
graph_ptp = gtsam.NonlinearFactorGraph()
graph_ptp_3d_model = gtsam.NonlinearFactorGraph()

graph_proj_robust = gtsam.NonlinearFactorGraph()
graph_ptp_robust = gtsam.NonlinearFactorGraph()
graph_ptp_3d_model_robust = gtsam.NonlinearFactorGraph()


# Add priors
pose0_key = symbol('x', 0)
prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-4]*6))
graph_proj.add(gtsam.PriorFactorPose3(pose0_key, poses_gt[0], prior_noise))
graph_ptp.add(gtsam.PriorFactorPose3(pose0_key, poses_gt[0], prior_noise))
graph_ptp_3d_model.add(gtsam.PriorFactorPose3(pose0_key, poses_gt[0], prior_noise))

graph_proj_robust.add(gtsam.PriorFactorPose3(pose0_key, poses_gt[0], prior_noise))
graph_ptp_robust.add(gtsam.PriorFactorPose3(pose0_key, poses_gt[0], prior_noise))
graph_ptp_3d_model_robust.add(gtsam.PriorFactorPose3(pose0_key, poses_gt[0], prior_noise))

plot_3d_measurement_cov = False

is_outlier = []


# Simulate observations and build both graphs
for i, pose in enumerate(poses_gt):
    pose_key = symbol('x', i)
    initial_estimates.insert(pose_key, pose.retract(np.random.randn(6) * 0.1))
    # initial_estimates_ptp.insert(pose_key, pose.retract(np.random.randn(6) * 0.1))

    camera = gtsam.PinholeCameraCal3_S2(pose, K)
    camera_local = gtsam.PinholeCameraCal3_S2(gtsam.Pose3.Identity(), K)

    for j, point in enumerate(points_gt):
        point_key = symbol('p', j)

        depth = point[2]
        proj = camera.project(point)
        z = proj + np.random.normal(0, pixel_noise_sigma, 2)

        # test!!
        # test_point = camera.backproject(proj, depth)
        # assert np.array_equal(test_point, point), f"{test_point} != {point}"

        point_in_local = pose.transformTo(point)

        # print(f"Depth {depth}")
        depth_noisy = depth + np.random.normal(0, point_noise_sigma, 1)[0]
        # print(f"depth_noisy {depth_noisy}")

        # depth_noisy = depth
        noisy_point_local = camera_local.backproject(z, depth_noisy)

        test_noisy_point_local_x = (z[0] - K.px()) * depth_noisy / K.fx()
        test_noisy_point_local_y = (z[1] - K.py()) * depth_noisy / K.fy()
        test_noisy_point_local_z = depth_noisy

        test_noisy_point_local = np.array([test_noisy_point_local_x, test_noisy_point_local_y, test_noisy_point_local_z])


        if np.random.rand() < outlier_ratio:
            z += np.random.uniform(-50, 50, 2)  # simulate a non-Gaussian outlier
            is_outlier.append(True)
        else:
            is_outlier.append(False)

        # Add GenericProjectionFactor
        factor = gtsam.GenericProjectionFactorCal3_S2(
            z, pixel_measurement_noise, pose_key, point_key, K
        )
        graph_proj.add(factor)

        graph_proj_robust.add(gtsam.GenericProjectionFactorCal3_S2(
            z, robust_pixel_noise, pose_key, point_key, K
        ))

        # Add PointToPoseFactor (uses known 3D point)
        ptp_factor = pose_to_point_factor(ptp_measurement_noise, pose_key, point_key, noisy_point_local)
        graph_ptp.add(ptp_factor)

        graph_ptp_robust.add(pose_to_point_factor(robust_point_noise, pose_key, point_key, noisy_point_local))

        # Add 3d pose to point factor using 3D covariance model
        covariance_3d_model = construct_3d_projection_covariance(K, z[0], z[1], depth_noisy, pixel_noise_sigma, pixel_noise_sigma, point_noise_sigma)
        print(covariance_3d_model)
        ptp_factor_3d_model = pose_to_point_factor(covariance_3d_model, pose_key, point_key, noisy_point_local)
        graph_ptp_3d_model.add(ptp_factor_3d_model)

        covariance_3d_model_robust = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(k_huber),
            covariance_3d_model
        )
        graph_ptp_3d_model_robust.add(pose_to_point_factor(covariance_3d_model_robust, pose_key, point_key, noisy_point_local)
)

# # Add point priors (to anchor points in projection graph)
for j, point in enumerate(points_gt):
    point_key = symbol('p', j)
    initial_estimates.insert(point_key, point)
    # graph_proj.add(gtsam.PriorFactorPoint3(point_key, point, gtsam.noiseModel.Isotropic.Sigma(3, 1.0)))

# Optimize both
params = gtsam.LevenbergMarquardtParams()
params.setVerbosity("ERROR")

result_proj = gtsam.LevenbergMarquardtOptimizer(graph_proj, initial_estimates, params).optimize()
result_ptp = gtsam.LevenbergMarquardtOptimizer(graph_ptp, initial_estimates, params).optimize()
result_ptp_3d_model = gtsam.LevenbergMarquardtOptimizer(graph_ptp_3d_model, initial_estimates, params).optimize()

result_proj_robust = gtsam.LevenbergMarquardtOptimizer(graph_proj_robust, initial_estimates, params).optimize()
result_ptp_robust = gtsam.LevenbergMarquardtOptimizer(graph_ptp_robust, initial_estimates, params).optimize()
result_ptp_3d_model_robust = gtsam.LevenbergMarquardtOptimizer(graph_ptp_3d_model_robust, initial_estimates, params).optimize()


# Extract poses
def extract_trajectory(values):
    return np.array([values.atPose3(symbol('x', i)).translation() for i in range(num_poses)])

traj_gt = np.array([p.translation() for p in poses_gt])
traj_proj = extract_trajectory(result_proj)
traj_ptp = extract_trajectory(result_ptp)
traj_init = extract_trajectory(initial_estimates)
traj_ptp_3d_model = extract_trajectory(result_ptp_3d_model)

traj_proj_robust = extract_trajectory(result_proj_robust)
traj_ptp_robust = extract_trajectory(result_ptp_robust)
traj_ptp_3d_model_robust = extract_trajectory(result_ptp_3d_model_robust)


# # Plot
# fig = plt.figure(figsize=(10, 5))
# ax = fig.add_subplot(111)
# ax.plot(traj_gt[:, 0], traj_gt[:, 2], 'k-', label='Ground Truth')
# ax.plot(traj_init[:, 0], traj_init[:, 2], 'y--', label='Initial')
# ax.plot(traj_proj[:, 0], traj_proj[:, 2], 'b-', label='GenericProjection')
# ax.plot(traj_ptp[:, 0], traj_ptp[:, 2], 'g--', label='PointToPose')
# ax.set_xlabel('X')
# ax.set_ylabel('Z')
# ax.set_title('Trajectory Comparison')
# ax.legend()
# ax.grid()
# plt.tight_layout()
# plt.show()


fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the trajectories
ax.plot(traj_gt[:, 0], traj_gt[:, 1], traj_gt[:, 2], 'k-', label='Ground Truth')
ax.plot(traj_init[:, 0], traj_init[:, 1], traj_init[:, 2], 'y--', label='Initial')
ax.plot(traj_proj[:, 0], traj_proj[:, 1], traj_proj[:, 2], 'b-', label='GenericProjection')
ax.plot(traj_ptp[:, 0], traj_ptp[:, 1], traj_ptp[:, 2], 'g--', label='PointToPose')
ax.plot(traj_ptp_3d_model[:, 0], traj_ptp_3d_model[:, 1], traj_ptp_3d_model[:, 2], 'r--', label='PointToPose (3d cov model)')

ax.plot(traj_proj_robust[:, 0], traj_proj_robust[:, 1], traj_proj_robust[:, 2], 'b*-', label='GenericProjection (robust)')
ax.plot(traj_ptp_robust[:, 0], traj_ptp_robust[:, 1], traj_ptp_robust[:, 2], 'g*-', label='PointToPose (robust)')
ax.plot(traj_ptp_3d_model_robust[:, 0], traj_ptp_3d_model_robust[:, 1], traj_ptp_3d_model_robust[:, 2], 'r-*', label='PointToPose (3d cov model) (robust)')


# Axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# ax.set_title(r'3D Trajectory Comparison $\sigma_{\text{pixel}}$={}'.format(pixel_noise_sigma))
ax.set_title(rf'3D Trajectory Comparison $\sigma_{{\text{{pixel}}}}$ = {pixel_noise_sigma}, $\sigma_{{\text{{depth}}}}$ = {point_noise_sigma}, outlier ratio = {outlier_ratio}' )
# Decorations
ax.legend()
ax.grid(True)
fig.tight_layout()
# plt.tight_layout()
plt.show()
