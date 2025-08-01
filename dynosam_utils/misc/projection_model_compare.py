# import numpy as np
# import matplotlib.pyplot as plt
# import gtsam
# from gtsam import Cal3_S2, Point3, Pose3, Rot3, Values
# from gtsam import LevenbergMarquardtOptimizer, NonlinearFactorGraph
# from gtsam import PinholeCameraCal3_S2, symbol
# from gtsam.symbol_shorthand import X, L

# np.random.seed(42)

# def generate_trajectory(num_poses):
#     radius = 10.0
#     angles = np.linspace(0, 2 * np.pi, num_poses, endpoint=False)
#     return [Pose3(Rot3.Rz(theta), Point3(radius * np.cos(theta), radius * np.sin(theta), 0.0))
#             for theta in angles]

# def generate_points(num_points):
#     return [Point3(np.random.uniform(-5, 5),
#                    np.random.uniform(-5, 5),
#                    np.random.uniform(5, 10)) for _ in range(num_points)]

# def simulate_measurements(poses, points, K, noise_sigma, outlier_ratio):
#     measurements = {}
#     for i, pose in enumerate(poses):
#         cam = PinholeCameraCal3_S2(pose, K)
#         for j, point in enumerate(points):
#             try:
#                 z = cam.project(point)
#                 z += np.random.normal(0, noise_sigma, 2)
#                 if np.random.rand() < outlier_ratio:
#                     z += np.random.normal(0, 20 * noise_sigma, 2)
#                 measurements[(i, j)] = z
#             except RuntimeError:
#                 continue
#     return measurements

# def build_graph_projection(poses, points, measurements, K, noise_model):
#     graph = NonlinearFactorGraph()
#     initial = Values()

#     for i, pose in enumerate(poses):
#         # Add initial guess with noise
#         noise_translation = np.random.normal(0, 0.5, 3)
#         noisy_pose = Pose3(pose.rotation(), pose.translation() + noise_translation)
#         initial.insert(X(i), noisy_pose)

#     for j, point in enumerate(points):
#         initial.insert(L(j), point)

#     for (i, j), z in measurements.items():
#         factor = gtsam.GenericProjectionFactorCal3_S2(
#             gtsam.Point2(z), noise_model, X(i), L(j), K)
#         graph.add(factor)

#     return graph, initial

# def plot_trajectories(gt_poses, init_values, opt_values, label):
#     gt = np.array([pose.translation() for pose in gt_poses])
#     init = np.array([init_values.atPose3(X(i)).translation() for i in range(len(gt_poses))])
#     opt = np.array([opt_values.atPose3(X(i)).translation() for i in range(len(gt_poses))])

#     plt.plot(gt[:, 0], gt[:, 1], 'k-', label='Ground Truth')
#     plt.plot(init[:, 0], init[:, 1], 'rx--', label='Initial')
#     plt.plot(opt[:, 0], opt[:, 1], 'go-', label='Optimized')
#     plt.title(label)
#     plt.axis('equal')
#     plt.legend()

# # Parameters
# num_poses = 10
# num_points = 20
# noise_sigma = 1.0
# outlier_ratio = 0.2
# K = Cal3_S2(500, 500, 0, 320, 240)

# # Setup
# gt_poses = generate_trajectory(num_poses)
# points = generate_points(num_points)
# measurements = simulate_measurements(gt_poses, points, K, noise_sigma, outlier_ratio)

# # Noise models
# base_noise = gtsam.noiseModel.Isotropic.Sigma(2, noise_sigma)
# robust_noise = gtsam.noiseModel.Robust.Create(
#     gtsam.noiseModel.mEstimator.Huber.Create(1.345), base_noise)

# # Build graphs
# graph_gaussian, init_gaussian = build_graph_projection(gt_poses, points, measurements, K, base_noise)
# graph_robust, init_robust = build_graph_projection(gt_poses, points, measurements, K, robust_noise)

# # Optimize
# opt_gaussian = LevenbergMarquardtOptimizer(graph_gaussian, init_gaussian).optimize()
# opt_robust = LevenbergMarquardtOptimizer(graph_robust, init_robust).optimize()

# # Plot
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plot_trajectories(gt_poses, init_gaussian, opt_gaussian, "Gaussian Noise Model")
# plt.subplot(1, 2, 2)
# plot_trajectories(gt_poses, init_robust, opt_robust, "Robust Noise Model (Huber)")
# plt.tight_layout()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
import gtsam
from gtsam import symbol
from gtsam.noiseModel import Base as SharedNoiseModel, Diagonal


# Setup
np.random.seed(42)

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


num_poses = 10
num_points = 20
pixel_noise_sigma = 0.2
point_noise_sigma = 0.1
fx, fy, s, u0, v0 = 500, 500, 0, 320, 240
K = gtsam.Cal3_S2(fx, fy, s, u0, v0)
outlier_ratio = 0.2

# Ground truth trajectory (along x-axis)
poses_gt = [gtsam.Pose3(gtsam.Rot3(), np.array([i, i * 0.3, 0])) for i in range(num_poses)]
initial_estimates = gtsam.Values()
initial_estimates_ptp = gtsam.Values()

# 3D points in front of camera
points_gt = [np.array([np.random.uniform(0, num_poses), np.random.uniform(-1, 1), np.random.uniform(3, 5)])
             for _ in range(num_points)]

# Noise models
measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, pixel_noise_sigma)
robust_noise = gtsam.noiseModel.Robust.Create(
    gtsam.noiseModel.mEstimator.Huber.Create(1.345),
    measurement_noise
)
ptp_measurement_noise = gtsam.noiseModel.Isotropic.Sigma(3, point_noise_sigma)

# Factor graphs
graph_proj = gtsam.NonlinearFactorGraph()
graph_ptp = gtsam.NonlinearFactorGraph()

# Add priors
pose0_key = symbol('x', 0)
prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-4]*6))
graph_proj.add(gtsam.PriorFactorPose3(pose0_key, poses_gt[0], prior_noise))
graph_ptp.add(gtsam.PriorFactorPose3(pose0_key, poses_gt[0], prior_noise))

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
        np.equal(camera.backproject(proj, depth), point)

        depth_noisy = depth + np.random.normal(0, point_noise_sigma, 1)
        # depth_noisy = depth
        noisy_point_local = camera_local.backproject(z, depth_noisy)



        # NOW DO STEREO!!

        # point_local = pose.transformTo(point)

        # if np.random.rand() < outlier_ratio:
        #     z += np.random.uniform(-50, 50, 2)  # simulate a non-Gaussian outlier

        # Add GenericProjectionFactor
        factor = gtsam.GenericProjectionFactorCal3_S2(
            proj, measurement_noise, pose_key, point_key, K
        )
        graph_proj.add(factor)

        # Add PointToPoseFactor (uses known 3D point)
        ptp_factor = pose_to_point_factor(ptp_measurement_noise, pose_key, point_key, noisy_point_local)


        graph_ptp.add(ptp_factor)

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

# Extract poses
def extract_trajectory(values):
    return np.array([values.atPose3(symbol('x', i)).translation() for i in range(num_poses)])

traj_gt = np.array([p.translation() for p in poses_gt])
traj_proj = extract_trajectory(result_proj)
traj_ptp = extract_trajectory(result_ptp)
traj_init = extract_trajectory(initial_estimates)

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


fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the trajectories
ax.plot(traj_gt[:, 0], traj_gt[:, 1], traj_gt[:, 2], 'k-', label='Ground Truth')
ax.plot(traj_init[:, 0], traj_init[:, 1], traj_init[:, 2], 'y--', label='Initial')
ax.plot(traj_proj[:, 0], traj_proj[:, 1], traj_proj[:, 2], 'b-', label='GenericProjection')
ax.plot(traj_ptp[:, 0], traj_ptp[:, 1], traj_ptp[:, 2], 'g--', label='PointToPose')

# Axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Trajectory Comparison')

# Decorations
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()
