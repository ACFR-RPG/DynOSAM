# import numpy as np
# import matplotlib.pyplot as plt

# # -----------------------------
# # Utility functions
# # -----------------------------

# def hat(w):
#     """Skew symmetric matrix"""
#     return np.array([
#         [0, -w[2], w[1]],
#         [w[2], 0, -w[0]],
#         [-w[1], w[0], 0]
#     ])

# def exp_so3(w):
#     """Exponential map for SO(3)"""
#     theta = np.linalg.norm(w)
#     if theta < 1e-8:
#         return np.eye(3)
#     K = hat(w / theta)
#     return np.eye(3) + np.sin(theta)*K + (1 - np.cos(theta))*(K @ K)

# def make_transform(R, t):
#     T = np.eye(4)
#     T[:3, :3] = R
#     T[:3, 3] = t
#     return T

# def invert_transform(T):
#     R = T[:3, :3]
#     t = T[:3, 3]
#     T_inv = np.eye(4)
#     T_inv[:3, :3] = R.T
#     T_inv[:3, 3] = -R.T @ t
#     return T_inv

# def compose(T1, T2):
#     return T1 @ T2

# def add_noise_to_pose(T, rot_sigma=0.05, trans_sigma=0.05):
#     """Add Gaussian noise in se(3)"""
#     w = np.random.randn(3) * rot_sigma
#     t = np.random.randn(3) * trans_sigma
#     R_noise = exp_so3(w)
#     T_noise = make_transform(R_noise, t)
#     return compose(T, T_noise)

# # -----------------------------
# # Generate ground truth
# # -----------------------------

# N = 20

# X_gt = []  # camera poses
# L_gt = []  # object poses

# for k in range(N):
#     # Camera moves in a circle
#     angle = 0.2 * k
#     R = exp_so3(np.array([0, 0, angle]))
#     t = np.array([np.cos(angle)*5, np.sin(angle)*5, 1.0])
#     X_gt.append(make_transform(R, t))

#     # Object moves forward + slight rotation
#     R_obj = exp_so3(np.array([0.05*k, 0.02*k, 0]))
#     t_obj = np.array([0.5*k, 0.2*k, 0.0])
#     L_gt.append(make_transform(R_obj, t_obj))

# # -----------------------------
# # Compute ground truth motions
# # -----------------------------

# H_gt = []
# for k in range(1, N):
#     H = compose(L_gt[k], invert_transform(L_gt[k-1]))
#     H_gt.append(H)


# # object motion as observed by gt camera
# H_gt_cam = []
# for k in range(1, N):
#     H = compose(invert_transform(X_gt[k-1]), compose(H_gt[k-1], X_gt[k-1]))
#     H_gt_cam.append(H)

# X_noisy = [add_noise_to_pose(X) for X in X_gt]

# # object motion as observed by noisy camera
# H_est_cam = []
# for k in range(1, N):
#     H = compose(invert_transform(X_noisy[k-1]), compose(H_gt[k-1], X_noisy[k-1]))
#     H_est_cam.append(H)

# print(H_gt_cam[4])
# print(H_est_cam[4])

# # -----------------------------
# # Simulate observations (object in camera frame)
# # Z_k = X_k^{-1} * L_k
# # -----------------------------

# # Z_gt = []
# # for k in range(N):
# #     Z = compose(invert_transform(X_gt[k]), L_gt[k])
# #     Z_gt.append(Z)

# # # -----------------------------
# # # Add noise to camera poses
# # # -----------------------------


# # # Recompute noisy observations
# # Z_noisy = []
# # for k in range(N):
# #     Z = compose(invert_transform(X_noisy[k]), L_gt[k])
# #     Z_noisy.append(Z)

# # -----------------------------
# # Estimate motion from noisy observations
# # H_est = (X_k * Z_k) * (X_{k-1} * Z_{k-1})^{-1}
# # -----------------------------

# # H_est = []
# # for k in range(1, N):
# #     Lk_est = compose(X_noisy[k], Z_noisy[k])
# #     Lkm1_est = compose(X_noisy[k-1], Z_noisy[k-1])
# #     H = compose(Lk_est, invert_transform(Lkm1_est))
# #     H_est.append(H)
# # H_est = []
# # for k in range(1, N):
# #     Lk_est = compose(X_noisy[k], Z_gt[k])
# #     Lkm1_est = compose(X_noisy[k-1], Z_gt[k-1])
# #     H = compose(Lk_est, invert_transform(Lkm1_est))
# #     H_est.append(H)

# # -----------------------------
# # Reconstruct object trajectory from H_est
# # -----------------------------

# # L_recon = [L_gt[0]]  # anchor at true initial pose

# # for k in range(1, N):
# #     L_new = compose(H_est[k-1], L_recon[k-1])
# #     L_recon.append(L_new)


# L_recon = [L_gt[0]]  # anchor at true initial pose

# for k in range(1, N):
#     H_w = compose(X_noisy[k-1], compose(H_gt_cam[k-1], invert_transform(X_noisy[k-1])))
#     L_new = compose(H_w, L_recon[k-1])
#     L_recon.append(L_new)

# # -----------------------------
# # Extract positions for plotting
# # -----------------------------

# def extract_positions(T_list):
#     return np.array([T[:3, 3] for T in T_list])

# X_gt_pos = extract_positions(X_gt)
# X_noisy_pos = extract_positions(X_noisy)
# L_gt_pos = extract_positions(L_gt)
# L_recon_pos = extract_positions(L_recon)

# # -----------------------------
# # Plot
# # -----------------------------

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.plot(X_gt_pos[:,0], X_gt_pos[:,1], X_gt_pos[:,2], label="Camera GT")
# ax.plot(X_noisy_pos[:,0], X_noisy_pos[:,1], X_noisy_pos[:,2], '--', label="Camera Noisy")

# ax.plot(L_gt_pos[:,0], L_gt_pos[:,1], L_gt_pos[:,2], label="Object GT")
# ax.plot(L_recon_pos[:,0], L_recon_pos[:,1], L_recon_pos[:,2], '--', label="Object Reconstructed")

# ax.scatter(L_gt_pos[:,0], L_gt_pos[:,1], L_gt_pos[:,2])
# ax.scatter(L_recon_pos[:,0], L_recon_pos[:,1], L_recon_pos[:,2])

# ax.set_title("Rigid Body Motion Test")
# ax.legend()
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")

# plt.show()

import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# Lie utilities
# =========================================================

def hat(w):
    return np.array([
        [0, -w[2], w[1]],
        [w[2], 0, -w[0]],
        [-w[1], w[0], 0]
    ])

def exp_so3(w):
    theta = np.linalg.norm(w)
    if theta < 1e-8:
        return np.eye(3)
    K = hat(w / theta)
    return np.eye(3) + np.sin(theta)*K + (1 - np.cos(theta))*(K @ K)

def make_transform(R, t):
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = t
    return T

def invert(T):
    R = T[:3,:3]
    t = T[:3,3]
    Tinv = np.eye(4)
    Tinv[:3,:3] = R.T
    Tinv[:3,3] = -R.T @ t
    return Tinv

def compose(A, B):
    return A @ B

# =========================================================
# Motion Models
# =========================================================

class SE3Motion:
    """Standard rigid motion"""
    @staticmethod
    def between(T2, T1):
        return compose(T2, invert(T1))

    @staticmethod
    def apply(H, T):
        return compose(H, T)


class DirectProductMotion:
    """
    SO(3) x R^3 (decoupled translation)
    Translation is NOT rotated by R
    """

    @staticmethod
    def between(T2, T1):
        R2, t2 = T2[:3,:3], T2[:3,3]
        R1, t1 = T1[:3,:3], T1[:3,3]

        R = R2 @ R1.T
        t = t2 - t1  # <-- key difference

        return make_transform(R, t)

    @staticmethod
    def apply(H, T):
        R_h, t_h = H[:3,:3], H[:3,3]
        R, t = T[:3,:3], T[:3,3]

        R_new = R_h @ R
        t_new = t + t_h  # <-- no rotation

        return make_transform(R_new, t_new)

# =========================================================
# Noise
# =========================================================

def add_noise(T, rot_sigma=0.05, trans_sigma=0.05):
    w = np.random.randn(3) * rot_sigma
    t = np.random.randn(3) * trans_sigma
    Rn = exp_so3(w)
    Tn = make_transform(Rn, t)
    return compose(T, Tn)

# =========================================================
# Scenario generation
# =========================================================

def generate_scene(N):
    X, L = [], []

    for k in range(N):
        # Camera
        ang = 0.2 * k
        Rc = exp_so3([0,0,ang])
        tc = np.array([5*np.cos(ang), 5*np.sin(ang), 1.0])
        X.append(make_transform(Rc, tc))

        # Object
        Ro = exp_so3([0.05*k, 0.02*k, 0])
        to = np.array([0.5*k, 0.2*k, 0])
        L.append(make_transform(Ro, to))

    return X, L

# =========================================================
# Frame conversion
# =========================================================

def motion_in_camera_frame(H_world, X):
    """H_cam = X^{-1} H_world X"""
    return compose(invert(X), compose(H_world, X))

def motion_back_to_world(H_cam, X):
    """H_world = X H_cam X^{-1}"""
    return compose(X, compose(H_cam, invert(X)))

# =========================================================
# Experiment runner
# =========================================================

def run_experiment(motion_model, N=20):

    # --- generate GT ---
    X_gt, L_gt = generate_scene(N)

    # --- GT motion in world ---
    H_gt = [motion_model.between(L_gt[k], L_gt[k-1])
            for k in range(1, N)]

    # --- project into camera frame ---
    H_cam_gt = [motion_in_camera_frame(H_gt[k-1], X_gt[k-1])
                for k in range(1, N)]

    # --- noisy camera ---
    X_noisy = [add_noise(X) for X in X_gt]

    # --- interpret motion using noisy camera ---
    H_cam_est = [motion_in_camera_frame(H_gt[k-1], X_noisy[k-1])
                 for k in range(1, N)]

    # --- reconstruct in world ---
    L_rec = [L_gt[0]]

    for k in range(1, N):
        H_world = motion_back_to_world(H_cam_gt[k-1], X_noisy[k-1])
        L_new = motion_model.apply(H_world, L_rec[k-1])
        L_rec.append(L_new)

    return X_gt, X_noisy, L_gt, L_rec

# =========================================================
# Plot
# =========================================================

def plot_results(X_gt, X_noisy, L_gt, L_rec, title):

    def pos(Ts):
        return np.array([T[:3,3] for T in Ts])

    Xg, Xn = pos(X_gt), pos(X_noisy)
    Lg, Lr = pos(L_gt), pos(L_rec)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(Xg[:,0], Xg[:,1], Xg[:,2], label="Camera GT")
    ax.plot(Xn[:,0], Xn[:,1], Xn[:,2], '--', label="Camera Noisy")

    ax.plot(Lg[:,0], Lg[:,1], Lg[:,2], label="Object GT")
    ax.plot(Lr[:,0], Lr[:,1], Lr[:,2], '--', label="Object Recon")

    ax.set_title(title)
    ax.legend()
    plt.show()

# =========================================================
# Run both models
# =========================================================

np.random.seed(0)

# --- SE3 ---
X_gt, X_noisy, L_gt, L_rec = run_experiment(SE3Motion)
plot_results(X_gt, X_noisy, L_gt, L_rec, "SE(3) Motion")

# --- Direct product ---
X_gt, X_noisy, L_gt, L_rec = run_experiment(DirectProductMotion)
plot_results(X_gt, X_noisy, L_gt, L_rec, "SO(3) x R^3 Motion")
