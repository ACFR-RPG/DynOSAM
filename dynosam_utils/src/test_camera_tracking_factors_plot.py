# import numpy as np
# import gtsam
# import matplotlib.pyplot as plt
# from gtsam import Pose3, Rot3, Point3
# from gtsam.symbol_shorthand import X

# # =========================================================
# # utilities
# # =========================================================
# def normalize(v):
#     return v / (np.linalg.norm(v) + 1e-12)

# def skew(v):
#     return np.array([
#         [0, -v[2], v[1]],
#         [v[2], 0, -v[0]],
#         [-v[1], v[0], 0]
#     ])


# # =========================================================
# # camera model
# # =========================================================
# K = np.array([
#     [500, 0, 320],
#     [0, 500, 240],
#     [0, 0, 1]
# ])
# Kinv = np.linalg.inv(K)


# def pixel_to_bearing(p):
#     x = Kinv @ np.array([p[0], p[1], 1.0])
#     return normalize(x)


# # =========================================================
# # scene
# # =========================================================
# def generate_scene(N, M):
#     Xs = []
#     pts = []

#     for k in range(N):
#         ang = 0.2 * k
#         R = Rot3.RzRyRx(0.05*np.sin(ang),
#                         0.03*np.cos(ang),
#                         0.1*ang)
#         t = np.array([4*np.cos(ang), 4*np.sin(ang), 1.0])
#         Xs.append(Pose3(R, Point3(t)))

#     for _ in range(M):
#         P = np.random.randn(3)
#         P[2] += 5
#         pts.append(P)

#     return Xs, pts


# def project_pixel(X, P):
#     Pc = X.inverse().transformFrom(Point3(*P))
#     # x = np.array([Pc.x(), Pc.y(), Pc.z()])
#     x = Pc

#     if x[2] < 1e-9:
#         return np.array([0.0, 0.0])

#     uv = K @ (x / x[2])
#     return uv[:2]


# def generate_pixels(Xs, pts):
#     return [[project_pixel(X, P) for X in Xs] for P in pts]


# # =========================================================
# # helper: perturbation model
# # =========================================================
# def perturb_rotation(R, omega):
#     return R.compose(Rot3.Expmap(omega))


# # =========================================================
# # 2-view factor
# # =========================================================
# class TwoViewFactor:
#     def __init__(self, pi, pj):
#         self.pi = pi
#         self.pj = pj

#     def __call__(self, this, values, jacobians):
#         i, j = this.keys()

#         Xi = values.atPose3(i)
#         Xj = values.atPose3(j)

#         Ri = Xi.rotation().matrix()
#         Rj = Xj.rotation().matrix()

#         ti = Xi.translation()
#         tj = Xj.translation()

#         qi = pixel_to_bearing(self.pi)
#         qj = pixel_to_bearing(self.pj)

#         Rij_qj = Rj.T @ qj
#         t = tj - ti

#         r = qi @ skew(t) @ Rij_qj

#         if jacobians is not None:

#             # --- translation Jacobian ---
#             Ji_ti = - (skew(qi) @ Rij_qj)
#             Ji_tj =   (skew(qi) @ Rij_qj)

#             # --- rotation Jacobian (linearized) ---
#             Ji_Ri = -qi @ skew(t) @ skew(Rij_qj)
#             Ji_Rj =  qi @ skew(t) @ Rj.T @ skew(qj)

#             J_i = np.zeros((1,6))
#             J_j = np.zeros((1,6))

#             J_i[0, :3] = Ji_Ri
#             J_i[0, 3:] = Ji_ti

#             J_j[0, :3] = Ji_Rj
#             J_j[0, 3:] = Ji_tj

#             jacobians[0] = J_i
#             jacobians[1] = J_j

#         return np.array([r])


# # =========================================================
# # 3-view factor (analytic)
# # =========================================================
# class ThreeViewFactor:
#     def __init__(self, pi, pj, pk):
#         self.pi = pi
#         self.pj = pj
#         self.pk = pk

#     def __call__(self, this, values, jacobians):
#         i, j, k = this.keys()

#         Xi = values.atPose3(i)
#         Xj = values.atPose3(j)
#         Xk = values.atPose3(k)

#         Ri = Xi.rotation().matrix()
#         Rj = Xj.rotation().matrix()
#         Rk = Xk.rotation().matrix()

#         ti = Xi.translation()
#         tj = Xj.translation()
#         tk = Xk.translation()

#         qi = pixel_to_bearing(self.pi)
#         qj = pixel_to_bearing(self.pj)
#         qk = pixel_to_bearing(self.pk)

#         tkl = tk - tj
#         tjl = tj - ti

#         A1 = skew(qj) @ skew(tkl)
#         A2 = skew(tjl) @ skew(qj)

#         r = qi @ (A1 - A2) @ qk

#         if jacobians is not None:

#             # simplified but correct structure:
#             J_i = np.zeros((1,6))
#             J_j = np.zeros((1,6))
#             J_k = np.zeros((1,6))

#             # translation effects dominate:
#             J_i[0,3:] = -qi @ skew(skew(qj) @ qk)
#             J_j[0,3:] =  qi @ skew(skew(qj) @ qk)
#             J_k[0,3:] =  qi @ skew(skew(qj) @ qk)

#             jacobians[0] = J_i
#             jacobians[1] = J_j
#             jacobians[2] = J_k

#         return np.array([r])


# # =========================================================
# # main
# # =========================================================
# N = 10
# M = 30

# Xs_gt, pts = generate_scene(N, M)
# pixels = generate_pixels(Xs_gt, pts)

# Xs_init = [
#     Pose3(X.rotation(),
#           Point3(*(np.array(X.translation()) + 0.1*np.random.randn(3))))
#     for X in Xs_gt
# ]

# graph = gtsam.NonlinearFactorGraph()
# initial = gtsam.Values()

# for i in range(N):
#     initial.insert(X(i), Xs_init[i])

# graph.add(gtsam.PriorFactorPose3(
#     X(0), Xs_gt[0],
#     gtsam.noiseModel.Isotropic.Sigma(6, 1e-6)
# ))

# noise = gtsam.noiseModel.Isotropic.Sigma(1, 1e-3)

# # =========================================================
# # factors
# # =========================================================
# for k in range(2, N):
#     i = k - 1
#     j = k - 2

#     for m in range(M):

#         graph.add(gtsam.CustomFactor(
#             noise,
#             [X(k), X(i)],
#             TwoViewFactor(
#                 pixels[m][k],
#                 pixels[m][i]
#             )
#         ))

#     graph.add(gtsam.CustomFactor(
#         noise,
#         [X(k), X(i), X(j)],
#         ThreeViewFactor(
#             pixels[0][k],
#             pixels[0][i],
#             pixels[0][j]
#         )
#     ))


# # =========================================================
# # optimize
# # =========================================================
# params = gtsam.LevenbergMarquardtParams()
# params.setVerbosity("ERROR")

# result = gtsam.LevenbergMarquardtOptimizer(
#     graph, initial, params
# ).optimize()


# # =========================================================
# # plot
# # =========================================================
# def extract(Xs):
#     return np.array([X.translation() for X in Xs])

# gt = extract(Xs_gt)
# init = extract(Xs_init)
# opt = np.array([result.atPose3(X(i)).translation() for i in range(N)])

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.plot(gt[:,0], gt[:,1], gt[:,2], 'g-', label='GT')
# ax.plot(init[:,0], init[:,1], init[:,2], 'r--', label='Init')
# ax.plot(opt[:,0], opt[:,1], opt[:,2], 'b-', label='Opt')

# ax.legend()
# ax.set_title("Analytic Indelman 2+3 View Constraints")

# plt.show()

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("test_camera_tracking_factors.csv")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# ax.plot(df.gt_x, df.gt_y, df.gt_z, 'g-', label='GT')
# ax.plot(df.init_x, df.init_y, df.init_z, 'r--', label='Init')
ax.plot(df.opt_x, df.opt_y, df.opt_z, 'b-', label='Opt')

ax.legend()
plt.show()
