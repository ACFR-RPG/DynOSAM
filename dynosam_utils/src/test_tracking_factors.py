import numpy as np
import gtsam
import matplotlib.pyplot as plt
from gtsam import Pose3, Rot3, Point3
from gtsam.symbol_shorthand import X

K = np.array([
    [500, 0, 320],
    [0, 500, 240],
    [0, 0, 1]
])

Kinv = np.linalg.inv(K)

# =========================================================
# Geometry helpers
# =========================================================
def normalize(v):
    return v / (np.linalg.norm(v) + 1e-12)

def cross(a, b):
    return np.cross(a, b)


def exp_so3(w):
    theta = np.linalg.norm(w)
    if theta < 1e-9:
        return np.eye(3)

    k = w / theta
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])

    return np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K@K)


def make_transform(R, t):
    return Pose3(Rot3(R), Point3(t))


def numerical_jacobian_pose3(func, pose, eps=1e-6):
    J = np.zeros((1, 6), order="F")

    for i in range(6):
        d = np.zeros(6)
        d[i] = eps

        J[0, i] = (
            func(pose.retract(d)) -
            func(pose.retract(-d))
        ) / (2 * eps)

    return J


# =========================================================
# 2-view factor (correct epipolar-style constraint)
# =========================================================
class TwoViewResidual:
    def __init__(self, qk, ql, Xk, Xl):
        self.qk = normalize(qk)
        self.ql = normalize(ql)
        self.Xk = Xk
        self.Xl = Xl

    def compute_error(self, Lk, Ll):

        Tk = self.Xk.inverse().compose(Lk)
        Tl = self.Xl.inverse().compose(Ll)

        Tkl = Tk.between(Tl)

        R = Tkl.rotation().matrix()
        t = Tkl.translation()

        ql_hat = normalize(R @ self.ql)

        # epipolar constraint
        return self.qk @ cross(t, ql_hat)

    def __call__(self, this, values, jacobians):
        k, l = this.keys()

        Lk = values.atPose3(k)
        Ll = values.atPose3(l)

        err = self.compute_error(Lk, Ll)

        if jacobians is not None:
            jacobians[0] = numerical_jacobian_pose3(
                lambda x: self.compute_error(x, Ll), Lk
            )
            jacobians[1] = numerical_jacobian_pose3(
                lambda x: self.compute_error(Lk, x), Ll
            )

        return np.array([err])


# =========================================================
# 3-view factor (FIXED consistent formulation)
# =========================================================
class ThreeViewResidual:
    """
    FIX:
    - all quantities expressed in frame k
    - no mixing of independent translation terms
    - uses relative transforms only
    """

    def __init__(self, qk, ql, qm, Xk, Xl, Xm):
        self.qk = normalize(qk)
        self.ql = normalize(ql)
        self.qm = normalize(qm)
        self.Xk = Xk
        self.Xl = Xl
        self.Xm = Xm

    def compute_error(self, Lk, Ll, Lm):

        Tk = self.Xk.inverse().compose(Lk)
        Tl = self.Xl.inverse().compose(Ll)
        Tm = self.Xm.inverse().compose(Lm)

        Tkl = Tk.between(Tl)
        Tkm = Tk.between(Lm)

        Rkl = Tkl.rotation().matrix()
        Rkm = Tkm.rotation().matrix()

        tkl = Tkl.translation()
        tkm = Tkm.translation()

        ql_k = normalize(Rkl @ self.ql)
        qm_k = normalize(Rkm @ self.qm)

        # corrected symmetric consistency constraint
        term1 = self.qk @ cross(tkl, ql_k)
        term2 = self.qk @ cross(tkm, qm_k)

        return term1 - term2

    def __call__(self, this, values, jacobians):
        k, l, m = this.keys()

        Lk = values.atPose3(k)
        Ll = values.atPose3(l)
        Lm = values.atPose3(m)

        err = self.compute_error(Lk, Ll, Lm)

        if jacobians is not None:
            jacobians[0] = numerical_jacobian_pose3(
                lambda x: self.compute_error(x, Ll, Lm), Lk
            )
            jacobians[1] = numerical_jacobian_pose3(
                lambda x: self.compute_error(Lk, x, Lm), Ll
            )
            jacobians[2] = numerical_jacobian_pose3(
                lambda x: self.compute_error(Lk, Ll, x), Lm
            )

        return np.array([err])


# =========================================================
# Scene generation (better excitation)
# =========================================================
def generate_scene(N):
    X = []
    L = []

    for k in range(N):

        ang = 0.25 * k

        # camera: spiral + height variation
        Rc = exp_so3([0.05*np.sin(ang), 0.05*np.cos(ang), 0.1*ang])
        tc = np.array([
            4*np.cos(ang),
            4*np.sin(ang),
            1.0 + 0.3*np.sin(0.3*ang)
        ])

        X.append(make_transform(Rc, tc))

        # object: independent 3D motion (less collinear!)
        Ro = exp_so3([
            0.03*k,
            0.025*k,
            0.02*np.sin(0.3*k)
        ])

        to = np.array([
            0.3*k,
            0.15*k*np.cos(0.2*k),
            0.2*np.sin(0.2*k)
        ])

        L.append(make_transform(Ro, to))

    return X, L


# =========================================================
# Points
# =========================================================
def generate_points(n):
    pts = []
    for _ in range(n):
        p = np.random.randn(3)
        p[2] += 3.0
        pts.append(p)
    return pts


# =========================================================
# Measurements (CORRECTED: normalize!)
# =========================================================
def generate_measurements(Ls, Xs, points):
    qs = []

    for P in points:
        qs_j = []

        for L, X in zip(Ls, Xs):
            p_cam = X.inverse().transformFrom(
                L.transformFrom(Point3(*P))
            )
            qs_j.append(normalize(p_cam))
            # qs_j.append(normalize(np.array([p_cam.x(), p_cam.y(), p_cam.z()])))

        qs.append(qs_j)

    return qs


# =========================================================
# Main
# =========================================================
N = 8
num_points = 40

Xs, Ls_gt = generate_scene(N)
points = generate_points(num_points)

qs_all = generate_measurements(Ls_gt, Xs, points)

Ls_init = [make_transform(
    L.rotation().matrix(),
    L.translation() + 0.05*np.random.randn(3)
) for L in Ls_gt]


graph = gtsam.NonlinearFactorGraph()
initial = gtsam.Values()

for i in range(N):
    initial.insert(X(i), Ls_init[i])


# prior (fix gauge)
graph.add(gtsam.PriorFactorPose3(
    X(0),
    Ls_gt[0],
    gtsam.noiseModel.Isotropic.Sigma(6, 1e-6)
))

noise = gtsam.noiseModel.Isotropic.Sigma(1, 1e-2)

# =========================================================
# Factors
# =========================================================
for k in range(2, N):
    l = k - 1
    m = k - 2

    for j in range(num_points):

        graph.add(gtsam.CustomFactor(
            noise,
            [X(k), X(l)],
            TwoViewResidual(
                qs_all[j][k],
                qs_all[j][l],
                Xs[k],
                Xs[l]
            )
        ))

        graph.add(gtsam.CustomFactor(
            noise,
            [X(k), X(l), X(m)],
            ThreeViewResidual(
                qs_all[j][k],
                qs_all[j][l],
                qs_all[j][m],
                Xs[k],
                Xs[l],
                Xs[m]
            )
        ))


# =========================================================
# Optimize
# =========================================================
params = gtsam.LevenbergMarquardtParams()
params.setVerbosity("ERROR")

result = gtsam.LevenbergMarquardtOptimizer(
    graph,
    initial,
    params
).optimize()


# =========================================================
# Plot
# =========================================================
def extract(Ls):
    return np.array([L.translation() for L in Ls])


gt = extract(Ls_gt)
init = extract(Ls_init)
opt = extract([result.atPose3(X(i)) for i in range(N)])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(gt[:,0], gt[:,1], gt[:,2], 'g-', label='GT')
ax.plot(init[:,0], init[:,1], init[:,2], 'r--', label='Init')
ax.plot(opt[:,0], opt[:,1], opt[:,2], 'b-', label='Optimized')

ax.legend()
ax.set_title("Corrected Multi-view SE(3) Object Estimation")

plt.show()
