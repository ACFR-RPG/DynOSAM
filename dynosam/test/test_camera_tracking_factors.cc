
#include <config_utilities/parsing/yaml.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/geometry/Cal3DS2.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/nonlinear/ExpressionFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/expressions.h>

#include <Eigen/Dense>
#include <fstream>
#include <random>
#include <thread>
#include <vector>

using namespace std;
using namespace gtsam;
using namespace gtsam::symbol_shorthand;

namespace dyno_testing {

// =========================================================
// utilities
// =========================================================
Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
  Eigen::Matrix3d S;
  S << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
  return S;
}

Eigen::Vector3d normalize(const Eigen::Vector3d& v) { return v / v.norm(); }

// =========================================================
// camera projection (pinhole normalized)
// =========================================================
Eigen::Vector3d pixelToBearing(const Eigen::Vector2d& p) {
  double fx = 500, fy = 500, cx = 320, cy = 240;

  Eigen::Vector3d x((p.x() - cx) / fx, (p.y() - cy) / fy, 1.0);
  return dyno_testing::normalize(x);
}

// =========================================================
// synthetic scene
// =========================================================
vector<Pose3> generateTrajectory(int N) {
  vector<Pose3> X;
  for (int k = 0; k < N; k++) {
    double ang = 0.2 * k;

    Rot3 R = Rot3::RzRyRx(0.05 * sin(ang), 0.03 * cos(ang), 0.1 * ang);

    Point3 t(4 * cos(ang), 4 * sin(ang), 1.0);
    X.emplace_back(R, t);
  }
  return X;
}

vector<Point3> generatePoints(int M) {
  std::mt19937 gen(0);
  std::normal_distribution<double> d(0.0, 1.0);

  vector<Point3> pts;
  for (int i = 0; i < M; i++) {
    pts.emplace_back(d(gen), d(gen), 5.0 + fabs(d(gen)));
  }
  return pts;
}

// =========================================================
// projection
// =========================================================
Eigen::Vector2d project(const Pose3& X, const Point3& P) {
  Point3 Pc = X.inverse().transformFrom(P);
  Eigen::Vector3d x(Pc.x(), Pc.y(), Pc.z());

  if (x.z() <= 1e-6) return Eigen::Vector2d(0, 0);

  double fx = 500, fy = 500, cx = 320, cy = 240;

  return {fx * x.x() / x.z() + cx, fy * x.y() / x.z() + cy};
}

// =========================================================
// 2-view factor
// g = q_i^T [t_ij]_x q_j
struct TwoViewFactor : gtsam::NoiseModelFactor2<Pose3, Pose3> {
  Eigen::Vector3d q_i_cam, q_j_cam;

  TwoViewFactor(Key i, Key j, const Eigen::Vector3d& qi,
                const Eigen::Vector3d& qj, const SharedNoiseModel& model)
      : NoiseModelFactor2(model, i, j), q_i_cam(qi), q_j_cam(qj) {}

  Vector evaluateError(const Pose3& Xi, const Pose3& Xj,
                       boost::optional<Matrix&> H1,
                       boost::optional<Matrix&> H2) const {
    Eigen::Matrix3d Ri = Xi.rotation().matrix();
    Eigen::Matrix3d Rj = Xj.rotation().matrix();

    Eigen::Vector3d qi = Ri.transpose() * q_i_cam;
    Eigen::Vector3d qj = Rj.transpose() * q_j_cam;

    Eigen::Vector3d ti = Xi.translation();
    Eigen::Vector3d tj = Xj.translation();

    Eigen::Vector3d t = tj - ti;

    double r = qi.dot(skew(t) * qj);

    if (H1 || H2) {
      // numerical Jacobians (clean + safe)
      auto f = [&](const Pose3& A, const Pose3& B) -> Vector {
        Eigen::Matrix3d RA = A.rotation().matrix();
        Eigen::Matrix3d RB = B.rotation().matrix();

        Eigen::Vector3d qa = RA.transpose() * q_i_cam;
        Eigen::Vector3d qb = RB.transpose() * q_j_cam;

        Eigen::Vector3d ta = A.translation();
        Eigen::Vector3d tb = B.translation();

        Eigen::Vector3d tt = tb - ta;

        return (Vector(1) << qa.dot(skew(tt) * qb)).finished();
      };

      if (H1)
        *H1 = numericalDerivative11<Vector, Pose3>(
            [&](const Pose3& X) { return f(X, Xj); }, Xi);

      if (H2)
        *H2 = numericalDerivative11<Vector, Pose3>(
            [&](const Pose3& X) { return f(Xi, X); }, Xj);
    }

    return (Vector(1) << r).finished();
  }
};

// =========================================================
// 3-view factor
// g = qk^T ( [ql]_x [t_lm]_x - [tlk]_x [ql]_x ) qm
// =========================================================
struct ThreeViewFactor : gtsam::NoiseModelFactor3<Pose3, Pose3, Pose3> {
  Eigen::Vector3d qi_cam, qj_cam, qk_cam;

  ThreeViewFactor(Key i, Key j, Key k, const Eigen::Vector3d& qi,
                  const Eigen::Vector3d& qj, const Eigen::Vector3d& qk,
                  const SharedNoiseModel& model)
      : NoiseModelFactor3(model, i, j, k), qi_cam(qi), qj_cam(qj), qk_cam(qk) {}

  Vector evaluateError(const Pose3& Xi, const Pose3& Xj, const Pose3& Xk,
                       boost::optional<Matrix&> H1, boost::optional<Matrix&> H2,
                       boost::optional<Matrix&> H3) const {
    auto f = [&](const Pose3& A, const Pose3& B, const Pose3& C) -> Vector {
      Eigen::Matrix3d RA = A.rotation().matrix();
      Eigen::Matrix3d RB = B.rotation().matrix();
      Eigen::Matrix3d RC = C.rotation().matrix();

      Eigen::Vector3d qa = RA.transpose() * qi_cam;
      Eigen::Vector3d qb = RB.transpose() * qj_cam;
      Eigen::Vector3d qc = RC.transpose() * qk_cam;

      Eigen::Vector3d ta = A.translation();
      Eigen::Vector3d tb = B.translation();
      Eigen::Vector3d tc = C.translation();

      Eigen::Vector3d tkl = tc - tb;
      Eigen::Vector3d tjl = tb - ta;

      auto skew = [](const Eigen::Vector3d& v) {
        Eigen::Matrix3d S;
        S << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
        return S;
      };

      Eigen::Matrix3d Aterm = skew(qb) * skew(tkl) - skew(tjl) * skew(qb);

      double r = qa.dot(Aterm * qc);

      return (Vector(1) << r).finished();
    };

    Vector error = f(Xi, Xj, Xk);

    // =====================================================
    // FIXED NUMERICAL DERIVATIVES
    // =====================================================

    if (H1)
      *H1 = numericalDerivative31<Vector, Pose3, Pose3, Pose3>(
          [&](const Pose3& A, const Pose3& B, const Pose3& C) {
            return f(A, B, C);
          },
          Xi, Xj, Xk);

    if (H2)
      *H2 = numericalDerivative32<Vector, Pose3, Pose3, Pose3>(
          [&](const Pose3& A, const Pose3& B, const Pose3& C) {
            return f(A, B, C);
          },
          Xi, Xj, Xk);

    if (H3)
      *H3 = numericalDerivative33<Vector, Pose3, Pose3, Pose3>(
          [&](const Pose3& A, const Pose3& B, const Pose3& C) {
            return f(A, B, C);
          },
          Xi, Xj, Xk);

    return error;
  }
};

}  // namespace dyno_testing

TEST(TestCameraTrackingFactors, main) {
  using namespace dyno_testing;
  int N = 10;
  int M = 30;

  auto Xgt = generateTrajectory(N);
  auto pts = generatePoints(M);

  // initial guess
  vector<Pose3> Xinit = Xgt;
  for (auto& X : Xinit) {
    X = Pose3(X.rotation(), X.translation() + Point3(0.1, 0.1, 0.1));
  }

  NonlinearFactorGraph graph;
  Values initial;

  for (int i = 0; i < N; i++) initial.insert(X(i), Xinit[i]);

  auto noise = noiseModel::Isotropic::Sigma(1, 1e-3);

  using PriorFactorPose3 = gtsam::PriorFactor<gtsam::Pose3>;

  // priors
  graph.add(
      PriorFactorPose3(X(0), Xgt[0], noiseModel::Isotropic::Sigma(6, 1e-6)));

  // measurements
  for (int k = 2; k < N; k++) {
    int i = k - 1;
    int j = k - 2;

    for (int m = 0; m < M; m++) {
      auto qi = pixelToBearing(project(Xgt[k], pts[m]));
      auto qj = pixelToBearing(project(Xgt[i], pts[m]));
      auto qk = pixelToBearing(project(Xgt[j], pts[m]));

      graph.add(boost::make_shared<TwoViewFactor>(X(k), X(i), qi, qj, noise));

      // graph.add(boost::make_shared<ThreeViewFactor>(
      //     X(k), X(i), X(j),
      //     qi,qj,qk, noise));
    }
  }

  LevenbergMarquardtOptimizer opt(graph, initial);
  Values result = opt.optimize();

  // =====================================================
  // save output
  // =====================================================
  ofstream f(
      "/home/user/dev_ws/src/core/dynosam_utils/src/"
      "test_camera_tracking_factors.csv");
  f << "k,gt_x,gt_y,gt_z,init_x,init_y,init_z,opt_x,opt_y,opt_z\n";

  for (int i = 0; i < N; i++) {
    auto gt = Xgt[i].translation();
    auto ini = Xinit[i].translation();
    auto op = result.at<gtsam::Pose3>(X(i)).translation();

    LOG(INFO) << op;

    f << i << "," << gt.x() << "," << gt.y() << "," << gt.z() << "," << ini.x()
      << "," << ini.y() << "," << ini.z() << "," << op.x() << "," << op.y()
      << "," << op.z() << "\n";
  }

  f.close();
  // return 0;
}
