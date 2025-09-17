/*
 *   Copyright (c) 2025 ACFR-RPG, University of Sydney, Jesse Morris
 (jesse.morris@sydney.edu.au)
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a
 copy
 *   of this software and associated documentation files (the "Software"), to
 deal
 *   in the Software without restriction, including without limitation the
 rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:

 *   The above copyright notice and this permission notice shall be included in
 all
 *   copies or substantial portions of the Software.

 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE
 *   SOFTWARE.
 */

#include "dynosam/backend/rgbd/MPCEstimator.hpp"

#include <glog/logging.h>

#include "dynosam/factors/HybridFormulationFactors.hpp"
#include "dynosam/utils/Numerical.hpp"

// Vec2LimitFactor (2DV and 2DA limits)
// ControlChangeFactor (2DA Smooth)
// FollowFactor (need zero Jacobian)
// Pose2DynXVAFactor (Dynamic factor) + zero Jacobian version

DEFINE_uint32(mpc_horizon, 4, "Dyno mpc time horizon");

DEFINE_uint32(
    mpc_local_goal_horizin, 120,
    "Number of steps ahead to calculate the local goal from the global plan");

DEFINE_double(mpc_vel2d_prior_sigma, 1.0, "Sigma for the cam pose prior");

DEFINE_double(mpc_vel2d_limit_sigma, 1.0, "Sigma for the cam pose prior");
DEFINE_double(mpc_accel2d_limit_sigma, 1.0, "Sigma for the cam pose prior");

DEFINE_double(mpc_accel2d_cost_sigma, 1.0, "Sigma for the cam pose prior");
DEFINE_double(mpc_accel2d_smoothing_sigma, 1.0, "Sigma for the cam pose prior");

DEFINE_double(mpc_dynamic_factor_sigma, 1.0, "Sigma for mpc dynamic factor");
DEFINE_double(mpc_object_prediction_constant_motion_sigma, 0.01,
              "Sigma object prediction smoothing");
DEFINE_double(mpc_static_sdf_X_safety_distance_sigma, 0.1,
              "Sigma for sdf X factor");
DEFINE_double(mpc_static_sdf_H_safety_distance_sigma, 0.1,
              "Sigma for sdf H obstacle factor");

DEFINE_double(mpc_dynamic_obstacle_sigma, 0.1,
              "Sigma for dynamic obstacle factor");

DEFINE_double(mpc_follow_sigma, 1.0, "Sigma for follow factor");
DEFINE_double(mpc_goal_sigma, 1.0, "Sigma for goal factor");

DEFINE_double(mpc_desired_follow_distance, 3.0,
              "Desired follow distance for follow factor");
DEFINE_double(mpc_desired_follow_heading, 0.1,
              "Desired follow heading for follow factor");

DEFINE_double(
    mpc_dt, 0.1,
    "Expected dt for mpc algorithm. SHould be the same as the frame-rate");

DEFINE_int32(mission_type, 0, "0: follow object, 1: navigate (global path)");

DEFINE_string(mpc_path_to_sdf_map,
              "/root/sdf_maps/tugboat_warehouse_with_boxes_map",
              "Path to sdf map (without suffix)");

DEFINE_double(mpc_static_sdf_X_safety_distance, 1.0,
              "Safety distance constant for a static obstacle");
DEFINE_double(mpc_static_sdf_H_safety_distance, 1.0,
              "Safety distance constant for a dynamic obstacle");

DEFINE_double(mpc_dynamic_obstacle_safety_distance, 1.0,
              "Safety distance constant for keeping X away from dynamic "
              "obstacles. Only in Navigation mode");

DEFINE_bool(mpc_use_directed_factors, true,
            "Mostly for testing/experiments. Whether or not to use proposed "
            "directed factors");

namespace dyno {

gtsam::Pose2 convertToSE2OpenCV(const gtsam::Pose3& pose_3) {
  return gtsam::Pose2(pose_3.z(), -pose_3.x(), -pose_3.rotation().pitch());
}

class SDFMap2D {
 public:
  DYNO_POINTER_TYPEDEFS(SDFMap2D)

  SDFMap2D(const std::string& bin_path, const std::string& meta_path) {
    CHECK(loadFromFiles(bin_path, meta_path));
  }

  // Returns signed distance at world coordinates (x, y)
  double distanceAt(double x, double y) const {
    double gx, gy;
    if (!worldToGrid(x, y, gx, gy)) {
      // Outside map bounds, return large positive distance
      // we assume outside the map there are no obsacles
      // but we have map boundary so really this should be -10!
      return -10.0;
    }

    int i = static_cast<int>(std::floor(gx));
    int j = static_cast<int>(std::floor(gy));
    double dx = gx - i;
    double dy = gy - j;

    // Bilinear interpolation
    double v00 = valueAt(i, j);
    double v10 = valueAt(i + 1, j);
    double v01 = valueAt(i, j + 1);
    double v11 = valueAt(i + 1, j + 1);

    double val = (1 - dx) * (1 - dy) * v00 + dx * (1 - dy) * v10 +
                 (1 - dx) * dy * v01 + dx * dy * v11;

    return val;
  }

  // transform that puts a query into the map frame
  // we assume the query will be in opencv coordinate convention but the map
  // frame will be in robotic convention!
  SDFMap2D& setQueryOffset(const gtsam::Pose3& T_map_query) {
    T_map_query_ = T_map_query;
    return *this;
  }

  double getDistanceFromPose(const gtsam::Pose3& T_query) const {
    // T_query will be an SE(3) in opecv convention
    // We then hope/assume that T_map_query_ rightly puts T_query into the
    // robotic convention!!
    gtsam::Pose3 T_map_se3 = T_map_query_ * T_query;

    const double x = T_map_se3.x();
    const double y = T_map_se3.y();
    const double dist = this->distanceAt(x, y);
    return dist;
  }

  // Returns gradient vector at world coordinates (x, y)
  gtsam::Vector2 gradientAt(double x, double y) const {
    const double delta = 0.1;
    double dx =
        (distanceAt(x + delta, y) - distanceAt(x - delta, y)) / (2 * delta);
    double dy =
        (distanceAt(x, y + delta) - distanceAt(x, y - delta)) / (2 * delta);

    gtsam::Vector2 grad(dx, dy);
    // SCALE IT DOWN TO MOVE SLOWER
    const double step_scale = 0.1;  // smaller = slower
    return grad * step_scale;
  }

  // Accessors for map info
  int getWidth() const { return width_; }
  int getHeight() const { return height_; }
  double getResolution() const { return resolution_; }
  double getOriginX() const { return origin_x_; }
  double getOriginY() const { return origin_y_; }

  // Returns signed distance at discrete grid coordinates
  double distanceAtGrid(int gx, int gy) const {
    return static_cast<double>(valueAt(gx, gy));
  }

 private:
  bool loadFromFiles(const std::string& bin_path,
                     const std::string& meta_path) {
    std::ifstream meta_file(meta_path);
    if (!meta_file) {
      LOG(FATAL) << "Failed to open meta file: " << meta_path;
      return false;
    }

    meta_file >> width_ >> height_;
    meta_file >> origin_x_ >> origin_y_;
    meta_file >> resolution_;
    meta_file.close();

    std::ifstream bin_file(bin_path, std::ios::binary);
    if (!bin_file) {
      LOG(FATAL) << "Failed to open bin file: " << bin_path;
      return false;
    }

    data_.resize(width_ * height_);
    bin_file.read(reinterpret_cast<char*>(data_.data()),
                  data_.size() * sizeof(float));
    if (!bin_file) {
      LOG(FATAL) << "Failed to read binary data";
      return false;
    }

    bin_file.close();
    return true;
  }

 private:
  std::vector<float> data_;
  int width_, height_;
  double origin_x_, origin_y_;
  double resolution_;

  gtsam::Pose3 T_map_query_;

  // Converts world coordinates to grid indices (floating point)
  bool worldToGrid(double x, double y, double& gx, double& gy) const {
    gx = (x - origin_x_) / resolution_;
    gy = (y - origin_y_) / resolution_;
    return gx >= 0 && gx < width_ - 1 && gy >= 0 && gy < height_ - 1;
  }

  // Returns the value at integer grid indices (with bounds check)
  double valueAt(int i, int j) const {
    if (i < 0) i = 0;
    if (i >= width_) i = width_ - 1;
    if (j < 0) j = 0;
    if (j >= height_) j = height_ - 1;
    return static_cast<double>(data_[j * width_ + i]);
  }
};

namespace mpc_factors {

gtsam::Pose2 predictRobotPose(gtsam::Pose2 current_pose,
                              gtsam::Vector2 current_control, double dt) {
  // const double linear_velocity      = current_control(0);
  // const double angular_velocity     = current_control(1);
  // const double dx = linear_velocity * dt * cos(current_pose.theta());
  // const double dy = linear_velocity * dt * sin(current_pose.theta());
  // const double dtheta = angular_velocity * dt;
  // return gtsam::Pose2(current_pose.x() + dx, current_pose.y() + dy,
  // current_pose.theta() + dtheta);

  const double linear_velocity = current_control(0);
  const double angular_velocity = current_control(1);
  const double angular_displacement = angular_velocity * dt / 2;

  const double dx = linear_velocity * dt * cos(angular_displacement);
  const double dy = linear_velocity * dt * sin(angular_displacement);
  const double dtheta = angular_velocity * dt;

  // Apply displacement in x and y based on the current heading (theta)
  const double new_x =
      dx * cos(current_pose.theta()) - dy * sin(current_pose.theta());
  const double new_y =
      dx * sin(current_pose.theta()) + dy * cos(current_pose.theta());
  const double new_theta = dtheta;

  // Print the current pose, current control, and expected pose
  // std::cout << "Current Pose: (" << current_pose.x() << ", " <<
  // current_pose.y() << ", " << current_pose.theta() << ")\n"; std::cout <<
  // "Current Control: (Linear Velocity: " << linear_velocity << ", Angular
  // Velocity: " << angular_velocity << ")\n"; std::cout << "Expected Pose: ("
  // << current_pose.x() + new_x << ", " << current_pose.y() + new_y << ", " <<
  // current_pose.theta() + new_theta << ")\n";

  return gtsam::Pose2(current_pose.x() + new_x, current_pose.y() + new_y,
                      current_pose.theta() + new_theta);
}

class SDFObstacleFactorBase : public gtsam::NoiseModelFactor1<gtsam::Pose3> {
 public:
  using shared_ptr = boost::shared_ptr<SDFObstacleFactorBase>;
  using This = SDFObstacleFactorBase;
  using Base = gtsam::NoiseModelFactor1<gtsam::Pose3>;

  SDFObstacleFactorBase(gtsam::Key key, std::shared_ptr<SDFMap2D> sdf_map,
                        double safety_distance, gtsam::SharedNoiseModel model)
      : Base(model, key),
        sdf_map_(CHECK_NOTNULL(sdf_map)),
        safety_distance_(safety_distance) {}

  virtual double calculateDistance(const gtsam::Pose3& pose) const = 0;

  // Error function
  gtsam::Vector evaluateError(
      const gtsam::Pose3& pose,
      boost::optional<gtsam::Matrix&> J1 = boost::none) const override {

    // Note: We do not set up jacobian to 0 when the error is 0 because then the Numerical Derivative struggles
    // and doesn't work for one of the sides of the obstacle.
    if (J1) {
      Eigen::Matrix<double, 1, 6> df_f =
          gtsam::numericalDerivative11<gtsam::Vector1, gtsam::Pose3>(
              std::bind(&SDFObstacleFactorBase::residual, this,
                        std::placeholders::_1),
              pose);
      *J1 = df_f;
    }
    gtsam::Vector e = residual(pose);
    // LOG(INFO) << "dist " << distance << " error " << e;
    return e;
  }

  // Compute residual (static utility function)
  gtsam::Vector residual(const gtsam::Pose3& pose) const {
    double distance = this->calculateDistance(pose);
    return gtsam::Vector1(std::max(0.0, safety_distance_ - distance));
  }

  inline gtsam::Key poseKey() const { return key1(); }

 protected:
  std::shared_ptr<SDFMap2D> sdf_map_;

 private:
  double safety_distance_;
};

class SDFStaticObstacleXFactor : public SDFObstacleFactorBase {
 public:
  using shared_ptr = boost::shared_ptr<SDFStaticObstacleXFactor>;
  using This = SDFStaticObstacleXFactor;
  using Base = SDFObstacleFactorBase;

  SDFStaticObstacleXFactor(gtsam::Key X_k_key,
                           std::shared_ptr<SDFMap2D> sdf_map,
                           double safety_distance,
                           gtsam::SharedNoiseModel model)
      : Base(X_k_key, sdf_map, safety_distance, model) {}

  double calculateDistance(const gtsam::Pose3& X_k) const override {
    return sdf_map_->getDistanceFromPose(X_k);
  }

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::make_shared<SDFStaticObstacleXFactor>(*this);
  }
};

class SDFStaticObstacleHFactor : public SDFObstacleFactorBase {
 public:
  using shared_ptr = boost::shared_ptr<SDFStaticObstacleHFactor>;
  using This = SDFStaticObstacleHFactor;
  using Base = SDFObstacleFactorBase;

  SDFStaticObstacleHFactor(gtsam::Key H_W_e_k_key,
                           std::shared_ptr<SDFMap2D> sdf_map,
                           double safety_distance, const gtsam::Pose3& L_e,
                           gtsam::SharedNoiseModel model)
      : Base(H_W_e_k_key, sdf_map, safety_distance, model), L_e_(L_e) {}

  double calculateDistance(const gtsam::Pose3& H_W_e_k) const override {
    auto L_W_k = H_W_e_k * L_e_;
    return sdf_map_->getDistanceFromPose(L_W_k);
  }

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::make_shared<SDFStaticObstacleHFactor>(*this);
  }

 private:
  gtsam::Pose3 L_e_;
};

void MissionFactorBase::print(const std::string& s,
                              const KeyFormatter& keyFormatter) const {
  std::cout << s << "MissionFactorBase(" << keyFormatter(this->key1()) << ","
            << keyFormatter(this->key2()) << ")\n";
  this->noiseModel_->print("  noise model: ");
}

bool MissionFactorBase::isFuture(FrameId frame_k) const {
  ObjectId object_id;
  FrameId frame_id;
  CHECK(reconstructMotionInfo(objectMotionKey(), object_id, frame_id));
  (void)object_id;
  return frame_id > frame_k;
}

struct DynamicObstacleFactorResidual {
  static gtsam::Vector residual(const gtsam::Pose3& X_k_se3,
                                const gtsam::Pose3& H_W_e_k,
                                const gtsam::Pose3& L_e,
                                double safety_distance) {
    const gtsam::Pose3 L_W_k_se3 = H_W_e_k * L_e;
    gtsam::Pose2 X_k_se2 = convertToSE2OpenCV(X_k_se3);
    gtsam::Pose2 L_W_k_se2 = convertToSE2OpenCV(L_W_k_se3);

    double distance = X_k_se2.range(L_W_k_se2);
    // double distance = X_k_se3.range(L_W_k_se3);
    if (distance < safety_distance) {
      return gtsam::Vector1(safety_distance - distance);  // - Works
    } else {
      // No error if outside safety distance
      return gtsam::Vector1(0.0);
    }
  }
};

class DynamicObstacleFactorBase : public MissionFactorBase,
                                  public DynamicObstacleFactorResidual {
 public:
  using shared_ptr = boost::shared_ptr<DynamicObstacleFactorBase>;
  using This = DynamicObstacleFactorBase;
  using Base = MissionFactorBase;

  DynamicObstacleFactorBase(gtsam::Key X_k_key, gtsam::Key H_W_e_k_key,
                            const gtsam::Pose3& L_e, double safety_distance,
                            gtsam::SharedNoiseModel model)
      : Base(X_k_key, H_W_e_k_key, L_e, model),
        safety_distance_(safety_distance) {}
  // Error function
  virtual gtsam::Vector evaluateError(
      const gtsam::Pose3& X_k, const gtsam::Pose3& H_W_e_k,
      boost::optional<gtsam::Matrix&> J1 = boost::none,
      boost::optional<gtsam::Matrix&> J2 = boost::none) const override {
    if (J1) {
      Eigen::Matrix<double, 1, 6> df_f =
          gtsam::numericalDerivative21<gtsam::Vector1, gtsam::Pose3,
                                       gtsam::Pose3>(
              std::bind(&DynamicObstacleFactorBase::residual,
                        std::placeholders::_1, std::placeholders::_2, L_e_,
                        safety_distance_),
              X_k, H_W_e_k);
      *J1 = df_f;
    }

    if (J2) {
      Eigen::Matrix<double, 1, 6> df_f =
          gtsam::numericalDerivative22<gtsam::Vector1, gtsam::Pose3,
                                       gtsam::Pose3>(
              std::bind(&DynamicObstacleFactorBase::residual,
                        std::placeholders::_1, std::placeholders::_2, L_e_,
                        safety_distance_),
              X_k, H_W_e_k);
      *J2 = df_f;
    }

    return residual(X_k, H_W_e_k, L_e_, safety_distance_);
  }

 private:
  double safety_distance_;
};

template <size_t... ZeroIndices>
class DynamicObstacleFactor : public DynamicObstacleFactorBase {
 public:
  using This = DynamicObstacleFactor<ZeroIndices...>;
  using Base = DynamicObstacleFactorBase;
  typedef boost::shared_ptr<This> shared_ptr;

  static constexpr bool HasIndices = sizeof...(ZeroIndices) > 0;
  static constexpr std::array<size_t, sizeof...(ZeroIndices)> indices = {
      ZeroIndices...};

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::make_shared<This>(*this);
  }

  DynamicObstacleFactor(gtsam::Key X_k_key, gtsam::Key H_W_e_k_key,
                        const gtsam::Pose3& L_e, double safety_distance,
                        gtsam::SharedNoiseModel model)
      : Base(X_k_key, H_W_e_k_key, L_e, safety_distance, model) {}

  gtsam::Vector evaluateError(
      const gtsam::Pose3& X_k, const gtsam::Pose3& H_W_e_k,
      boost::optional<gtsam::Matrix&> J1 = boost::none,
      boost::optional<gtsam::Matrix&> J2 = boost::none) const override {
    gtsam::Vector error = Base::evaluateError(X_k, H_W_e_k, J1, J2);

    if constexpr (HasIndices) {
      for (size_t i : indices) {
        if (i == 1 && J1) J1->setZero();
        if (i == 2 && J2) J2->setZero();
      }
    }
    return error;
  }
};

template <typename VALUE, size_t... ZeroIndices>
class BetweenFactorDirected : public gtsam::NoiseModelFactorN<VALUE, VALUE> {
 public:
  typedef VALUE T;
  typedef BetweenFactorDirected<VALUE, ZeroIndices...> This;
  typedef gtsam::NoiseModelFactorN<VALUE, VALUE> Base;
  typedef boost::shared_ptr<This> shared_ptr;

  static constexpr bool HasIndices = sizeof...(ZeroIndices) > 0;
  static constexpr std::array<size_t, sizeof...(ZeroIndices)> indices = {
      ZeroIndices...};

  BetweenFactorDirected(Key key1, Key key2, const VALUE& measured,
                        const SharedNoiseModel& model = nullptr)
      : Base(model, key1, key2), measured_(measured) {}

  void print(
      const std::string& s = "",
      const KeyFormatter& keyFormatter = DefaultKeyFormatter) const override {
    std::cout << s << "BetweenFactorDirected(" << keyFormatter(this->key1())
              << "," << keyFormatter(this->key2()) << ")\n";
    gtsam::traits<T>::Print(measured_, "  measured: ");
    this->noiseModel_->print("  noise model: ");
  }

  gtsam::Vector evaluateError(
      const T& p1, const T& p2,
      boost::optional<gtsam::Matrix&> H1 = boost::none,
      boost::optional<gtsam::Matrix&> H2 = boost::none) const override {
    T hx = gtsam::traits<T>::Between(p1, p2, H1, H2);  // h(x)
    // manifold equivalent of h(x)-z -> log(z,h(x))
    gtsam::Vector error = gtsam::traits<T>::Local(measured_, hx);

    using Jacobian = typename gtsam::traits<T>::ChartJacobian::Jacobian;
    if constexpr (HasIndices) {
      for (size_t i : indices) {
        if (i == 1 && H1) *H1 = Jacobian::Zero();
        if (i == 2 && H2) *H2 = Jacobian::Zero();
      }
      return error;
    }
  }

 protected:
  VALUE measured_; /** The measurement */
};

using Vec2BetweenFactorDirected1 = BetweenFactorDirected<gtsam::Vector2, 1>;

/**
 * @brief A factor that enforces inequality constraints on a gtsam::Vector2
 * (control input: linear and angular velocity).
 *
 * The factor penalizes violations of the specified control limits:
 * - Linear velocity: minLinearControl <= v <= maxLinearControl
 * - Angular velocity: minAngularControl <= omega <= maxAngularControl
 */
class Vec2LimitFactor : public gtsam::NoiseModelFactor1<gtsam::Vector2> {
 public:
  using shared_ptr = boost::shared_ptr<Vec2LimitFactor>;
  using This = Vec2LimitFactor;
  using Base = gtsam::NoiseModelFactor1<gtsam::Vector2>;

  // Constructor
  Vec2LimitFactor(gtsam::Key key, double min1, double max1, double min2,
                  double max2, gtsam::SharedNoiseModel noiseModel,
                  double dt = 1.0)
      : Base(noiseModel, key),
        min1_(min1),
        max1_(max1),
        min2_(min2),
        max2_(max2),
        dt_(dt) {}

  // Clone method
  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::make_shared<Vec2LimitFactor>(*this);
  }

  void print(const std::string& s = "",
             const gtsam::KeyFormatter& keyFormatter =
                 DynoLikeKeyFormatter) const override {
    std::cout << s << "Vec2LimitFactor\n";
    Base::print("", keyFormatter);
  }

  // Evaluate error and optionally compute Jacobian
  gtsam::Vector evaluateError(
      const gtsam::Vector2& variable,
      boost::optional<gtsam::Matrix&> J1 = boost::none) const override {
    if (J1) {
      Eigen::Matrix<double, 2, 2> df_cl =
          gtsam::numericalDerivative11<gtsam::Vector2, gtsam::Vector2>(
              std::bind(&Vec2LimitFactor::residual, this, std::placeholders::_1,
                        min1_, max1_, min2_, max2_, dt_),
              variable);
      *J1 = df_cl;
    }

    return residual(variable, min1_, max1_, min2_, max2_, dt_);
  }

  gtsam::Vector residual(const gtsam::Vector2& variable, double min1,
                         double max1, double min2, double max2,
                         double dt) const {
    // Extract control inputs
    double first = variable(0) * dt;   // Linear acc to delta vel
    double second = variable(1) * dt;  // Angular acc to delta vel

    // Initialize error vector
    gtsam::Vector2 error(0.0, 0.0);

    // Compute error for linear control
    if (first < min1 * dt) {
      error(0) = min1 * dt - first;  // positive residual, push up
    } else if (first > max1 * dt) {
      error(0) = max1 * dt - first;  // negative residual, push down
    }

    if (second < min2 * dt) {
      error(1) = min2 * dt - second;  // positive residual, push up
    } else if (second > max2 * dt) {
      error(1) = max2 * dt - second;  // negative residual, push down
    }
    return error;
  }

 private:
  double min1_;  // Minimum linear control (velocity)
  double max1_;  // Maximum linear control (velocity)
  double min2_;  // Minimum angular control (velocity)
  double max2_;  // Maximum angular control (velocity)
  double dt_;    // Time step
};

class HybridMotionFollowJac0Factor : public MissionFactorBase {
 public:
  using shared_ptr = boost::shared_ptr<HybridMotionFollowJac0Factor>;
  using This = HybridMotionFollowJac0Factor;
  using Base = MissionFactorBase;

  // Constructor
  HybridMotionFollowJac0Factor(gtsam::Key X_W_key_follower,
                               gtsam::Key H_W_e_k_key_leader,
                               const gtsam::Pose3& L_e_leader,
                               double des_distance, double des_heading,
                               gtsam::SharedNoiseModel model)
      : Base(X_W_key_follower, H_W_e_k_key_leader, L_e_leader, model),
        des_distance_(des_distance),
        des_heading_(des_heading) {}

  // Clone method
  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::make_shared<HybridMotionFollowJac0Factor>(*this);
  }

  void print(const std::string& s = "",
             const gtsam::KeyFormatter& keyFormatter =
                 DynoLikeKeyFormatter) const override {
    std::cout << s << "HybridMotionFollowJac0Factor\n";
    Base::print("", keyFormatter);
  }

  // Error function
  gtsam::Vector evaluateError(
      const gtsam::Pose3& X_W_follower, const gtsam::Pose3& H_W_e_k_leader,
      boost::optional<gtsam::Matrix&> J1 = boost::none,
      boost::optional<gtsam::Matrix&> J2 = boost::none) const override {
    // Compute Jacobians if requested
    if (J1) {
      Eigen::Matrix<double, 2, 6> df_f =
          gtsam::numericalDerivative21<gtsam::Vector2, gtsam::Pose3,
                                       gtsam::Pose3>(
              std::bind(&HybridMotionFollowJac0Factor::residual, this,
                        std::placeholders::_1, std::placeholders::_2,
                        des_distance_, des_heading_),
              X_W_follower, H_W_e_k_leader);
      *J1 = df_f;
    }

    if (J2) {
      // Eigen::Matrix<double, 2, 6> df_l =
      //     gtsam::numericalDerivative22<gtsam::Vector2, gtsam::Pose3,
      //                                  gtsam::Pose3>(
      //         std::bind(&HybridMotionFollowJac0Factor::residual, this,
      //                   std::placeholders::_1, std::placeholders::_2,
      //                   des_distance_, des_heading_),
      //         X_W_follower, H_W_e_k_leader);
      // *J2 = df_l;
      *J2 = Eigen::Matrix<double, 2, 6>::Zero();
    }

    return residual(X_W_follower, H_W_e_k_leader, des_distance_, des_heading_);
  }

  // Compute residual (static utility function)
  gtsam::Vector residual(const gtsam::Pose3& X_W_follower_3d,
                         const gtsam::Pose3& H_W_e_k_leader_3d,
                         double des_distance,
                         const gtsam::Rot2& des_heading) const {
    // gtsam::Pose2 X_W_follower(X_W_follower_3d.x(), X_W_follower_3d.y(),
    //                           X_W_follower_3d.rotation().yaw());
    gtsam::Pose2 X_W_follower(X_W_follower_3d.z(), -X_W_follower_3d.x(),
                              -X_W_follower_3d.rotation().pitch());

    gtsam::Pose3 L_W_3d = H_W_e_k_leader_3d * L_e_;

    // gtsam::Pose2 L_W(L_W_3d.x(), L_W_3d.y(),
    //                         L_W_3d.rotation().yaw());
    gtsam::Pose2 L_W(L_W_3d.z(), -L_W_3d.x(), -L_W_3d.rotation().pitch());
    double distance = X_W_follower.range(L_W);
    gtsam::Rot2 bearing = X_W_follower.bearing(L_W);
    gtsam::Vector1 bearing_error =
        gtsam::Rot2::Logmap(bearing.between(des_heading));

    // Field of view
    // double relAngle = bearing.theta();
    // double error = 0.0;
    // if (std::abs(relAngle) <= M_PI_2 / 2.0) {
    //     // Inside FoV — soft quadratic penalty
    //     double normalized = relAngle / (M_PI_2 / 2.0);  // between -1 and 1
    //     error = 1.0 * normalized * normalized;
    // } else {
    //     // Outside FoV — harder quadratic penalty past FoV limit
    //     double excess = std::abs(relAngle) - M_PI_2 / 2.0;
    //     error = 1.0 + 50.0 * excess * excess;
    // }

    // ******************* Field of view with uncertainity

    // ***************** The end of field of view

    return gtsam::Vector2(distance - des_distance,
                          bearing_error(0)  // error
    );
  }

  inline gtsam::Key X_W_key_follower() const { return key1(); }
  inline gtsam::Key H_W_e_k_key_leader() const { return key2(); }

 private:
  double des_distance_;
  gtsam::Rot2 des_heading_;
};

struct MotionModelFactorResidual {
  static gtsam::Vector residual(const gtsam::Pose3& pose3_1,
                                const gtsam::Pose3& pose3_2,
                                const gtsam::Vector2& vel1,
                                const gtsam::Vector2& vel2,
                                const gtsam::Vector2& acc1, const double dt) {
    gtsam::Pose2 pose1 = convertToSE2OpenCV(pose3_1);
    gtsam::Pose2 pose2 = convertToSE2OpenCV(pose3_2);
    // Update velocities using acceleration and time step
    gtsam::Vector2 pred_vel2 = vel1 + acc1 * dt;

    gtsam::Pose2 pred_pose = predictRobotPose(pose1, vel2, dt);

    gtsam::Vector3 pose_error =
        gtsam::traits<gtsam::Pose2>::Local(pose2, pred_pose);

    return gtsam::Vector8(pose_error(0),  // x error
                          pose_error(1),  // y error
                          pose_error(2),  // theta error
                          vel2[0] - pred_vel2[0], vel2[1] - pred_vel2[1],
                          pose3_2.rotation().yaw() - pose3_1.rotation().yaw(),
                          pose3_2.rotation().roll() - pose3_1.rotation().roll(),
                          pose3_2.y() - pose3_1.y());
  }
};

class MotionModelFactorBase
    : public gtsam::NoiseModelFactor5<gtsam::Pose3, gtsam::Pose3,
                                      gtsam::Vector2, gtsam::Vector2,
                                      gtsam::Vector2>,
      public MotionModelFactorResidual {
 public:
  using shared_ptr = boost::shared_ptr<MotionModelFactorBase>;
  using This = MotionModelFactorBase;
  using Base =
      gtsam::NoiseModelFactor5<gtsam::Pose3, gtsam::Pose3, gtsam::Vector2,
                               gtsam::Vector2, gtsam::Vector2>;

  // Constructor
  MotionModelFactorBase(gtsam::Key pose1Key, gtsam::Key pose2Key,
                        gtsam::Key vel1Key, gtsam::Key vel2Key,
                        gtsam::Key acc1Key, gtsam::SharedNoiseModel model,
                        double dt)
      : Base(model, pose1Key, pose2Key, vel1Key, vel2Key, acc1Key), dt_(dt) {}

  virtual void print(const std::string& s = "",
                     const gtsam::KeyFormatter& keyFormatter =
                         DynoLikeKeyFormatter) const override {
    std::cout << s << "MotionModelFactorBase\n";
    Base::print("", keyFormatter);
  }

  // Evaluate error and optionally compute Jacobians
  virtual gtsam::Vector evaluateError(
      const gtsam::Pose3& pose1, const gtsam::Pose3& pose2,
      const gtsam::Vector2& vel1, const gtsam::Vector2& vel2,
      const gtsam::Vector2& acc1,
      boost::optional<gtsam::Matrix&> J1 = boost::none,
      boost::optional<gtsam::Matrix&> J2 = boost::none,
      boost::optional<gtsam::Matrix&> J3 = boost::none,
      boost::optional<gtsam::Matrix&> J4 = boost::none,
      boost::optional<gtsam::Matrix&> J5 = boost::none) const override {
    // Compute Jacobians if requested
    // Compute Jacobians if requested
    if (J1) {
      Eigen::Matrix<double, 8, 6> df_pp =
          gtsam::numericalDerivative51<gtsam::Vector8, gtsam::Pose3,
                                       gtsam::Pose3, gtsam::Vector2,
                                       gtsam::Vector2, gtsam::Vector2>(
              std::bind(&MotionModelFactorBase::residual, std::placeholders::_1,
                        std::placeholders::_2, std::placeholders::_3,
                        std::placeholders::_4, std::placeholders::_5, dt_),
              pose1, pose2, vel1, vel2, acc1);
      *J1 = df_pp;
    }

    if (J2) {
      Eigen::Matrix<double, 8, 6> df_cp =
          gtsam::numericalDerivative52<gtsam::Vector8, gtsam::Pose3,
                                       gtsam::Pose3, gtsam::Vector2,
                                       gtsam::Vector2, gtsam::Vector2>(
              std::bind(&MotionModelFactorBase::residual, std::placeholders::_1,
                        std::placeholders::_2, std::placeholders::_3,
                        std::placeholders::_4, std::placeholders::_5, dt_),
              pose1, pose2, vel1, vel2, acc1);
      *J2 = df_cp;
    }

    if (J3) {
      Eigen::Matrix<double, 8, 2> df_pc =
          gtsam::numericalDerivative53<gtsam::Vector8, gtsam::Pose3,
                                       gtsam::Pose3, gtsam::Vector2,
                                       gtsam::Vector2, gtsam::Vector2>(
              std::bind(&MotionModelFactorBase::residual, std::placeholders::_1,
                        std::placeholders::_2, std::placeholders::_3,
                        std::placeholders::_4, std::placeholders::_5, dt_),
              pose1, pose2, vel1, vel2, acc1);
      *J3 = df_pc;
    }

    if (J4) {
      Eigen::Matrix<double, 8, 2> df_cc =
          gtsam::numericalDerivative54<gtsam::Vector8, gtsam::Pose3,
                                       gtsam::Pose3, gtsam::Vector2,
                                       gtsam::Vector2, gtsam::Vector2>(
              std::bind(&MotionModelFactorBase::residual, std::placeholders::_1,
                        std::placeholders::_2, std::placeholders::_3,
                        std::placeholders::_4, std::placeholders::_5, dt_),
              pose1, pose2, vel1, vel2, acc1);
      *J4 = df_cc;
    }

    if (J5) {
      Eigen::Matrix<double, 8, 2> df_ca =
          gtsam::numericalDerivative55<gtsam::Vector8, gtsam::Pose3,
                                       gtsam::Pose3, gtsam::Vector2,
                                       gtsam::Vector2, gtsam::Vector2>(
              std::bind(&MotionModelFactorBase::residual, std::placeholders::_1,
                        std::placeholders::_2, std::placeholders::_3,
                        std::placeholders::_4, std::placeholders::_5, dt_),
              pose1, pose2, vel1, vel2, acc1);
      *J5 = df_ca;
    }

    // Return the residual
    return residual(pose1, pose2, vel1, vel2, acc1, dt_);
  }

 private:
  double dt_;  // Time step
};

template <size_t... ZeroIndices>
class MotionModelFactor : public MotionModelFactorBase {
 public:
  using This = MotionModelFactor<ZeroIndices...>;
  using Base = MotionModelFactorBase;
  typedef boost::shared_ptr<This> shared_ptr;

  // Only define indices if pack is non-empty
  static constexpr bool HasIndices = sizeof...(ZeroIndices) > 0;
  static constexpr std::array<size_t, sizeof...(ZeroIndices)> indices = {
      ZeroIndices...};

  // Constructor
  MotionModelFactor(gtsam::Key pose1Key, gtsam::Key pose2Key,
                    gtsam::Key vel1Key, gtsam::Key vel2Key, gtsam::Key acc1Key,
                    gtsam::SharedNoiseModel model, double dt)
      : Base(pose1Key, pose2Key, vel1Key, vel2Key, acc1Key, model, dt) {}

  // Clone method
  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::make_shared<This>(*this);
  }

  // Evaluate error and optionally compute Jacobians
  gtsam::Vector evaluateError(
      const gtsam::Pose3& pose1, const gtsam::Pose3& pose2,
      const gtsam::Vector2& vel1, const gtsam::Vector2& vel2,
      const gtsam::Vector2& acc1,
      boost::optional<gtsam::Matrix&> J1 = boost::none,
      boost::optional<gtsam::Matrix&> J2 = boost::none,
      boost::optional<gtsam::Matrix&> J3 = boost::none,
      boost::optional<gtsam::Matrix&> J4 = boost::none,
      boost::optional<gtsam::Matrix&> J5 = boost::none) const override {
    gtsam::Vector error =
        Base::evaluateError(pose1, pose2, vel1, vel2, acc1, J1, J2, J3, J4, J5);

    if constexpr (HasIndices) {
      for (size_t i : indices) {
        if (i == 1 && J1) J1->setZero();
        if (i == 2 && J2) J2->setZero();
        if (i == 3 && J3) J3->setZero();
        if (i == 4 && J4) J4->setZero();
        if (i == 5 && J5) J5->setZero();
      }
    }
    return error;
  }
};

// mimics a prior factor on SE(2) when the variables are in SE(3)
// leaves some variables unconstrained
// NOTE: we assume the goal pose has already been converted into the opencv
// coordiante convention (this happens at the ROS level where we do a tf look up
// between the map and child frame of dynosam)
class GoalFactorSE2 : public gtsam::NoiseModelFactor1<gtsam::Pose3> {
 public:
  GoalFactorSE2(gtsam::Key X_k_key, const gtsam::Pose3& goal_pose_opencv,
                const gtsam::SharedNoiseModel& noiseModel)
      : gtsam::NoiseModelFactor1<gtsam::Pose3>(noiseModel, X_k_key),
        goal_pose_se2(convertToSE2OpenCV(goal_pose_opencv)) {}

  // Clone method
  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::make_shared<GoalFactorSE2>(*this);
  }

  gtsam::Vector evaluateError(
      const gtsam::Pose3& X_k_se3,
      boost::optional<gtsam::Matrix&> J1 = boost::none) const {
    auto residual = [](const gtsam::Pose3& X_k_se3,
                       const gtsam::Pose2& goal_pose_se2) {
      gtsam::Pose2 X_k_se2 = convertToSE2OpenCV(X_k_se3);
      LOG(INFO) << "X_k_se2 " << X_k_se2;
      return -gtsam::traits<gtsam::Pose2>::Local(X_k_se2,
                                                 gtsam::Pose2(20, 20, 0));
    };

    // Compute Jacobians if requested
    if (J1) {
      Eigen::Matrix<double, 3, 6> df_p =
          gtsam::numericalDerivative11<gtsam::Vector3, gtsam::Pose3>(
              std::bind(residual, std::placeholders::_1, goal_pose_se2),
              X_k_se3);
      *J1 = df_p;
    }

    gtsam::Vector3 e = residual(X_k_se3, goal_pose_se2);
    return e;
  }

 private:
  gtsam::Pose2 goal_pose_se2;
};

struct HybridConstantMotionFactorResidual {
  static gtsam::Vector residual(const gtsam::Pose3& e_H_km2_world,
                                const gtsam::Pose3& e_H_km1_world,
                                const gtsam::Pose3& e_H_k_world,
                                const gtsam::Pose3& L_e) {
    gtsam::Pose3 L_km2_est = e_H_km2_world * L_e;
    gtsam::Pose3 L_km1_est = e_H_km1_world * L_e;
    gtsam::Pose3 L_k_est = e_H_k_world * L_e;

    // this is now same as smoothing factor from Morris RA-L (2025)
    gtsam::Pose3 k_2_H_k_1 = L_km2_est.inverse() * L_km1_est;
    gtsam::Pose3 k_1_H_k = L_km1_est.inverse() * L_k_est;

    gtsam::Pose3 relative_motion = k_2_H_k_1.inverse() * k_1_H_k;

    return gtsam::traits<gtsam::Pose3>::Local(gtsam::Pose3::Identity(),
                                              relative_motion);
  }
};

class HybridConstantMotionFactorBase
    : public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3>,
      public HybridConstantMotionFactorResidual {
 public:
  typedef boost::shared_ptr<HybridConstantMotionFactorBase> shared_ptr;
  typedef HybridConstantMotionFactorBase This;
  typedef gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3>
      Base;

  gtsam::Pose3 L_e_;

  HybridConstantMotionFactorBase(gtsam::Key e_H_km2_world_key,
                                 gtsam::Key e_H_km1_world_key,
                                 gtsam::Key e_H_k_world_key,
                                 const gtsam::Pose3& L_e,
                                 gtsam::SharedNoiseModel model)
      : Base(model, e_H_km2_world_key, e_H_km1_world_key, e_H_k_world_key),
        L_e_(L_e) {}

  virtual gtsam::Vector evaluateError(
      const gtsam::Pose3& e_H_km2_world, const gtsam::Pose3& e_H_km1_world,
      const gtsam::Pose3& e_H_k_world,
      boost::optional<gtsam::Matrix&> J1 = boost::none,
      boost::optional<gtsam::Matrix&> J2 = boost::none,
      boost::optional<gtsam::Matrix&> J3 = boost::none) const override {
    if (J1) {
      *J1 = gtsam::numericalDerivative31<gtsam::Vector6, gtsam::Pose3,
                                         gtsam::Pose3, gtsam::Pose3>(
          std::bind(&HybridConstantMotionFactorBase::residual,
                    std::placeholders::_1, std::placeholders::_2,
                    std::placeholders::_3, L_e_),
          e_H_km2_world, e_H_km1_world, e_H_k_world);
    }

    if (J2) {
      *J2 = gtsam::numericalDerivative32<gtsam::Vector6, gtsam::Pose3,
                                         gtsam::Pose3, gtsam::Pose3>(
          std::bind(&HybridConstantMotionFactorBase::residual,
                    std::placeholders::_1, std::placeholders::_2,
                    std::placeholders::_3, L_e_),
          e_H_km2_world, e_H_km1_world, e_H_k_world);
    }

    if (J3) {
      *J3 = gtsam::numericalDerivative33<gtsam::Vector6, gtsam::Pose3,
                                         gtsam::Pose3, gtsam::Pose3>(
          std::bind(&HybridConstantMotionFactorBase::residual,
                    std::placeholders::_1, std::placeholders::_2,
                    std::placeholders::_3, L_e_),
          e_H_km2_world, e_H_km1_world, e_H_k_world);
    }

    // in this case k = n1 and n1=2 since we overlap between the estiamtion and
    // the prediction
    return residual(e_H_km2_world, e_H_km1_world, e_H_k_world, L_e_);
  }
};

template <size_t... ZeroIndices>
class HybridConstantMotionFactor : public HybridConstantMotionFactorBase {
 public:
  using This = HybridConstantMotionFactor<ZeroIndices...>;
  using Base = HybridConstantMotionFactorBase;
  typedef boost::shared_ptr<This> shared_ptr;

  // Only define indices if pack is non-empty
  static constexpr bool HasIndices = sizeof...(ZeroIndices) > 0;
  static constexpr std::array<size_t, sizeof...(ZeroIndices)> indices = {
      ZeroIndices...};

  template <typename... Args>
  HybridConstantMotionFactor(Args&&... args)
      : Base(std::forward<Args>(args)...) {}

  virtual gtsam::Vector evaluateError(
      const gtsam::Pose3& e_H_km2_world, const gtsam::Pose3& e_H_km1_world,
      const gtsam::Pose3& e_H_k_world,
      boost::optional<gtsam::Matrix&> J1 = boost::none,
      boost::optional<gtsam::Matrix&> J2 = boost::none,
      boost::optional<gtsam::Matrix&> J3 = boost::none) const override {
    gtsam::Vector error = Base::evaluateError(e_H_km2_world, e_H_km1_world,
                                              e_H_k_world, J1, J2, J3);

    if constexpr (HasIndices) {
      for (size_t i : indices) {
        if (i == 1 && J1) *J1 = Eigen::Matrix<double, 6, 6>::Zero();
        if (i == 2 && J2) *J2 = Eigen::Matrix<double, 6, 6>::Zero();
        if (i == 3 && J3) *J3 = Eigen::Matrix<double, 6, 6>::Zero();
      }
    }

    return error;
  }
};

}  // namespace mpc_factors

StateQuery<gtsam::Vector2> MPCAccessor::getControlCommand(
    FrameId frame_k) const {
  auto control_command = this->makeControlCommandKey(frame_k);
  return this->query<gtsam::Vector2>(control_command);
}

// TODO: gross dont need this function!!!!
StateQuery<gtsam::Vector2> MPCFormulation::getControlCommand(
    FrameId frame_k) const {
  auto accessor = this->derivedAccessor<MPCAccessor>();
  return accessor->getControlCommand(frame_k);
}

std::pair<ObjectMotionMap, ObjectPoseMap> MPCFormulation::getObjectPredictions(
    FrameId frame_k) const {
  auto object_to_follow = mpc_data_.object_to_follow;
  auto accessor = this->accessorFromTheta();

  if (!this->hasObjectKeyFrame(object_to_follow, frame_k)) {
    VLOG(10) << "j=" << object_to_follow << " has no keyframe at k=" << frame_k
             << ". Unable to make object predictions!";
    return {};
  } else {
    // assume these are the same for all future poses... yes (as we havebt
    // observed them yet so there is no mechaism to change L_e)
    const auto [reference_frame_e, L_e] =
        this->getObjectKeyFrame(object_to_follow, frame_k);

    auto mpc_horizon = mpc_data_.mpc_horizon;
    FrameId frame_N = frame_k + mpc_horizon;

    ObjectPoseMap object_poses;
    ObjectMotionMap object_motions;

    for (FrameId frame_id = frame_k; frame_id < frame_N; frame_id++) {
      auto motion_key_k = ObjectMotionSymbol(object_to_follow, frame_id);

      if (auto query = accessor->query<gtsam::Pose3>(motion_key_k); query) {
        gtsam::Pose3 H = *query;
        gtsam::Pose3 L = H * L_e;
        object_poses.insert22(object_to_follow, frame_id, L);

        Motion3ReferenceFrame H_ref(H, MotionRepresentationStyle::KF,
                                    ReferenceFrame::GLOBAL, reference_frame_e,
                                    frame_id);
        object_motions.insert22(object_to_follow, frame_id, H_ref);
      }
    }

    return {object_motions, object_poses};
  }
}

gtsam::Pose3Vector MPCFormulation::getPredictedCameraPoses(
    FrameId frame_k) const {
  auto mpc_horizon = mpc_data_.mpc_horizon;
  FrameId frame_N = frame_k + mpc_horizon;

  auto accessor = this->accessorFromTheta();

  gtsam::Pose3Vector camera_poses;
  for (FrameId frame_id = frame_k; frame_id < frame_N; frame_id++) {
    auto camera_key = CameraPoseSymbol(frame_id);

    if (auto query = accessor->query<gtsam::Pose3>(camera_key); query) {
      camera_poses.push_back(*query);
    }
  }

  return camera_poses;
}

MPCFormulation::MPCFormulation(const FormulationParams& params,
                               typename Map::Ptr map,
                               const NoiseModels& noise_models,
                               const Sensors& sensors,
                               const FormulationHooks& hooks)
    : Base(params, map, noise_models, sensors, hooks),
      mpc_data_({FLAGS_mpc_horizon, 1}),
      dt_(FLAGS_mpc_dt),
      mission_type_(static_cast<MissionType>(FLAGS_mission_type)) {
  vel2d_prior_noise_ =
      gtsam::noiseModel::Isotropic::Sigma(2u, FLAGS_mpc_vel2d_prior_sigma);

  vel2d_limit_noise_ =
      gtsam::noiseModel::Isotropic::Sigma(2u, FLAGS_mpc_vel2d_limit_sigma);
  accel2d_limit_noise_ =
      gtsam::noiseModel::Isotropic::Sigma(2u, FLAGS_mpc_accel2d_limit_sigma);

  accel2d_cost_noise_ =
      gtsam::noiseModel::Isotropic::Sigma(2u, FLAGS_mpc_accel2d_cost_sigma);
  accel2d_smoothing_noise_ = gtsam::noiseModel::Isotropic::Sigma(
      2u, FLAGS_mpc_accel2d_smoothing_sigma);

  dynamic_factor_noise_ =
      gtsam::noiseModel::Isotropic::Sigma(8u, FLAGS_mpc_dynamic_factor_sigma);

  // object_prediction_constant_motion_noise_ =
  // gtsam::noiseModel::Isotropic::Sigma(6u,
  // FLAGS_mpc_object_prediction_smoothing_sigma);
  object_prediction_constant_motion_noise_ =
      gtsam::noiseModel::Isotropic::Sigma(
          6u, FLAGS_mpc_object_prediction_constant_motion_sigma);
  // follow_noise_ =
  //     gtsam::noiseModel::Isotropic::Sigma(2u, FLAGS_mpc_follow_sigma);
  // TODO: This noise shouldn't be isotropic. Distance is more important than
  // heading. If heading cost is too strong, robot won't go around the obstacle
  follow_noise_ = gtsam::noiseModel::Diagonal::Sigmas(
      gtsam::Vector2(FLAGS_mpc_follow_sigma, 5e-2));

  goal_noise_ = gtsam::noiseModel::Isotropic::Sigma(3u, FLAGS_mpc_goal_sigma);
  static_obstacle_X_noise_ = gtsam::noiseModel::Isotropic::Sigma(
      1u, FLAGS_mpc_static_sdf_X_safety_distance_sigma);
  static_obstacle_H_noise_ = gtsam::noiseModel::Isotropic::Sigma(
      1u, FLAGS_mpc_static_sdf_H_safety_distance_sigma);

  dynamic_obstacle_factor_ =
      gtsam::noiseModel::Isotropic::Sigma(1u, FLAGS_mpc_dynamic_obstacle_sigma);

  lin_vel_ = Limits{-0.3, 1.0};
  ang_vel_ = Limits{-0.5, 0.5};
  lin_acc_ = Limits{-1.0, 0.5};
  ang_acc_ = Limits{-0.5, 0.5};

  // lin_vel_ = Limits{-0.3, 1.2};
  // ang_vel_ = Limits{-0.5, 0.5};
  // lin_acc_ = Limits{-1.2, 0.8};
  // ang_acc_ = Limits{-0.5, 0.5};


  // lin_vel_ = Limits{-0.5, 1.72};
  // ang_vel_ = Limits{-0.8, 0.8};
  // lin_acc_ = Limits{-1.2, 0.8};
  // ang_acc_ = Limits{-1.0, 1.0};


  // lin_vel_ = Limits{-0.3, 1.6};
  // ang_vel_ = Limits{-1.0, 1.0};
  // lin_acc_ = Limits{-1.2, 1.2};
  // ang_acc_ = Limits{-0.8, 0.8};

  // lin_vel_ = Limits{-0.3, 1.2};
  // ang_vel_ = Limits{-0.5, 0.5};
  // lin_acc_ = Limits{-1.2, 0.8};
  // ang_acc_ = Limits{-0.5, 0.5};

  // For Following
  // lin_vel_ = Limits{-0.3, 1.8};
  // ang_vel_ = Limits{-0.8, 0.8};
  // lin_acc_ = Limits{-1.2, 0.7};
  // ang_acc_ = Limits{-0.8, 0.8};

  // For estimation task
  // lin_vel_ = Limits{-0.5, 1.72};
  // ang_vel_ = Limits{-0.8, 0.8};
  // lin_acc_ = Limits{-1.2, 0.8};
  // ang_acc_ = Limits{-1.0, 1.0};

  // lin_vel_ = Limits{-0.3, 1.6};
  // ang_vel_ = Limits{-1.0, 1.0};
  // lin_acc_ = Limits{-1.2, 1.2};
  // ang_acc_ = Limits{-0.8, 0.8};

  desired_follow_distance_ = FLAGS_mpc_desired_follow_distance;
  desired_follow_heading_ = FLAGS_mpc_desired_follow_heading;

  LOG(INFO) << "Creating MPC formulation with time horizon "
            << mpc_data_.mpc_horizon;

  if (mission_type_ == MissionType::FOLLOW) {
    LOG(INFO) << "Mission type is follow!";
  } else if (mission_type_ == MissionType::NAVIGATE) {
    LOG(INFO) << "Mission type is Navigate!";
  } else {
    LOG(FATAL) << "Unknown mission type for MPCFormulation!";
  }

  const std::string sdf_map_path = FLAGS_mpc_path_to_sdf_map;
  const std::string sdf_bin_path = sdf_map_path + ".bin";
  const std::string sdf_meta_path = sdf_map_path + ".txt";
  sdf_map_ = std::make_shared<SDFMap2D>(sdf_bin_path, sdf_meta_path);
  CHECK(sdf_map_);
  LOG(INFO) << "Made sdf map from path " << sdf_map_path;
}

void MPCFormulation::updateGlobalPath(Timestamp timestamp,
                                      const gtsam::Pose3Vector& global_path) {
  last_global_path_update_ = timestamp;
  global_path_ = global_path;
}

void MPCFormulation::otherUpdatesContext(
    const OtherUpdateContextType& context, UpdateObservationResult& result,
    gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors) {
  using namespace mpc_factors;

  // frames relative to the current real frame
  const FrameId frame_k = context.getFrameId();

  if (frame_k < 1) {
    return;
  }

  const FrameId frame_k_m1 = frame_k - 1u;
  const FrameId frame_k_m2 = frame_k - 2u;
  auto formatter = this->formatter();
  // TODO: now assume we reun with batch. Later must check the indicies
  if (factors_per_frame_.exists(frame_k_m1)) {
    result.batch_update_params.factors_to_remove =
        factors_per_frame_.at(frame_k_m1);

    // for (const auto& f : result.batch_update_params.factors_to_remove)
    //   f->print("Factors wanting removel ", this->formatter());
  }

  auto is_future_frame = [&frame_k](FrameId frame_id) -> bool {
    return frame_id > frame_k;
  };
  auto is_past_frame = [&frame_k](FrameId frame_id) -> bool {
    return frame_id < frame_k;
  };
  auto is_current_frame = [&frame_k](FrameId frame_id) -> bool {
    return frame_id == frame_k;
  };

  gtsam::NonlinearFactorGraph factors_to_remove_this_frame;

  // 1. Initalise values of k + N
  //. Poses X(k+1) -> X(k+N)
  //. 2DV V(k) -> V(k+N)
  //. 2DA A(k) -> A(k+N-1)
  //. Object prediction L(k) -> L(k+N)
  // 2. Add factors k + N
  //. Prior factor on V(k)
  //. Limit factors on V(k) -> V(N)
  //. Limit factors on A(k) -> A(N-1)
  //. Cost factors on A(k) -> A(N-1)
  //. Smoothing factors between A(...)
  //. Dynamic factors {X(k), V(k), A(k), X(k+1), V(k+1)} -> {N}
  //. Goal factor on X(N) (later)
  //. Follow factor on {X(k+1), L(k+1)} -> {X(N), L(N)}
  //. Prediction factor { L(k), L(k+1)} -> { L(N-1), L(N)}

  auto accessor = this->derivedAccessor<MPCAccessor>();
  CHECK(accessor);

  const auto mpc_horizon = mpc_data_.mpc_horizon;
  const auto object_to_follow = mpc_data_.object_to_follow;

  FrameId frame_N = frame_k + mpc_horizon;

  const bool& use_directed_factors = FLAGS_mpc_use_directed_factors;
  LOG(INFO) << "Using directed factors: " << std::boolalpha
            << use_directed_factors;

  // first try and get map offset pose if we dont already have
  // this sets the static offset between dynosam odom and the map frame (which
  // is called 'odom')
  if (!T_map_camera_ && viz_) {
    gtsam::Pose3 T;
    if (viz_->queryGlobalOffset(T)) {
      T_map_camera_ = T;
      LOG(INFO) << "Found offset between dynosam pose and map frame!! " << T;
      CHECK(sdf_map_);
      sdf_map_->setQueryOffset(T);
    }
  }

  bool sdf_map_valid = (bool)T_map_camera_;
  if (sdf_map_valid)
    LOG(INFO) << "Sdf map is valid. Can add obstacle factors if desired";

  // values init camera pose, 2dvelocity, 2d acceletation
  for (FrameId frame_id = frame_k; frame_id < frame_N; frame_id++) {
    FrameId frame_id_m1 = frame_id - 1u;

    auto camera_key = CameraPoseSymbol(frame_id);
    auto control_key = this->makeControlCommandKey(frame_id);
    auto accel_key = this->makeAccelerationKey(frame_id);

    auto camera_key_previous = CameraPoseSymbol(frame_id_m1);
    auto control_key_previous = this->makeControlCommandKey(frame_id_m1);
    auto accel_key_previous = this->makeAccelerationKey(frame_id_m1);

    // check if we have x(k), v(k), a(k)
    gtsam::Pose3 X_k;
    if (auto sensor_pose_query =
            accessor->queryWithTheta<gtsam::Pose3>(camera_key, new_values);
        !sensor_pose_query) {
      // LOG(INFO) << "Inserting future key " << formatter(camera_key);

      auto sensor_pose_query_previous = accessor->queryWithTheta<gtsam::Pose3>(
          camera_key_previous, new_values);
      CHECK(sensor_pose_query_previous);
      new_values.insert(camera_key, *sensor_pose_query_previous);

      if (is_future_frame(frame_id)) {
        result.keys_to_not_marginalize.insert(camera_key);
      }

      X_k = *sensor_pose_query_previous;
    } else {
      X_k = *sensor_pose_query;
    }

    gtsam::Vector2 V_k;
    if (auto velocity2d_query =
            accessor->queryWithTheta<gtsam::Vector2>(control_key, new_values);
        !velocity2d_query) {
      // LOG(INFO) << "Inserting future key " << formatter(control_key);

      auto control_query_previous = accessor->queryWithTheta<gtsam::Vector2>(
          control_key_previous, new_values);
      gtsam::Vector2 control_value;
      if (control_query_previous) {
        control_value = *control_query_previous;
      } else {
        control_value = gtsam::Vector2(0, 0);
      }
      new_values.insert(control_key, control_value);

      if (is_future_frame(frame_id)) {
        result.keys_to_not_marginalize.insert(control_key);
      }

      // limit factor
      auto limit_factor = boost::make_shared<Vec2LimitFactor>(
          control_key, lin_vel_.min, lin_vel_.max, ang_vel_.min, ang_vel_.max,
          vel2d_limit_noise_, dt_);

      V_k = control_value;
    } else {
      V_k = *velocity2d_query;
    }

    // at each timestep (update) assume we have removed all old factors
    // perterining to forward prediction stuff

    // add velocity and acceleration prior only on first of time horizon
    if (frame_id == frame_k) {
      // velocity prior
      auto velocity_prior_factor =
          boost::make_shared<gtsam::PriorFactor<gtsam::Vector2>>(
              control_key, V_k, vel2d_prior_noise_);

      new_factors.add(velocity_prior_factor);
      factors_to_remove_this_frame.add(velocity_prior_factor);

      // add prior on hanging (outdated) velocity and acceleration factor
      // this should only be called once per update iteration and handles the
      // case that we no longer need old (ie < current time k) velocity and
      // accelration values these will no longer be connected to anything but we
      // cannot delete them from the Smoother and we dont want to marginalize
      // them
      if (auto control_query_previous =
              accessor->queryWithTheta<gtsam::Vector2>(control_key_previous,
                                                       new_values);
          control_query_previous) {
        auto stabilising_velocity_prior =
            boost::make_shared<gtsam::PriorFactor<gtsam::Vector2>>(
                control_key_previous, *control_query_previous,
                vel2d_prior_noise_);

        new_factors.add(stabilising_velocity_prior);
      }

      if (auto accel_query_previous = accessor->queryWithTheta<gtsam::Vector2>(
              accel_key_previous, new_values);
          accel_query_previous) {
        auto stabilising_accel_prior =
            boost::make_shared<gtsam::PriorFactor<gtsam::Vector2>>(
                accel_key_previous, *accel_query_previous, vel2d_prior_noise_);

        new_factors.add(stabilising_accel_prior);
      }
    }

    if (frame_id != frame_k) {
      CHECK_GT(frame_id, frame_k);
      // limit factors on planning only
      auto velocity_limit_factor = boost::make_shared<Vec2LimitFactor>(
          control_key, lin_vel_.min, lin_vel_.max, ang_vel_.min, ang_vel_.max,
          vel2d_limit_noise_, dt_);

      new_factors.add(velocity_limit_factor);
      factors_to_remove_this_frame.add(velocity_limit_factor);

      // add static obstacle factor on planned poses from frame_k + 1 -> frame_N
      if (sdf_map_valid) {
        auto static_obstacle_factor =
            boost::make_shared<SDFStaticObstacleXFactor>(
                camera_key, CHECK_NOTNULL(sdf_map_),
                FLAGS_mpc_static_sdf_X_safety_distance,
                static_obstacle_X_noise_);

        new_factors.add(static_obstacle_factor);
        factors_to_remove_this_frame.add(static_obstacle_factor);
        // LOG(INFO) << "Adding static obstacle factor " <<
        // formatter(camera_key);
      }
    }

    // add acceleration up to k+N-1
    if (frame_id < frame_N - 1) {
      // acceleration value at time-step k
      gtsam::Vector2 A_k;
      if (auto acceleration2d_query =
              accessor->queryWithTheta<gtsam::Vector2>(accel_key, new_values);
          !acceleration2d_query) {
        // LOG(INFO) << "Inserting future key " << formatter(accel_key);

        auto accel_query_previous = accessor->queryWithTheta<gtsam::Vector2>(
            accel_key_previous, new_values);
        gtsam::Vector2 accel_value;
        if (accel_query_previous) {
          accel_value = *accel_query_previous;
        } else {
          accel_value = gtsam::Vector2(0, 0);
        }
        new_values.insert(accel_key, accel_value);

        if (is_future_frame(frame_id)) {
          result.keys_to_not_marginalize.insert(accel_key);
        }

        A_k = accel_value;
      } else {
        A_k = *acceleration2d_query;
      }

      // acceleration prior
      auto acceleration_cost_factor =
          boost::make_shared<gtsam::PriorFactor<gtsam::Vector2>>(
              accel_key, A_k, accel2d_cost_noise_);

      // limit factor
      auto acceleration_limit_factor = boost::make_shared<Vec2LimitFactor>(
          accel_key, lin_acc_.min, lin_acc_.max, ang_acc_.min, ang_acc_.max,
          accel2d_limit_noise_, dt_);

      new_factors.add(acceleration_cost_factor);
      new_factors.add(acceleration_limit_factor);

      factors_to_remove_this_frame.add(acceleration_cost_factor);
      factors_to_remove_this_frame.add(acceleration_limit_factor);

      // we are at least the second frame of the horizon
      if (frame_id == frame_k) {
        // add direcrd between factor between current acceleration and previous
        // if we have a previous acceletation relies on also having a prior
        // factor on a(k-1) this should be set in the previous condition which
        // adds stabilising factors on velocity and acceleration on the first
        // iteration (ie. when frame_id == frame_k)
        bool has_previous_acceleration =
            static_cast<bool>(accessor->queryWithTheta<gtsam::Vector2>(
                accel_key_previous, new_values));
        if (has_previous_acceleration) {
          auto factor = boost::make_shared<Vec2BetweenFactorDirected1>(
              accel_key_previous, accel_key,
              gtsam::traits<gtsam::Vector2>::Identity(),
              accel2d_smoothing_noise_);

          new_factors.add(factor);
          factors_to_remove_this_frame.add(factor);
        }
      } else {
        // here accel_key_previous == a(k)
        // and accel_key == a(k+1) since frame_id iterates from frame_k to
        // frame_N
        auto factor = boost::make_shared<gtsam::BetweenFactor<gtsam::Vector2>>(
            accel_key_previous, accel_key,
            gtsam::traits<gtsam::Vector2>::Identity(),
            accel2d_smoothing_noise_);

        new_factors.add(factor);
        factors_to_remove_this_frame.add(factor);
      }
    }

    // add dynamic factor
    if (frame_id > frame_k) {
      // TODO: should do checks all these keys actually exist...

      gtsam::NonlinearFactor::shared_ptr factor = nullptr;

      if (frame_id == frame_k + 1) {
        if (use_directed_factors) {
          // add zero jacobian on first 'pair' of iteration
          factor = boost::make_shared<MotionModelFactor<1, 3>>(
              camera_key_previous, camera_key, control_key_previous,
              control_key, accel_key_previous, dynamic_factor_noise_, dt_);
        } else {
          factor = boost::make_shared<MotionModelFactor<>>(
              camera_key_previous, camera_key, control_key_previous,
              control_key, accel_key_previous, dynamic_factor_noise_, dt_);
        }

      } else {
        factor = boost::make_shared<MotionModelFactor<>>(
            camera_key_previous, camera_key, control_key_previous, control_key,
            accel_key_previous, dynamic_factor_noise_, dt_);
      }

      CHECK(factor);
      new_factors.add(factor);
      factors_to_remove_this_frame.add(factor);
    }
  }

  auto map = this->map();
  auto frame_node_k = map->getFrame(frame_k);
  CHECK(frame_node_k) << "Frame node null at k=" << frame_k;

  // logic flags that determine how we handle adding of predicted object motion
  // factors
  bool object_available = false;
  bool object_reappeared = false;
  bool object_disappeared = false;
  bool object_new = false;

  /**
   * We need at least two frames before a motion is added (it is always added as
   * a pair the first time an object is seen/re-appears) So to check that an
   * object is new/re-appeared we need to check back 2 frames but to check that
   * an object has disappeared we only need to check back once frame.
   *
   * Cannot rely on object observed functions from the map as this does not tell
   * us if the object is in the optimisation problem.
   *
   * NOTE: if a motion is in an opt it may come from the estimator or from the
   * prediction depending if k > frame_k
   *
   */
  const bool object_in_opt_at_km2 = static_cast<bool>(
      accessor->getObjectMotion(frame_k_m2, object_to_follow));
  const bool object_in_opt_at_km1 = static_cast<bool>(
      accessor->getObjectMotion(frame_k_m1, object_to_follow));
  const bool object_in_opt_at_k =
      static_cast<bool>(accessor->getObjectMotion(frame_k, object_to_follow));
  const bool object_in_opt_has_embedded_frame =
      this->hasObjectKeyFrame(object_to_follow, frame_k);
  const bool object_in_opt_at_current =
      object_in_opt_has_embedded_frame && object_in_opt_at_k;

  // handle removal of motion keys in the past.
  if (object_in_opt_at_k) {
    // auto motion_key_k = accessor->getObjectMotion(frame_k,
    // object_to_follow).key(); LOG(INFO) << "Adding motion key " <<
    // formatter(motion_key_k) << " into marginalization set";
    // result.additional_keys_to_marginalize[motion_key_k] = frame_k;
  }

  // object not observed or has no keyframe
  // it may be observed but checking for its embedded frame tells us its in the
  // optimisation problem
  // we use object motion expected to indicate that the object is seen at k-1
  // and k and not just in the optimsiation as they may happen from estimation
  // or prediction!
  const bool seen_and_in_opt =
      frame_node_k->objectMotionExpected(object_to_follow) &&
      object_in_opt_at_current;
  // object exists at all (may not be observed at this frame!)
  // logic to track that object is seen and we should do predictions etc...
  if (objects_update_data_.exists(object_to_follow)) {
    const ObjectUpdateData& update_data =
        objects_update_data_.at(object_to_follow);

    auto object_node = map->getObject(object_to_follow);
    CHECK(object_node);

    if (frame_k_m1 == object_node->getFirstSeenFrame()) {
      object_new = true;
      LOG(INFO) << "Object j=" << object_to_follow << " is new";
    }

    // object must be optimized at least N times before we start to use its
    // prediction count is updated in RegularHybrid::postUpdate so only
    // increments after an optimization!
    constexpr static int kMinNumberMotionsOptimized = 3;
    if (update_data.count > kMinNumberMotionsOptimized && seen_and_in_opt) {
      object_available = true;
    }

    CHECK(this->hasObjectKeyFrame(object_to_follow, frame_k));

    FrameId last_update_frame = update_data.frame_id;

    // duplicate logic from RegularHybridEstimator::preUpdate
    if (object_available && !object_new && (frame_k > 0) &&
        (last_update_frame < (frame_k - 1u))) {
      LOG(INFO) << "Object j=" << object_to_follow << " reappeared!";
      object_reappeared = true;
    }

    // not seen in this frame but was updated at this frame or last frame
    if (!seen_and_in_opt && (frame_k > 0) &&
        (last_update_frame >= (frame_k - 1u))) {
      // for an object to disppear it must have existed at least once
      // we can put this logic inside the objects_update_data_.exists == true
      // condition
      object_disappeared = true;
      LOG(INFO) << "Object j=" << object_to_follow << " is disappeared";
    }

    // //if object re-appeared check its frame count. If not enough set
    // object_available
    // //this is only to handle the re-appearing case and we want to wait
    // kMinNumberMotionsOptimized frames again before
    // //forward predicring!!
    // if(object_reappeared) {
    //   const size_t& object_count =
    //   object_prediction_count_.at(object_to_follow); if(object_count <
    //   kMinNumberMotionsOptimized) {
    //     object_available = false;
    //   }
    // }
  }

  {
    // sanity check
    // object cannot have disappeared and also be available
    if (object_available) CHECK(!object_disappeared);

    // if(!object_available) CHECK(!object_reappeared);
  }

  if (!object_available) {
    VLOG(10) << "j=" << object_to_follow << "is unavailable at k=" << frame_k;

    bool mission_factors_expired = false;
    // keep mission factors while variables are still within the time horizon
    if (previous_mission_factors_.exists(object_to_follow)) {
      VLOG(10) << "Found previous mission factors "
               << info_string(frame_k, object_to_follow);

      const MissionFactorGraph& current_mission_factors =
          previous_mission_factors_.at(object_to_follow);
      MissionFactorGraph factors_within_horizon;

      for (auto mission_factor : current_mission_factors) {
        if (mission_factor->isFuture(frame_k)) {
          VLOG(10) << "Keeping mission factor "
                   << formatter(mission_factor->cameraPoseKey()) << " "
                   << formatter(mission_factor->objectMotionKey());
          // TODO: lots of adding new factors and then deleting factors every
          // time
          //  this is very slow!!!
          factors_within_horizon.add(mission_factor);
          new_factors.add(mission_factor);
          factors_to_remove_this_frame.add(mission_factor);
        } else {
          // delete factor immediately!
          VLOG(10) << "Deleting mission factor "
                   << formatter(mission_factor->cameraPoseKey()) << " "
                   << formatter(mission_factor->objectMotionKey());
          result.batch_update_params.factors_to_remove.add(mission_factor);
        }
      }
      // update set of current mission factors in previous_mission_factors_
      previous_mission_factors_.at(object_to_follow) = factors_within_horizon;

      if (factors_within_horizon.empty()) {
        VLOG(10) << "No more mission factors for "
                 << info_string(frame_k, object_to_follow);
        mission_factors_expired = true;
        // explicitly delete set of mission factors so we know there are none!!
        previous_mission_factors_.erase(object_to_follow);
      }
    } else {
      mission_factors_expired = true;
    }

    // if we have no more (or just none at all) mission factors AND we're in
    // follow mode add a stabilising prior factor on the last pose
    if (mission_factors_expired && mission_type_ == MissionType::FOLLOW) {
      VLOG(10) << "Adding stabilising prior factor ";
      // add factor on last pose if no object (for now) to stabilise planning
      // using value of latest real pose (ie frame k)
      auto camera_pose_k =
          accessor->query<gtsam::Pose3>(CameraPoseSymbol(frame_k));
      CHECK(camera_pose_k);
      auto stabilising_pose_prior =
          boost::make_shared<gtsam::PriorFactor<gtsam::Pose3>>(
              CameraPoseSymbol(frame_N - 1), *camera_pose_k,
              gtsam::noiseModel::Isotropic::Sigma(6u, 0.1));
      new_factors.add(stabilising_pose_prior);
      factors_to_remove_this_frame.add(stabilising_pose_prior);
    }

    // if disappeared add stabilising factors on predicted motions!!
    if (object_disappeared) {
      LOG(INFO) << object_to_follow
                << " assumed to have disappreared k=" << frame_k;

      // if we're adding stabilising object factors they should not already
      // exist
      CHECK(!stabilising_object_factors_.exists(object_to_follow));
      gtsam::NonlinearFactorGraph stabilising_factors;

      LOG(INFO) << "Adding stabilising motion factors " << frame_k << " -> "
                << frame_N - 1u;
      // Object observed in previous frame so only add stabilising factors on
      // the predicted motions this will end at frame_N - 1 becuase the last set
      // of predictions happend at the previous frame which is equivalanet to
      // (k-1)+1 to k-1+N
      for (FrameId frame_id = frame_k; frame_id < frame_N - 1u; frame_id++) {
        auto motion_key_to_be_stabilised =
            ObjectMotionSymbol(object_to_follow, frame_id);
        auto motion_to_be_stabilised_query =
            accessor->queryWithTheta<gtsam::Pose3>(motion_key_to_be_stabilised,
                                                   new_values);
        CHECK(motion_to_be_stabilised_query)
            << "No motion at  "
            << formatter(motion_to_be_stabilised_query.key());
        LOG(INFO) << "Addomg stabilising factor "
                  << formatter(motion_to_be_stabilised_query.key());

        auto stabilising_motion_prior =
            boost::make_shared<gtsam::PriorFactor<gtsam::Pose3>>(
                motion_key_to_be_stabilised, *motion_to_be_stabilised_query,
                gtsam::noiseModel::Isotropic::Sigma(6u, 0.1));
        new_factors.add(stabilising_motion_prior);
        stabilising_factors.add(stabilising_motion_prior);
      }

      stabilising_object_factors_.insert2(object_to_follow,
                                          stabilising_factors);
    }
  } else {
    // object available
    LOG(INFO) << "Object " << info_string(frame_k, object_to_follow)
              << " good. Adding predictions!";

    MissionFactorGraph mission_factors;
    // specicifically these are (existing) object motions that were in the
    // future and now are in the past and therefore have a different keyframe.
    // We will need to update the value to reflect the new keyframe
    gtsam::KeySet motions_with_new_keyframes;
    if (object_reappeared) {
      // remove all stabilising factors as we're about to add new ones
      if (stabilising_object_factors_.exists(object_to_follow)) {
        // HACK (sort of) of object re-seen within horizon overlap
        // ie. there are still predicted motions from last time we saw the
        // object but these have timestamp < k (ie now in the past) we keep
        // priors on these to stabilise them. Really this is WRONG (we should
        // remove them) by passing a new set of variables to the BackendModule
        // telling it which new variables to marginalize We current dont do this
        // as it messes up the following logic of which new variables to add..
        gtsam::NonlinearFactorGraph stabilising_motion_factors =
            stabilising_object_factors_.at(object_to_follow);
        for (auto factor : stabilising_motion_factors) {
          auto prior_factor =
              boost::dynamic_pointer_cast<gtsam::PriorFactor<gtsam::Pose3>>(
                  factor);
          CHECK(prior_factor) << "Unknown stabilising motion factor type!";
          auto motion_key = prior_factor->key1();

          ObjectId object_id;
          FrameId frame_id;
          CHECK(reconstructMotionInfo(motion_key, object_id, frame_id));
          (void)object_id;

          // remove as we're about to add new ones!
          // for motion prediction we also add a smoothing factor which connects
          // k-1, k and k+1 (prediction) to not destroy estimation we then also
          // need to remove prior on k-1
          if (is_future_frame(frame_id) || is_current_frame(frame_id) ||
              frame_id == frame_k_m1) {
            LOG(INFO) << "Removing stabilising factor "
                      << formatter(motion_key);
            // NOTE: here we directly update the result with this factor so it
            // removes it immediately adding to factors_to_remove_this_frame
            // will remove it at the next frame! this only makes sense for other
            // factors...
            result.batch_update_params.factors_to_remove.add(factor);
            motions_with_new_keyframes.insert(motion_key);
          }
        }
        // remove tracking of these factors as we'll never touch these
        // stabilishing factors again
        stabilising_object_factors_.erase(object_to_follow);

      } else {
        // NOTE: this should not actually happen! I think there is some
        // inconsistencies between how/when the appearence/disappearnce flags
        // are set. Maybe an obejct is seen and in the opt (from previous
        // prediction window) but not in the opt from new measurements. In this
        // case the last_update_frame will not be updated (since this is updated
        // when we get new measurements of the object) and therefore it will
        // re-appear twice (which I think is what happens) rather than
        // appearing->disappearing->appearing...
        LOG(FATAL) << "Object has reappeared but no previous stabilising "
                      "factors. This means it REappeared without disappearing!";
      }

      // if object has re-appeared we need to delete all previous mission
      // factors as we're about to add all new ones
      if (previous_mission_factors_.exists(object_to_follow)) {
        VLOG(10) << "Found previous mission factors "
                 << info_string(frame_k, object_to_follow) << ". Deleting all";

        const MissionFactorGraph& remaining_mission_factor =
            previous_mission_factors_.at(object_to_follow);
        result.batch_update_params.factors_to_remove.add(
            remaining_mission_factor);
        previous_mission_factors_.erase(object_to_follow);
      }
    }

    CHECK(frame_node_k->objectObserved(object_to_follow));

    // values and factors init object pose
    // dont get the latest motion/pose (the ones added by the base Hybrid
    // formulation) as these will not be optimized yet. Get the ones from the
    // previous frame which are better. Maybe bug in initalisation of latest
    // motion
    auto L_W_k_query = accessor->getObjectPose(frame_k_m1, object_to_follow);
    auto H_W_km1_k_query = accessor->getObjectMotion(frame_k, object_to_follow);

    // have already chceck that object should be in previous
    CHECK(L_W_k_query);
    CHECK(H_W_km1_k_query);

    // assume we have a timestamp_km1_ set before we get here
    // this is only true in implementation as we need at least two frames to get
    // a motion but we actually have a pose at all frames!!!
    Timestamp frame_dt = context.timestamp - timestamp_km1_;
    LOG(INFO) << "Calculated dt " << frame_dt;

    // if (L_W_k_query && H_W_km1_k_query) {
    // the estimated motions we are going to connect our constant motion factors
    // with
    auto motion_key_km1_est = ObjectMotionSymbol(object_to_follow, frame_k_m1);
    auto motion_key_k_est = ObjectMotionSymbol(object_to_follow, frame_k);

    const auto [reference_frame_e, L_e] =
        this->getObjectKeyFrame(object_to_follow, frame_k);

    for (const auto& motion_key : motions_with_new_keyframes) {
      VLOG(10) << "Correcting object motion with new linearization point "
               << formatter(motion_key);

      ObjectId object_id_internal;
      FrameId frame_id_internal;
      CHECK(reconstructMotionInfo(motion_key, object_id_internal,
                                  frame_id_internal));

      // ideally want to check what reference_frame_e these motions were
      // associated with but as we have
      // just made a new keyframe these frame_ids will be associated with the
      // new frame e and not the previous one!
      // const auto [reference_frame_e_internal, L_e_internal] =
      //   this->getObjectKeyFrame(object_id_internal, frame_id_internal);

      // CHECK_LT(reference_frame_e_internal, reference_frame_e);
      // //these motions must be in the past now
      // CHECK_LT(frame_id_internal, frame_k);
      // //HACK set to identity and let the optimizer figure it out...
      // really should be whatever computeInitialH calculates..?
      result.batch_update_params.values_relinearize.insert(
          motion_key, gtsam::Pose3::Identity());
    }

    gtsam::Pose3 L_W_k = *L_W_k_query;
    gtsam::Pose3 H_W_km1_k = *H_W_km1_k_query;
    // gtsam::Pose3 H_W_km1_k = gtsam::Pose3(gtsam::Rot3::Identity(),
    // gtsam::Point3(1, 0, 0));
    gtsam::Pose3 L_W_future = L_W_k;

    ObjectIds objects_predicted{object_to_follow};
    predicted_objects_at_frame_.insert2(frame_k, objects_predicted);

    for (FrameId frame_id = frame_k + 1; frame_id < frame_N; frame_id++) {
      // progate frame L using constant realtive motion
      // then convert the progogated frame into H
      L_W_future = H_W_km1_k * L_W_future;
      // following L_k = H_e_k * L_e
      gtsam::Pose3 H_W_future = L_W_future * L_e.inverse();

      auto motion_key_k_predicted =
          ObjectMotionSymbol(object_to_follow, frame_id);
      auto camera_pose_key_k_predicted = CameraPoseSymbol(frame_id);

      // k-1 and k-2 relative to the current forward predictive timetamp
      auto motion_key_k_m1_predicted =
          ObjectMotionSymbol(object_to_follow, frame_id - 1u);
      auto motion_key_k_m2_predicted =
          ObjectMotionSymbol(object_to_follow, frame_id - 2u);

      // motion does not exist so we should insert it
      if (auto hybrid_motion_query = accessor->queryWithTheta<gtsam::Pose3>(
              motion_key_k_predicted, new_values);
          !hybrid_motion_query) {
        LOG(INFO) << "Inserting future key "
                  << formatter(motion_key_k_predicted);
        new_values.insert(motion_key_k_predicted, H_W_future);

        // TODO: evenetually if this object is reobserved any intermdiate motion
        // predictions ie between i=k and j where the object is reobserved will
        // need to be explicitly removed
        if (is_future_frame(frame_id)) {
          result.keys_to_not_marginalize.insert(motion_key_k_predicted);
        }
      }

      using HybridConstantMotionFactorDirected1And2 =
          HybridConstantMotionFactor<1, 2>;
      using HybridConstantMotionFactorDirected1 = HybridConstantMotionFactor<1>;

      // first iteration:
      // motion_key_k_m2_predicted is km1 (ie the second last motion in the
      // estimation) motion_key_k_m1_predicted is k (ie. the last motion in the
      // estimation) we therefore prevent information flow back to these
      // variables
      gtsam::NonlinearFactor::shared_ptr constant_motion_factor = nullptr;
      if (frame_id == frame_k + 1) {
        CHECK_EQ(frame_id - 2u, frame_k - 1u);
        CHECK_EQ(frame_id - 1u, frame_k);
        constant_motion_factor =
            boost::make_shared<HybridConstantMotionFactorDirected1And2>(
                motion_key_k_m2_predicted, motion_key_k_m1_predicted,
                motion_key_k_predicted, L_e,
                object_prediction_constant_motion_noise_);

      }
      // second iteration:
      // motion_key_k_m2_predicted is km (ie. the last motion in the estimation)
      // motion_key_k_m1_predicted is k+1 (ie. the first predicted motion)
      // we therefore prevent information flow back to only
      // motion_key_k_m2_predicted
      else if (frame_id == frame_k + 2) {
        // here motion_key_k_m2_predicted == object_node(k) -> makeMotionKey()
        // ie the last motion in the optimisation problem
        constant_motion_factor =
            boost::make_shared<HybridConstantMotionFactorDirected1>(
                motion_key_k_m2_predicted, motion_key_k_m1_predicted,
                motion_key_k_predicted, L_e,
                object_prediction_constant_motion_noise_);

      }
      // third iteration:
      // all variales in prediction!
      else {
        constant_motion_factor =
            boost::make_shared<HybridConstantMotionFactor<>>(
                motion_key_k_m2_predicted, motion_key_k_m1_predicted,
                motion_key_k_predicted, L_e,
                object_prediction_constant_motion_noise_);
      }

      if (!use_directed_factors) {
        // replace with non-directed factor
        constant_motion_factor =
            boost::make_shared<HybridConstantMotionFactor<>>(
                motion_key_k_m2_predicted, motion_key_k_m1_predicted,
                motion_key_k_predicted, L_e,
                object_prediction_constant_motion_noise_);
      }
      CHECK(constant_motion_factor);
      new_factors.add(constant_motion_factor);
      factors_to_remove_this_frame.add(constant_motion_factor);

      // sdf factor on the dynamic obstacle to use the map to inform prediction
      // we assume the obstacle is smart and therefore will try and avoid places
      // where it cannot go!
      auto obstacle_H_factor = boost::make_shared<SDFStaticObstacleHFactor>(
          motion_key_k_predicted, sdf_map_,
          FLAGS_mpc_static_sdf_H_safety_distance, L_e,
          static_obstacle_H_noise_);
      new_factors.add(obstacle_H_factor);
      factors_to_remove_this_frame.add(obstacle_H_factor);

      {
        // sanity check
        CHECK(accessor->queryWithTheta<gtsam::Pose3>(
            camera_pose_key_k_predicted, new_values));
      }

      MissionFactorBase::shared_ptr mission_factor = nullptr;
      if (mission_type_ == MissionType::FOLLOW) {
        // add follow factors
        mission_factor = boost::make_shared<HybridMotionFollowJac0Factor>(
            camera_pose_key_k_predicted, motion_key_k_predicted, L_e,
            desired_follow_distance_, desired_follow_heading_, follow_noise_);

      } else if (mission_type_ == MissionType::NAVIGATE) {
        if (use_directed_factors) {
          // prevent motion from being affected
          mission_factor = boost::make_shared<DynamicObstacleFactor<2>>(
              camera_pose_key_k_predicted, motion_key_k_predicted, L_e,
              FLAGS_mpc_dynamic_obstacle_safety_distance,
              dynamic_obstacle_factor_);
        } else {
          mission_factor = boost::make_shared<DynamicObstacleFactor<>>(
              camera_pose_key_k_predicted, motion_key_k_predicted, L_e,
              FLAGS_mpc_dynamic_obstacle_safety_distance,
              dynamic_obstacle_factor_);
        }
      }
      CHECK(mission_factor);
      new_factors.add(mission_factor);
      factors_to_remove_this_frame.add(mission_factor);
      mission_factors.add(mission_factor);
    }
    // replace previous mission factors
    if (!previous_mission_factors_.exists(object_to_follow)) {
      previous_mission_factors_.insert2(object_to_follow, MissionFactorGraph{});
    }

    if (mission_factors.size() > 0) {
      previous_mission_factors_[object_to_follow] = mission_factors;
    } else {
      previous_mission_factors_.erase(object_to_follow);
    }
  }

  if (mission_type_ == MissionType::NAVIGATE) {
    const int local_horizon = static_cast<int>(FLAGS_mpc_local_goal_horizin);
    // handle global path
    gtsam::Pose3 local_goal;
    bool local_goal_result = getLocalGoalFromGlobalPath(
        accessor->getSensorPose(frame_k).value(), local_horizon, local_goal);

    if (local_goal_result) {
      local_goal_ = local_goal;

      VLOG(20) << "Calculated local goal from global plan with horizin: "
               << local_horizon;

      // final camera pose in the plan
      // TODO: check this is in the values?
      auto camera_key_N = CameraPoseSymbol(frame_N - 1u);

      // auto goal_factor = boost::make_shared<GoalFactorSE2>(
      //   camera_key_N,
      //   local_goal,
      //   goal_noise_
      // );
      auto goal_factor = boost::make_shared<gtsam::PriorFactor<gtsam::Pose3>>(
          camera_key_N, local_goal,
          gtsam::noiseModel::Isotropic::Sigma(6u, FLAGS_mpc_goal_sigma));
      LOG(INFO) << "Added goal factor for " << formatter(camera_key_N);
      new_factors.add(goal_factor);
      factors_to_remove_this_frame.add(goal_factor);

    } else {
      LOG(WARNING) << "MissionType set to navigate but global plan not found "
                      "or local goal cannot be set!";
      auto camera_pose_k =
          accessor->query<gtsam::Pose3>(CameraPoseSymbol(frame_k));
      CHECK(camera_pose_k);
      auto stabilising_pose_prior =
          boost::make_shared<gtsam::PriorFactor<gtsam::Pose3>>(
              CameraPoseSymbol(frame_N - 1), *camera_pose_k,
              gtsam::noiseModel::Isotropic::Sigma(6u, 0.1));
      new_factors.add(stabilising_pose_prior);
      factors_to_remove_this_frame.add(stabilising_pose_prior);
    }
  }

  factors_per_frame_[frame_k] = factors_to_remove_this_frame;
  timestamp_km1_ = context.timestamp;
  LOG(INFO) << "MPCFormulation::otherUpdatesContext";
}

void MPCFormulation::preUpdate(const PreUpdateData& data) {
  Base::preUpdate(data);
  if (viz_) viz_->inPreUpdate();
}

void MPCFormulation::postUpdate(const PostUpdateData& data) {
  Base::postUpdate(data);
  if (viz_) viz_->inPostUpdate();

  // if we get here the update must have been good
  FrameId frame_k = data.frame_id;
  FrameId frame_k_m1 = frame_k - 1u;
  // remove tracking of old factors
  if (factors_per_frame_.exists(frame_k_m1)) {
    VLOG(10) << "Removing interal old factors";
    factors_per_frame_.erase(frame_k_m1);
  }

  FrameId frame_N = frame_k + mpc_data_.mpc_horizon - 1;

  auto formatter = this->formatter();
  // values init camera pose, 2dvelocity, 2d acceletation
  for (FrameId frame_id = frame_k; frame_id < frame_N; frame_id++) {
    auto camera_key = CameraPoseSymbol(frame_id);
    auto control_key = this->makeControlCommandKey(frame_id);
    auto accel_key = this->makeAccelerationKey(frame_id);

    // LOG(INFO) << formatter(camera_key) << " " <<
    // theta_.at<gtsam::Pose3>(camera_key); LOG(INFO) << formatter(control_key)
    // << " " << theta_.at<gtsam::Vector2>(control_key); LOG(INFO) <<
    // formatter(accel_key) << " " << theta_.at<gtsam::Vector2>(accel_key);
  }

  // 1. Collect current estimates and publish to rviz
  // 2. Collect control command and send

  if (viz_) viz_->spin(data.timestamp, data.frame_id, this);
}

bool MPCFormulation::getLocalGoalFromGlobalPath(const gtsam::Pose3& X_k,
                                                size_t horizon,
                                                gtsam::Pose3& goal) {
  if (!global_path_) {
    LOG(WARNING) << "Cannot get local goal as global path not set!";
    return false;
  }

  const auto& path = *global_path_;
  auto find_closest_index = [](const gtsam::Pose3& X_k,
                               const gtsam::Pose3Vector& path) -> size_t {
    size_t best_index = 0;
    double best_dist2 = std::numeric_limits<double>::max();

    const auto& translation = X_k.translation();  // Point3

    for (size_t i = 0; i < path.size(); ++i) {
      const auto& t = path[i].translation();
      double dist2 = (t - translation).squaredNorm();

      if (dist2 < best_dist2) {
        best_dist2 = dist2;
        best_index = i;
      }
    }

    return best_index;
  };

  // find index in path which has the closest euclidean distance to the current
  // pose
  size_t closest_index = find_closest_index(X_k, path);
  size_t desired_local_goal_index = closest_index + horizon;

  size_t max_path_index = path.size() - 1u;
  // handle case where we are close to end of goal
  size_t local_goal_index = std::min(desired_local_goal_index, max_path_index);
  CHECK_LE(local_goal_index, max_path_index);

  gtsam::Pose3 goal_n = path.at(local_goal_index);

  if (local_goal_index == 0) {
    goal = goal_n;
  } else {
    gtsam::Pose3 goal_nm1 = path.at(local_goal_index - 1);

    double diff_x = goal_n.x() - goal_nm1.x();
    double diff_z = goal_n.z() - goal_nm1.z();

    double theta = std::atan2(diff_x, diff_z);
    gtsam::Rot3 rotation = gtsam::Rot3::Pitch(theta);
    goal = gtsam::Pose3(rotation, goal_n.translation());
  }

  return true;
}

}  // namespace dyno
