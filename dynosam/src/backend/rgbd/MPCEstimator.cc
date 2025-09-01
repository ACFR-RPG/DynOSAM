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

DEFINE_double(mpc_vel2d_prior_sigma, 1.0, "Sigma for the cam pose prior");

DEFINE_double(mpc_vel2d_limit_sigma, 1.0, "Sigma for the cam pose prior");
DEFINE_double(mpc_accel2d_limit_sigma, 1.0, "Sigma for the cam pose prior");

DEFINE_double(mpc_accel2d_cost_sigma, 1.0, "Sigma for the cam pose prior");
DEFINE_double(mpc_accel2d_smoothing_sigma, 1.0, "Sigma for the cam pose prior");

DEFINE_double(mpc_dynamic_factor_sigma, 1.0, "Sigma for mpc dynamic factor");
DEFINE_double(mpc_object_prediction_constant_motion_sigma, 0.01,
              "Sigma object prediction smoothing");
DEFINE_double(mpc_static_obstacle_sigma, 0.1,
              "Sigma for static obstacle factor");

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

DEFINE_double(mpc_static_safety_distance, 1.0,
              "Safety distance constant for a static obstacel");

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
      return 10.0;
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

class SDFStaticObstacleFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3> {
 public:
  using shared_ptr = boost::shared_ptr<SDFStaticObstacleFactor>;
  using This = SDFStaticObstacleFactor;
  using Base = gtsam::NoiseModelFactor1<gtsam::Pose3>;

  SDFStaticObstacleFactor(gtsam::Key X_k_key, std::shared_ptr<SDFMap2D> sdf_map,
                          double safety_distance, gtsam::SharedNoiseModel model)
      : Base(model, X_k_key),
        sdf_map_(CHECK_NOTNULL(sdf_map)),
        safety_distance_(safety_distance) {}

  // Clone method
  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::make_shared<SDFStaticObstacleFactor>(*this);
  }

  // Error function
  gtsam::Vector evaluateError(
      const gtsam::Pose3& X_k,
      boost::optional<gtsam::Matrix&> J1 = boost::none) const override {
    // TODO: repeated calculations!!
    double distance = sdf_map_->getDistanceFromPose(X_k);

    if (J1) {
      // NOTE: condition on distance directly without limit (std::max) as in
      // residual function!
      if (distance >= safety_distance_) {
        // Zero residual => zero Jacobian
        *J1 = (gtsam::Matrix(1, 6) << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0).finished();
      } else {
        Eigen::Matrix<double, 1, 6> df_f =
            gtsam::numericalDerivative11<gtsam::Vector1, gtsam::Pose3>(
                std::bind(&SDFStaticObstacleFactor::residual, this,
                          std::placeholders::_1),
                X_k);
        *J1 = df_f;
      }
    }
    gtsam::Vector e = residual(X_k);
    LOG(INFO) << "dist " << distance << " error " << e;
    return e;
  }

  // Compute residual (static utility function)
  gtsam::Vector residual(const gtsam::Pose3& X_k) const {
    double distance = sdf_map_->getDistanceFromPose(X_k);
    return gtsam::Vector1(std::max(0.0, safety_distance_ - distance));
  }

  inline gtsam::Key poseKey() const { return key1(); }

 private:
  std::shared_ptr<SDFMap2D> sdf_map_;
  double safety_distance_;
};

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

class HybridMotionFollowJac0Factor
    : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3> {
 public:
  using shared_ptr = boost::shared_ptr<HybridMotionFollowJac0Factor>;
  using This = HybridMotionFollowJac0Factor;
  using Base = gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3>;

  // Constructor
  HybridMotionFollowJac0Factor(gtsam::Key X_W_key_follower,
                               gtsam::Key H_W_e_k_key_leader,
                               const gtsam::Pose3& L_e_leader,
                               double des_distance, double des_heading,
                               gtsam::SharedNoiseModel model)
      : Base(model, X_W_key_follower, H_W_e_k_key_leader),
        L_e_leader_(L_e_leader),
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

    gtsam::Pose3 L_W_3d = H_W_e_k_leader_3d * L_e_leader_;

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
  gtsam::Pose3 L_e_leader_;
  double des_distance_;
  gtsam::Rot2 des_heading_;
};

class Pose3DynXVAFactor
    : public gtsam::NoiseModelFactor5<gtsam::Pose3, gtsam::Pose3,
                                      gtsam::Vector2, gtsam::Vector2,
                                      gtsam::Vector2> {
 public:
  using shared_ptr = boost::shared_ptr<Pose3DynXVAFactor>;
  using This = Pose3DynXVAFactor;
  using Base =
      gtsam::NoiseModelFactor5<gtsam::Pose3, gtsam::Pose3, gtsam::Vector2,
                               gtsam::Vector2, gtsam::Vector2>;

  // Constructor
  Pose3DynXVAFactor(gtsam::Key pose1Key, gtsam::Key pose2Key,
                    gtsam::Key vel1Key, gtsam::Key vel2Key, gtsam::Key acc1Key,
                    gtsam::SharedNoiseModel model, double dt)
      : Base(model, pose1Key, pose2Key, vel1Key, vel2Key, acc1Key), dt_(dt) {}

  // Clone method
  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::make_shared<Pose3DynXVAFactor>(*this);
  }

  void print(const std::string& s = "",
             const gtsam::KeyFormatter& keyFormatter =
                 DynoLikeKeyFormatter) const override {
    std::cout << s << "Pose3DynXVAFactor\n";
    Base::print("", keyFormatter);
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
    // Compute Jacobians if requested
    // Compute Jacobians if requested
    if (J1) {
      Eigen::Matrix<double, 8, 6> df_pp =
          gtsam::numericalDerivative51<gtsam::Vector8, gtsam::Pose3,
                                       gtsam::Pose3, gtsam::Vector2,
                                       gtsam::Vector2, gtsam::Vector2>(
              std::bind(&Pose3DynXVAFactor::residual, this,
                        std::placeholders::_1, std::placeholders::_2,
                        std::placeholders::_3, std::placeholders::_4,
                        std::placeholders::_5),
              pose1, pose2, vel1, vel2, acc1);
      *J1 = df_pp;
    }

    if (J2) {
      Eigen::Matrix<double, 8, 6> df_cp =
          gtsam::numericalDerivative52<gtsam::Vector8, gtsam::Pose3,
                                       gtsam::Pose3, gtsam::Vector2,
                                       gtsam::Vector2, gtsam::Vector2>(
              std::bind(&Pose3DynXVAFactor::residual, this,
                        std::placeholders::_1, std::placeholders::_2,
                        std::placeholders::_3, std::placeholders::_4,
                        std::placeholders::_5),
              pose1, pose2, vel1, vel2, acc1);
      *J2 = df_cp;
    }

    if (J3) {
      Eigen::Matrix<double, 8, 2> df_pc =
          gtsam::numericalDerivative53<gtsam::Vector8, gtsam::Pose3,
                                       gtsam::Pose3, gtsam::Vector2,
                                       gtsam::Vector2, gtsam::Vector2>(
              std::bind(&Pose3DynXVAFactor::residual, this,
                        std::placeholders::_1, std::placeholders::_2,
                        std::placeholders::_3, std::placeholders::_4,
                        std::placeholders::_5),
              pose1, pose2, vel1, vel2, acc1);
      *J3 = df_pc;
    }

    if (J4) {
      Eigen::Matrix<double, 8, 2> df_cc =
          gtsam::numericalDerivative54<gtsam::Vector8, gtsam::Pose3,
                                       gtsam::Pose3, gtsam::Vector2,
                                       gtsam::Vector2, gtsam::Vector2>(
              std::bind(&Pose3DynXVAFactor::residual, this,
                        std::placeholders::_1, std::placeholders::_2,
                        std::placeholders::_3, std::placeholders::_4,
                        std::placeholders::_5),
              pose1, pose2, vel1, vel2, acc1);
      *J4 = df_cc;
    }

    if (J5) {
      Eigen::Matrix<double, 8, 2> df_ca =
          gtsam::numericalDerivative55<gtsam::Vector8, gtsam::Pose3,
                                       gtsam::Pose3, gtsam::Vector2,
                                       gtsam::Vector2, gtsam::Vector2>(
              std::bind(&Pose3DynXVAFactor::residual, this,
                        std::placeholders::_1, std::placeholders::_2,
                        std::placeholders::_3, std::placeholders::_4,
                        std::placeholders::_5),
              pose1, pose2, vel1, vel2, acc1);
      *J5 = df_ca;
    }

    // Return the residual
    return residual(pose1, pose2, vel1, vel2, acc1);
  }

  // Compute residual
  gtsam::Vector residual(const gtsam::Pose3& pose3_1,
                         const gtsam::Pose3& pose3_2,
                         const gtsam::Vector2& vel1, const gtsam::Vector2& vel2,
                         const gtsam::Vector2& acc1) const {
    gtsam::Pose2 pose1 = convertToSE2OpenCV(pose3_1);
    gtsam::Pose2 pose2 = convertToSE2OpenCV(pose3_2);
    // Update velocities using acceleration and time step
    gtsam::Vector2 pred_vel2 = vel1 + acc1 * dt_;

    gtsam::Pose2 pred_pose = predictRobotPose(pose1, vel2, dt_);

    // Following 2 lines do the same as
    // "gtsam::traits<gtsam::Pose2>::Local(next_pose, pred_pose);" gtsam::Pose2
    // error_pose = next_pose.between(pred_pose); gtsam::Vector3 pose_error =
    // gtsam::Pose2::Logmap(error_pose);

    gtsam::Vector3 pose_error =
        gtsam::traits<gtsam::Pose2>::Local(pose2, pred_pose);

    // return gtsam::traits<gtsam::Pose2>::Local(pose2, expectedPose);

    // Compute the residual as the difference between the expected and actual
    // current pose

    return gtsam::Vector8(pose_error(0),  // x error
                          pose_error(1),  // y error
                          pose_error(2),  // theta error
                          vel2[0] - pred_vel2[0], vel2[1] - pred_vel2[1],
                          pose3_2.rotation().yaw() - pose3_1.rotation().yaw(),
                          pose3_2.rotation().roll() - pose3_1.rotation().roll(),
                          pose3_2.y() - pose3_1.y());
  }

  // Accessor methods for keys
  inline gtsam::Key pose1Key() const { return key1(); }
  inline gtsam::Key pose2Key() const { return key2(); }
  inline gtsam::Key vel1Key() const { return key3(); }
  inline gtsam::Key vel2Key() const { return key4(); }
  inline gtsam::Key acc1Key() const { return key5(); }

 private:
  double dt_;  // Time step
};

// TODO:(Jesse) make DynXVAJac0Factor templated on the pose type and
//  some way to encapsulate the zeroJacobian stuff...
// NOTE: residuals have the same math betweeen Pose3DynXVAJac0Factor and
// Pose2DynXVAFactor we just collapse the Pose3 to a Pose3
class Pose3DynXVAJac0Factor
    : public gtsam::NoiseModelFactor5<gtsam::Pose3, gtsam::Pose3,
                                      gtsam::Vector2, gtsam::Vector2,
                                      gtsam::Vector2> {
 public:
  using shared_ptr = boost::shared_ptr<Pose3DynXVAJac0Factor>;
  using This = Pose3DynXVAJac0Factor;
  using Base =
      gtsam::NoiseModelFactor5<gtsam::Pose3, gtsam::Pose3, gtsam::Vector2,
                               gtsam::Vector2, gtsam::Vector2>;

  // Constructor
  Pose3DynXVAJac0Factor(gtsam::Key pose1Key, gtsam::Key pose2Key,
                        gtsam::Key vel1Key, gtsam::Key vel2Key,
                        gtsam::Key acc1Key, gtsam::SharedNoiseModel model,
                        double dt)
      : Base(model, pose1Key, pose2Key, vel1Key, vel2Key, acc1Key), dt_(dt) {}

  // Clone method
  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::make_shared<Pose3DynXVAJac0Factor>(*this);
  }

  void print(const std::string& s = "",
             const gtsam::KeyFormatter& keyFormatter =
                 DynoLikeKeyFormatter) const override {
    std::cout << s << "Pose3DynXVAJac0Factor\n";
    Base::print("", keyFormatter);
  }

  // Evaluate error and optionally compute Jacobians
  gtsam::Vector evaluateError(
      const gtsam::Pose3& pose1, const gtsam::Pose3& pose2,
      const gtsam::Vector2& vel1, const gtsam::Vector2& vel2,
      const gtsam::Vector2& acc1, boost::optional<gtsam::Matrix&> J1,
      boost::optional<gtsam::Matrix&> J2, boost::optional<gtsam::Matrix&> J3,
      boost::optional<gtsam::Matrix&> J4,
      boost::optional<gtsam::Matrix&> J5) const {
    // Compute Jacobians if requested
    if (J1) {
      *J1 = Eigen::Matrix<double, 8, 6>::Zero();
    }

    if (J2) {
      Eigen::Matrix<double, 8, 6> df_cp =
          gtsam::numericalDerivative52<gtsam::Vector8, gtsam::Pose3,
                                       gtsam::Pose3, gtsam::Vector2,
                                       gtsam::Vector2, gtsam::Vector2>(
              std::bind(&Pose3DynXVAJac0Factor::residual, this,
                        std::placeholders::_1, std::placeholders::_2,
                        std::placeholders::_3, std::placeholders::_4,
                        std::placeholders::_5),
              pose1, pose2, vel1, vel2, acc1);
      *J2 = df_cp;
    }

    if (J3) {
      *J3 = Eigen::Matrix<double, 8, 2>::Zero();
    }

    if (J4) {
      Eigen::Matrix<double, 8, 2> df_cc =
          gtsam::numericalDerivative54<gtsam::Vector8, gtsam::Pose3,
                                       gtsam::Pose3, gtsam::Vector2,
                                       gtsam::Vector2, gtsam::Vector2>(
              std::bind(&Pose3DynXVAJac0Factor::residual, this,
                        std::placeholders::_1, std::placeholders::_2,
                        std::placeholders::_3, std::placeholders::_4,
                        std::placeholders::_5),
              pose1, pose2, vel1, vel2, acc1);
      *J4 = df_cc;
    }

    if (J5) {
      Eigen::Matrix<double, 8, 2> df_ca =
          gtsam::numericalDerivative55<gtsam::Vector8, gtsam::Pose3,
                                       gtsam::Pose3, gtsam::Vector2,
                                       gtsam::Vector2, gtsam::Vector2>(
              std::bind(&Pose3DynXVAJac0Factor::residual, this,
                        std::placeholders::_1, std::placeholders::_2,
                        std::placeholders::_3, std::placeholders::_4,
                        std::placeholders::_5),
              pose1, pose2, vel1, vel2, acc1);
      *J5 = df_ca;
    }

    // Return the residual
    return residual(pose1, pose2, vel1, vel2, acc1);
  }

  // Compute residual
  gtsam::Vector residual(const gtsam::Pose3& pose3_1,
                         const gtsam::Pose3& pose3_2,
                         const gtsam::Vector2& vel1, const gtsam::Vector2& vel2,
                         const gtsam::Vector2& acc1) const {
    gtsam::Pose2 pose1 = convertToSE2OpenCV(pose3_1);
    gtsam::Pose2 pose2 = convertToSE2OpenCV(pose3_2);
    // Update velocities using acceleration and time step
    gtsam::Vector2 pred_vel2 = vel1 + acc1 * dt_;

    gtsam::Pose2 pred_pose = predictRobotPose(pose1, pred_vel2, dt_);

    // Following 2 lines do the same as
    // "gtsam::traits<gtsam::Pose2>::Local(next_pose, pred_pose);" gtsam::Pose2
    // error_pose = next_pose.between(pred_pose); gtsam::Vector3 pose_error =
    // gtsam::Pose2::Logmap(error_pose);

    gtsam::Vector3 pose_error =
        gtsam::traits<gtsam::Pose2>::Local(pose2, pred_pose);

    // return gtsam::traits<gtsam::Pose2>::Local(pose2, expectedPose);

    // Compute the residual as the difference between the expected and actual
    // current pose

    return gtsam::Vector8(pose_error(0),  // x error
                          pose_error(1),  // y error
                          pose_error(2),  // theta error
                          vel2[0] - pred_vel2[0], vel2[1] - pred_vel2[1],
                          pose3_2.rotation().yaw() - pose3_1.rotation().yaw(),
                          pose3_2.rotation().roll() - pose3_1.rotation().roll(),
                          pose3_2.y() - pose3_1.y());
  }

  // Accessor methods for keys
  inline gtsam::Key pose1Key() const { return key1(); }
  inline gtsam::Key pose2Key() const { return key2(); }
  inline gtsam::Key vel1Key() const { return key3(); }
  inline gtsam::Key vel2Key() const { return key4(); }
  inline gtsam::Key acc1Key() const { return key5(); }

 private:
  double dt_;  // Time step
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

class HybridSmoothingJac0Factor
    : public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3,
                                      gtsam::Pose3> {
 public:
  typedef boost::shared_ptr<HybridSmoothingJac0Factor> shared_ptr;
  typedef HybridSmoothingJac0Factor This;
  typedef gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3>
      Base;

  gtsam::Pose3 L_e_;

  HybridSmoothingJac0Factor(gtsam::Key e_H_km2_world_key,
                            gtsam::Key e_H_km1_world_key,
                            gtsam::Key e_H_k_world_key, const gtsam::Pose3& L_e,
                            gtsam::SharedNoiseModel model)
      : Base(model, e_H_km2_world_key, e_H_km1_world_key, e_H_k_world_key),
        L_e_(L_e) {}

  gtsam::Vector evaluateError(
      const gtsam::Pose3& e_H_km2_world, const gtsam::Pose3& e_H_km1_world,
      const gtsam::Pose3& e_H_k_world,
      boost::optional<gtsam::Matrix&> J1 = boost::none,
      boost::optional<gtsam::Matrix&> J2 = boost::none,
      boost::optional<gtsam::Matrix&> J3 = boost::none) const override {
    if (J1) {
      *J1 = Eigen::Matrix<double, 6, 6>::Zero();
    }

    if (J2) {
      *J2 = gtsam::numericalDerivative32<gtsam::Vector6, gtsam::Pose3,
                                         gtsam::Pose3, gtsam::Pose3>(
          std::bind(&HybridSmoothingJac0Factor::residual, std::placeholders::_1,
                    std::placeholders::_2, std::placeholders::_3, L_e_),
          e_H_km2_world, e_H_km1_world, e_H_k_world);
    }

    if (J3) {
      *J3 = gtsam::numericalDerivative33<gtsam::Vector6, gtsam::Pose3,
                                         gtsam::Pose3, gtsam::Pose3>(
          std::bind(&HybridSmoothingJac0Factor::residual, std::placeholders::_1,
                    std::placeholders::_2, std::placeholders::_3, L_e_),
          e_H_km2_world, e_H_km1_world, e_H_k_world);
    }

    return residual(e_H_km2_world, e_H_km1_world, e_H_k_world, L_e_);
  }

  static gtsam::Vector residual(const gtsam::Pose3& e_H_km2_world,
                                const gtsam::Pose3& e_H_km1_world,
                                const gtsam::Pose3& e_H_k_world,
                                const gtsam::Pose3& L_e) {
    const gtsam::Pose3 L_k_2 = e_H_km2_world * L_e;
    const gtsam::Pose3 L_k_1 = e_H_km1_world * L_e;
    const gtsam::Pose3 L_k = e_H_k_world * L_e;

    gtsam::Pose3 k_2_H_k_1 = L_k_2.inverse() * L_k_1;
    gtsam::Pose3 k_1_H_k = L_k_1.inverse() * L_k;

    gtsam::Pose3 relative_motion = k_2_H_k_1.inverse() * k_1_H_k;

    return gtsam::traits<gtsam::Pose3>::Local(gtsam::Pose3::Identity(),
                                              relative_motion);
  }
};

class HybridSmoothingHolonomicFactor
    : public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3,
                                      gtsam::Pose3> {
 public:
  typedef boost::shared_ptr<HybridSmoothingHolonomicFactor> shared_ptr;
  typedef HybridSmoothingHolonomicFactor This;
  typedef gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3>
      Base;

  gtsam::Pose3 L_e_;
  bool should_zero_first_jacobian_;

  HybridSmoothingHolonomicFactor(gtsam::Key e_H_km2_world_key,
                                 gtsam::Key e_H_km1_world_key,
                                 gtsam::Key e_H_k_world_key,
                                 const gtsam::Pose3& L_e,
                                 gtsam::SharedNoiseModel model,
                                 bool should_zero_first_jacobian = false)
      : Base(model, e_H_km2_world_key, e_H_km1_world_key, e_H_k_world_key),
        L_e_(L_e),
        should_zero_first_jacobian_(should_zero_first_jacobian) {}

  gtsam::Vector evaluateError(
      const gtsam::Pose3& e_H_km2_world, const gtsam::Pose3& e_H_km1_world,
      const gtsam::Pose3& e_H_k_world,
      boost::optional<gtsam::Matrix&> J1 = boost::none,
      boost::optional<gtsam::Matrix&> J2 = boost::none,
      boost::optional<gtsam::Matrix&> J3 = boost::none) const override {
    if (J1) {
      if (should_zero_first_jacobian_) {
        *J1 = Eigen::Matrix<double, 8, 6>::Zero();
      } else {
        *J1 = gtsam::numericalDerivative31<gtsam::Vector8, gtsam::Pose3,
                                           gtsam::Pose3, gtsam::Pose3>(
            std::bind(&HybridSmoothingHolonomicFactor::residual,
                      std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3, L_e_),
            e_H_km2_world, e_H_km1_world, e_H_k_world);
      }
    }

    if (J2) {
      *J2 = gtsam::numericalDerivative32<gtsam::Vector8, gtsam::Pose3,
                                         gtsam::Pose3, gtsam::Pose3>(
          std::bind(&HybridSmoothingHolonomicFactor::residual,
                    std::placeholders::_1, std::placeholders::_2,
                    std::placeholders::_3, L_e_),
          e_H_km2_world, e_H_km1_world, e_H_k_world);
    }

    if (J3) {
      *J3 = gtsam::numericalDerivative33<gtsam::Vector8, gtsam::Pose3,
                                         gtsam::Pose3, gtsam::Pose3>(
          std::bind(&HybridSmoothingHolonomicFactor::residual,
                    std::placeholders::_1, std::placeholders::_2,
                    std::placeholders::_3, L_e_),
          e_H_km2_world, e_H_km1_world, e_H_k_world);
    }

    return residual(e_H_km2_world, e_H_km1_world, e_H_k_world, L_e_);
  }

  static gtsam::Vector residual(const gtsam::Pose3& e_H_km2_world,
                                const gtsam::Pose3& e_H_km1_world,
                                const gtsam::Pose3& e_H_k_world,
                                const gtsam::Pose3& L_e) {
    const gtsam::Pose3 L_k_2 = e_H_km2_world * L_e;
    const gtsam::Pose3 L_k_1 = e_H_km1_world * L_e;
    const gtsam::Pose3 L_k = e_H_k_world * L_e;

    gtsam::Pose3 k_2_H_k_1 = L_k_2.inverse() * L_k_1;
    gtsam::Pose3 k_1_H_k = L_k_1.inverse() * L_k;

    gtsam::Pose3 relative_motion = k_2_H_k_1.inverse() * k_1_H_k;
    // now we add holonomic constraints in the
    //  gtsam::Pose2 relaitive_motion2(
    //    relative_motion.z(),
    //    -relative_motion.x(),
    //    -relative_motion.rotation().pitch());

    // now we add holonomic constraints in the opencv convention
    // to constrain the pose in world frame
    // gtsam::Pose2 relaitive_motion2(
    //   relative_motion.z(),
    //   -relative_motion.x(),
    //   -relative_motion.rotation().pitch());

    // gtsam::Vector3 relative_motion2_error =
    //   gtsam::traits<gtsam::Pose2>::Local(
    //     gtsam::Pose2::Identity(),
    //     relaitive_motion2);

    gtsam::Vector6 relative_motion_error = gtsam::traits<gtsam::Pose3>::Local(
        gtsam::Pose3::Identity(), relative_motion);

    // return gtsam::Vector9(
    //     relative_motion2_error(0),  // x error
    //     relative_motion2_error(1),  // y error
    //     relative_motion2_error(2),  // theta error
    //     // L_k_2.rotation().roll() - L_k_1.rotation().roll(),
    //     // L_k_1.rotation().roll() - L_k.rotation().roll(),
    //     // L_k_2.rotation().yaw() - L_k_1.rotation().yaw(),
    //     // L_k_1.rotation().yaw() - L_k.rotation().yaw(),
    //     0, 0, 0, 0,
    //     L_k_2.y() - L_k_1.y(),
    //     L_k_1.y() - L_k.y()
    // );
    return gtsam::Vector8(relative_motion_error(0), relative_motion_error(1),
                          relative_motion_error(2), relative_motion_error(3),
                          relative_motion_error(4), relative_motion_error(5),
                          L_k_2.y() - L_k_1.y(), L_k_1.y() - L_k.y());
  }
};

struct HybridConstantMotionFactorResidual {
  static gtsam::Vector residual(const gtsam::Pose3& e_H_km1_world_est,
                                const gtsam::Pose3& e_H_k_world_est,
                                const gtsam::Pose3& e_H_n1_world_future,
                                const gtsam::Pose3& e_H_n2_world_future,
                                const gtsam::Pose3& L_e) {
    gtsam::Pose3 L_km1_est = e_H_km1_world_est * L_e;
    gtsam::Pose3 L_k_est = e_H_k_world_est * L_e;

    gtsam::Pose3 L_n1_est = e_H_n1_world_future * L_e;
    gtsam::Pose3 L_n2_est = e_H_n2_world_future * L_e;

    // measured (really expected) relative motion as given by the most recent
    // estimate
    gtsam::Pose3 measured_relative_motion = L_km1_est.inverse() * L_k_est;

    // implement between factor!
    gtsam::Pose3 hx =
        gtsam::traits<gtsam::Pose3>::Between(L_n1_est, L_n2_est);  // h(x)
    return gtsam::traits<gtsam::Pose3>::Local(measured_relative_motion, hx);
  }
};

struct HybridConstantMotionFactorResidualBetter {
  static gtsam::Vector residual(const gtsam::Pose3& e_H_km2_world,
                                const gtsam::Pose3& e_H_km1_world,
                                const gtsam::Pose3& e_H_k_world,
                                const gtsam::Pose3& L_e) {
    gtsam::Pose3 L_km2_est = e_H_km2_world * L_e;
    gtsam::Pose3 L_km1_est = e_H_km1_world * L_e;
    gtsam::Pose3 L_k_est = e_H_k_world * L_e;

    // measured (really expected) relative motion as given by the most recent
    // estimate
    gtsam::Pose3 measured_relative_motion = L_km2_est.inverse() * L_km1_est;

    // implement between factor!
    gtsam::Pose3 hx =
        gtsam::traits<gtsam::Pose3>::Between(L_km1_est, L_k_est);  // h(x)
    return gtsam::traits<gtsam::Pose3>::Local(measured_relative_motion, hx);
  }
};

class HybridConstantMotionFactorBase
    : public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3>,
      public HybridConstantMotionFactorResidualBetter {
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

// constant relative motion from estimation (propogated to the future)
// always zero jacobian on estimated motions (ie <=k)
class HybridConstantMotionFactor4
    : public gtsam::NoiseModelFactor4<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3,
                                      gtsam::Pose3>,
      public HybridConstantMotionFactorResidual {
 public:
  typedef boost::shared_ptr<HybridConstantMotionFactor4> shared_ptr;
  typedef HybridConstantMotionFactor4 This;
  typedef gtsam::NoiseModelFactor4<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3,
                                   gtsam::Pose3>
      Base;

  gtsam::Pose3 L_e_;

  // first two keys are motions at k-1 and k (ie the most recent two motions) in
  // the dynosam estimator (ie actually seen) the next two are the k+n and k+n+1
  // predicted motions at n steps in the future (n is from 1 up to N-1 ie mpc
  // horizon)
  HybridConstantMotionFactor4(gtsam::Key e_H_km1_world_est_key,
                              gtsam::Key e_H_k_world_est_key,
                              gtsam::Key e_H_n1_world_future_key,
                              gtsam::Key e_H_n2_world_future_key,
                              const gtsam::Pose3& L_e,
                              gtsam::SharedNoiseModel model)
      : Base(model, e_H_km1_world_est_key, e_H_k_world_est_key,
             e_H_n1_world_future_key, e_H_n2_world_future_key),
        L_e_(L_e) {}

  gtsam::Vector evaluateError(
      const gtsam::Pose3& e_H_km1_world_est,
      const gtsam::Pose3& e_H_k_world_est,
      const gtsam::Pose3& e_H_n1_world_future,
      const gtsam::Pose3& e_H_n2_world_future,
      boost::optional<gtsam::Matrix&> J1 = boost::none,
      boost::optional<gtsam::Matrix&> J2 = boost::none,
      boost::optional<gtsam::Matrix&> J3 = boost::none,
      boost::optional<gtsam::Matrix&> J4 = boost::none) const override {
    if (J1) {
      *J1 = Eigen::Matrix<double, 6, 6>::Zero();
    }

    if (J2) {
      *J2 = Eigen::Matrix<double, 6, 6>::Zero();
    }

    if (J3) {
      *J3 = gtsam::numericalDerivative43<gtsam::Vector6, gtsam::Pose3,
                                         gtsam::Pose3, gtsam::Pose3,
                                         gtsam::Pose3>(
          std::bind(&HybridConstantMotionFactor4::residual,
                    std::placeholders::_1, std::placeholders::_2,
                    std::placeholders::_3, std::placeholders::_4, L_e_),
          e_H_km1_world_est, e_H_k_world_est, e_H_n1_world_future,
          e_H_n2_world_future);
    }

    if (J4) {
      *J4 = gtsam::numericalDerivative44<gtsam::Vector6, gtsam::Pose3,
                                         gtsam::Pose3, gtsam::Pose3,
                                         gtsam::Pose3>(
          std::bind(&HybridConstantMotionFactor4::residual,
                    std::placeholders::_1, std::placeholders::_2,
                    std::placeholders::_3, std::placeholders::_4, L_e_),
          e_H_km1_world_est, e_H_k_world_est, e_H_n1_world_future,
          e_H_n2_world_future);
    }

    return residual(e_H_km1_world_est, e_H_k_world_est, e_H_n1_world_future,
                    e_H_n2_world_future, L_e_);
  }
};

// template<typename NLF, size_t...N>
// class DirectedEdgeFactor;

// HybridConstantMotionFactorJac12 =
// DirectedEdgeFactor<HybridConstantMotionFactor, 1, 2>;
// HybridConstantMotionFactorJac0 =
// DirectedEdgeFactor<HybridConstantMotionFactor, 0>;

// varient factor for when k == n1 (ie the first prediction) where the factor
// overlaps the estimation and prediction as we want
class HybridConstantMotionFactor3
    : public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3>,
      public HybridConstantMotionFactorResidual {
 public:
  typedef boost::shared_ptr<HybridConstantMotionFactor3> shared_ptr;
  typedef HybridConstantMotionFactor3 This;
  typedef gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3>
      Base;

  gtsam::Pose3 L_e_;

  HybridConstantMotionFactor3(gtsam::Key e_H_km1_world_est_key,
                              gtsam::Key e_H_k_world_est_key,
                              gtsam::Key e_H_n1_world_future_key,
                              const gtsam::Pose3& L_e,
                              gtsam::SharedNoiseModel model)
      : Base(model, e_H_km1_world_est_key, e_H_k_world_est_key,
             e_H_n1_world_future_key),
        L_e_(L_e) {}

  gtsam::Vector evaluateError(
      const gtsam::Pose3& e_H_km1_world_est,
      const gtsam::Pose3& e_H_k_world_est,
      const gtsam::Pose3& e_H_n1_world_future,
      boost::optional<gtsam::Matrix&> J1 = boost::none,
      boost::optional<gtsam::Matrix&> J2 = boost::none,
      boost::optional<gtsam::Matrix&> J3 = boost::none) const override {
    if (J1) {
      *J1 = Eigen::Matrix<double, 6, 6>::Zero();
    }

    if (J2) {
      *J2 = Eigen::Matrix<double, 6, 6>::Zero();
    }

    if (J3) {
      // Jacobian of the fourth input (ie e_H_n1_world_future) which is the 3rd
      // jacobian of the actual factor
      *J3 = gtsam::numericalDerivative44<gtsam::Vector6, gtsam::Pose3,
                                         gtsam::Pose3, gtsam::Pose3,
                                         gtsam::Pose3>(
          std::bind(&HybridConstantMotionFactor3::residual,
                    std::placeholders::_1, std::placeholders::_2,
                    std::placeholders::_3, std::placeholders::_4, L_e_),
          e_H_km1_world_est, e_H_k_world_est, e_H_k_world_est,
          e_H_n1_world_future);
    }

    // in this case k = n1 and n1=2 since we overlap between the estiamtion and
    // the prediction
    return residual(e_H_km1_world_est, e_H_k_world_est, e_H_k_world_est,
                    e_H_n1_world_future, L_e_);
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
  follow_noise_ =
      gtsam::noiseModel::Isotropic::Sigma(2u, FLAGS_mpc_follow_sigma);

  goal_noise_ = gtsam::noiseModel::Isotropic::Sigma(3u, FLAGS_mpc_goal_sigma);
  static_obstacle_noise_ =
      gtsam::noiseModel::Isotropic::Sigma(1u, FLAGS_mpc_static_obstacle_sigma);

  lin_vel_ = Limits{-0.0, 1};
  ang_vel_ = Limits{-0.5, 0.5};
  lin_acc_ = Limits{-0.5, 1};
  ang_acc_ = Limits{-0.5, 0.5};

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

  // first try and get map offset pose if we dont already have
  // this sets the static offset between dynosam odom and the map frame (which
  // is called 'odom')
  if (!T_map_camera_) {
    gtsam::Pose3 T;
    if (viz_->queryGlobalOffset(T)) {
      T_map_camera_ = T;
      LOG(INFO) << "Found offset between dynosam pose and map frame!! " << T;
      CHECK(sdf_map_);
      sdf_map_->setQueryOffset(T);
    }
  }

  bool sdf_map_valid = (bool)T_map_camera_;
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

      // velocity prior
      auto prior_factor =
          boost::make_shared<gtsam::PriorFactor<gtsam::Vector2>>(
              control_key, control_value, vel2d_prior_noise_);

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

    // add velocity prior only on first of time horizon
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
            boost::make_shared<SDFStaticObstacleFactor>(
                camera_key, CHECK_NOTNULL(sdf_map_),
                FLAGS_mpc_static_safety_distance, static_obstacle_noise_);

        new_factors.add(static_obstacle_factor);
        factors_to_remove_this_frame.add(static_obstacle_factor);
        LOG(INFO) << "Adding static obstacle factor " << formatter(camera_key);
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
      if (frame_id > frame_k) {
        // acceleration smoothing costs
        // LOG(INFO) << "Adding accel smoothing cost"
        //           << formatter(accel_key_previous) << " -> "
        //           << formatter(accel_key);

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
        // add zero jacobian on first 'pair' of iteration
        factor = boost::make_shared<Pose3DynXVAJac0Factor>(
            camera_key_previous, camera_key, control_key_previous, control_key,
            accel_key_previous, dynamic_factor_noise_, dt_);

        // LOG(INFO) << "Adding Pose3 zero-jacobain factor at k=" << frame_id
        //           << " k-1=" << frame_id_m1;
      } else {
        factor = boost::make_shared<Pose3DynXVAFactor>(
            camera_key_previous, camera_key, control_key_previous, control_key,
            accel_key_previous, dynamic_factor_noise_, dt_);
        // LOG(INFO) << "Adding Pose3 factor at k=" << frame_id
        //           << " k-1=" << frame_id_m1;
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
  const bool object_in_opt_at_k = static_cast<bool>(
      accessor->getObjectMotion(frame_k_m1, object_to_follow));
  const bool object_in_opt_has_embedded_frame =
      this->hasObjectKeyFrame(object_to_follow, frame_k);
  const bool object_in_opt_at_current =
      object_in_opt_has_embedded_frame && object_in_opt_at_k;
  // object not observed or has no keyframe
  // it may be observed but checking for its embedded frame tells us its in the
  // optimisation problem
  const bool seen_and_in_opt = frame_node_k->objectObserved(object_to_follow) &&
                               object_in_opt_at_current;
  // object exists at all (may not be observed at this frame!)
  // logic to track that object is seen and we should do predictions etc...
  if (objects_update_data_.exists(object_to_follow)) {
    const ObjectUpdateData& update_data =
        objects_update_data_.at(object_to_follow);

    auto object_node = map->getObject(object_to_follow);
    CHECK(object_node);

    // object must be optimized at least N times before we start to use its
    // prediction count is updated in RegularHybrid::postUpdate so only
    // increments after an optimization!
    constexpr static int kMinNumberMotionsOptimized = 2;
    if (update_data.count > kMinNumberMotionsOptimized && seen_and_in_opt) {
      LOG(INFO) << "Object seen enough times, is in opt etc..., "
                << info_string(frame_k, object_to_follow);
      object_available = true;
    }

    CHECK(this->hasObjectKeyFrame(object_to_follow, frame_k));

    if (frame_k_m1 == object_node->getFirstSeenFrame()) {
      object_new = true;
      LOG(INFO) << "Object j=" << object_to_follow << " is new";
    }

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
  }

  {
    // sanity check
    // object cannot have disappeared and also be available
    if (object_available) CHECK(!object_disappeared);

    // if(!object_available) CHECK(!object_reappeared);
  }

  if (!object_available) {
    VLOG(10) << "j=" << object_to_follow << "is unavailable at k=" << frame_k
             << ". Unable to follow!";
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
            << "No motion at  " << motion_to_be_stabilised_query.key();

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

    if (object_reappeared) {
      // remove all stabilising factors as we're about to add new ones
      CHECK(stabilising_object_factors_.exists(object_to_follow));
      gtsam::NonlinearFactorGraph stabilising_motion_factors =
          stabilising_object_factors_.at(object_to_follow);
      factors_to_remove_this_frame += stabilising_motion_factors;

      stabilising_object_factors_.erase(object_to_follow);
    }

    CHECK(frame_node_k->objectObserved(object_to_follow));
    // values and factors init object pose
    // dont get the latest motion/pose (the ones added by the base Hybrid
    // formulation) as these will not be optimized yet. Get the ones from the
    // previous frame which are better. Maybe bug in initalisation of latest
    // motion
    auto L_W_k_query = accessor->getObjectPose(frame_k_m1, object_to_follow);
    auto H_W_km1_k_query =
        accessor->getObjectMotion(frame_k_m1, object_to_follow);

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
    gtsam::Pose3 L_W_k = *L_W_k_query;
    gtsam::Pose3 H_W_km1_k = *H_W_km1_k_query;
    // gtsam::Pose3 H_W_km1_k = gtsam::Pose3(gtsam::Rot3::Identity(),
    // gtsam::Point3(1, 0, 0));
    gtsam::Pose3 L_W_future = L_W_k;

    ObjectIds objects_predicted{object_to_follow};
    predicted_objects_at_frame_.insert2(frame_k, objects_predicted);

    // LOG(INFO) << "Predicting poses " << info_string(frame_k,
    // object_to_follow) << " using constant H_W_km1_k " << H_W_km1_k
    //   << " from frame e=" << reference_frame_e;
    // LOG(INFO) << "Starting pose " <<  L_W_k << " using motion key " <<
    // formatter(H_W_km1_k_query.key()) << " pose key " <<
    // formatter(L_W_k_query.key());

    for (FrameId frame_id = frame_k + 1; frame_id < frame_N; frame_id++) {
      // progate frame L using constant realtive motion
      // then convert the progogated frame into H
      L_W_future = H_W_km1_k * L_W_future;
      // following L_k = H_e_k * L_e
      gtsam::Pose3 H_W_future = L_W_future * L_e.inverse();

      // LOG(INFO) << "Future pose k=" << frame_id << " " << L_W_future;

      auto motion_key_k_predicted =
          ObjectMotionSymbol(object_to_follow, frame_id);
      auto camera_pose_key_k_predicted = CameraPoseSymbol(frame_id);

      // k-1 and k-2 relative to the current forward predictive timetamp
      auto motion_key_k_m1_predicted =
          ObjectMotionSymbol(object_to_follow, frame_id - 1u);
      auto motion_key_k_m2_predicted =
          ObjectMotionSymbol(object_to_follow, frame_id - 2u);

      if (auto hybrid_motion_query = accessor->queryWithTheta<gtsam::Pose3>(
              motion_key_k_predicted, new_values);
          !hybrid_motion_query) {
        LOG(INFO) << "Inserting future key "
                  << formatter(motion_key_k_predicted);
        new_values.insert(motion_key_k_predicted, H_W_future);

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
      CHECK(constant_motion_factor);
      new_factors.add(constant_motion_factor);
      factors_to_remove_this_frame.add(constant_motion_factor);

      {
        // sanity check
        CHECK(accessor->queryWithTheta<gtsam::Pose3>(
            camera_pose_key_k_predicted, new_values));
      }

      if (mission_type_ == MissionType::FOLLOW) {
        // add follow factors
        auto follow_factor = boost::make_shared<HybridMotionFollowJac0Factor>(
            camera_pose_key_k_predicted, motion_key_k_predicted, L_e,
            desired_follow_distance_, desired_follow_heading_, follow_noise_);
        new_factors.add(follow_factor);
        factors_to_remove_this_frame.add(follow_factor);
      }
    }
  }

  if (mission_type_ == MissionType::NAVIGATE) {
    // handle global path
    gtsam::Pose3 local_goal;
    bool local_goal_result = getLocalGoalFromGlobalPath(
        accessor->getSensorPose(frame_k).value(), 60, local_goal);

    if (local_goal_result) {
      local_goal_ = local_goal;

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
    }
  }

  factors_per_frame_[frame_k] = factors_to_remove_this_frame;
  timestamp_km1_ = context.timestamp;
  LOG(INFO) << "MPCFormulation::otherUpdatesContext";
}

void MPCFormulation::postUpdate(const PostUpdateData& data) {
  Base::postUpdate(data);

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

    // calcualte bearing in pose2 for the plan (we assume everything will happen
    // in pose3) assyme goals are in opencv convention (this happens at the ROS
    // level where we use the tf look to)
    gtsam::Pose2 goal_n_se2 = convertToSE2OpenCV(goal_n);
    gtsam::Pose2 goal_nm1_se2 = convertToSE2OpenCV(goal_nm1);

    // theta in opencv
    // since we do calculations in opencv when we convert SE(3) to SE(2) we use
    // the -pitch component of the 3D rotation. Then we go back to SE(3) we just
    // use -bearing again
    double bearing = goal_nm1_se2.bearing(goal_n_se2).theta();
    gtsam::Rot3 rotation = gtsam::Rot3::Pitch(-bearing);

    goal = gtsam::Pose3(rotation, goal_n.translation());
  }

  return true;
}

}  // namespace dyno
