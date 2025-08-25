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

// Vec2LimitFactor (2DV and 2DA limits)
// ControlChangeFactor (2DA Smooth)
// FollowFactor (need zero Jacobian)
// Pose2DynXVAFactor (Dynamic factor) + zero Jacobian version

namespace dyno {

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

class Pose3FollowJac0Factor
    : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3> {
 public:
  using shared_ptr = boost::shared_ptr<Pose3FollowJac0Factor>;
  using This = Pose3FollowJac0Factor;
  using Base = gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3>;

  // Constructor
  Pose3FollowJac0Factor(gtsam::Key poseFollowerKey, gtsam::Key poseLeaderKey,
                        double des_distance, double des_heading,
                        gtsam::SharedNoiseModel model)
      : Base(model, poseFollowerKey, poseLeaderKey),
        des_distance_(des_distance),
        des_heading_(des_heading) {}

  // Clone method
  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::make_shared<Pose3FollowJac0Factor>(*this);
  }

  void print(const std::string& s = "",
             const gtsam::KeyFormatter& keyFormatter =
                 DynoLikeKeyFormatter) const override {
    std::cout << s << "Pose3FollowJac0Factor\n";
    Base::print("", keyFormatter);
  }

  // Error function
  gtsam::Vector evaluateError(
      const gtsam::Pose3& poseFollower, const gtsam::Pose3& poseLeader,
      boost::optional<gtsam::Matrix&> J1 = boost::none,
      boost::optional<gtsam::Matrix&> J2 = boost::none) const override {
    // Compute Jacobians if requested
    if (J1) {
      Eigen::Matrix<double, 2, 6> df_f =
          gtsam::numericalDerivative21<gtsam::Vector2, gtsam::Pose3,
                                       gtsam::Pose3>(
              std::bind(&Pose3FollowJac0Factor::residual, this,
                        std::placeholders::_1, std::placeholders::_2,
                        des_distance_, des_heading_),
              poseFollower, poseLeader);
      *J1 = df_f;
    }

    if (J2) {
      Eigen::Matrix<double, 2, 6> df_l =
          gtsam::numericalDerivative22<gtsam::Vector2, gtsam::Pose3,
                                       gtsam::Pose3>(
              std::bind(&Pose3FollowJac0Factor::residual, this,
                        std::placeholders::_1, std::placeholders::_2,
                        des_distance_, des_heading_),
              poseFollower, poseLeader);
      *J2 = df_l;
    }

    return residual(poseFollower, poseLeader, des_distance_, des_heading_);
  }

  // Compute residual (static utility function)
  gtsam::Vector residual(const gtsam::Pose3& poseFollower_3d,
                         const gtsam::Pose3& poseLeader_3d, double des_distance,
                         const gtsam::Rot2& des_heading) const {
    gtsam::Pose2 poseFollower(poseFollower_3d.x(), poseFollower_3d.y(),
                              poseFollower_3d.rotation().yaw());
    gtsam::Pose2 poseLeader(poseLeader_3d.x(), poseLeader_3d.y(),
                            poseLeader_3d.rotation().yaw());
    double distance = poseFollower.range(poseLeader);
    gtsam::Rot2 bearing = poseFollower.bearing(poseLeader);
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

  inline gtsam::Key poseFollowerKey() const { return key1(); }
  inline gtsam::Key poseLeaderKey() const { return key2(); }

 private:
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
    gtsam::Pose2 pose1(pose3_1.x(), pose3_1.y(), pose3_1.rotation().yaw());
    gtsam::Pose2 pose2(pose3_2.x(), pose3_2.y(), pose3_2.rotation().yaw());
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

    return gtsam::Vector8(
        pose_error(0),  // x error
        pose_error(1),  // y error
        pose_error(2),  // theta error
        vel2[0] - pred_vel2[0], vel2[1] - pred_vel2[1],
        pose3_2.rotation().roll() - pose3_1.rotation().roll(),
        pose3_2.rotation().pitch() - pose3_1.rotation().pitch(),
        pose3_2.z() - pose3_1.z());
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
    gtsam::Pose2 pose1(pose3_1.x(), pose3_1.y(), pose3_1.rotation().yaw());
    gtsam::Pose2 pose2(pose3_2.x(), pose3_2.y(), pose3_2.rotation().yaw());
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

    return gtsam::Vector8(
        pose_error(0),  // x error
        pose_error(1),  // y error
        pose_error(2),  // theta error
        vel2[0] - pred_vel2[0], vel2[1] - pred_vel2[1],
        pose3_2.rotation().roll() - pose3_1.rotation().roll(),
        pose3_2.rotation().pitch() - pose3_1.rotation().pitch(),
        pose3_2.z() - pose3_1.z());
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

}  // namespace mpc_factors

StateQuery<gtsam::Vector2> MPCAccessor::getControlCommand(
    FrameId frame_k) const {
  return StateQuery<gtsam::Vector2>::NotInMap(
      this->makeControlCommandKey(frame_k));
}

MPCFormulation::MPCFormulation(const FormulationParams& params,
                               typename Map::Ptr map,
                               const NoiseModels& noise_models,
                               const Sensors& sensors,
                               const FormulationHooks& hooks)
    : Base(params, map, noise_models, sensors, hooks) {
  camera_pose_prior_noise_ = gtsam::noiseModel::Isotropic::Sigma(6u, 1.0);
  vel2d_prior_noise_ = gtsam::noiseModel::Isotropic::Sigma(2u, 1.0);
  accel2d_prior_noise_ = gtsam::noiseModel::Isotropic::Sigma(2u, 1.0);

  vel2d_limit_noise_ = gtsam::noiseModel::Isotropic::Sigma(2u, 1.0);
  accel2d_limit_noise_ = gtsam::noiseModel::Isotropic::Sigma(2u, 1.0);

  accel2d_cost_noise_ = gtsam::noiseModel::Isotropic::Sigma(2u, 1.0);
  accel2d_smoothing_noise_ = gtsam::noiseModel::Isotropic::Sigma(2u, 1.0);

  dynamic_factor_noise_ = gtsam::noiseModel::Isotropic::Sigma(8u, 1.0);

  lin_vel_ = Limits{-1, 1};
  ang_vel_ = Limits{-1, 1};
  lin_acc_ = Limits{-1, 1};
  ang_acc_ = Limits{-1, 1};
}

void MPCFormulation::otherUpdatesContext(
    const OtherUpdateContextType& context, UpdateObservationResult& result,
    gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors) {
  using namespace mpc_factors;

  FrameId frame_k = context.getFrameId();
  FrameId frame_k_m1 = frame_k - 1u;
  auto formatter = this->formatter();
  // TODO: now assume we reun with batch. Later must check the indicies
  if (factors_per_frame_.exists(frame_k_m1)) {
    result.batch_update_params.factors_to_remove =
        factors_per_frame_.at(frame_k_m1);

    for (const auto& f : result.batch_update_params.factors_to_remove)
      f->print("Factors wanting removel ", this->formatter());
  }

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
  //. Follow factor on {X(k), L(k)} -> {X(N), L(N)}
  //. Prediction factor { L(k), L(k+1)} -> { L(N-1), L(N)}

  auto accessor = this->derivedAccessor<MPCAccessor>();
  CHECK(accessor);

  FrameId frame_N = frame_k + mpc_horizon;

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
      LOG(INFO) << "Inserting future key " << formatter(camera_key);

      auto sensor_pose_query_previous = accessor->queryWithTheta<gtsam::Pose3>(
          camera_key_previous, new_values);
      CHECK(sensor_pose_query_previous);
      new_values.insert(camera_key, *sensor_pose_query_previous);

      X_k = *sensor_pose_query_previous;
    } else {
      X_k = *sensor_pose_query;
    }

    gtsam::Vector2 V_k;
    if (auto velocity2d_query =
            accessor->queryWithTheta<gtsam::Vector2>(control_key, new_values);
        !velocity2d_query) {
      LOG(INFO) << "Inserting future key " << formatter(control_key);

      auto control_query_previous = accessor->queryWithTheta<gtsam::Vector2>(
          control_key_previous, new_values);
      gtsam::Vector2 control_value;
      if (control_query_previous) {
        control_value = *control_query_previous;
      } else {
        control_value = gtsam::Vector2(0, 0);
      }
      new_values.insert(control_key, control_value);

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
                accel_key_previous, *accel_query_previous,
                accel2d_prior_noise_);

        new_factors.add(stabilising_accel_prior);
      }
    }

    // limit factor
    auto velocity_limit_factor = boost::make_shared<Vec2LimitFactor>(
        control_key, lin_vel_.min, lin_vel_.max, ang_vel_.min, ang_vel_.max,
        vel2d_limit_noise_, dt_);

    new_factors.add(velocity_limit_factor);
    factors_to_remove_this_frame.add(velocity_limit_factor);

    // add acceleration up to k+N-1
    if (frame_id < frame_N - 1) {
      // acceleration value at time-step k
      gtsam::Vector2 A_k;
      if (auto acceleration2d_query =
              accessor->queryWithTheta<gtsam::Vector2>(accel_key, new_values);
          !acceleration2d_query) {
        LOG(INFO) << "Inserting future key " << formatter(accel_key);

        auto accel_query_previous = accessor->queryWithTheta<gtsam::Vector2>(
            accel_key_previous, new_values);
        gtsam::Vector2 accel_value;
        if (accel_query_previous) {
          accel_value = *accel_query_previous;
        } else {
          accel_value = gtsam::Vector2(0, 0);
        }
        new_values.insert(accel_key, accel_value);
        A_k = accel_value;
      } else {
        A_k = *acceleration2d_query;
      }

      // acceleration prior
      // TODO: cost factor?
      auto acceleration_prior_factor =
          boost::make_shared<gtsam::PriorFactor<gtsam::Vector2>>(
              accel_key, A_k, accel2d_prior_noise_);

      // limit factor
      auto acceleration_limit_factor = boost::make_shared<Vec2LimitFactor>(
          accel_key, lin_acc_.min, lin_acc_.max, ang_acc_.min, ang_acc_.max,
          accel2d_limit_noise_, dt_);

      new_factors.add(acceleration_prior_factor);
      new_factors.add(acceleration_limit_factor);

      factors_to_remove_this_frame.add(acceleration_prior_factor);
      factors_to_remove_this_frame.add(acceleration_limit_factor);

      // we are at least the second frame of the horizon
      if (frame_id > frame_k) {
        // acceleration smoothing costs
        LOG(INFO) << "Adding accel smoothing cost"
                  << formatter(accel_key_previous) << " -> "
                  << formatter(accel_key);

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

        LOG(INFO) << "Adding Pose3 zero-jacobain factor at k=" << frame_id
                  << " k-1=" << frame_id_m1;
      } else {
        factor = boost::make_shared<Pose3DynXVAFactor>(
            camera_key_previous, camera_key, control_key_previous, control_key,
            accel_key_previous, dynamic_factor_noise_, dt_);
        LOG(INFO) << "Adding Pose3 factor at k=" << frame_id
                  << " k-1=" << frame_id_m1;
      }

      CHECK(factor);
      new_factors.add(factor);
      factors_to_remove_this_frame.add(factor);
    }
  }

  // values and factors init object pose
  auto L_W_k = accessor->getObjectPose(frame_k, object_to_follow_);
  auto H_k_1_k = accessor->getRelativeLocalMotion(frame_k, object_to_follow_);
  if (L_W_k && H_k_1_k) {
    for (FrameId frame_id = frame_k; frame_id < frame_N; frame_id++) {
      if (frame_id == frame_k) {
      }
    }

  } else {
    VLOG(10) << "j=" << object_to_follow_ << " not found at k=" << frame_k
             << ". Unable to follow!";
  }

  factors_per_frame_[frame_k] = factors_to_remove_this_frame;
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

  // 1. Collect current estimates and publish to rviz
  // 2. Collect control command and send

  if (viz_) viz_->spin(data.frame_id, this);
}

}  // namespace dyno
