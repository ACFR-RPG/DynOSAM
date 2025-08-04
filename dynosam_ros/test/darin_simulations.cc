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

#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <dynosam/backend/ParallelHybridBackendModule.hpp>
#include <dynosam/test/backend_runners.hpp>
#include <dynosam/test/helpers.hpp>
#include <dynosam/test/simulator.hpp>
#include <dynosam/utils/GtsamUtils.hpp>

#include "dynosam_ros/displays/DisplaysImpl.hpp"

using namespace dyno;

TEST(DarinSimulations, test1) {
  dyno_testing::ScenarioBody::Ptr camera =
      std::make_shared<dyno_testing::ScenarioBody>(
          std::make_unique<dyno_testing::ConstantMotionBodyVisitor>(
              gtsam::Pose3::Identity(),
              // motion only in x
              gtsam::Pose3(gtsam::Rot3::RzRyRx(0.3, 0.1, 0.0),
                           gtsam::Point3(1.5, 2.5, 0))));
  // dyno_testing::ScenarioBody::Ptr camera =
  //   std::make_shared<dyno_testing::ScenarioBody>(
  //       std::make_unique<dyno_testing::ConstantMotionBodyVisitor>(
  //         gtsam::Pose3(gtsam::Rot3::Identity(), gtsam::Point3(0, 0, 0)),
  //           // motion only in x
  //           gtsam::Pose3::Identity()));
  // needs to be at least 3 overlap so we can meet requirements in graph
  // TODO: how can we do 1 point but with lots of overlap (even infinity
  // overlap?)

  const double H_R_sigma = 0.05;
  const double H_t_sigma = 0.08;
  const double dynamic_point_sigma = 0.1;

  const double X_R_sigma = 0.01;
  const double X_t_sigma = 0.01;

  dyno_testing::RGBDScenario::NoiseParams noise_params;
  noise_params.H_R_sigma = H_R_sigma;
  noise_params.H_t_sigma = H_t_sigma;
  noise_params.dynamic_point_sigma = dynamic_point_sigma;
  noise_params.X_R_sigma = X_R_sigma;
  noise_params.X_t_sigma = X_t_sigma;

  dyno_testing::RGBDScenario scenario(
      camera,
      std::make_shared<dyno_testing::SimpleStaticPointsGenerator>(50, 12),
      noise_params);

  std::random_device rd;
  std::mt19937 gen(rd());  // Mersenne Twister RNG
  std::uniform_real_distribution<> dist_pose(-30, 30);
  std::uniform_real_distribution<> dist_motion(-3, 3);
  auto generate_random_objects = [&dist_pose, &dist_motion, &scenario, &gen]() {
    static int object_id = 1;

    gtsam::Pose3 pose =
        dyno::utils::createRandomAroundIdentity<gtsam::Pose3>(dist_pose(gen));
    gtsam::Pose3 motion =
        dyno::utils::createRandomAroundIdentity<gtsam::Pose3>(dist_motion(gen));

    dyno_testing::ObjectBody::Ptr obj =
        std::make_shared<dyno_testing::ObjectBody>(
            std::make_unique<dyno_testing::ConstantMotionBodyVisitor>(
                pose,
                // motion only in x
                motion),
            std::make_unique<dyno_testing::RandomOverlapObjectPointsVisitor>(10,
                                                                             4),
            dyno_testing::ObjectBodyParams(0, 20));

    scenario.addObjectBody(object_id, obj);
    object_id++;
  };

  for (int i = 0; i < 9; i++) {
    generate_random_objects();
  }

  // add one obect
  const size_t num_points = 10;
  const size_t obj1_overlap = 4;
  const size_t obj2_overlap = 4;
  const size_t obj3_overlap = 4;
  dyno_testing::ObjectBody::Ptr object1 =
      std::make_shared<dyno_testing::ObjectBody>(
          std::make_unique<dyno_testing::ConstantMotionBodyVisitor>(
              gtsam::Pose3(gtsam::Rot3::Identity(), gtsam::Point3(-15, 30, 0)),
              // motion only in x
              gtsam::Pose3(gtsam::Rot3::RzRyRx(0.0, 0.0, 0.0),
                           gtsam::Point3(2, 0, 0))),
          std::make_unique<dyno_testing::RandomOverlapObjectPointsVisitor>(
              num_points, obj1_overlap),
          dyno_testing::ObjectBodyParams(0, 20));

  dyno_testing::ObjectBody::Ptr object2 =
      std::make_shared<dyno_testing::ObjectBody>(
          std::make_unique<dyno_testing::ConstantMotionBodyVisitor>(
              gtsam::Pose3(gtsam::Rot3::Identity(), gtsam::Point3(-16, 28, 0)),
              // motion only in x
              gtsam::Pose3(gtsam::Rot3::Identity(), gtsam::Point3(1, -1.8, 0))),
          std::make_unique<dyno_testing::RandomOverlapObjectPointsVisitor>(
              num_points, obj2_overlap),
          dyno_testing::ObjectBodyParams(0, 20));

  //   dyno_testing::ObjectBody::Ptr object3 =
  //       std::make_shared<dyno_testing::ObjectBody>(
  //           std::make_unique<dyno_testing::ConstantMotionBodyVisitor>(
  //               gtsam::Pose3(gtsam::Rot3::RzRyRx(0.3, 0.2, 0.1),
  //                            gtsam::Point3(1.1, 0.2, 1.2)),
  //               // motion only in x
  //               gtsam::Pose3(gtsam::Rot3::RzRyRx(0.2, 0.1, 0.0),
  //                            gtsam::Point3(0.2, 0.3, 0))),
  //           std::make_unique<dyno_testing::RandomOverlapObjectPointsVisitor>(
  //               num_points, obj3_overlap),
  //           dyno_testing::ObjectBodyParams(0, 19));

  dyno_testing::ObjectBody::Ptr object3 =
      std::make_shared<dyno_testing::ObjectBody>(
          std::make_unique<dyno_testing::ConstantMotionBodyVisitor>(
              gtsam::Pose3(gtsam::Rot3::Identity(), gtsam::Point3(-17, 36, 0)),
              // motion only in x
              gtsam::Pose3(gtsam::Rot3::Identity(), gtsam::Point3(2, 0, 0))),
          std::make_unique<dyno_testing::RandomOverlapObjectPointsVisitor>(
              num_points, obj2_overlap),
          dyno_testing::ObjectBodyParams(0, 20));

  dyno_testing::ObjectBody::Ptr object4 =
      std::make_shared<dyno_testing::ObjectBody>(
          std::make_unique<dyno_testing::ConstantMotionBodyVisitor>(
              gtsam::Pose3(gtsam::Rot3::RzRyRx(0.0, 0, 0.0),
                           gtsam::Point3(-30, 40, 0)),
              // motion only in x
              gtsam::Pose3(gtsam::Rot3::Identity(), gtsam::Point3(2, -2.5, 0))),
          std::make_unique<dyno_testing::RandomOverlapObjectPointsVisitor>(
              num_points, obj2_overlap),
          dyno_testing::ObjectBodyParams(0, 20));

  //   scenario.addObjectBody(1, object1);
  //   scenario.addObjectBody(2, object2);
  //   scenario.addObjectBody(3, object3);
  //   scenario.addObjectBody(4, object4);

  dyno::BackendParams backend_params;
  backend_params.use_robust_kernals_ = true;
  backend_params.useLogger(true);
  backend_params.min_dynamic_observations = 3u;
  backend_params.dynamic_point_noise_sigma = dynamic_point_sigma;
  backend_params.odometry_rotation_sigma_ = X_R_sigma;
  backend_params.odometry_translation_sigma_ = X_t_sigma;

  auto backend = std::make_shared<dyno::ParallelHybridBackendModule>(
      backend_params, dyno_testing::makeDefaultCameraPtr());

  // dyno_testing::RGBDBackendTester tester;

  //  auto backend = std::make_shared<dyno::RegularBackendModule>(
  //     backend_params, dyno_testing::makeDefaultCameraPtr(),
  //     dyno::RegularBackendModule::UpdaterType::HYBRID);

  auto node = std::make_shared<rclcpp::Node>("dynosam");
  auto viz = std::make_shared<dyno::BackendDisplayRos>(DisplayParams{}, node);

  for (size_t i = 0; i < 20; i++) {
    dyno::RGBDInstanceOutputPacket::Ptr output_gt, output_noisy;
    std::tie(output_gt, output_noisy) = scenario.getOutput(i);

    auto result = backend->spinOnce(output_noisy);
    LOG(INFO) << "Finished backebend run on" << i;
    // viz->spinOnce(result);
  }
}
