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
#include <gtest/gtest.h>
#include <gtsam/base/debug.h>
#include <gtsam/linear/GaussianEliminationTree.h>
#include <gtsam/nonlinear/ISAM2-impl.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearISAM.h>

#include "MRISAM2.h"
#include "dynosam/backend/BackendPipeline.hpp"
#include "dynosam/backend/FactorGraphTools.hpp"
#include "dynosam/backend/Formulation.hpp"
#include "dynosam/backend/RGBDBackendModule.hpp"
#include "dynosam/common/Map.hpp"
#include "dynosam/factors/LandmarkMotionTernaryFactor.hpp"
#include "dynosam/frontend/FrontendPipeline.hpp"
#include "internal/helpers.hpp"
#include "internal/simulator.hpp"

// TEST(MRBayesTree, testSimpl) {

//     using namespace dyno;

//     dyno_testing::ScenarioBody::Ptr camera =
//     std::make_shared<dyno_testing::ScenarioBody>(
//         std::make_unique<dyno_testing::ConstantMotionBodyVisitor>(
//             gtsam::Pose3::Identity(),
//             // motion only in x
//             gtsam::Pose3(gtsam::Rot3::RzRyRx(0.3, 0.1, 0.0),
//                          gtsam::Point3(0.1, 0.05, 0))));

//     const double H_R_sigma = 0.05;
//     const double H_t_sigma = 0.08;
//     const double dynamic_point_sigma = 0.1;

//     const double X_R_sigma = 0.0;
//     const double X_t_sigma = 0.0;

//     dyno_testing::RGBDScenario::NoiseParams noise_params;
//     noise_params.H_R_sigma = H_R_sigma;
//     noise_params.H_t_sigma = H_t_sigma;
//     noise_params.dynamic_point_sigma = dynamic_point_sigma;
//     noise_params.X_R_sigma = X_R_sigma;
//     noise_params.X_t_sigma = X_t_sigma;

//     dyno_testing::RGBDScenario scenario(
//         camera,
//         std::make_shared<dyno_testing::SimpleStaticPointsGenerator>(25, 15),
//         noise_params);

//     // add one obect
//     const size_t num_points = 10;
//     const size_t obj1_overlap = 5;
//     const size_t obj2_overlap = 4;
//     const size_t obj3_overlap = 5;
//     dyno_testing::ObjectBody::Ptr object1 =
//         std::make_shared<dyno_testing::ObjectBody>(
//             std::make_unique<dyno_testing::ConstantMotionBodyVisitor>(
//                 gtsam::Pose3(gtsam::Rot3::Identity(), gtsam::Point3(2, 0,
//                 0)),
//                 // motion only in x
//                 gtsam::Pose3(gtsam::Rot3::RzRyRx(0.2, 0.1, 0.0),
//                             gtsam::Point3(0.2, 0, 0))),
//             std::make_unique<dyno_testing::RandomOverlapObjectPointsVisitor>(
//                 num_points, obj1_overlap),
//             dyno_testing::ObjectBodyParams{});

//     dyno_testing::ObjectBody::Ptr object2 =
//         std::make_shared<dyno_testing::ObjectBody>(
//             std::make_unique<dyno_testing::ConstantMotionBodyVisitor>(
//                 gtsam::Pose3(gtsam::Rot3::Identity(), gtsam::Point3(1, 0.4,
//                 0.1)),
//                 // motion only in x
//                 gtsam::Pose3(gtsam::Rot3::Identity(), gtsam::Point3(0.2, 0,
//                 0))),
//             std::make_unique<dyno_testing::RandomOverlapObjectPointsVisitor>(
//                 num_points, obj2_overlap),
//             dyno_testing::ObjectBodyParams{.enters_scenario = 8,
//                                         .leaves_scenario = 15});

//     dyno_testing::ObjectBody::Ptr object3 =
//         std::make_shared<dyno_testing::ObjectBody>(
//             std::make_unique<dyno_testing::ConstantMotionBodyVisitor>(
//                 gtsam::Pose3(gtsam::Rot3::RzRyRx(0.3, 0.2, 0.1),
//                             gtsam::Point3(1.1, 0.2, 1.2)),
//                 // motion only in x
//                 gtsam::Pose3(gtsam::Rot3::RzRyRx(0.2, 0.1, 0.0),
//                             gtsam::Point3(0.2, 0.3, 0))),
//             std::make_unique<dyno_testing::RandomOverlapObjectPointsVisitor>(
//                 num_points, obj3_overlap),
//             dyno_testing::ObjectBodyParams{.enters_scenario = 13,
//                                         .leaves_scenario = 19});

//     scenario.addObjectBody(1, object1);
//     // scenario.addObjectBody(2, object2);
//     // scenario.addObjectBody(3, object3);

//     dyno::BackendParams backend_params;
//     backend_params.useLogger(false);
//     backend_params.min_dynamic_obs_ = 1u;
//     backend_params.dynamic_point_noise_sigma_ = dynamic_point_sigma;
//     backend_params.odometry_rotation_sigma_ = X_R_sigma;
//     backend_params.odometry_translation_sigma_ = X_t_sigma;

//     struct RunIncrementalTest {
//     struct Data {
//         dyno::RGBDBackendModule::Ptr backend;
//         std::shared_ptr<gtsam::ISAM2> isam2;
//         std::shared_ptr<gtsam::MRISAM2> mrisam2;
//         gtsam::Values opt_values;
//     };

//     void addBackend(dyno::RGBDBackendModule::Ptr backend) {
//         std::shared_ptr<Data> data = std::make_shared<Data>();

//         gtsam::ISAM2Params isam2_params;
//         isam2_params.evaluateNonlinearError = true;
//         isam2_params.factorization =
//         gtsam::ISAM2Params::Factorization::CHOLESKY; data->isam2 =
//         std::make_shared<gtsam::ISAM2>(isam2_params);

//         backend->callback =
//             [data](const dyno::Formulation<dyno::Map3d2d>::UniquePtr&
//             formulation,
//                 dyno::FrameId frame_id,
//                 const GraphUpdateResult& graph_update) -> void {
//         LOG(INFO) << "Running isam2 update " << frame_id << " for formulation
//         "
//                     << formulation->getFullyQualifiedName();

//         gtsam::Values new_values = graph_update.values();
//         gtsam::NonlinearFactorGraph new_factors = graph_update.factors();

//         CHECK_NOTNULL(data);
//         CHECK_NOTNULL(data->isam2);
//         auto isam = data->isam2;
//         // gtsam::ISAM2Result result;
//         // {
//         //     dyno::utils::TimingStatsCollector timer(
//         //         "isam2_oc_test_update." +
//         formulation->getFullyQualifiedName());
//         //     result = isam->update(new_factors, new_values);
//         // }

//         // LOG(INFO) << "ISAM2 result. Error before " <<
//         result.getErrorBefore()
//         //             << " error after " << result.getErrorAfter();
//         // data->opt_values = isam->calculateEstimate();

//         // isam->getFactorsUnsafe().saveGraph(
//         //     dyno::getOutputFilePath("isam_graph_" +
//         std::to_string(frame_id) +
//         //                             "_" +
//         formulation->getFullyQualifiedName() +
//         //                             ".dot"),
//         //     dyno::DynoLikeKeyFormatter);

//         // if (!isam->empty()) {
//         //     dyno::factor_graph_tools::saveBayesTree(
//         //         *isam,
//         //         dyno::getOutputFilePath(
//         //             "oc_bayes_tree_" + std::to_string(frame_id) + "_" +
//         //             formulation->getFullyQualifiedName() + ".dot"),
//         //         dyno::DynoLikeKeyFormatter);
//         // }

//         auto& mr_isam = data->mrisam2;

//         ObjectIds object_ids = graph_update.dynamicObjectIds();
//         if(object_ids.size() > 0 && !mr_isam) {
//             CHECK_EQ(object_ids.size(), 1u);
//             LOG(INFO) << "Making MRISAM with object ids " <<
//             object_ids.at(0); gtsam::KeySet keys_order;
//             keys_order.insert(CameraPoseSymbol(0));
//             keys_order.insert(ObjectMotionSymbol(1, 0));

//             gtsam::Values new_values = graph_update.values();
//             gtsam::NonlinearFactorGraph new_factors = graph_update.factors();

//             gtsam::MRISAM2::RootKeySetMap other_root_keys_map;
//             other_root_keys_map[0] =  gtsam::KeySet{CameraPoseSymbol(0)};
//             other_root_keys_map[1] =  gtsam::KeySet{ObjectMotionSymbol(1,
//             0)};

//             LOG(INFO) << "Making MRISAM";
//             // LOG(INFO) << "With values "
//             // new_values.print("New keys ", dyno::DynoLikeKeyFormatter);
//             // LOG(INFO) << "With keys " <<
//             dyno::container_to_string(new_values.keys());

//             gtsam::Values all_values = formulation->getTheta();
//             all_values.print("All keys ", dyno::DynoLikeKeyFormatter);
//             auto all_factors = formulation->getGraph();
//             // gtsam::Values all_values = formulation->getTheta();

//             auto order = gtsam::Ordering::Colamd(all_factors);

//             mr_isam = std::make_shared<MRISAM2>(all_factors, all_values,
//             order, 0, other_root_keys_map, MRISAM2Params{});

//             mr_isam->saveGraph(dyno::getOutputFilePath(
//                             "oc_multi_bayes_tree_" + std::to_string(frame_id)
//                             + "_" + formulation->getFullyQualifiedName() +
//                             ".dot"));
//         }
//         else if(mr_isam) {
//             //update static
//             MRISAM2Result result = mr_isam->updateRoot(
//                 0, graph_update.staticFactors(), graph_update.staticValues(),
//                 true);

//                 mr_isam->saveGraph(
//                 dyno::getOutputFilePath(
//                     "oc_multi_bayes_tree_" + std::to_string(frame_id) +
//                     "_obj0_" + formulation->getFullyQualifiedName() +
//                     ".dot"),
//                 result.top_cliques);

//             //update dynamic
//             // result = mr_isam->updateRoot(
//             //     1, graph_update.factors(1).value(),
//             graph_update.values(1).value(), true);
//             //     mr_isam->saveGraph(
//             //     dyno::getOutputFilePath(
//             //         "oc_multi_bayes_tree_" + std::to_string(frame_id) +
//             "_obj1_" +
//             //         formulation->getFullyQualifiedName() + ".dot"),
//             //     result.top_cliques);

//         }

//         };

//         data->backend = backend;
//         datas.push_back(data);
//     }

//     void processAll(dyno::RGBDInstanceOutputPacket::Ptr output_packet) {
//         for (auto d : datas) {
//         d->backend->spinOnce(output_packet);
//         }
//     }

//     void finishAll() {
//         for (auto& d : datas) {
//         auto backend = d->backend;
//         dyno::BackendMetaData backend_info;
//         backend->new_updater_->accessorFromTheta()->postUpdateCallback(
//             backend_info);
//         backend->new_updater_->logBackendFromMap(backend_info);

//         backend_info.logging_suffix = "isam_opt";
//         backend->new_updater_->updateTheta(d->opt_values);
//         backend->new_updater_->accessorFromTheta()->postUpdateCallback(
//             backend_info);
//         backend->new_updater_->logBackendFromMap(backend_info);
//         }
//     }

//     std::vector<std::shared_ptr<Data>> datas;
//     };

//     RunIncrementalTest tester;
//     tester.addBackend(std::make_shared<dyno::RGBDBackendModule>(
//         backend_params, dyno_testing::makeDefaultCameraPtr(),
//         dyno::RGBDBackendModule::UpdaterType::ObjectCentric));

//     // tester.addBackend(std::make_shared<dyno::RGBDBackendModule>(
//     //     backend_params, dyno_testing::makeDefaultCameraPtr(),
//     //     dyno::RGBDBackendModule::UpdaterType::OC_SD));

//     // tester.addBackend(std::make_shared<dyno::RGBDBackendModule>(
//     //     backend_params, dyno_testing::makeDefaultCameraPtr(),
//     //     dyno::RGBDBackendModule::UpdaterType::OC_D));

//     // tester.addBackend(std::make_shared<dyno::RGBDBackendModule>(
//     //     backend_params, dyno_testing::makeDefaultCameraPtr(),
//     //     dyno::RGBDBackendModule::UpdaterType::OC_S));

//     for (size_t i = 0; i < 20; i++) {
//     dyno::RGBDInstanceOutputPacket::Ptr output_gt, output_noisy;
//     std::tie(output_gt, output_noisy) = scenario.getOutput(i);

//     tester.processAll(output_noisy);
//     }

//     tester.finishAll();
//     // dyno::writeStatisticsSamplesToFile("statistics_samples.csv");
//     // dyno::writeStatisticsModuleSummariesToFile();
// }

TEST(MRBayesTree, testSimpl) {
  using namespace dyno;

  dyno_testing::ScenarioBody::Ptr camera =
      std::make_shared<dyno_testing::ScenarioBody>(
          std::make_unique<dyno_testing::ConstantMotionBodyVisitor>(
              gtsam::Pose3::Identity(),
              // motion only in x
              gtsam::Pose3(gtsam::Rot3::RzRyRx(0.3, 0.1, 0.0),
                           gtsam::Point3(0.1, 0.05, 0))));

  const double H_R_sigma = 0.05;
  const double H_t_sigma = 0.08;
  const double dynamic_point_sigma = 0.1;

  const double X_R_sigma = 0.0;
  const double X_t_sigma = 0.0;

  dyno_testing::RGBDScenario::NoiseParams noise_params;
  noise_params.H_R_sigma = H_R_sigma;
  noise_params.H_t_sigma = H_t_sigma;
  noise_params.dynamic_point_sigma = dynamic_point_sigma;
  noise_params.X_R_sigma = X_R_sigma;
  noise_params.X_t_sigma = X_t_sigma;

  dyno_testing::RGBDScenario scenario(
      camera,
      std::make_shared<dyno_testing::SimpleStaticPointsGenerator>(25, 15),
      noise_params);

  // add one obect
  const size_t num_points = 10;
  const size_t obj1_overlap = 5;
  const size_t obj2_overlap = 4;
  const size_t obj3_overlap = 5;
  dyno_testing::ObjectBody::Ptr object1 =
      std::make_shared<dyno_testing::ObjectBody>(
          std::make_unique<dyno_testing::ConstantMotionBodyVisitor>(
              gtsam::Pose3(gtsam::Rot3::Identity(), gtsam::Point3(2, 0, 0)),
              // motion only in x
              gtsam::Pose3(gtsam::Rot3::RzRyRx(0.2, 0.1, 0.0),
                           gtsam::Point3(0.2, 0, 0))),
          std::make_unique<dyno_testing::RandomOverlapObjectPointsVisitor>(
              num_points, obj1_overlap),
          dyno_testing::ObjectBodyParams{});

  dyno_testing::ObjectBody::Ptr object2 =
      std::make_shared<dyno_testing::ObjectBody>(
          std::make_unique<dyno_testing::ConstantMotionBodyVisitor>(
              gtsam::Pose3(gtsam::Rot3::Identity(), gtsam::Point3(1, 0.4, 0.1)),
              // motion only in x
              gtsam::Pose3(gtsam::Rot3::Identity(), gtsam::Point3(0.2, 0, 0))),
          std::make_unique<dyno_testing::RandomOverlapObjectPointsVisitor>(
              num_points, obj2_overlap),
          dyno_testing::ObjectBodyParams{.enters_scenario = 8,
                                         .leaves_scenario = 15});

  dyno_testing::ObjectBody::Ptr object3 =
      std::make_shared<dyno_testing::ObjectBody>(
          std::make_unique<dyno_testing::ConstantMotionBodyVisitor>(
              gtsam::Pose3(gtsam::Rot3::RzRyRx(0.3, 0.2, 0.1),
                           gtsam::Point3(1.1, 0.2, 1.2)),
              // motion only in x
              gtsam::Pose3(gtsam::Rot3::RzRyRx(0.2, 0.1, 0.0),
                           gtsam::Point3(0.2, 0.3, 0))),
          std::make_unique<dyno_testing::RandomOverlapObjectPointsVisitor>(
              num_points, obj3_overlap),
          dyno_testing::ObjectBodyParams{.enters_scenario = 13,
                                         .leaves_scenario = 19});

  scenario.addObjectBody(1, object1);
  // scenario.addObjectBody(2, object2);
  // scenario.addObjectBody(3, object3);

  dyno::BackendParams backend_params;
  backend_params.useLogger(false);
  backend_params.min_dynamic_obs_ = 1u;
  backend_params.dynamic_point_noise_sigma_ = dynamic_point_sigma;
  backend_params.odometry_rotation_sigma_ = X_R_sigma;
  backend_params.odometry_translation_sigma_ = X_t_sigma;
}
