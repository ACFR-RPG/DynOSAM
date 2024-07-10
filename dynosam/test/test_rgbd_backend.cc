/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris (jesse.morris@sydney.edu.au)
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a copy
 *   of this software and associated documentation files (the "Software"), to deal
 *   in the Software without restriction, including without limitation the rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:

 *   The above copyright notice and this permission notice shall be included in all
 *   copies or substantial portions of the Software.

 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *   SOFTWARE.
 */

#include "internal/simulator.hpp"
#include "internal/helpers.hpp"

#include "dynosam/backend/BackendPipeline.hpp"
#include "dynosam/backend/IncrementalOptimizer.hpp"
#include "dynosam/backend/FactorGraphTools.hpp"
#include "dynosam/backend/RGBDBackendModule.hpp"

#include <gtsam/linear/GaussianEliminationTree.h>

#include <gtsam/nonlinear/ISAM2-impl.h>


#include "dynosam/factors/LandmarkMotionTernaryFactor.hpp"
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

#include <gtsam_unstable/slam/PoseToPointFactor.h>


#include <glog/logging.h>
#include <gtest/gtest.h>
#include <exception>
#include <vector>
#include <iterator>


TEST(RGBDBackendModule, constructSimpleGraph) {

    //make camera with a constant motion model starting at zero
    dyno_testing::ScenarioBody::Ptr camera = std::make_shared<dyno_testing::ScenarioBody>(
        std::make_unique<dyno_testing::ConstantMotionBodyVisitor>(
            gtsam::Pose3::Identity(),
            //motion only in x
            gtsam::Pose3(gtsam::Rot3::Identity(), gtsam::Point3(0.1, 0, 0))
        )
    );
    //needs to be at least 3 overlap so we can meet requirements in graph
    //TODO: how can we do 1 point but with lots of overlap (even infinity overlap?)
    dyno_testing::RGBDScenario scenario(
        camera,
        std::make_shared<dyno_testing::SimpleStaticPointsGenerator>(6, 1)
    );

    //add one obect
    const size_t num_points = 3;
    dyno_testing::ObjectBody::Ptr object1 = std::make_shared<dyno_testing::ObjectBody>(
        std::make_unique<dyno_testing::ConstantMotionBodyVisitor>(
            gtsam::Pose3(gtsam::Rot3::Identity(), gtsam::Point3(10, 0, 0)),
            //motion only in x
            gtsam::Pose3(gtsam::Rot3::Identity(), gtsam::Point3(0.2, 0, 0))
        ),
        std::make_unique<dyno_testing::ConstantObjectPointsVisitor>(num_points)
    );

    dyno_testing::ObjectBody::Ptr object2 = std::make_shared<dyno_testing::ObjectBody>(
        std::make_unique<dyno_testing::ConstantMotionBodyVisitor>(
            gtsam::Pose3(gtsam::Rot3::Identity(), gtsam::Point3(10, 0, 0)),
            //motion only in x
            gtsam::Pose3(gtsam::Rot3::Identity(), gtsam::Point3(0.2, 0, 0))
        ),
        std::make_unique<dyno_testing::ConstantObjectPointsVisitor>(num_points)
    );

    scenario.addObjectBody(1, object1);
    // scenario.addObjectBody(2, object2);

    dyno::Map3d2d::Ptr map = dyno::Map3d2d::create();
    // std::shared_ptr<dyno::IncrementalOptimizer<dyno::Landmark>> optimizer = std::make_shared<dyno::IncrementalOptimizer<dyno::Landmark>>();
    // std::shared_ptr<dyno_testing::DummyOptimizer<dyno::Landmark>> optimizer = std::make_shared<dyno_testing::DummyOptimizer<dyno::Landmark>>();
    // std::shared_ptr<dyno::ISAMOptimizer<dyno::Landmark>> optimizer = std::make_shared<dyno::ISAMOptimizer<dyno::Landmark>>();
    dyno::RGBDBackendModule backend(
        dyno::BackendParams().useLogger(false),
        map,
        dyno_testing::makeDefaultCameraPtr(),
        dyno::RGBDBackendModule::UpdaterType::ObjectCentric
        );

    for(size_t i = 0; i < 7; i++) {
        auto output = scenario.getOutput(i);

        std::stringstream ss;
        ss << output->T_world_camera_ << "\n";
        ss << dyno::container_to_string(output->static_landmarks_);

        // LOG(INFO) << ss.str();
        backend.spinOnce(output);

        LOG(INFO) << "Spun backend";

        // //Is this the full graph?
        // gtsam::NonlinearFactorGraph full_graph = backend.getMap()->getGraph();
        // gtsam::Values values = backend.smoother_->getLinearizationPoint();

        //get variables in this frame
        // const auto& map = backend.getMap();
        // const auto frame_k_node = map->getFrame(i);
        // const auto dyn_lmk_nodes = frame_k_node->dynamic_landmarks;
        // const auto object_nodes = frame_k_node->objects_seen;

        // gtsam::KeyVector dyn_lmks_this_frame;

        // for(const auto& dyn_lmk_node : dyn_lmk_nodes) {
        //     gtsam::Key key = dyn_lmk_node->makeDynamicKey(i);
        //     if(map->exists(key)) {
        //         dyn_lmks_this_frame.push_back(key);
        //     }
        // }

        // dyn_lmks_this_frame.push_back(map->getFrame(i)->makePoseKey());

        // gtsam::KeyVector object_motions_this_frame;

        // for(const auto& node : object_nodes) {
        //     const gtsam::Key key = frame_k_node->makeObjectMotionKey(node->getId());
        //     if(map->exists(key)) {
        //         dyn_lmks_this_frame.push_back(key);
        //     }
        // }


        // gtsam::VariableIndex affectedFactorsVarIndex(full_graph);
        // gtsam::Ordering order = gtsam::Ordering::ColamdConstrainedLast(affectedFactorsVarIndex, dyn_lmks_this_frame);
        // auto linearized = full_graph.linearize(values);

        // auto bayesTree =
        //     gtsam::ISAM2JunctionTree(
        //         gtsam::GaussianEliminationTree(*linearized, affectedFactorsVarIndex, order))
        //         .eliminate(gtsam::EliminatePreferCholesky)
        //         .first;

        // LOG(INFO) << bayesTree->roots().size();
        // LOG(WARNING) << "Number nnz bayes tree " << bayesTree->roots().at(0)->calculate_nnz();
        // LOG(WARNING) << "Number nnz isam2 tree " << optimizer->getSmoother().roots().at(0)->calculate_nnz();

        // // bayesTree->saveGraph(dyno::getOutputFilePath("elimated_tree.dot"), dyno::DynoLikeKeyFormatter);
        // dyno::factor_graph_tools::saveBayesTree(*bayesTree, dyno::getOutputFilePath("elimated_tree.dot"), dyno::DynoLikeKeyFormatter);

        // const auto& dynosam2_result = optimizer->getResult();

        // gtsam::FastMap<gtsam::Key, std::string> coloured_affected_keys;
        // for(const auto& key : dynosam2_result.reeliminatedKeys) {
        //     coloured_affected_keys.insert2(key, "red");
        // }

        // // const auto& smoother = optimizer->getSmoother().bayesTree();
        // const auto& smoother = optimizer->getSmoother();

        // // smoother.calculateBestEstimate();
        // if(smoother.roots().empty()) {
        //     continue;
        // }

        // dyno::factor_graph_tools::saveBayesTree(
        //     smoother,
        //     dyno::getOutputFilePath("rgbd_bayes_tree_" + std::to_string(i) + ".dot"),
        //     dyno::DynoLikeKeyFormatter,
        //     coloured_affected_keys);

        // {
        //     gtsam::KeySet marginalize_keys = dyno::factor_graph_tools::travsersal::getLeafKeys(smoother);
        //     gtsam::PrintKeySet(marginalize_keys, "Leaf keys", dyno::DynoLikeKeyFormatter);

        //     gtsam::FastMap<gtsam::Key, std::string> coloured_affected_keys;
        //     for(const auto& key : marginalize_keys) {
        //         coloured_affected_keys.insert2(key, "blue");
        //     }

        //     dyno::factor_graph_tools::saveBayesTree(
        //         smoother,
        //         dyno::getOutputFilePath("rgbd_bayes_tree_leaf.dot"),
        //         dyno::DynoLikeKeyFormatter,
        //         coloured_affected_keys);

        // }

        // smoother.getFactorsUnsafe().saveGraph(dyno::getOutputFilePath("smoother_graph_" + std::to_string(i) + ".dot"), dyno::DynoLikeKeyFormatter);

        // backend.saveGraph("rgbd_graph_" + std::to_string(i) + ".dot");
        // backend.saveTree("rgbd_bayes_tree_" + std::to_string(i) + ".dot");
    }

    // gtsam::NonlinearFactorGraph full_graph = backend.new_updater_->getGraph();
    // full_graph.saveGraph(dyno::getOutputFilePath("construct_simple_graph_test_object_pose.dot"), dyno::DynoLikeKeyFormatter);

    // const auto[delayed_values, delayed_graph] = backend.constructGraph(2, 6, true);
    // delayed_graph.saveGraph(dyno::getOutputFilePath("construct_simple_delayed_graph_test.dot"), dyno::DynoLikeKeyFormatter);

     gtsam::NonlinearFactorGraph full_graph = backend.new_updater_->getGraph();
    full_graph.saveGraph(dyno::getOutputFilePath("construct_simple_graph_object_centric.dot"), dyno::DynoLikeKeyFormatter);

    const auto[delayed_values, delayed_graph] = backend.constructGraph(2, 6, true);
    delayed_graph.saveGraph(dyno::getOutputFilePath("construct_simple_delayed_graph_object_centric.dot"), dyno::DynoLikeKeyFormatter);






    // auto graph = backend.getMap()->getGraph();
    // auto estimates = backend.getMap()->getValues();

    // gtsam::GaussianFactorGraph factors = *graph.linearize(estimates);
    // gtsam::VariableIndex affectedFactorsVarIndex(factors);
    // //NOTE: NOT the same as in sam2 because of the affectedFactorsVarIndex
    // //maybe we can reconstruct the affectedFactorsVarIndex by looking at the factor index's and making a factor graph
    // //from this? This needs to be the factors that are affected
    // const gtsam::Ordering ordering =
    //   gtsam::Ordering::ColamdConstrained(affectedFactorsVarIndex, gtsam::FastMap<gtsam::Key, int>{});

    // gtsam::GaussianEliminationTree etree(factors, affectedFactorsVarIndex, ordering);

    // auto bayesTree = gtsam::ISAM2JunctionTree(etree)
    //                    .eliminate(gtsam::EliminatePreferCholesky)
    //                    .first;
    // bayesTree->saveGraph(dyno::getOutputFilePath("elimated_tree.dot"), dyno::DynoLikeKeyFormatter);
    // // etree.sa();

    // auto gfg = graph.linearize(estimates);

    // gtsam::Ordering isam2_ordering(dyno::factor_graph_tools::travsersal::getEliminatonOrder(*backend.smoother_));

    // //this ordering is not right. how do we get the ordering used by the bayes tree?
    // //OH DUH, the bayes tree IS the order!!! So which tree traversal algorithm do we use?
    // //Do we not start at the roots and check the frontal variables?
    // dyno::factor_graph_tools::computeRFactor(gfg, isam2_ordering);



}




TEST(RGBDBackendModule, testCliques) {
    using namespace dyno;
    //the simplest dynamic graph
    gtsam::NonlinearFactorGraph graph;
    gtsam::Values initial;

    gtsam::Pose3 H_0_1 = gtsam::Pose3(gtsam::Rot3::Rodrigues(0.3,0.2,0.1), gtsam::Point3 (0, 1, 1));
    gtsam::Pose3 H_1_2 = gtsam::Pose3(gtsam::Rot3::Rodrigues(0.5,0.1,0.1), gtsam::Point3 (1, 1.5, 1));
    gtsam::Pose3 H_0_2 = H_0_1 * H_1_2;

    gtsam::Point3 dyn_point_1_world = gtsam::Point3(2, 1, 3);
    gtsam::Point3 dyn_point_2_world = gtsam::Point3(1, 1, 3);
    gtsam::Point3 dyn_point_3_world = gtsam::Point3(3, 0.5, 2);

    static gtsam::Point3 p0 = gtsam::Point3(1,3,4);
    static gtsam::Rot3   R0 = gtsam::Rot3::Expmap( ( gtsam::Vector(3) << 0.2,0.1,0.12 ).finished() );
    static gtsam::Point3 p1 = gtsam::Point3(1,2,1);
    static gtsam::Rot3   R1 = gtsam::Rot3::Expmap( ( gtsam::Vector(3) << 0.1,0.2,1.570796 ).finished() );
    static gtsam::Point3 p2 = gtsam::Point3(2,2,1);
    static gtsam::Rot3   R2 = gtsam::Rot3::Expmap( ( gtsam::Vector(3) << 0.2,0.3,3.141593 ).finished() );
    static gtsam::Point3 p3 = gtsam::Point3(-1,1,0);
    static gtsam::Rot3   R3 = gtsam::Rot3::Expmap( ( gtsam::Vector(3) << 0.1,0.4,4.712389 ).finished() );

    static gtsam::Pose3 pose0 = gtsam::Pose3(R0,p0);
    static gtsam::Pose3 pose1 = gtsam::Pose3(R1,p1);
    static gtsam::Pose3 pose2 = gtsam::Pose3(R2,p2);
    static gtsam::Pose3 pose3 = gtsam::Pose3(R3,p3);

    auto landmark_noise = gtsam::noiseModel::Isotropic::Sigma(3u, 10);

    //static point seen at frames 0, 1 and 2
    graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
            CameraPoseSymbol(0),
            StaticLandmarkSymbol(0),
            gtsam::Point3(1, 2, 3),
            landmark_noise
        );
    graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
            CameraPoseSymbol(1),
            StaticLandmarkSymbol(0),
            gtsam::Point3(2, 2, 3),
            landmark_noise
        );

    graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
            CameraPoseSymbol(2),
            StaticLandmarkSymbol(0),
            gtsam::Point3(3, 2, 3),
            landmark_noise
        );

    //motion between frames 0 and 1
    //add motion factor for the 3 tracklet with tracklet id = 1, 2, 3
    graph.emplace_shared<LandmarkMotionTernaryFactor>(
        DynamicLandmarkSymbol(0, 1),
        DynamicLandmarkSymbol(1, 1),
        ObjectMotionSymbol(1, 1),
        landmark_noise
    );

    //motion between frames 1 and 2
    graph.emplace_shared<LandmarkMotionTernaryFactor>(
        DynamicLandmarkSymbol(1, 1),
        DynamicLandmarkSymbol(2, 1),
        ObjectMotionSymbol(1, 2),
        landmark_noise
    );

    //tracklet 2
    graph.emplace_shared<LandmarkMotionTernaryFactor>(
        DynamicLandmarkSymbol(0, 2),
        DynamicLandmarkSymbol(1, 2),
        ObjectMotionSymbol(1, 1),
        landmark_noise
    );
    //motion between frames 1 and 2
    graph.emplace_shared<LandmarkMotionTernaryFactor>(
        DynamicLandmarkSymbol(1, 2),
        DynamicLandmarkSymbol(2, 2),
        ObjectMotionSymbol(1, 2),
        landmark_noise
    );


    //tracklet 3
    graph.emplace_shared<LandmarkMotionTernaryFactor>(
        DynamicLandmarkSymbol(0, 3),
        DynamicLandmarkSymbol(1, 3),
        ObjectMotionSymbol(1, 1),
        landmark_noise
    );
    //motion between frames 1 and 2
    graph.emplace_shared<LandmarkMotionTernaryFactor>(
        DynamicLandmarkSymbol(1, 3),
        DynamicLandmarkSymbol(2, 3),
        ObjectMotionSymbol(1, 2),
        landmark_noise
    );

    //tracklet 1
    //add dynamic point obs frame 0
    graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
        CameraPoseSymbol(0),
        DynamicLandmarkSymbol(0, 1),
        pose0.inverse() * dyn_point_1_world,
        landmark_noise
    );

     //add dynamic point obs frame 1
    graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
            CameraPoseSymbol(1),
            DynamicLandmarkSymbol(1, 1),
            pose1.inverse() * H_0_1 * dyn_point_1_world,
            landmark_noise
        );

     //add dynamic point obs frame 2
    graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
            CameraPoseSymbol(2),
            DynamicLandmarkSymbol(2, 1),
            pose2.inverse() * H_0_2 * dyn_point_1_world,
            landmark_noise
    );

    //tracklet 2
    //add dynamic point obs frame 0
    graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
        CameraPoseSymbol(0),
        DynamicLandmarkSymbol(0, 2),
        pose0.inverse() * dyn_point_2_world,
        landmark_noise
    );

     //add dynamic point obs frame 1
    graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
            CameraPoseSymbol(1),
            DynamicLandmarkSymbol(1, 2),
            pose1.inverse() * H_0_1 * dyn_point_2_world,
            landmark_noise
        );

     //add dynamic point obs frame 2
    graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
            CameraPoseSymbol(2),
            DynamicLandmarkSymbol(2, 2),
            pose2.inverse() * H_0_2 * dyn_point_2_world,
            landmark_noise
    );


    //tracklet 3
    //add dynamic point obs frame 0
    graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
        CameraPoseSymbol(0),
        DynamicLandmarkSymbol(0, 3),
        pose0.inverse() * dyn_point_3_world,
        landmark_noise
    );

     //add dynamic point obs frame 1
    graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
            CameraPoseSymbol(1),
            DynamicLandmarkSymbol(1, 3),
            pose1.inverse() * H_0_1 * dyn_point_3_world,
            landmark_noise
        );

     //add dynamic point obs frame 2
    graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
            CameraPoseSymbol(2),
            DynamicLandmarkSymbol(2, 3),
            pose2.inverse() * H_0_2 * dyn_point_3_world,
            landmark_noise
    );


    static gtsam::SharedNoiseModel pose_model(gtsam::noiseModel::Isotropic::Sigma(6, 0.1));

    // add two poses
    graph.add(gtsam::BetweenFactor<gtsam::Pose3>(CameraPoseSymbol(0), CameraPoseSymbol(1), pose0.between(pose1), pose_model));
    graph.add(gtsam::BetweenFactor<gtsam::Pose3>(CameraPoseSymbol(1), CameraPoseSymbol(2), pose1.between(pose2), pose_model));

    // add prior on first pose
    graph.addPrior(CameraPoseSymbol(0), pose0, pose_model);

    //now add inital values

    //add static point
    initial.insert(StaticLandmarkSymbol(0), gtsam::Point3(0, 1, 1));

    //add dynamic points

    //frame 0
    initial.insert(DynamicLandmarkSymbol(0, 1), dyn_point_1_world);
    initial.insert(DynamicLandmarkSymbol(0, 2), dyn_point_2_world);
    initial.insert(DynamicLandmarkSymbol(0, 3), dyn_point_3_world);

    //frame 1
    initial.insert(DynamicLandmarkSymbol(1, 1), H_0_1 * dyn_point_1_world);
    initial.insert(DynamicLandmarkSymbol(1, 2), H_0_1 * dyn_point_2_world);
    initial.insert(DynamicLandmarkSymbol(1, 3), H_0_1 * dyn_point_3_world);

    //frame 2
    initial.insert(DynamicLandmarkSymbol(2, 1), H_0_2 * dyn_point_1_world);
    initial.insert(DynamicLandmarkSymbol(2, 2), H_0_2 * dyn_point_2_world);
    initial.insert(DynamicLandmarkSymbol(2, 3), H_0_2 * dyn_point_3_world);

    //add two motions
    initial.insert(ObjectMotionSymbol(1, 1), gtsam::Pose3::Identity());
    initial.insert(ObjectMotionSymbol(1, 2), gtsam::Pose3::Identity());

    //add three poses
    initial.insert(CameraPoseSymbol(0), pose0);
    initial.insert(CameraPoseSymbol(1), pose1);
    initial.insert(CameraPoseSymbol(2), pose2);

    dyno::NonlinearFactorGraphManager nlfgm(graph, initial);
    nlfgm.writeDynosamGraphFile(dyno::getOutputFilePath("test_graph.g2o"));

    // graph.saveGraph(dyno::getOutputFilePath("small_graph.dot"), dyno::DynoLikeKeyFormatter);
    gtsam::ISAM2BayesTree::shared_ptr bayesTree = nullptr;
    {//
        // gtsam::ISAM2 isam2;
        gtsam::VariableIndex affectedFactorsVarIndex(graph);
        gtsam::Ordering order = gtsam::Ordering::Colamd(affectedFactorsVarIndex);
        auto linearized = graph.linearize(initial);

        bayesTree =
        gtsam::ISAM2JunctionTree(
            gtsam::GaussianEliminationTree(*linearized, affectedFactorsVarIndex, order))
            .eliminate(gtsam::EliminatePreferCholesky)
            .first;

        bayesTree->saveGraph(dyno::getOutputFilePath("elimated_tree.dot"), dyno::DynoLikeKeyFormatter);
    }

    // {

    //     gtsam::ISAM2UpdateParams isam_update_params;

    //     gtsam::FastMap<gtsam::Key, int> constrainedKeys;
    //     //this includes the motions, where do we want these?
    //     //maybe BETWEEN the dynnamic keys
    //     //we want motions always on the right but impossible since parent?
    //     for(const auto& [keys, value] : initial) {
    //         constrainedKeys.insert2(keys, 1);
    //     }
    //     //put previous dynamic keys lower in the graph
    //     constrainedKeys.insert2(DynamicLandmarkSymbol(0, 1), 0);
    //     constrainedKeys.insert2(DynamicLandmarkSymbol(1, 1), 0);

    //     constrainedKeys.insert2(DynamicLandmarkSymbol(0, 2), 0);
    //     constrainedKeys.insert2(DynamicLandmarkSymbol(1, 2), 0);

    //     constrainedKeys.insert2(DynamicLandmarkSymbol(0, 2), 0);
    //     constrainedKeys.insert2(DynamicLandmarkSymbol(1, 2), 0);

    //     ///put current keys later in the graph
    //     constrainedKeys.insert2(DynamicLandmarkSymbol(2, 1), 2);
    //     constrainedKeys.insert2(DynamicLandmarkSymbol(2, 2), 2);
    //     constrainedKeys.insert2(DynamicLandmarkSymbol(2, 3), 2);

    //     isam_update_params.constrainedKeys = constrainedKeys;
    //     isam2.update(graph, initial, isam_update_params);
    // }

    // isam2.saveGraph(dyno::getOutputFilePath("small_tree_original.dot"), dyno::DynoLikeKeyFormatter);

    //add new factors to  motion
    //the simplest dynamic graph
    gtsam::NonlinearFactorGraph new_graph;
    gtsam::Values new_values;

    gtsam::Pose3 H_2_3 = gtsam::Pose3(gtsam::Rot3::Rodrigues(0.4,0.15,0.1), gtsam::Point3 (0, 3, 1));
    gtsam::Pose3 H_0_3 = H_0_2 * H_2_3;

    //add dynamic obs
    new_graph.emplace_shared<LandmarkMotionTernaryFactor>(
        DynamicLandmarkSymbol(2, 1),
        DynamicLandmarkSymbol(3, 1),
        ObjectMotionSymbol(1, 3),
        landmark_noise
    );

     new_graph.emplace_shared<LandmarkMotionTernaryFactor>(
        DynamicLandmarkSymbol(2, 2),
        DynamicLandmarkSymbol(3, 2),
        ObjectMotionSymbol(1, 3),
        landmark_noise
    );

    new_graph.emplace_shared<LandmarkMotionTernaryFactor>(
        DynamicLandmarkSymbol(2, 3),
        DynamicLandmarkSymbol(3, 3),
        ObjectMotionSymbol(1, 3),
        landmark_noise
    );

    //add pose-point constraints
    new_graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
        CameraPoseSymbol(3),
        DynamicLandmarkSymbol(3, 1),
        pose3.inverse() * H_0_3 * dyn_point_1_world,
        landmark_noise
    );
    new_graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
            CameraPoseSymbol(3),
            DynamicLandmarkSymbol(3,2),
            pose3.inverse() * H_0_3 * dyn_point_2_world,
            landmark_noise
        );

    new_graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
            CameraPoseSymbol(3),
            DynamicLandmarkSymbol(3, 3),
            pose3.inverse() * H_0_3 * dyn_point_3_world,
            landmark_noise
    );

    //add static obs
    new_graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
            CameraPoseSymbol(3),
            StaticLandmarkSymbol(0),
            gtsam::Point3(4, 2, 3),
            landmark_noise
        );

    //add odom
    new_graph.add(gtsam::BetweenFactor<gtsam::Pose3>(CameraPoseSymbol(2), CameraPoseSymbol(3), pose2.between(pose3), pose_model));

    //add initials
    new_values.insert(DynamicLandmarkSymbol(3, 1), H_0_3 * dyn_point_1_world);
    new_values.insert(DynamicLandmarkSymbol(3, 2), H_0_3 * dyn_point_2_world);
    new_values.insert(DynamicLandmarkSymbol(3, 3), H_0_3 * dyn_point_3_world);
    new_values.insert(ObjectMotionSymbol(1, 3), gtsam::Pose3::Identity());

    new_values.insert(CameraPoseSymbol(3), pose3);

    {
        gtsam::NonlinearFactorGraph full_graph(graph);
        full_graph.add_factors(new_graph);

        gtsam::Values all_values(initial);
        all_values.insert(new_values);

        gtsam::VariableIndex affectedFactorsVarIndex(full_graph);
        gtsam::Ordering order = gtsam::Ordering::ColamdConstrainedLast(affectedFactorsVarIndex, full_graph.keyVector(), true);
        auto linearized = full_graph.linearize(all_values);

        bayesTree =
        gtsam::ISAM2JunctionTree(
            gtsam::GaussianEliminationTree(*linearized, affectedFactorsVarIndex, order))
            .eliminate(gtsam::EliminatePreferCholesky)
            .first;

        bayesTree->saveGraph(dyno::getOutputFilePath("elimated_tree_1.dot"), dyno::DynoLikeKeyFormatter);
    }

    // gtsam::NonlinearFactorGraph all = graph;
    // all.add_factors(new_graph);
    // all.saveGraph(dyno::getOutputFilePath("small_graph_updated.dot"), dyno::DynoLikeKeyFormatter);

    //  {

    //     gtsam::ISAM2UpdateParams isam_update_params;

    //     gtsam::FastMap<gtsam::Key, int> constrainedKeys;
    //     //this includes the motions, where do we want these?
    //     //maybe BETWEEN the dynnamic keys
    //     //we want motions always on the right but impossible since parent?
    //     for(const auto& [keys, value] : new_values) {
    //         constrainedKeys.insert2(keys, 1);
    //     }
    //     constrainedKeys.insert2(DynamicLandmarkSymbol(2, 1), 0);
    //     constrainedKeys.insert2(DynamicLandmarkSymbol(2, 2), 0);
    //     constrainedKeys.insert2(DynamicLandmarkSymbol(2, 3), 0);

    //     ///put current keys later in the graph
    //     constrainedKeys.insert2(DynamicLandmarkSymbol(3, 1), 2);
    //     constrainedKeys.insert2(DynamicLandmarkSymbol(3, 2), 2);
    //     constrainedKeys.insert2(DynamicLandmarkSymbol(3, 3), 2);

    //     isam_update_params.constrainedKeys = constrainedKeys;
    //     isam2.update(new_graph, new_values, isam_update_params);
    // }

    // isam2.saveGraph(dyno::getOutputFilePath("small_tree_updated.dot"), dyno::DynoLikeKeyFormatter);





}


TEST(RGBDBackendModule, writeOutSimpleGraph) {
    using namespace dyno;
    //the simplest dynamic graph
    gtsam::NonlinearFactorGraph graph;
    gtsam::Values initial;

    gtsam::Pose3 H_0_1 = gtsam::Pose3(gtsam::Rot3::Rodrigues(0.3,0.2,0.1), gtsam::Point3 (0, 1, 1));
    gtsam::Pose3 H_1_2 = gtsam::Pose3(gtsam::Rot3::Rodrigues(0.5,0.1,0.1), gtsam::Point3 (1, 1.5, 1));
    gtsam::Pose3 H_0_2 = H_0_1 * H_1_2;

    gtsam::Point3 dyn_point_1_world = gtsam::Point3(2, 1, 3);
    gtsam::Point3 dyn_point_2_world = gtsam::Point3(1, 1, 3);
    gtsam::Point3 dyn_point_3_world = gtsam::Point3(3, 0.5, 2);

    static gtsam::Point3 p0 = gtsam::Point3(1,3,4);
    static gtsam::Rot3   R0 = gtsam::Rot3::Expmap( ( gtsam::Vector(3) << 0.2,0.1,0.12 ).finished() );
    static gtsam::Point3 p1 = gtsam::Point3(1,2,1);
    static gtsam::Rot3   R1 = gtsam::Rot3::Expmap( ( gtsam::Vector(3) << 0.1,0.2,1.570796 ).finished() );
    static gtsam::Point3 p2 = gtsam::Point3(2,2,1);
    static gtsam::Rot3   R2 = gtsam::Rot3::Expmap( ( gtsam::Vector(3) << 0.2,0.3,3.141593 ).finished() );
    static gtsam::Point3 p3 = gtsam::Point3(-1,1,0);
    static gtsam::Rot3   R3 = gtsam::Rot3::Expmap( ( gtsam::Vector(3) << 0.1,0.4,4.712389 ).finished() );

    static gtsam::Pose3 pose0 = gtsam::Pose3(R0,p0);
    static gtsam::Pose3 pose1 = gtsam::Pose3(R1,p1);
    static gtsam::Pose3 pose2 = gtsam::Pose3(R2,p2);
    static gtsam::Pose3 pose3 = gtsam::Pose3(R3,p3);

    auto landmark_noise = gtsam::noiseModel::Isotropic::Sigma(3u, 10);

    //static point seen at frames 0, 1 and 2
    graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
            CameraPoseSymbol(0),
            StaticLandmarkSymbol(0),
            gtsam::Point3(1, 2, 3),
            landmark_noise
        );
    graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
            CameraPoseSymbol(1),
            StaticLandmarkSymbol(0),
            gtsam::Point3(2, 2, 3),
            landmark_noise
        );

    graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
            CameraPoseSymbol(2),
            StaticLandmarkSymbol(0),
            gtsam::Point3(3, 2, 3),
            landmark_noise
        );

    //motion between frames 0 and 1
    //add motion factor for the 3 tracklet with tracklet id = 1, 2, 3
    graph.emplace_shared<LandmarkMotionTernaryFactor>(
        DynamicLandmarkSymbol(0, 1),
        DynamicLandmarkSymbol(1, 1),
        ObjectMotionSymbol(1, 1),
        landmark_noise
    );

    //motion between frames 1 and 2
    graph.emplace_shared<LandmarkMotionTernaryFactor>(
        DynamicLandmarkSymbol(1, 1),
        DynamicLandmarkSymbol(2, 1),
        ObjectMotionSymbol(1, 2),
        landmark_noise
    );

    //tracklet 2
    graph.emplace_shared<LandmarkMotionTernaryFactor>(
        DynamicLandmarkSymbol(0, 2),
        DynamicLandmarkSymbol(1, 2),
        ObjectMotionSymbol(1, 1),
        landmark_noise
    );
    //motion between frames 1 and 2
    graph.emplace_shared<LandmarkMotionTernaryFactor>(
        DynamicLandmarkSymbol(1, 2),
        DynamicLandmarkSymbol(2, 2),
        ObjectMotionSymbol(1, 2),
        landmark_noise
    );


    //tracklet 3
    graph.emplace_shared<LandmarkMotionTernaryFactor>(
        DynamicLandmarkSymbol(0, 3),
        DynamicLandmarkSymbol(1, 3),
        ObjectMotionSymbol(1, 1),
        landmark_noise
    );
    //motion between frames 1 and 2
    graph.emplace_shared<LandmarkMotionTernaryFactor>(
        DynamicLandmarkSymbol(1, 3),
        DynamicLandmarkSymbol(2, 3),
        ObjectMotionSymbol(1, 2),
        landmark_noise
    );

    //tracklet 1
    //add dynamic point obs frame 0
    graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
        CameraPoseSymbol(0),
        DynamicLandmarkSymbol(0, 1),
        pose0.inverse() * dyn_point_1_world,
        landmark_noise
    );

     //add dynamic point obs frame 1
    graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
            CameraPoseSymbol(1),
            DynamicLandmarkSymbol(1, 1),
            pose1.inverse() * H_0_1 * dyn_point_1_world,
            landmark_noise
        );

     //add dynamic point obs frame 2
    graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
            CameraPoseSymbol(2),
            DynamicLandmarkSymbol(2, 1),
            pose2.inverse() * H_0_2 * dyn_point_1_world,
            landmark_noise
    );

    //tracklet 2
    //add dynamic point obs frame 0
    graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
        CameraPoseSymbol(0),
        DynamicLandmarkSymbol(0, 2),
        pose0.inverse() * dyn_point_2_world,
        landmark_noise
    );

     //add dynamic point obs frame 1
    graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
            CameraPoseSymbol(1),
            DynamicLandmarkSymbol(1, 2),
            pose1.inverse() * H_0_1 * dyn_point_2_world,
            landmark_noise
        );

     //add dynamic point obs frame 2
    graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
            CameraPoseSymbol(2),
            DynamicLandmarkSymbol(2, 2),
            pose2.inverse() * H_0_2 * dyn_point_2_world,
            landmark_noise
    );


    //tracklet 3
    //add dynamic point obs frame 0
    graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
        CameraPoseSymbol(0),
        DynamicLandmarkSymbol(0, 3),
        pose0.inverse() * dyn_point_3_world,
        landmark_noise
    );

     //add dynamic point obs frame 1
    graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
            CameraPoseSymbol(1),
            DynamicLandmarkSymbol(1, 3),
            pose1.inverse() * H_0_1 * dyn_point_3_world,
            landmark_noise
        );

     //add dynamic point obs frame 2
    graph.emplace_shared<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
            CameraPoseSymbol(2),
            DynamicLandmarkSymbol(2, 3),
            pose2.inverse() * H_0_2 * dyn_point_3_world,
            landmark_noise
    );


    static gtsam::SharedNoiseModel pose_model(gtsam::noiseModel::Isotropic::Sigma(6, 0.1));

    // add two poses
    graph.add(gtsam::BetweenFactor<gtsam::Pose3>(CameraPoseSymbol(0), CameraPoseSymbol(1), pose0.between(pose1), pose_model));
    graph.add(gtsam::BetweenFactor<gtsam::Pose3>(CameraPoseSymbol(1), CameraPoseSymbol(2), pose1.between(pose2), pose_model));

    // add prior on first pose
    graph.addPrior(CameraPoseSymbol(0), pose0, pose_model);

    //now add inital values

    //add static point
    initial.insert(StaticLandmarkSymbol(0), gtsam::Point3(0, 1, 1));

    //add dynamic points

    //frame 0
    initial.insert(DynamicLandmarkSymbol(0, 1), dyn_point_1_world);
    initial.insert(DynamicLandmarkSymbol(0, 2), dyn_point_2_world);
    initial.insert(DynamicLandmarkSymbol(0, 3), dyn_point_3_world);

    //frame 1
    initial.insert(DynamicLandmarkSymbol(1, 1), H_0_1 * dyn_point_1_world);
    initial.insert(DynamicLandmarkSymbol(1, 2), H_0_1 * dyn_point_2_world);
    initial.insert(DynamicLandmarkSymbol(1, 3), H_0_1 * dyn_point_3_world);

    //frame 2
    initial.insert(DynamicLandmarkSymbol(2, 1), H_0_2 * dyn_point_1_world);
    initial.insert(DynamicLandmarkSymbol(2, 2), H_0_2 * dyn_point_2_world);
    initial.insert(DynamicLandmarkSymbol(2, 3), H_0_2 * dyn_point_3_world);

    //add two motions
    initial.insert(ObjectMotionSymbol(1, 1), gtsam::Pose3::Identity());
    initial.insert(ObjectMotionSymbol(1, 2), gtsam::Pose3::Identity());

    //add three poses
    initial.insert(CameraPoseSymbol(0), pose0);
    initial.insert(CameraPoseSymbol(1), pose1);
    initial.insert(CameraPoseSymbol(2), pose2);

    dyno::NonlinearFactorGraphManager nlfgm(graph, initial);
    // nlfgm.writeDynosamGraphFile(dyno::getOutputFilePath("test_graph.g2o"));

    graph.saveGraph(dyno::getOutputFilePath("simple_dynamic_graph.dot"), dyno::DynoLikeKeyFormatter);
    gtsam::GaussianFactorGraph::shared_ptr linearised_graph = nlfgm.linearize();
    gtsam::JacobianFactor jacobian_factor(*linearised_graph);
    gtsam::Matrix information = jacobian_factor.information();

    LOG(INFO) << information.rows() << " " << information.cols();

    writeMatrixWithPythonFormat(information, dyno::getOutputFilePath("simple_dynamic_graph_information.txt"));



}
