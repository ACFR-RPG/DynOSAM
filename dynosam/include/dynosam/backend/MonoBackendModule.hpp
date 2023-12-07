/*
 *   Copyright (c) 2023 ACFR-RPG, University of Sydney, Jesse Morris (jesse.morris@sydney.edu.au)
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

#pragma once

#include "dynosam/backend/BackendModule.hpp"
#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/backend/MonoBackend-Definitions.hpp"
#include "dynosam/common/Camera.hpp"

#include "dynosam/frontend/MonoInstance-Definitions.hpp" // for MonocularInstanceOutputPacket


#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

namespace dyno {

class MonoBackendModule : public BackendModule {

public:
    MonoBackendModule(const BackendParams& backend_params, Camera::Ptr camera);
    ~MonoBackendModule();

    using SpinReturn = BackendModule::SpinReturn;


private:
    SpinReturn boostrapSpin(BackendInputPacket::ConstPtr input) override;
    SpinReturn nominalSpin(BackendInputPacket::ConstPtr input) override;

    SpinReturn monoBoostrapSpin(MonocularInstanceOutputPacket::ConstPtr input);
    SpinReturn monoNominalSpin(MonocularInstanceOutputPacket::ConstPtr input);

    //adds pose to the new values and a prior on this pose to the new_factors
    void addInitialPose(const gtsam::Pose3& T_world_camera, FrameId frame_id, gtsam::Values& new_values,  gtsam::NonlinearFactorGraph& new_factors);

    //T_world_camera is the estimate from the frontend and should be associated with the curr_frame_id
    void addOdometry(const gtsam::Pose3& T_world_camera, FrameId curr_frame_id, FrameId prev_frame_id, gtsam::Values& new_values,  gtsam::NonlinearFactorGraph& new_factors);


    // void handleStaticMeasurementsBoostrap(MonocularInstanceOutputPacket::ConstPtr input, gtsam::Values& new_values,  gtsam::NonlinearFactorGraph& new_factors);
    // void handleStaticMeasurementsNominal(MonocularInstanceOutputPacket::ConstPtr input, gtsam::Values& new_values,  gtsam::NonlinearFactorGraph& new_factors);


    void updateSmartStaticObservations(
        const StatusKeypointMeasurements& measurements,
        const FrameId frame_id,
        TrackletIds& new_smart_factors,
        TrackletIds& updated_smart_factors,
        StatusKeypointMeasurements& new_projection_measurements);


    //adds smart factors from smart factor map and updates slots according to thew NLFG passed in
    void addNewSmartStaticFactors(const TrackletIds& new_smart_factors,  gtsam::NonlinearFactorGraph& factors);

    //try to triangualte on smart factor factors that have just been updated
    //if we can triangulate, update and convert existing factor to projection factor
    //new_and_current_state should be the most up to date values (including the most recent camera pose which may not be in state_)
    //that we want to use to triangulate the point
    //this will also add the projection factors to the graph so no need to run another "add projection measurements"
    TrackletIds tryTriangulateExistingSmartStaticFactors(const TrackletIds& updated_smart_factors, const gtsam::Values& new_and_current_state, gtsam::Values& new_values, gtsam::NonlinearFactorGraph& factors);

    //the slot in the smart_factor_map should be associted with the factor in factors
    //will be removed from the smart factor map but remain in the tracklet_to_status_map_ where the status will be updated to PROJECTION
    bool convertSmartToProjectionFactor(const TrackletId smart_factor_to_convert, const gtsam::Point3& lmk_world, gtsam::Values& new_values, gtsam::NonlinearFactorGraph& factors);


    void addStaticProjectionMeasurements(const FrameId frame_id, const StatusKeypointMeasurements& new_projection_measurements, gtsam::NonlinearFactorGraph& factors);

    void setFactorParams(const BackendParams& backend_params);


    void saveAllToGraphFile(MonocularInstanceOutputPacket::ConstPtr input);

    //dynamic measuerement stuff
    void updateDynamicObjectTrackletMap(MonocularInstanceOutputPacket::ConstPtr input);


    bool attemptObjectTriangulation(
        FrameId current_frame,
        FrameId previous_frame,
        ObjectId object_id,
        const EssentialDecompositionResult&  motion_estimate,
        const gtsam::Pose3& T_world_camera_curr,
        const gtsam::Pose3& T_world_camera_prev,
        StatusLandmarkEstimates& triangulated_values,
        gtsam::Rot3& R_motion);


    void addInitalObjectValues(
        FrameId current_frame_id,
        ObjectId object_id,
        const StatusLandmarkEstimates& triangulated_values,
        const gtsam::Pose3& prev_H_world_current,
        gtsam::Values& new_values, gtsam::NonlinearFactorGraph& factors);


private:
    Camera::Ptr camera_;
    boost::shared_ptr<Camera::CalibrationType> gtsam_calibration_; //Ugh, this version of gtsam still uses some boost

    // //data structures to hold measurements and landmarks
    // TrackletIdSmartFactorMap tracklet_smart_factor_map_;
    // TrackletIdLabelMap tracklet_to_label_map_;
    // TrackletIdToTypeMap tracklet_to_type_map_;
    // TrackletIdSlotMap tracklet_to_slot_; //also indicates if tracklet factor is in the map (either as smart or projection factor). -1 means not in the map
    // ProjectionFactorStatusMap projection_factor_map_;
    TrackletIdToProjectionStatus tracklet_to_status_map_;
    SmartProjectionFactorMap smart_factor_map_;


    //params for factors
    SmartProjectionFactorParams static_projection_params_; //! Projection factor params for static points
    gtsam::SharedNoiseModel static_smart_noise_; //! Projection factor noise for static points
    gtsam::SharedNoiseModel odometry_noise_; //! Between factor noise for between two consequative poses
    gtsam::SharedNoiseModel initial_pose_prior_;


    DynamicObjectTrackletManager<Keypoint> do_tracklet_manager_;


    // State.
    //!< current state of the system.
    gtsam::Values state_;
    gtsam::NonlinearFactorGraph state_graph_;

    MonocularInstanceOutputPacket::ConstPtr previous_input_;
    std::stringstream fg_ss_;



};


} //dyno
