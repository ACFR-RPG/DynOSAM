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

#include "dynosam/frontend/RGBDInstanceFrontendModule.hpp"
#include "dynosam/frontend/RGBDInstance-Definitions.hpp"
#include "dynosam/frontend/vision/Vision-Definitions.hpp"
#include "dynosam/frontend/vision/MotionSolver.hpp"
#include "dynosam/utils/SafeCast.hpp"
#include "dynosam/utils/TimingStats.hpp"
#include "dynosam/logger/Logger.hpp"

#include <opencv4/opencv2/opencv.hpp>
#include <tbb/tbb.h>
#include <glog/logging.h>

#include "dynosam/common/Flags.hpp" //for common flags

DEFINE_bool(use_frontend_logger, false , "If true, the frontend logger will be used");
DEFINE_bool(use_dynamic_track, true, "If true, the dynamic tracking will be used");

DEFINE_bool(log_projected_masks, false, "If true, projected masks will be saved at every frame");

namespace dyno {


RGBDInstanceFrontendModule::RGBDInstanceFrontendModule(const FrontendParams& frontend_params, Camera::Ptr camera, ImageDisplayQueue* display_queue)
    : FrontendModule(frontend_params, display_queue),
      camera_(camera),
      motion_solver_(frontend_params, camera->getParams()),
      object_motion_solver_(frontend_params, camera->getParams())
    {
    CHECK_NOTNULL(camera_);
    tracker_ = std::make_unique<FeatureTracker>(frontend_params, camera_, display_queue);

    if(FLAGS_use_frontend_logger) {
        logger_ = std::make_unique<RGBDFrontendLogger>();
    }
}

RGBDInstanceFrontendModule::~RGBDInstanceFrontendModule() {
    if(FLAGS_save_frontend_json) {
        LOG(INFO) << "Saving frontend output as json";
        const std::string file_path = getOutputFilePath(kRgbdFrontendOutputJsonFile);
        JsonConverter::WriteOutJson(output_packet_record_, file_path, JsonConverter::Format::BSON);
    }

}

FrontendModule::ImageValidationResult
RGBDInstanceFrontendModule::validateImageContainer(const ImageContainer::Ptr& image_container) const {
    return ImageValidationResult(image_container->hasDepth(), "Depth is required");
}

FrontendModule::SpinReturn
RGBDInstanceFrontendModule::boostrapSpin(FrontendInputPacketBase::ConstPtr input) {

    ImageContainer::Ptr image_container = input->image_container_;

    //if we only have instance semgentation (not motion) then we need to make a motion mask out of the semantic mask
    //we cannot do this for the first frame so we will just treat the semantic mask and the motion mask
    //and then subsequently elimate non-moving objects later on
    TrackingInputImages tracking_images;
    if(image_container->hasSemanticMask()) {
        CHECK(!image_container->hasMotionMask());
        //TODO: some bug when going from semantic mask to motion mask as motion mask is empty in the tracker after this process!!!
        //its becuase we dont actually use the tracking_images!!
        auto intermediate_tracking_images = image_container->makeSubset<ImageType::RGBMono, ImageType::OpticalFlow, ImageType::SemanticMask>();
        tracking_images = TrackingInputImages(
            intermediate_tracking_images.getImageWrapper<ImageType::RGBMono>(),
            intermediate_tracking_images.getImageWrapper<ImageType::OpticalFlow>(),
            ImageWrapper<ImageType::MotionMask>(
                intermediate_tracking_images.get<ImageType::SemanticMask>()
            )
        );
    }
    else {
        tracking_images = image_container->makeSubset<ImageType::RGBMono, ImageType::OpticalFlow, ImageType::MotionMask>();
    }


    Frame::Ptr frame =  tracker_->track(input->getFrameId(), input->getTimestamp(), *image_container);
    if(gt_packet_map_.find(frame->getFrameId()) != gt_packet_map_.end()) {
        frame->T_world_camera_ = gt_packet_map_.at(frame->getFrameId()).X_world_;
    }
    CHECK(frame->updateDepths(base_params_.depth_background_thresh, base_params_.depth_obj_thresh));

    return {State::Nominal, nullptr};
}


FrontendModule::SpinReturn
RGBDInstanceFrontendModule::nominalSpin(FrontendInputPacketBase::ConstPtr input) {
    ImageContainer::Ptr image_container = input->image_container_;
    //if we only have instance semgentation (not motion) then we need to make a motion mask out of the semantic mask
    //we cannot do this for the first frame so we will just treat the semantic mask and the motion mask
    //and then subsequently elimate non-moving objects later on
    TrackingInputImages tracking_images;
    if(image_container->hasSemanticMask()) {
        CHECK(!image_container->hasMotionMask());

        auto intermediate_tracking_images = image_container->makeSubset<ImageType::RGBMono, ImageType::OpticalFlow, ImageType::SemanticMask>();
        tracking_images = TrackingInputImages(
            intermediate_tracking_images.getImageWrapper<ImageType::RGBMono>(),
            intermediate_tracking_images.getImageWrapper<ImageType::OpticalFlow>(),
            ImageWrapper<ImageType::MotionMask>(
                intermediate_tracking_images.get<ImageType::SemanticMask>()
            )
        );
    }
    else {
        tracking_images = image_container->makeSubset<ImageType::RGBMono, ImageType::OpticalFlow, ImageType::MotionMask>();
    }


    Frame::Ptr frame = nullptr;
    {
        utils::TimingStatsCollector tracking_timer("tracking_timer");
        frame =  tracker_->track(input->getFrameId(), input->getTimestamp(), *image_container);

    }
    CHECK(frame);

    Frame::Ptr previous_frame = tracker_->getPreviousFrame();
    CHECK(previous_frame);

    LOG(INFO) << to_string(tracker_->getTrackerInfo());

    {
        utils::TimingStatsCollector update_depths_timer("depth_updater");
        frame->updateDepths(base_params_.depth_background_thresh, base_params_.depth_obj_thresh);

    }
    //updates frame->T_world_camera_
    if(!solveCameraMotion(frame, previous_frame)) {
        LOG(ERROR) << "Could not solve for camera";
    }

    // if(gt_packet_map_.exists(frame->getFrameId())) frame->T_world_camera_ = gt_packet_map_.at(frame->getFrameId()).X_world_;

    if(FLAGS_use_dynamic_track) {
        //TODO: bring back byte tracker??
        utils::TimingStatsCollector track_dynamic_timer("tracking_dynamic");
        vision_tools::trackDynamic(base_params_,*previous_frame, frame);
    }


    MotionEstimateMap motion_estimates;
    solveObjectMotions(frame, previous_frame, motion_estimates);

    //update the object_poses trajectory map which will be send to the viz
    propogateObjectPoses(motion_estimates, frame->getFrameId());

    if(logger_) {
        //TODO: hack to set ground truths as empty if non provided to ensure that values still log with no gt
        std::optional<GroundTruthPacketMap> ground_truths;
        if(gt_packet_map_.size() > 0) ground_truths = gt_packet_map_;

        logger_->logCameraPose(frame->getFrameId(), frame->getPose(), ground_truths);
        logger_->logObjectMotion(frame->getFrameId(), motion_estimates, ground_truths);
        logger_->logTrackingLengthHistogram(frame);
        logger_->logFrameIdToTimestamp(frame->getFrameId(), frame->getTimestamp());
    }

    DebugImagery debug_imagery;
    debug_imagery.tracking_image = tracker_->computeImageTracks(*previous_frame, *frame);
    if(display_queue_) display_queue_->push(ImageToDisplay("tracks", debug_imagery.tracking_image));

    debug_imagery.detected_bounding_boxes = frame->drawDetectedObjectBoxes();
    //use the tracking images from the frame NOT the input tracking images since the feature tracking
    //will do some modifications on the images (particularily the mask during mask propogation)
    // debug_imagery.input_images = frame->tracking_images_;
    debug_imagery.input_images = tracking_images;



    RGBDInstanceOutputPacket::Ptr output = constructOutput(*frame, motion_estimates, frame->T_world_camera_, input->optional_gt_, debug_imagery);

    if(FLAGS_save_frontend_json) output_packet_record_.insert({output->getFrameId(), output});

    if(FLAGS_log_projected_masks) vision_tools::writeOutProjectMaskAndDepthMap(
        frame->image_container_.get<ImageType::Depth>(),
        frame->image_container_.get<ImageType::MotionMask>(),
        *frame->getCamera(),
        frame->getFrameId());

    if(logger_) {
        //TODO: hack to set ground truths as empty if non provided to ensure that values still log with no gt
        std::optional<GroundTruthPacketMap> ground_truths;
        if(gt_packet_map_.size() > 0) ground_truths = gt_packet_map_;

        logger_->logPoints(output->getFrameId(), output->T_world_camera_, output->dynamic_landmarks_);
        //object_poses_ are in frontend module
        logger_->logObjectPose(output->getFrameId(), object_poses_, ground_truths);
        logger_->logObjectBbxes(output->getFrameId(), output->getObjectBbxes());
    }
    return {State::Nominal, output};
}


bool RGBDInstanceFrontendModule::solveCameraMotion(Frame::Ptr frame_k, const Frame::Ptr& frame_k_1) {

    Pose3SolverResult result;
    if(base_params_.use_ego_motion_pnp) {
        result = motion_solver_.geometricOutlierRejection3d2d(frame_k_1, frame_k);
    }
    else {
        //TODO: untested
        LOG(FATAL) << "Not tested";
        result = motion_solver_.geometricOutlierRejection3d3d(frame_k_1, frame_k);
    }

    VLOG(15) << (base_params_.use_ego_motion_pnp ? "3D2D" : "3D3D") << "camera pose estimate at frame " << frame_k->frame_id_
          << (result.status == TrackingStatus::VALID ? " success " : " failure ") << ":\n"
          << "- Tracking Status: "
          << to_string(result.status) << '\n'
          << "- Total Correspondences: " << result.inliers.size() + result.outliers.size() << '\n'
          << "\t- # inliers: " << result.inliers.size() << '\n'
          << "\t- # outliers: " << result.outliers.size() << '\n';

    if(result.status == TrackingStatus::VALID) {

        frame_k->T_world_camera_ = result.best_result;
        TrackletIds tracklets = frame_k->static_features_.collectTracklets();
        CHECK_GE(tracklets.size(), result.inliers.size() + result.outliers.size()); //tracklets shoudl be more (or same as) correspondances as there will be new points untracked
        frame_k->static_features_.markOutliers(result.outliers);

        if(FLAGS_refine_with_optical_flow) {
            OpticalFlowAndPoseOptimizer flow_optimizer(OpticalFlowAndPoseOptimizerParams{});
            auto flow_opt_result = flow_optimizer.optimizeAndUpdate<CalibrationType>(
                frame_k_1,
                frame_k,
                result.inliers,
                result.best_result
            );
            frame_k->T_world_camera_ = flow_opt_result.best_result.refined_pose;

        }
        return true;
    }
    else {
        frame_k->T_world_camera_ = gtsam::Pose3::Identity();
        return false;
    }

}

bool RGBDInstanceFrontendModule::solveObjectMotion(Frame::Ptr frame_k, Frame::Ptr frame_k_1,  ObjectId object_id, MotionEstimateMap& motion_estimates) {

    Pose3SolverResult result;
    if(base_params_.use_object_motion_pnp) {
        result = object_motion_solver_.geometricOutlierRejection3d2d(frame_k_1, frame_k,  frame_k->T_world_camera_, object_id);
    }
    else {
        result = object_motion_solver_.geometricOutlierRejection3d3d(frame_k_1, frame_k,  frame_k->T_world_camera_, object_id);
    }

    VLOG(15) << (base_params_.use_object_motion_pnp ? "3D2D" : "3D3D") << " object motion estimate " << object_id << " at frame " << frame_k->frame_id_
          << (result.status == TrackingStatus::VALID ? " success " : " failure ") << ":\n"
          << "- Tracking Status: "
          << to_string(result.status) << '\n'
          << "- Total Correspondences: " << result.inliers.size() + result.outliers.size() << '\n'
          << "\t- # inliers: " << result.inliers.size() << '\n'
          << "\t- # outliers: " << result.outliers.size() << '\n';

    //sanity check
    //if valid, remove outliers and add to motion estimation
    if(result.status == TrackingStatus::VALID) {
        frame_k->dynamic_features_.markOutliers(result.outliers);
        ReferenceFrameValue<Motion3> estimate(result.best_result, ReferenceFrame::GLOBAL);
        motion_estimates.insert({object_id, estimate});
        return true;
    }
    else {
        return false;
    }
}

void RGBDInstanceFrontendModule::solveObjectMotions(Frame::Ptr frame_k, Frame::Ptr frame_k_1, MotionEstimateMap& motion_estimates) {
    ObjectIds failed_object_tracks;

    //if only 1 object, no point parallelising
    if(motion_estimates.size() <= 1) {
        for(const auto& [object_id, observations] : frame_k->object_observations_) {

            if(!solveObjectMotion(frame_k, frame_k_1, object_id, motion_estimates)) {
                VLOG(5) << "Could not solve motion for object " << object_id <<
                    " from frame " << frame_k_1->getFrameId() << " -> " << frame_k->getFrameId();
                failed_object_tracks.push_back(object_id);
            }
        }
    }
    else {
        std::mutex mutex;
        //paralleilise the process of each function call.
        tbb::parallel_for_each(frame_k->object_observations_.begin(), frame_k->object_observations_.end(),
            [&](const std::pair<ObjectId, DynamicObjectObservation>& pair) {
                const auto object_id = pair.first;
                if(!solveObjectMotion(frame_k, frame_k_1, object_id, motion_estimates)) {
                    VLOG(5) << "Could not solve motion for object " << object_id <<
                        " from frame " << frame_k_1->getFrameId() << " -> " << frame_k->getFrameId();

                    std::lock_guard<std::mutex> lk(mutex);
                    failed_object_tracks.push_back(object_id);
                }
            }
        );
    }

    ///remove objects from the object observations list
    //does not remove the features etc but stops the object being propogated to the backend
    //as we loop over the object observations in the constructOutput function
    for(auto object_id : failed_object_tracks) {
        frame_k->object_observations_.erase(object_id);
    }

}



RGBDInstanceOutputPacket::Ptr RGBDInstanceFrontendModule::constructOutput(
    const Frame& frame,
    const MotionEstimateMap& estimated_motions,
    const gtsam::Pose3& T_world_camera,
    const GroundTruthInputPacket::Optional& gt_packet,
    const DebugImagery::Optional& debug_imagery)
{
    StatusKeypointMeasurements static_keypoint_measurements;
    StatusLandmarkEstimates static_landmarks;
    for(const Feature::Ptr& f : frame.usableStaticFeaturesBegin()) {
        const TrackletId tracklet_id = f->trackletId();
        const Keypoint kp = f->keypoint();
        Landmark lmk_camera;
        camera_->backProject(kp, f->depth(), &lmk_camera);
        CHECK(f->isStatic());
        CHECK(Feature::IsUsable(f));

        //dont include features that have only been seen once as we havent had a chance to validate it yet
        if(f->age() < 1) {
            continue;
        }


        static_keypoint_measurements.push_back(
            KeypointStatus::Static(
                kp,
                frame.getFrameId(),
                tracklet_id
            )
        );

        static_landmarks.push_back(
            LandmarkStatus::StaticInLocal(
                lmk_camera,
                frame.getFrameId(),
                tracklet_id,
                LandmarkStatus::Method::MEASURED
            )
        );
    }


    StatusKeypointMeasurements dynamic_keypoint_measurements;
    StatusLandmarkEstimates dynamic_landmarks;
    for(const auto& [object_id, obs] : frame.object_observations_) {
        CHECK_EQ(object_id, obs.instance_label_);
        //TODO: add back in?
        // CHECK(obs.marked_as_moving_);

        for(const TrackletId tracklet : obs.object_features_) {
            if(frame.isFeatureUsable(tracklet)) {
                const Feature::Ptr f = frame.at(tracklet);
                CHECK(!f->isStatic());
                CHECK_EQ(f->objectId(), object_id);

                //dont include features that have only been seen once as we havent had a chance to validate it yet
                if(f->age() < 1) {
                    continue;
                }

                const TrackletId tracklet_id = f->trackletId();
                const Keypoint kp = f->keypoint();
                Landmark lmk_camera;
                camera_->backProject(kp, f->depth(), &lmk_camera);

                dynamic_keypoint_measurements.push_back(
                    KeypointStatus::Dynamic(
                        kp,
                        frame.frame_id_,
                        tracklet_id,
                        object_id
                    )
                );

                dynamic_landmarks.push_back(
                    LandmarkStatus::DynamicInLocal(
                        lmk_camera,
                        frame.frame_id_,
                        tracklet_id,
                        object_id,
                        LandmarkStatus::Method::MEASURED
                    )
                );
            }
        }

    }

    //update trajectory of camera poses to be visualised by the frontend viz module
    camera_poses_.push_back(T_world_camera);

    return std::make_shared<RGBDInstanceOutputPacket>(
        static_keypoint_measurements,
        dynamic_keypoint_measurements,
        static_landmarks,
        dynamic_landmarks,
        T_world_camera,
        frame.timestamp_,
        frame.frame_id_,
        estimated_motions,
        object_poses_,
        camera_poses_,
        camera_,
        gt_packet,
        debug_imagery
    );
}


// void RGBDInstanceFrontendModule::propogateObjectPoses(const MotionEstimateMap& motion_estimates, FrameId frame_id) {
//     for(const auto& [object_id, motion_estimate] : motion_estimates) {
//         propogateObjectPose(motion_estimate, object_id, frame_id);
//     }
// }

// void RGBDInstanceFrontendModule::propogateObjectPose(const gtsam::Pose3& prev_H_world_curr, ObjectId object_id, FrameId frame_id) {
//     //start from previous frame -> if we have a motion we must have observations in k-1
//     const FrameId frame_id_k_1 = frame_id - 1;
//     if(!object_poses_.exists(object_id)) {

//         gtsam::Pose3 starting_pose;

//         //need gt pose for rotation even if not for translation
//         CHECK(gt_packet_map_.exists(frame_id_k_1)) << "Cannot initalise object poses for viz using gt as the ground truth does not exist for frame " << frame_id_k_1;
//         const GroundTruthInputPacket& gt_packet_k_1 = gt_packet_map_.at(frame_id_k_1);

//         ObjectPoseGT object_pose_gt_k;
//         if(!gt_packet_k_1.getObject(object_id, object_pose_gt_k)) {
//             LOG(ERROR) << "Object Id " << object_id <<  " cannot be found in the gt packet. Unable to initalise object starting point use gt pose. Skipping!";
//             return;
//             // throw std::runtime_error("Object Id " + std::to_string(object_id) + " cannot be found in the gt packet. Unable to initalise object starting point use gt pose");
//         }

//         starting_pose = object_pose_gt_k.L_world_;

//         //if not use gt translation, find centroid of object
//         if(!FLAGS_init_object_pose_from_gt) {
//             const auto frame_k_1 = tracker_->getPreviousFrame();
//             const auto frame_k = tracker_->getCurrentFrame();

//             auto object_points = FeatureFilterIterator(
//                 const_cast<FeatureContainer&>(frame_k_1->dynamic_features_),
//                 [object_id, &frame_k](const Feature::Ptr& f) -> bool {
//                     return Feature::IsUsable(f) && f->instance_label_ == object_id &&
//                     frame_k->exists(f->tracklet_id_) && frame_k->isFeatureUsable(f->tracklet_id_);
//                 });

//             gtsam::Point3 centroid(0, 0, 0);
//             size_t count = 0;
//             for(const auto& feature : object_points) {
//                 gtsam::Point3 lmk = frame_k_1->backProjectToWorld(feature->tracklet_id_);
//                 centroid += lmk;
//                 count++;
//             }

//             centroid /= count;
//             //update starting pose
//             starting_pose = gtsam::Pose3(object_pose_gt_k.L_world_.rotation(), centroid);


//         }
//         //new object
//         object_poses_.insert2(object_id , gtsam::FastMap<FrameId, gtsam::Pose3>{});
//         object_poses_.at(object_id).insert2(frame_id_k_1, starting_pose);
//     }

//     auto& pose_map = object_poses_.at(object_id);
//     if(!pose_map.exists(frame_id_k_1)) {
//         //this can happen if the object misses a frame (e.g becomes too small - this is problematic while we do not properly propogate out own tracking labels!!)
//         LOG(ERROR)<< "Previous frame " << frame_id_k_1 << " does not exist for object " << object_id << ". They should be in order?";
//         return;
//     }

//     const gtsam::Pose3& object_pose_k_1 = pose_map.at(frame_id_k_1);
//     gtsam::Pose3 object_pose_k = prev_H_world_curr * object_pose_k_1;
//     pose_map.insert2(frame_id, object_pose_k);
// }

void RGBDInstanceFrontendModule::propogateObjectPoses(const MotionEstimateMap& motion_estimates, FrameId frame_id) {

    gtsam::Point3Vector object_centroids_k_1, object_centroids_k;

    for(const auto& [object_id, motion_estimate] : motion_estimates) {
        const auto frame_k_1 = tracker_->getPreviousFrame();
        const auto frame_k = tracker_->getCurrentFrame();

        auto object_points = FeatureFilterIterator(
            const_cast<FeatureContainer&>(frame_k_1->dynamic_features_),
            [object_id, &frame_k](const Feature::Ptr& f) -> bool {
                return Feature::IsUsable(f) && f->objectId() == object_id &&
                    frame_k->exists(f->trackletId()) && frame_k->isFeatureUsable(f->trackletId());
            });

        gtsam::Point3 centroid_k_1(0, 0, 0);
        gtsam::Point3 centroid_k(0, 0, 0);
        size_t count = 0;
        for(const auto& feature : object_points) {
            gtsam::Point3 lmk_k_1 = frame_k_1->backProjectToCamera(feature->trackletId());
            centroid_k_1 += lmk_k_1;

            gtsam::Point3 lmk_k = frame_k->backProjectToCamera(feature->trackletId());
            centroid_k += lmk_k;

            count++;
        }

        centroid_k_1 /= count;
        centroid_k /= count;

        centroid_k_1 = frame_k_1->getPose() * centroid_k_1;
        centroid_k = frame_k->getPose() * centroid_k;


        object_centroids_k_1.push_back(centroid_k_1);
        object_centroids_k.push_back(centroid_k);
    }

    if(FLAGS_init_object_pose_from_gt) {
        dyno::propogateObjectPoses(
            object_poses_,
            motion_estimates,
            object_centroids_k_1,
            object_centroids_k,
            frame_id,
            gt_packet_map_);
    }
    else {
        dyno::propogateObjectPoses(
            object_poses_,
            motion_estimates,
            object_centroids_k_1,
            object_centroids_k,
            frame_id);
    }

}


} //dyno
