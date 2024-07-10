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
#pragma once

#include "dynosam/common/MapNodes.hpp"
#include "dynosam/common/Map.hpp"
#include "dynosam/backend/BackendDefinitions.hpp"



namespace dyno {


template<typename MEASUREMENT>
int FrameNode<MEASUREMENT>::getId() const {
    return (int)frame_id;
}

template<typename MEASUREMENT>
bool FrameNode<MEASUREMENT>::objectObserved(ObjectId object_id) const {
    return objects_seen.exists(object_id);
}

template<typename MEASUREMENT>
bool FrameNode<MEASUREMENT>::objectObservedInPrevious(ObjectId object_id) const {
    const auto frame_id_k_1 = frame_id - 1u;
    FrameNodePtr<MEASUREMENT> frame_node_k_1 = this->map_ptr_->template getFrame(frame_id_k_1);

    if(!frame_node_k_1) { return false;}
    return frame_node_k_1->objectObserved(object_id);
}

template<typename MEASUREMENT>
gtsam::Key FrameNode<MEASUREMENT>::makePoseKey() const {
    return CameraPoseSymbol(frame_id);
}


template<typename MEASUREMENT>
gtsam::Key FrameNode<MEASUREMENT>::makeObjectMotionKey(ObjectId object_id) const {
    //TODO: no point checking if this key exists yet as we might want it arbiratarily!!
    // if(!objectObserved(object_id)) {
    //     throw DynosamException("Object motion key requested" + std::to_string(object_id) + " at frame " +  std::to_string(frame_id) + " but object is not observed in this frame");
    // }
    return ObjectMotionSymbol(object_id, this->frame_id);
}

template<typename MEASUREMENT>
gtsam::Key FrameNode<MEASUREMENT>::makeObjectPoseKey(ObjectId object_id) const {
    //  if(!objectObserved(object_id)) {
    //     throw DynosamException("Object pose key requested" + std::to_string(object_id) + " at frame " +  std::to_string(frame_id) + " but object is not observed in this frame");
    // }
    return ObjectPoseSymbol(object_id, this->frame_id);
}


template<typename MEASUREMENT>
bool FrameNode<MEASUREMENT>::objectMotionExpected(ObjectId object_id) const {
    return objectObserved(object_id) && objectObservedInPrevious(object_id);
}

// template<typename MEASUREMENT>
// StateQuery<gtsam::Pose3> FrameNode<MEASUREMENT>::getPoseEstimate() const {
//     return this->map_ptr_->template query<gtsam::Pose3>(makePoseKey());
// }

// template<typename MEASUREMENT>
// StateQuery<gtsam::Pose3> FrameNode<MEASUREMENT>::getObjectPoseEstimate(ObjectId object_id) const {
//     return this->map_ptr_->template query<gtsam::Pose3>(makeObjectPoseKey(object_id));
// }

// template<typename MEASUREMENT>
// StateQuery<gtsam::Pose3> FrameNode<MEASUREMENT>::getObjectMotionEstimate(ObjectId object_id) const {
//     return this->map_ptr_->template query<gtsam::Pose3>(makeObjectMotionKey(object_id));
// }

// template<typename MEASUREMENT>
// StateQuery<Landmark> FrameNode<MEASUREMENT>::getDynamicLandmarkEstimate(TrackletId tracklet_id) const {
//     auto iter = dynamic_landmarks.find(tracklet_id);
//     checkAndThrow<MissingLandmarkException>(iter != dynamic_landmarks.end(), tracklet_id, frame_id, false);
//     const LandmarkNodePtr<MEASUREMENT> lmk_node = *iter;
//     return lmk_node->getDynamicLandmarkEstimate(frame_id);
// }

// template<typename MEASUREMENT>
// StatusLandmarkEstimates FrameNode<MEASUREMENT>::getAllDynamicLandmarkEstimates() const {
//     //quicker to go through the dynamic_landmarks than the objects as we already know they exist
//     //in this frame
//     StatusLandmarkEstimates estimates;
//     for(const auto& lmk_ptr : dynamic_landmarks) {
//         lmk_ptr->appendDynamicLandmarkEstimate(estimates, this->frame_id);
//     }

//     return estimates;
// }

// template<typename MEASUREMENT>
// StatusLandmarkEstimates FrameNode<MEASUREMENT>::getDynamicLandmarkEstimates(ObjectId object_id) const {
//     CHECK(objectObserved(object_id)); //TODO: handle exception?

//     //not sure what will be faster? going through all landmarks and finding object id
//     //or going to this object and iterating over lmks there.
//     //Doing the former requires iterating over all the seen lmks (which should be in this frame)
//     //and then just checking if the lmk has the right object
//     //Doing the latter requires a call to getSeenFrames() and THEN finding the right frame in that set. This means
//     //iterating over all the lmks for that dynamic object...
//     StatusLandmarkEstimates estimates;
//     for(const auto& lmk_ptr : dynamic_landmarks) {
//         if(lmk_ptr->getObjectId() == object_id) {
//             lmk_ptr->appendDynamicLandmarkEstimate(estimates, this->frame_id);
//         }
//     }
//     return estimates;
// }

// template<typename MEASUREMENT>
// StateQuery<Landmark> FrameNode<MEASUREMENT>::getStaticLandmarkEstimate(TrackletId tracklet_id) const {
//     auto iter = static_landmarks.find(tracklet_id);
//     checkAndThrow<MissingLandmarkException>(iter != static_landmarks.end(), tracklet_id, frame_id, true);
//     const LandmarkNodePtr<MEASUREMENT> lmk_node = *iter;
//     return lmk_node->getStaticLandmarkEstimate();
// }


// template<typename MEASUREMENT>
// StatusLandmarkEstimates FrameNode<MEASUREMENT>::getAllStaticLandmarkEstimates() const {
//     StatusLandmarkEstimates estimates;
//     for(const auto& lmk_ptr : static_landmarks) {
//         lmk_ptr->appendStaticLandmarkEstimate(estimates);
//     }
//     return estimates;
// }

template<typename MEASUREMENT>
std::vector<typename FrameNode<MEASUREMENT>::LandmarkMeasurementPair> FrameNode<MEASUREMENT>::getStaticMeasurements() const {
    std::vector<LandmarkMeasurementPair> measurements;
    for(const auto& lmk_ptr : static_landmarks) {
        MEASUREMENT m = lmk_ptr->getMeasurement(this->frame_id);
        measurements.push_back(std::make_pair(lmk_ptr, m));
    }
    return measurements;
}

template<typename MEASUREMENT>
std::vector<typename FrameNode<MEASUREMENT>::LandmarkMeasurementPair> FrameNode<MEASUREMENT>::getDynamicMeasurements() const {
    std::vector<LandmarkMeasurementPair> measurements;
    for(const auto& lmk_ptr : dynamic_landmarks) {
        const MEASUREMENT m = lmk_ptr->getMeasurement(this->frame_id);
        measurements.push_back(std::make_pair(lmk_ptr, m));
    }
    return measurements;

}

template<typename MEASUREMENT>
std::vector<typename FrameNode<MEASUREMENT>::LandmarkMeasurementPair> FrameNode<MEASUREMENT>::getDynamicMeasurements(ObjectId object_id) const {
    std::vector<LandmarkMeasurementPair> measurements;
    for(const auto& lmk_ptr : dynamic_landmarks) {
        if(lmk_ptr->getObjectId() == object_id) {
            const MEASUREMENT m = lmk_ptr->getMeasurement(this->frame_id);
            measurements.push_back(std::make_pair(lmk_ptr, m));
        }

    }
    return measurements;
}

// template<typename MEASUREMENT>
// StatusLandmarkEstimates FrameNode<MEASUREMENT>::getAllLandmarkEstimates() const {
//     StatusLandmarkEstimates static_landmarks = getAllStaticLandmarkEstimates();
//     StatusLandmarkEstimates dynamic_landmarks = getAllDynamicLandmarkEstimates();

//     //add all dynamic lmks to the static ones
//     static_landmarks.insert(static_landmarks.end(), dynamic_landmarks.begin(), dynamic_landmarks.end());
//     return static_landmarks;

// }


/// LandmarkNode
template<typename MEASUREMENT>
int LandmarkNode<MEASUREMENT>::getId() const {
    return (int)tracklet_id;
}

template<typename MEASUREMENT>
ObjectId LandmarkNode<MEASUREMENT>::getObjectId() const {
    return object_id;
}

template<typename MEASUREMENT>
bool LandmarkNode<MEASUREMENT>::isStatic() const {
    return object_id == background_label;
}

template<typename MEASUREMENT>
size_t LandmarkNode<MEASUREMENT>::numObservations() const {
    return frames_seen_.size();
}

template<typename MEASUREMENT>
void LandmarkNode<MEASUREMENT>::add(FrameNodePtr<MEASUREMENT> frame_node, const MEASUREMENT& measurement) {
    frames_seen_.insert(frame_node);

    //add measurement to map
    //first check that we dont already have a measurement at this frame
    if(measurements_.exists(frame_node)) {
        throw DynosamException("Unable to add new measurement to landmark node " + std::to_string(this->getId()) + " at frame " + std::to_string(frame_node->getId()) + " as a measurement already exists at this frame!");
    }

    measurements_.insert2(frame_node, measurement);

    CHECK_EQ(frames_seen_.size(), measurements_.size());
}


// template<typename MEASUREMENT>
// StateQuery<Landmark> LandmarkNode<MEASUREMENT>::getStaticLandmarkEstimate() const {
//     const gtsam::Key lmk_key = makeStaticKey();
//     return this->map_ptr_->template query<Landmark>(lmk_key);
// }

// template<typename MEASUREMENT>
// StateQuery<Landmark> LandmarkNode<MEASUREMENT>::getDynamicLandmarkEstimate(FrameId frame_id) const {
//     auto iter = frames_seen_.find(frame_id);
//     CHECK(iter != frames_seen_.end()) << "Requested dynamic landmark with id " << this->tracklet_id << " does not exist at frame " << frame_id;
//     const gtsam::Key lmk_key = makeDynamicKey(frame_id);
//     return this->map_ptr_->template query<Landmark>(lmk_key);
// }

template<typename MEASUREMENT>
bool LandmarkNode<MEASUREMENT>::seenAtFrame(FrameId frame_id) const {
    return frames_seen_.exists(frame_id);
}

template<typename MEASUREMENT>
bool LandmarkNode<MEASUREMENT>::hasMeasurement(FrameId frame_id) const {
    return this->seenAtFrame(frame_id);
}

template<typename MEASUREMENT>
const MEASUREMENT& LandmarkNode<MEASUREMENT>::getMeasurement(FrameNodePtr<MEASUREMENT> frame_node) const {
    CHECK_NOTNULL(frame_node);
    if(!hasMeasurement(frame_node->frame_id)) {
        throw DynosamException("Missing measurement in landmark node with id " + std::to_string(tracklet_id) + " at frame " +  std::to_string(frame_node->frame_id));
    }
    return measurements_.at(frame_node);
}

template<typename MEASUREMENT>
const MEASUREMENT& LandmarkNode<MEASUREMENT>::getMeasurement(FrameId frame_id) const {
    if(!seenAtFrame(frame_id)) {
        throw DynosamException("Missing measurement in landmark node with id " + std::to_string(tracklet_id) + " at frame " +  std::to_string(frame_id));
    }
    return getMeasurement(this->map_ptr_->template getFrame(frame_id));
}

template<typename MEASUREMENT>
gtsam::Key LandmarkNode<MEASUREMENT>::makeStaticKey() const {
    const auto key = StaticLandmarkSymbol(this->tracklet_id);
    // if(!this->isStatic()) {
    //     throw InvalidLandmarkQuery(key, "Static estimate requested but landmark is dynamic!");
    // }
    return key;
}


template<typename MEASUREMENT>
gtsam::Key LandmarkNode<MEASUREMENT>::makeDynamicKey(FrameId frame_id) const {
    return (gtsam::Key)makeDynamicSymbol(frame_id);

}

template<typename MEASUREMENT>
DynamicPointSymbol LandmarkNode<MEASUREMENT>::makeDynamicSymbol(FrameId frame_id) const {
    const auto key = DynamicLandmarkSymbol(frame_id, this->tracklet_id);
    // if(this->isStatic()) {
    //     throw InvalidLandmarkQuery(key, "Dynamic estimate requested but landmark is static!");
    // }
    return key;

}

// //TODO: coded in estiamtes are in global frame!!
// template<typename MEASUREMENT>
// bool LandmarkNode<MEASUREMENT>::appendStaticLandmarkEstimate(StatusLandmarkEstimates& estimates) const {
//     StateQuery<Landmark> lmk_status_query = this->getStaticLandmarkEstimate();
//     if(lmk_status_query) {
//         estimates.push_back(
//             LandmarkStatus::StaticInGlobal(
//                 lmk_status_query.get(), //estimate
//                 LandmarkStatus::MeaninglessFrame,
//                 this->getId(), //tracklet id
//                 LandmarkStatus::Method::OPTIMIZED
//             ) //status
//         );
//         return true;
//     }
//     return false;
// }

// //TODO: coded in estiamtes are in global frame!!
// template<typename MEASUREMENT>
// bool LandmarkNode<MEASUREMENT>::appendDynamicLandmarkEstimate(StatusLandmarkEstimates& estimates, FrameId frame_id) const {
//     StateQuery<Landmark> lmk_status_query = this->getDynamicLandmarkEstimate(frame_id);
//     if(lmk_status_query) {
//         estimates.push_back(
//             LandmarkStatus::DynamicInGLobal(
//                 lmk_status_query.get(), //estimate
//                 frame_id,
//                 this->getId(), //tracklet id
//                 this->getObjectId(),
//                 LandmarkStatus::Method::OPTIMIZED
//             ) //status
//         );
//         return true;
//     }
//     return false;
// }

/// ObjectNode
template<typename MEASUREMENT>
int ObjectNode<MEASUREMENT>::getId() const {
    return (int)object_id;
}

// template<typename MEASUREMENT>
// StateQuery<gtsam::Pose3> ObjectNode<MEASUREMENT>::getMotionEstimate(FrameId frame_id) const {
//     const auto seen_frames = this->getSeenFrames();

//     auto frame_itr = seen_frames.find(frame_id);
//     if(frame_itr == seen_frames.end()) {
//         throw DynosamException("Motion estimate query failed: object " + std::to_string(getId()) + " was not seen at frame " + std::to_string(frame_id));
//     }

//     const auto& frame_ptr = *frame_itr;
//     return frame_ptr->getObjectMotionEstimate(this->getId());
// }

// template<typename MEASUREMENT>
// bool ObjectNode<MEASUREMENT>::hasMotionEstimate(FrameId frame_id, gtsam::Pose3* motion) const {
//     StateQuery<gtsam::Pose3> query = this->getMotionEstimate(frame_id);
//     if(query) {
//         //only set motion if valid, return true anyway
//         if(motion) *motion = *query;
//         return true;
//     }
//     return false;
// }

// template<typename MEASUREMENT>
// bool ObjectNode<MEASUREMENT>::hasMotionEstimate(FrameId frame_id, gtsam::Pose3& motion) const {
//     return this->hasMotionEstimate(frame_id, &motion);
// }

// template<typename MEASUREMENT>
// StateQuery<gtsam::Pose3> ObjectNode<MEASUREMENT>::getPoseEstimate(FrameId frame_id) const {
//     //basically the same code as in getMotionEstimate
//     const auto seen_frames = this->getSeenFrames();

//     auto frame_itr = seen_frames.find(frame_id);
//     if(frame_itr == seen_frames.end()) {
//         throw DynosamException("Pose estimate query failed: object " + std::to_string(getId()) + " was not seen at frame " + std::to_string(frame_id));
//     }

//     const auto& frame_ptr = *frame_itr;
//     return frame_ptr->getObjectPoseEstimate(this->getId());
// }

// template<typename MEASUREMENT>
// bool ObjectNode<MEASUREMENT>::hasPoseEstimate(FrameId frame_id, gtsam::Pose3* pose) const {
//     try {
//         //this throws exception if query pose does not exist for the requested frame_id
//         StateQuery<gtsam::Pose3> query = this->getPoseEstimate(frame_id);
//         if(query) {
//             //only set pose if valid, return true anyway
//             if(pose) *pose = *query;
//             return true;
//         }
//         return false;
//     }
//     catch (DynosamException&) {
//         return false;
//     }
// }

// template<typename MEASUREMENT>
// bool ObjectNode<MEASUREMENT>::hasPoseEstimate(FrameId frame_id, gtsam::Pose3& pose) const{
//     return this->hasPoseEstimate(frame_id, &pose);
// }

template<typename MEASUREMENT>
FrameNodePtrSet<MEASUREMENT> ObjectNode<MEASUREMENT>::getSeenFrames() const {
    FrameNodePtrSet<MEASUREMENT> seen_frames;
    for(const auto& lmks : dynamic_landmarks) {
        seen_frames.merge(lmks->getSeenFrames());
    }
    return seen_frames;
}

// template<typename MEASUREMENT>
// gtsam::FastMap<FrameId, gtsam::Pose3> ObjectNode<MEASUREMENT>::computeComposedPoseMap(const GroundTruthPacketMap::Optional& gt_packet_map) const {
//     //similar logic to how we compute the poses from the gt in the frontend
//     //if init_translation_from_gt -> take the translation part from gt else compute the centroid of the point cloud at the first seen frame
//     //always take rotation from gt currently

//     //first seen frame (ie t = 0, for this object)
//     const FrameId frame_k_0 = this->getFirstSeenFrame();
//     const ObjectId obejct_id = this->getId();

//     const auto& seen_frames = this->getSeenFrames();

//     auto frame_itr = seen_frames.begin();
//     //advance itr one so we're now at the second frame
//     std::advance(frame_itr, 1);

//     for(auto itr = frame_itr; itr != seen_frames.end(); itr++) {
//         auto frame_node_ptr_k = *itr;
//         const FrameId frame_id_k = frame_node_ptr_k->getId();

//         auto prev_itr = itr;
//         std::advance(prev_itr, -1);
//         CHECK(prev_itr != seen_frames.end());

//         auto frame_node_ptr_k_1 = *prev_itr;
//         const FrameId frame_id_k_1 = frame_node_ptr_k_1->getID();
//         CHECK_EQ(frame_id_k_1 + 1, frame_id_k);


//         const auto [centroid_k, result_k] = frame_node_ptr_k->computeObjectCentroid(object_id);
//         const auto [centroid_k_1, result_k_1] = frame_node_ptr_k_1->computeObjectCentroid(object_id);

//         CHECK(result_k);
//         CHECK(result_k_1);

//     }


// }


// template<typename MEASUREMENT>
// gtsam::FastMap<FrameId, gtsam::Pose3> ObjectNode<MEASUREMENT>::computeComposedPoseMap(const GroundTruthPacketMap::Optional& gt_packet_map) const {
//     //similar logic to how we compute the poses from the gt in the frontend
//     //if init_translation_from_gt -> take the translation part from gt else compute the centroid of the point cloud at the first seen frame
//     //always take rotation from gt currently

//     //first seen frame (ie t = 0, for this object)
//     const FrameId frame_k_0 = this->getFirstSeenFrame();
//     const ObjectId obejct_id = this->getId();

//     const auto& seen_frames = this->getSeenFrames();

//     //need gt pose for rotation even if not for translation
//     CHECK(gt_packet_map.exists(frame_k_0)) << "Cannot initalise object poses for viz using gt as the ground truth does not exist for frame " << frame_k_0;
//     const GroundTruthInputPacket& gt_packet_k_0 = gt_packet_map.at(frame_k_0);

//     ObjectPoseGT object_pose_gt_0;
//     if(!gt_packet_k_0.getObject(object_id, object_pose_gt_0)) {
//         LOG(ERROR) << "Object Id " << object_id <<  " cannot be found in the gt packet. Unable to initalise object starting point use gt pose. Skipping!";
//         return gtsam::FastMap<FrameId, gtsam::Pose3>{};
//     }

//     gtsam::Pose3 starting_pose = object_pose_gt_0.L_world_;

//     auto init_translation_from_centroid = [=, &seen_frames](const ObjectPoseGT& object_pose_gt, FrameId frame_id, ObjectId object_id, gtsam::Pose3& pose) -> bool {
//         CHECK_EQ(object_pose_gt.frame_id_, frame_id);
//         CHECK_EQ(object_pose_gt.object_id_, object_id);

//         //get estimates from the map using the latest values
//         auto frame_itr = seen_frames.find(frame_id);
//         const auto& frame_node = *frame_itr;
//         if(!frame_node) {
//             LOG(WARNING) << "Cannot compute centroid of object " << object_id << " at frame " << frame_id << " as it did not appear at this frame";
//             return false;
//         }

//         const StatusLandmarkEstimates& dynamic_lmks = frame_node->getDynamicLandmarkEstimates(obejct_id);
//         //convert to point cloud -> should be a map with only one map in it
//         CloudPerObject object_clouds = groupObjectCloud(dynamic_lmks, frame_node->getPoseEstimate().get());
//         if(object_clouds.size() == 0) {
//             LOG(WARNING) << "Cannot object clouds from dynamic landmarks of " << object_id << " and frame " << frame_id << "!! "
//                 << " # Dynamic lmks in the map for this object at this frame was " << dynamic_lmks.size();
//             return false;
//         }
//         CHECK_EQ(object_clouds.size(), 1);
//         CHECK(object_clouds.exists(obejct_id));

//         const auto dynamic_point_cloud = object_clouds.at(object_id);
//         pcl::PointXYZ centroid;
//         pcl::computeCentroid(dynamic_point_cloud, centroid);

//         gtsam::Point3 translation = pclPointToGtsam(centroid);

//         //update starting pose with the computed translation (ie.e centroid)
//         pose = gtsam::Pose3(object_pose_gt_0.L_world_.rotation(), translation);
//         return true;
//     };

//     if(!init_translation_from_gt) {
//         //set starting pose with the centroid of the object cloud at frame_k_0 and the rotation of the ground truth object pose
//         if(!init_translation_from_centroid(object_pose_gt_0, frame_k_0, obejct_id, starting_pose)) {
//             LOG(WARNING) << "Cannot compute starting centroid of object " << object_id << " and first frame " << frame_k_0 << "!!";
//             return gtsam::FastMap<FrameId, gtsam::Pose3>{};
//         }
//     }

//     gtsam::FastMap<FrameId, gtsam::Pose3> pose_map;
//     pose_map.insert2(frame_k_0, starting_pose);

//     auto frame_itr = seen_frames.begin();
//     //advance itr one so we're now at the second frame
//     std::advance(frame_itr, 1);

//     for(auto itr = frame_itr; itr != seen_frames.end(); itr++) {
//         auto frame_node_ptr = *itr;
//         const FrameId frame_id_k = frame_node_ptr->getId();
//         const FrameId frame_id_k_1 = frame_id_k - 1u;

//         const StateQuery<gtsam::Pose3> motion_query = frame_node_ptr->getObjectMotionEstimate(obejct_id);
//         if(!motion_query) {
//             LOG(WARNING) << "Cannot propofate pose of object " << object_id << " to frame " << frame_id_k << " as the motion query failed. Skipping";

//             //if the pose map has size 1, then we never propogate a motion
//             //no point returning this map as the evaluation will be skewed since the first (and only pose)
//             //is constructed from the gt
//             if(pose_map.size() == 1) {
//                 return gtsam::FastMap<FrameId, gtsam::Pose3>{};
//             }
//             else {
//                 return pose_map;
//             }

//         }
//         else {
//             const gtsam::Pose3& prev_H_world_curr = motion_query.get();
//             CHECK(pose_map.exists(frame_id_k_1));

//             const gtsam::Pose3& object_pose_k_1 = pose_map.at(frame_id_k_1);
//             gtsam::Pose3 object_pose_k = prev_H_world_curr * object_pose_k_1;
//             pose_map.insert2(frame_id_k, object_pose_k);

//         }
//     }

//     return pose_map;
// }

// template<typename MEASUREMENT>
// gtsam::FastMap<FrameId, gtsam::Pose3> ObjectNode<MEASUREMENT>::computeEstimatedPoseMap() const {
//     gtsam::FastMap<FrameId, gtsam::Pose3> pose_map;

//     const auto& seen_frames = this->getSeenFrames();
//     for(auto itr = seen_frames.begin(); itr != seen_frames.end(); itr++) {
//         auto frame_node_ptr = *itr;
//         const FrameId frame_id_k = frame_node_ptr->getId();

//         gtsam::Pose3 pose_estimate;
//         if(this->hasPoseEstimate(frame_id_k, pose_estimate)) {
//             pose_map.insert2(frame_id_k, pose_estimate);
//         }
//     }

//     return pose_map;

// }


template<typename MEASUREMENT>
FrameIds ObjectNode<MEASUREMENT>::getSeenFrameIds() const {
    return getSeenFrames().template collectIds<FrameId>();
}

template<typename MEASUREMENT>
LandmarkNodePtrSet<MEASUREMENT> ObjectNode<MEASUREMENT>::getLandmarksSeenAtFrame(FrameId frame_id) const {
    LandmarkNodePtrSet<MEASUREMENT> seen_lmks;

    for(const auto& lmk : dynamic_landmarks) {
        //all frames this lmk was seen in
        const FrameNodePtrSet<MEASUREMENT>& frames = lmk->getSeenFrames();
        //lmk was observed at this frame
        if(frames.find(frame_id) != frames.end()) {
            seen_lmks.insert(lmk);
        }
    }
    return seen_lmks;
}


} //dyno
