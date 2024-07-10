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

#include "dynosam/backend/BackendModule.hpp"
#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/backend/BackendInputPacket.hpp"
#include "dynosam/common/Map.hpp"

#include "dynosam/backend/DynoISAM2.hpp"
#include "dynosam/common/Flags.hpp"


#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
// #include <gtsam/nonlinear/ISAM2.h>

#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

namespace dyno {


using RGBDBackendModuleTraits = BackendModuleTraits<RGBDInstanceOutputPacket, LandmarkKeypoint>;

class RGBDBackendModule : public BackendModuleType<RGBDBackendModuleTraits> {

public:
    using Base = BackendModuleType<RGBDBackendModuleTraits>;
    using RGBDMap = Base::MapType;

    enum UpdaterType {
        MotionInWorld = 0,
        LLWorld = 1,
        ObjectCentric=2
    };

    RGBDBackendModule(const BackendParams& backend_params, RGBDMap::Ptr map, Camera::Ptr camera, const UpdaterType& updater_type, ImageDisplayQueue* display_queue = nullptr);
    ~RGBDBackendModule();

    using SpinReturn = Base::SpinReturn;

    //TODO: move to optimizer and put into pipeline manager where we know the type and bind write output to shutdown procedure
    void saveGraph(const std::string& file = "rgbd_graph.dot");
    void saveTree(const std::string& file = "rgbd_bayes_tree.dot");

    std::tuple<gtsam::Values, gtsam::NonlinearFactorGraph>
    constructGraph(FrameId from_frame, FrameId to_frame, bool set_initial_camera_pose_prior);

//TODO: for now
public:
    SpinReturn boostrapSpinImpl(RGBDInstanceOutputPacket::ConstPtr input) override;
    SpinReturn nominalSpinImpl(RGBDInstanceOutputPacket::ConstPtr input) override;

    struct UpdateObservationParams {
        //! If true, vision related updated will backtrack to the start of a new tracklet and all the measurements to the graph
        //! should make false in batch case where we want to be explicit about which frames are added!
        bool do_backtrack = false;
        bool enable_debug_info = true;
    };

    struct ConstructGraphOptions : public UpdateObservationParams {
        FrameId from_frame{0u};
        FrameId to_frame{0u};
        bool set_initial_camera_pose_prior = true;
    };

    struct UpdateObservationResult {
        gtsam::FastMap<ObjectId, std::set<FrameId>> objects_affected_per_frame; //per frame
        DebugInfo::Optional debug_info{};

        inline UpdateObservationResult& operator+=(const UpdateObservationResult& oth) {
            for(const auto& [key, value] : oth.objects_affected_per_frame) {
                objects_affected_per_frame[key].insert(value.begin(), value.end());
            }
            return *this;
        }

        void updateAffectedObject(FrameId frame_id, ObjectId object_id) {
            if(!objects_affected_per_frame.exists(object_id)) {
                objects_affected_per_frame.insert2(object_id,  std::set<FrameId>{});
            }
            objects_affected_per_frame[object_id].insert(frame_id);
        }
    };

    struct PointUpdateContext {
        LandmarkNode3d2d::Ptr lmk_node;
        FrameNode3d2d::Ptr frame_node_k_1;
        FrameNode3d2d::Ptr frame_node_k;

        gtsam::Pose3 X_k_measured; //! Camera pose from measurement (or initial)
        gtsam::Pose3 X_k_1_measured; //! Camera pose from measurement (or initial)

        //! If true then this frame is the first frame where a motion is available (i.e we have a pair of valid frames)
        //! e.g k-1 is the FIRST frame for this object and now, since we are at k, we can create a motion from k-1 to k
        bool is_starting_motion_frame {false};

        inline ObjectId getObjectId() const { return lmk_node->getObjectId(); }
        inline TrackletId getTrackletId() const { return lmk_node->tracklet_id; }

    };

    struct ObjectUpdateContext {
        FrameNode3d2d::Ptr frame_node_k;
        ObjectNode3d2d::Ptr object_node;

        //! Indicates that we have a valid motion pair from k-1 to k (this frame)
        //! and therefore k is at least the second frame for which this object has been consequatively tracked
        //! When this is false, it means that the frame k-1 (getFrameId() - 1u) did not track/observe this object
        //! and therfore no motion can be created between them. This happens on the first observation of this object.
        bool has_motion_pair{false};

        inline FrameId getFrameId() const { return frame_node_k->getId(); }
        inline ObjectId getObjectId() const { return object_node->getId(); }
    };

    class Accessor {
        public:
            DYNO_POINTER_TYPEDEFS(Accessor)

            Accessor(const gtsam::Values* theta, RGBDBackendModule* parent) : theta_(theta), parent_(parent) {}
            virtual ~Accessor() {}

            virtual StateQuery<gtsam::Pose3> getSensorPose(FrameId frame_id) const = 0;
            virtual StateQuery<gtsam::Pose3> getObjectMotion(FrameId frame_id, ObjectId object_id) const = 0;
            virtual StateQuery<gtsam::Pose3> getObjectPose(FrameId frame_id, ObjectId object_id) const = 0;

            virtual StateQuery<gtsam::Point3> getDynamicLandmark(FrameId frame_id, TrackletId tracklet_id) const = 0;
            virtual StateQuery<gtsam::Point3> getStaticLandmark(TrackletId tracklet_id) const;


            MotionEstimateMap getObjectMotions(FrameId frame_id) const;
            virtual EstimateMap<ObjectId, gtsam::Pose3> getObjectPoses(FrameId frame_id) const;

            //full object poses with interpolation (if possible!!) (used for vis and not evaluation)
            //TODO::
            //relies on all the virtual fucntions doing their thing and getting the desired value in the right frame etc
            //no interpolation etc...
            virtual ObjectPoseMap getObjectPoses() const;

            StatusLandmarkEstimates getDynamicLandmarkEstimates(FrameId frame_id) const;
            StatusLandmarkEstimates getDynamicLandmarkEstimates(FrameId frame_id, ObjectId object_id) const;

            StatusLandmarkEstimates getStaticLandmarkEstimates(FrameId frame_id) const;
            StatusLandmarkEstimates getFullStaticMap() const;

            StatusLandmarkEstimates getLandmarkEstimates(FrameId frame_id) const;


            bool hasObjectMotionEstimate(FrameId frame_id, ObjectId object_id, Motion3* motion = nullptr) const;
            bool hasObjectMotionEstimate(FrameId frame_id, ObjectId object_id, Motion3& motion) const;

             //TODO: test
            bool hasObjectPoseEstimate(FrameId frame_id, ObjectId object_id, gtsam::Pose3* pose = nullptr) const;
            bool hasObjectPoseEstimate(FrameId frame_id, ObjectId object_id, gtsam::Pose3& pose) const;

            /**
             * @brief Computes a the centroid of each object at this frame using the estimated dynamic points.
             *
             * Internally, uses the overloaded std::tuple<gtsam::Point3, bool> computeObjectCentroid function
             * and only includes centroids which are valid (ie returned with computeObjectCentroid()->second == true)
             *
             * @param frame_id FrameId
             * @return gtsam::FastMap<ObjectId, gtsam::Point3>
             */
            gtsam::FastMap<ObjectId, gtsam::Point3> computeObjectCentroids(FrameId frame_id) const;

            /**
             * @brief Computes the centroid of the requested object using the estimated dynamic points.
             *
             * Throws exception if object not found.
             *
             * @param frame_id FrameId
             * @param object_id ObjectId
             * @return std::tuple<gtsam::Point3, bool>
             */
            std::tuple<gtsam::Point3, bool> computeObjectCentroid(FrameId frame_id, ObjectId object_id) const;

            inline bool exists(gtsam::Key key) const { return theta_->exists(key); }

            template<typename ValueType>
            StateQuery<ValueType> query(gtsam::Key key) const {
                CHECK_NOTNULL(theta_);
                if(theta_->exists(key)) {
                    return StateQuery<ValueType>(key, theta_->at<ValueType>(key));
                }
                else {
                    return StateQuery<ValueType>::NotInMap(key);
                }
            }

            virtual void postUpdateCallback() {};

         protected:
            auto getMap() const { return parent_->getMap(); }

        private:
            const gtsam::Values* theta_;

        protected:
            RGBDBackendModule* parent_;
    };

    class Updater {

    public:

        DYNO_POINTER_TYPEDEFS(Updater)

        RGBDBackendModule* parent_;
        Updater(RGBDBackendModule* parent) : parent_(CHECK_NOTNULL(parent)) {}
        virtual ~Updater() = default;

        void setTheta(const gtsam::Values& linearization) {
            theta_ = linearization;
            accessorFromTheta()->postUpdateCallback();
        }
        void updateTheta(const gtsam::Values& linearization) {
            theta_.update(linearization);
            accessorFromTheta()->postUpdateCallback();
        }

        gtsam::Pose3 getInitialOrLinearizedSensorPose(FrameId frame_id) const;

        //  //adds pose to the new values and a prior on this pose to the new_factors
        void setInitialPose(const gtsam::Pose3& T_world_camera, FrameId frame_id_k, gtsam::Values& new_values);

        //TODO: specify noise model
        void setInitialPosePrior(const gtsam::Pose3& T_world_camera, FrameId frame_id_k, gtsam::NonlinearFactorGraph& new_factors);

        void addOdometry(FrameId from_frame, FrameId to_frame, gtsam::Values& new_values,  gtsam::NonlinearFactorGraph& new_factors);

        void addOdometry(FrameId frame_id_k, const gtsam::Pose3& T_world_camera, gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors);

        UpdateObservationResult updateStaticObservations(
            FrameId from_frame,
            FrameId to_frame,
            gtsam::Values& new_values,
            gtsam::NonlinearFactorGraph& new_factors,
            const UpdateObservationParams& update_params);

        UpdateObservationResult updateDynamicObservations(
            FrameId from_frame,
            FrameId to_frame,
            gtsam::Values& new_values,
            gtsam::NonlinearFactorGraph& new_factors,
            const UpdateObservationParams& update_params);


        UpdateObservationResult updateStaticObservations(
            FrameId frame_id_k,
            gtsam::Values& new_values,
            gtsam::NonlinearFactorGraph& new_factors,
            const UpdateObservationParams& update_params);

        UpdateObservationResult updateDynamicObservations(
            FrameId frame_id_k,
            gtsam::Values& new_values,
            gtsam::NonlinearFactorGraph& new_factors,
            const UpdateObservationParams& update_params);

        //log everything all frames!!
        void logBackendFromMap();
        virtual std::string loggerPrefix() const = 0;

        //creates a NEW accessor
        virtual Accessor::Ptr createAccessor(const gtsam::Values* values) const = 0;

        Accessor::Ptr accessorFromTheta() const;


        virtual void dynamicPointUpdateCallback(
            const PointUpdateContext& context,
            UpdateObservationResult& result,
            gtsam::Values& new_values,
            gtsam::NonlinearFactorGraph& new_factors) = 0;

        virtual void objectUpdateContext(
            const ObjectUpdateContext& context,
            UpdateObservationResult& result,
            gtsam::Values& new_values,
            gtsam::NonlinearFactorGraph& new_factors) = 0;

        virtual bool isDynamicTrackletInMap(const LandmarkNode3d2d::Ptr& lmk_node) const = 0;
        gtsam::FastMap<gtsam::Key, bool> is_other_values_in_map; //! the set of (static related) values managed by this updater. Allows checking if values have already been added over successifve function calls

        auto getMap() { return parent_->getMap(); }

        const gtsam::Values& getTheta() const { return theta_; }
        const gtsam::NonlinearFactorGraph& getGraph() const { return factors_; }

    protected:

    private:
        gtsam::Values theta_; //! Current linearisation that will be associated with the current graph
        gtsam::NonlinearFactorGraph factors_;

        mutable Accessor::Ptr accessor_theta_;
    };

    class LLAccessor : public Accessor {
        public:
            LLAccessor(const gtsam::Values* theta, RGBDBackendModule* parent) : Accessor(theta, parent) {}
            virtual ~LLAccessor() {}

            StateQuery<gtsam::Pose3> getSensorPose(FrameId frame_id) const override;
            virtual StateQuery<gtsam::Pose3> getObjectMotion(FrameId frame_id, ObjectId object_id) const override;
            virtual StateQuery<gtsam::Pose3> getObjectPose(FrameId frame_id, ObjectId object_id) const override;
            virtual StateQuery<gtsam::Point3> getDynamicLandmark(FrameId frame_id, TrackletId tracklet_id) const override;

        protected:


    };

    class LLUpdater : public Updater {
    public:
        DYNO_POINTER_TYPEDEFS(LLUpdater)

        LLUpdater(RGBDBackendModule* parent) : Updater(parent) {}
        virtual ~LLUpdater() {}

        inline Accessor::Ptr createAccessor(const gtsam::Values* values) const override {
            return std::make_shared<LLAccessor>(values, parent_);
        }

        void dynamicPointUpdateCallback(const PointUpdateContext& context, UpdateObservationResult& result, gtsam::Values& new_values,
            gtsam::NonlinearFactorGraph& new_factors) override;
        void objectUpdateContext(const ObjectUpdateContext& context, UpdateObservationResult& result, gtsam::Values& new_values,
            gtsam::NonlinearFactorGraph& new_factors) override;

        inline bool isDynamicTrackletInMap(const LandmarkNode3d2d::Ptr& lmk_node) const override {
            const TrackletId tracklet_id = lmk_node->tracklet_id;
            return is_dynamic_tracklet_in_map_.exists(tracklet_id);
        }

        std::string loggerPrefix() const override {
            return "rgbd_LL_world";
        }

    protected:
        //we need a separate way of tracking if a dynamic tracklet is in the map, since each point is modelled uniquely
        //simply used as an O(1) lookup, the value is not actually used. If the key exists, we assume that the tracklet is in the map
        gtsam::FastMap<TrackletId, bool> is_dynamic_tracklet_in_map_; //! thr set of dynamic points that have been added by this updater. We use a separate map containing the tracklets as the keys are non-unique

    };


    class MotionWorldAccessor : public LLAccessor {
        public:
            MotionWorldAccessor(const gtsam::Values* theta, RGBDBackendModule* parent) : LLAccessor(theta, parent) {}
            virtual ~MotionWorldAccessor() = default;

            virtual StateQuery<gtsam::Pose3> getObjectMotion(FrameId frame_id, ObjectId object_id) const override;
            virtual StateQuery<gtsam::Pose3> getObjectPose(FrameId frame_id, ObjectId object_id) const override;

            inline ObjectPoseMap getObjectPoses() const override { return object_pose_cache_; }
            EstimateMap<ObjectId, gtsam::Pose3> getObjectPoses(FrameId frame_id) const override;

            //this will update the object_pose_cache_ so call it every frame?
            void postUpdateCallback() override;

        private:
            ObjectPoseMap object_pose_cache_; //! Updated every time set/update theta is called via the postUpdateCallback
    };

    class MotionWorldUpdater : public LLUpdater {
    public:
        DYNO_POINTER_TYPEDEFS(MotionWorldUpdater)

        MotionWorldUpdater(RGBDBackendModule* parent) : LLUpdater(parent) {}

        inline Accessor::Ptr createAccessor(const gtsam::Values* values) const override {
            return std::make_shared<MotionWorldAccessor>(values, parent_);
        }

        void dynamicPointUpdateCallback(const PointUpdateContext& context, UpdateObservationResult& result, gtsam::Values& new_values,
            gtsam::NonlinearFactorGraph& new_factors) override;
        void objectUpdateContext(const ObjectUpdateContext& context, UpdateObservationResult& result, gtsam::Values& new_values,
            gtsam::NonlinearFactorGraph& new_factors) override;


        std::string loggerPrefix() const override {
            return "rgbd_motion_world";
        }
    };


    struct ObjectCentricProperties {
        inline gtsam::Symbol makeDynamicKey(TrackletId tracklet_id) const {
            return gtsam::Symbol(kDynamicLandmarkSymbolChar, tracklet_id);
        }
    };


    class ObjectCentricAccessor : public MotionWorldAccessor, public ObjectCentricProperties {
        public:
            ObjectCentricAccessor(
                const gtsam::Values* theta, RGBDBackendModule* parent,
                const gtsam::FastMap<ObjectId, std::pair<gtsam::Pose3, FrameId>>* L0_values) : MotionWorldAccessor(theta, parent), L0_values_(L0_values) {}
            virtual ~ObjectCentricAccessor() {}

            StateQuery<gtsam::Pose3> getObjectMotion(FrameId frame_id, ObjectId object_id) const override;
            StateQuery<gtsam::Pose3> getObjectPose(FrameId frame_id, ObjectId object_id) const override;
            StateQuery<gtsam::Point3> getDynamicLandmark(FrameId frame_id, TrackletId tracklet_id) const override;

        private:
            const gtsam::FastMap<ObjectId, std::pair<gtsam::Pose3, FrameId>>* L0_values_;
    };

    class ObjectCentricUpdater : public Updater, public ObjectCentricProperties {
    public:
        DYNO_POINTER_TYPEDEFS(ObjectCentricUpdater)

        ObjectCentricUpdater(RGBDBackendModule* parent) : Updater(parent) {}

        inline Accessor::Ptr createAccessor(const gtsam::Values* values) const override {
            return std::make_shared<ObjectCentricAccessor>(values, parent_, &L0_values_);
        }

        void dynamicPointUpdateCallback(const PointUpdateContext& context, UpdateObservationResult& result, gtsam::Values& new_values,
            gtsam::NonlinearFactorGraph& new_factors) override;
        void objectUpdateContext(const ObjectUpdateContext& context, UpdateObservationResult& result, gtsam::Values& new_values,
            gtsam::NonlinearFactorGraph& new_factors) override;

        inline bool isDynamicTrackletInMap(const LandmarkNode3d2d::Ptr& lmk_node) const override {
            const TrackletId tracklet_id = lmk_node->tracklet_id;
            return is_dynamic_tracklet_in_map_.exists(tracklet_id);
        }


        std::string loggerPrefix() const override {
            return "rgbd_object_centric";
        }

        private:
        //right now just uses the frame id of the first seen frame -> may cause problems down the line!!
            std::pair<gtsam::Pose3, FrameId> getL0OrInitalise(ObjectId object_id);

            //re-computes everytime
            gtsam::Pose3 computeInitialHFromFrontend(FrameId k, ObjectId object_id);

    private:
        gtsam::FastMap<ObjectId, std::pair<gtsam::Pose3, FrameId>> L0_values_;
        //we need a separate way of tracking if a dynamic tracklet is in the map, since each point is modelled uniquely
        //simply used as an O(1) lookup, the value is not actually used. If the key exists, we assume that the tracklet is in the map
        gtsam::FastMap<TrackletId, bool> is_dynamic_tracklet_in_map_; //! thr set of dynamic points that have been added by this updater. We use a separate map containing the tracklets as the keys are non-unique
    };

public:
    bool buildSlidingWindowOptimisation(FrameId frame_k, gtsam::Values& optimised_values, double& error_before, double& error_after);


    Updater::UniquePtr makeUpdater() {
        if(updater_type_ == UpdaterType::MotionInWorld) {
            LOG(INFO) << "Using MotionInWorld";
            return  std::make_unique<MotionWorldUpdater>(this);

        }
        else if(updater_type_ == UpdaterType::LLWorld) {
            LOG(INFO) << "Using LLWorld";
            return std::make_unique<LLUpdater>(this);
        }
        else if(updater_type_ == UpdaterType::ObjectCentric) {
            LOG(INFO) << "Using ObjectCentric";
            return std::make_unique<ObjectCentricUpdater>(this);
        }
        else {
            CHECK(false) << "Not implemented";
        }
    }

public:
    Camera::Ptr camera_;
    const UpdaterType updater_type_;
    // std::unique_ptr<DynoISAM2> smoother_;
    // DynoISAM2Result smoother_result_;
    // std::unique_ptr<gtsam::IncrementalFixedLagSmoother> smoother_;
    // UpdateImpl::UniquePtr updater_;
    Updater::UniquePtr new_updater_;
    FrameId first_frame_id_; //the first frame id that is received

    //new calibration every time
    inline auto getGtsamCalibration() const {
        const CameraParams& camera_params = camera_->getParams();
        return boost::make_shared<Camera::CalibrationType>(
            camera_params.constructGtsamCalibration<Camera::CalibrationType>()
        );
    }

    //logger here!!
    BackendLogger::UniquePtr logger_{nullptr};
    gtsam::FastMap<FrameId, gtsam::Pose3> initial_camera_poses_; //! Camera poses as estimated from the frontend per frame
    gtsam::FastMap<FrameId, MotionEstimateMap> initial_object_motions_; //! Object motions (in world) as estimated from the frontend per frame


    inline bool hasFrontendMotionEstimate(FrameId frame_id, ObjectId object_id, Motion3* motion) const {
        if(!initial_object_motions_.exists(frame_id)) { return false; }

        const auto& motion_map = initial_object_motions_.at(frame_id);
        if(!motion_map.exists(object_id)) { return false; }


        if(motion) {
            *motion = motion_map.at(object_id);
        }
        return true;
    }



    //base backend module does not correctly share properties between mono and rgbd (i.e static_pixel_noise_ is in backend module but is not used in this class)
    //TODO: really need a rgbd and mono base class for this reason
    gtsam::SharedNoiseModel static_point_noise_; //! 3d isotropic pixel noise on static points
    gtsam::SharedNoiseModel dynamic_point_noise_; //! 3d isotropic pixel noise on dynamic points

    DebugInfo debug_info_;



};

} //dyno
