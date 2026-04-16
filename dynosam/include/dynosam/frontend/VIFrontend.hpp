#pragma once

#include "dynosam/frontend/VIFrontendInput.hpp"
#include "dynosam/frontend/imu/ImuFrontend.hpp"
#include "dynosam/frontend/solvers/OpticalFlowAndPoseSolver.hpp"
#include "dynosam/frontend/solvers/PnPRansac.hpp"
#include "dynosam/frontend/vision/FeatureTracker.hpp"
#include "dynosam/pipeline/PipelineParams.hpp"
#include "dynosam_common/DynoState.hpp"
#include "dynosam_common/ModuleBase.hpp"
#include "dynosam_common/RealtimeOutput.hpp"
#include "dynosam_common/Trajectories.hpp"
#include "dynosam_cv/RGBDCamera.hpp"

namespace dyno {

struct InvalidImageContainerException : public DynosamException {
  InvalidImageContainerException(const ImageContainer& container,
                                 const std::string& what)
      : DynosamException("Image container with config: " +
                         container.toString() + "\n was invalid - " + what) {}
};

class VIFrontendLogger : public EstimationModuleLogger {
 public:
  DYNO_POINTER_TYPEDEFS(VIFrontendLogger)
  VIFrontendLogger(const std::string& logger_name = "frontend");
  virtual ~VIFrontendLogger();

  void logTrackingLengthHistogram(const Frame::Ptr frame);

 private:
  std::string tracking_length_hist_file_name_;
  json tracklet_length_json_;
};

class Frontend : public ModuleBase<VIFrontendInput, RealtimeOutput> {
 public:
  using Base = ModuleBase<VIFrontendInput, RealtimeOutput>;

  DYNO_POINTER_TYPEDEFS(Frontend)
  Frontend(const std::string& name, const DynoParams& params,
           ImageDisplayQueue* display_queue,
           const SharedGroundTruth& shared_ground_truth);
  virtual ~Frontend() = default;

 protected:
  bool pushImageToDisplayQueue(const std::string& wname, const cv::Mat& image);

  void logRealTimeOutput(const RealtimeOutput::Ptr& output);

 protected:
  virtual void validateInput(const VIFrontendInput::ConstPtr& input) const;

  const DynoParams dyno_params_;
  ImageDisplayQueue* display_queue_;
  const SharedGroundTruth shared_ground_truth_;

  VIFrontendLogger::UniquePtr logger_;
};

class VIFrontend : public Frontend {
 public:
  DYNO_POINTER_TYPEDEFS(VIFrontend)
  VIFrontend(const std::string& name, const DynoParams& params,
             Camera::Ptr camera, ImageDisplayQueue* display_queue,
             const SharedGroundTruth& shared_ground_truth);
  virtual ~VIFrontend() = default;

 protected:
  Frame::Ptr featureTrack(const VIFrontendInput::ConstPtr input,
                          std::optional<gtsam::Rot3> R_km1_k = std::nullopt);

  std::optional<gtsam::NavState> tryPropogateImu(
      const VIFrontendInput::ConstPtr input,
      const gtsam::NavState& nav_state_lIMU, ImuFrontend::PimPtr& pim_out);

  bool tryStereoMatch(Frame::Ptr frame, ImageContainer::Ptr image_container,
                      FeaturePtrs& stereo_features_out,
                      FeatureContainer& left_features_in_out);

  bool tryStereoMatchStaticFeatures(Frame::Ptr frame,
                                    ImageContainer::Ptr image_container,
                                    FeaturePtrs& stereo_features_out);

  // TODO: add back object poses?
  void fillDebugImagery(DebugImagery& debug_imagery, const Frame::Ptr& frame_k,
                        const Frame::Ptr& frame_km1) const;

  // NOTE: the ConstFeatureIterator is a little bit misleading as the features
  // do actually get
  //  modified in this function (since ConstFeatureIterator is const on the
  //  iterator but the features are non-const pointers)
  template <typename FeatureContainer, typename Predicate>
  size_t fillMeasurementsFromFeatureIterator(
      CameraMeasurementStatusVector* measurements,
      internal::FilterView<FeatureContainer, Predicate> it, FrameId frame_id,
      Timestamp timestamp, const gtsam::Vector2& pixel_sigmas,
      double depth_sigma, StatusLandmarkVector* landmarks = nullptr) const {
    CHECK(measurements);

    size_t num_added = 0;
    for (const Feature::Ptr& f : it) {
      CHECK_NOTNULL(f);
      const TrackletId tracklet_id = f->trackletId();
      const Keypoint& kp = f->keypoint();
      const ObjectId object_id = f->objectId();
      CHECK_EQ(f->objectId(), object_id);
      CHECK_EQ(f->frameId(), frame_id);
      CHECK(Feature::IsUsable(f));

      MeasurementWithCovariance<Keypoint> kp_measurement =
          MeasurementWithCovariance<Keypoint>::FromSigmas(kp, pixel_sigmas);
      CameraMeasurement camera_measurement(kp_measurement);

      // This can come from either stereo or rgbd
      if (f->hasDepth()) {
        // MeasurementWithCovariance<Landmark> landmark_measurement(
        //     // assume sigma_u and sigma_v are identical
        //     vision_tools::backProjectAndCovariance(
        //         *f, camera, pixel_sigmas(0), depth_sigma));
        // camera_measurement.landmark(landmark_measurement);
        Landmark landmark;
        rgbd_camera_->backProject(kp, f->depth(), &landmark);

        gtsam::Vector3 sigmas;
        sigmas << depth_sigma, depth_sigma, depth_sigma;

        MeasurementWithCovariance<Landmark> landmark_measurement =
            MeasurementWithCovariance<Landmark>::FromSigmas(landmark, sigmas);
        camera_measurement.landmark(landmark_measurement);

        if (landmarks) {
          landmarks->push_back(LandmarkStatus(landmark_measurement, frame_id,
                                              timestamp, tracklet_id, object_id,
                                              ReferenceFrame::LOCAL));
        }
      }

      if (f->hasRightKeypoint()) {
        CHECK(f->hasDepth()) << "Right keypoint set for feature but no depth!";
        MeasurementWithCovariance<Keypoint> right_kp_measurement =
            MeasurementWithCovariance<Keypoint>::FromSigmas(f->rightKeypoint(),
                                                            pixel_sigmas);
        camera_measurement.rightKeypoint(right_kp_measurement);
      }
      // no right keypoint and has rgbd camera and has depth, project
      // keypoint into right camera
      else if (rgbd_camera_ && f->hasDepth()) {
        bool right_projection_result = rgbd_camera_->projectRight(f);
        if (!right_projection_result) {
          // TODO: for now mark as outlier and ignore point
          f->markOutlier();
          continue;
        }

        CHECK(f->hasRightKeypoint());
        MeasurementWithCovariance<Keypoint> right_kp_measurement =
            MeasurementWithCovariance<Keypoint>::FromSigmas(f->rightKeypoint(),
                                                            pixel_sigmas);
        camera_measurement.rightKeypoint(right_kp_measurement);
      }

      if (f->keypointType() == KeyPointType::STATIC) {
        CHECK_EQ(object_id, background_label);
      } else {
        CHECK_NE(object_id, background_label);
      }

      measurements->push_back(CameraMeasurementStatus(
          camera_measurement, frame_id, timestamp, tracklet_id, object_id,
          ReferenceFrame::LOCAL));
      num_added++;
    }

    return num_added;
  }

 protected:
  Camera::Ptr camera_;

  //! PnpRansac solver for ego-motion
  PnPRansacSolver pnp_ransac_;
  //! OpticalFlowAndPoseSolver for ego-motion refinement
  OpticalFlowAndPoseSolver<Camera::CalibrationType> optical_flow_pose_solver_;

  ImuFrontend imu_frontend_;
  FeatureTracker::UniquePtr tracker_;

  //! Cached rgbd-camera
  std::shared_ptr<RGBDCamera> rgbd_camera_;

  //! Cached sigmas for static feature measurements
  double static_point_sigma_{0};
  gtsam::Vector2 static_pixel_sigmas_;
  //! Cached sigmas for dynamic feature measurements
  double dynamic_point_sigma_{0};
  gtsam::Vector2 dynamic_pixel_sigmas_;
};

}  // namespace dyno
