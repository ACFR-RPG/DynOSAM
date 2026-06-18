#include "dynosam/frontend/solvers/HybridObjectMotionSmoother.hpp"

#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>

#include "dynosam/factors/HybridFormulationFactors.hpp"
#include "dynosam_common/utils/Numerical.hpp"
#include "dynosam_common/utils/TimingStats.hpp"
#include "dynosam_opt/FactorGraphTools.hpp"
#include "dynosam_opt/IncrementalOptimization.hpp"
#include "dynosam_opt/NonlinearOptimizer.hpp"
#include "dynosam_opt/Symbols.hpp"

namespace dyno {

gtsam::Symbol PointSymbol(TrackletId tracklet_id) {
  // using the regular symbol is very important  as we use
  // gtsam::Symbol::ChrTest(kDynamicLandmarkSymbolChar)) to
  // return all the dynamic landmarks which does not work with
  // DynamicPointSymbol
  return gtsam::Symbol(kDynamicLandmarkSymbolChar,
                       static_cast<std::uint64_t>(tracklet_id));
}

// TODO: really should initalise with frame and tracklet ids...
HybridObjectMotionSmoother::HybridObjectMotionSmoother(ObjectId object_id,
                                                       Camera::Ptr camera,
                                                       double smootherLag)
    : HybridObjectMotionSolverImpl(object_id, camera),
      gtsam::FixedLagSmoother(smootherLag),
      logger_prefix_("hms_j" + std::to_string(object_id)),
      isam_(DefaultISAM2Params()),
      smoother_interface_(&isam_),
      kf_decision_logger_(CsvHeader("timestamp", "frame_id", "is_keyframe",
                                    "coverage", "shape_score", "scale_ratio",
                                    "frames_since_lkf", "last_kf")) {
  CHECK_NOTNULL(stereo_calibration_);
}

HybridObjectMotionSmoother::~HybridObjectMotionSmoother() {
  const std::string file_out = logger_prefix_ + "_kf_info.csv";
  OfstreamWrapper::WriteOutCsvWriter(kf_decision_logger_,
                                     getOutputFilePath(file_out));
  if (!debug_results_.empty()) {
    // const std::string file_name = logger_prefix_ + "_debug.bson";
    // LOG(INFO) << "Writing solver debug file: " << file_name;
    // const std::string file_path = getOutputFilePath(file_name);
    // JsonConverter::WriteOutJson(debug_results_, file_path,
    //                             JsonConverter::Format::BSON);
  }
}

size_t HybridObjectMotionSmoother::numKeyframes() const {
  return keyframe_ids_.size();
}

size_t HybridObjectMotionSmoother::numFramesSinceKeyframe() const {
  return frameId() - keyFrameId();
}

const FrameIds& HybridObjectMotionSmoother::frameIds() const {
  return frame_ids_;
}
const FrameIds& HybridObjectMotionSmoother::keyframeIds() const {
  return keyframe_ids_;
}

FrameId HybridObjectMotionSmoother::firstKeyframe() const {
  return keyframe_ids_.front();
}

PoseWithMotionTrajectory HybridObjectMotionSmoother::trajectory() const {
  // only from KF -> k (assume continuous?)
  PoseWithMotionTrajectory trajectory = frozen_trajectory_;

  const bool include_kf_in_local_traj = !trajectory.exists(keyFrameId());
  trajectory.insert(localTrajectoryImpl(include_kf_in_local_traj));

  return trajectory;
}

PoseWithMotionTrajectory HybridObjectMotionSmoother::localTrajectory() const {
  constexpr static bool kIncludeKFInTrajectory = true;
  return localTrajectoryImpl(kIncludeKFInTrajectory);
}

gtsam::Pose3 HybridObjectMotionSmoother::keyFrameMotion() const {
  const gtsam::Symbol sym(ObjectMotionSymbol(object_id_, frameId()));
  // TODO: bring back!
  //  CHECK(isam_.valueExists(sym));
  CHECK(smoother_state_.exists(sym));
  const gtsam::Pose3 H_W_KF_k = keyFrameMotionImpl(frameId(), smoother_state_);

  return H_W_KF_k;
}

Motion3ReferenceFrame HybridObjectMotionSmoother::frameToFrameMotionReference()
    const {
  const gtsam::Pose3 H_W_KF_k = keyFrameMotion();
  if (keyFrameId() == frameId()) {
    return Motion3ReferenceFrame(H_W_KF_k, Motion3ReferenceFrame::Style::F2F,
                                 ReferenceFrame::GLOBAL, keyFrameId(),
                                 frameId());
  }
  const gtsam::Pose3 L_W_KF = keyFramePose();

  CHECK_GT(frameId(), 0u);
  FrameId frame_id_km1 = frameId() - 1u;

  const gtsam::Symbol prev_motion_symbol(
      ObjectMotionSymbol(object_id_, frame_id_km1));

  //. TODO: bring back check!
  CHECK(state_since_lKF_.exists(prev_motion_symbol))
      << DynosamKeyFormatter(prev_motion_symbol);
  const gtsam::Pose3 H_W_KF_km1 =
      keyFrameMotionImpl(frame_id_km1, state_since_lKF_);

  gtsam::Pose3 H_W_km1_k = H_W_KF_k * H_W_KF_km1.inverse();
  return Motion3ReferenceFrame(H_W_km1_k, Motion3ReferenceFrame::Style::F2F,
                               ReferenceFrame::GLOBAL, frame_id_km1, frameId());
}

gtsam::FastMap<TrackletId, gtsam::Point3>
HybridObjectMotionSmoother::getObjectPoints() const {
  // TODO: smoother state or LKF state?
  // const std::map<gtsam::Key, gtsam::Point3> keyed_object_point_map =
  //     getObjectPointsFromSmootherState();

  // these are used for the realtime output but also used as initial for the
  // backend since we pass all points (for now) ensure that we only initalise
  // the point once?
  const std::map<gtsam::Key, gtsam::Point3> keyed_object_point_map =
      getObjectPointsFromState(all_m_L_points_);

  gtsam::FastMap<TrackletId, gtsam::Point3> object_point_map;
  for (const auto& [key, point] : keyed_object_point_map) {
    gtsam::Symbol sym(key);
    TrackletId tracklet_id = static_cast<TrackletId>(sym.index());
    object_point_map.insert2(tracklet_id, point);
  }
  return object_point_map;
}

gtsam::Pose3 HybridObjectMotionSmoother::keyFramePose() const {
  auto kf_data = keyframe_range_.find(frameId());
  CHECK(kf_data);

  const auto [_, LKF] = *kf_data;
  return LKF;
}

gtsam::Pose3 HybridObjectMotionSmoother::keyFrameCameraPose() const {
  return getCameraPose(this->keyFrameId());
}

gtsam::Pose3 HybridObjectMotionSmoother::getObjectPose(FrameId frame_id) const {
  auto kf_data = keyframe_range_.find(frame_id);
  CHECK(kf_data);

  auto motion_key = ObjectMotionSymbol(object_id_, frame_id);
  CHECK(all_states_.exists(motion_key)) << DynosamKeyFormatter(motion_key);

  const gtsam::Pose3 H_LKF_k = all_states_.at<gtsam::Pose3>(motion_key);

  const auto [_, LKF] = *kf_data;
  return H_LKF_k * LKF;
}

void HybridObjectMotionSmoother::receiveUpdate(
    const PoseChangeUpdateComplete& update_info) {
  // update does not involve this object (ie. this object did not have a
  // keyframe in the last batch of optimization)
  // TODO: we only get updates for objects that had new variables (ie were
  // keyframes)
  // but actually other objects may have changed due to joint-opt...!!!
  if (!update_info.objects.exists(object_id_)) {
    return;
  }

  const std::lock_guard<std::mutex> lock(backend_update_mutex_);
  has_backend_update_ = true;
  // could save some copying by using a const-ptr...
  backend_update_ = update_info;
}

double HybridObjectMotionSmoother::reprojectionError(Frame::Ptr frame) const {
  const FrameId frame_id_k = frame->getFrameId();

  const auto& object_points = this->getObjectPoints();
  const auto L_W_k = this->getObjectPose(frame_id_k);

  double repr_error = 0;

  // assume the pose is the same as the one used inside the smoother!
  auto frame_camera = frame->getFrameCamera();

  size_t count = 0;
  for (const Feature::Ptr& feature : frame->usableDynamicIterator(object_id_)) {
    const auto kp = feature->keypoint();
    const auto tracklet_id = feature->trackletId();

    if (object_points.exists(tracklet_id)) {
      const gtsam::Point3 m_W = L_W_k * object_points.at(tracklet_id);
      double repr = frame_camera.reprojectionError(m_W, kp).norm();

      repr_error += repr;
      count++;
    }
  }

  if (count == 0) {
    return std::numeric_limits<double>::infinity();
  } else {
    return repr_error / (double)count;
  }
}

double coverageKeyframeSupportSIMD(const std::vector<Eigen::Vector2d>& pts_kf,
                                   const std::vector<Eigen::Vector2d>& pts_cur,
                                   double radius = 10.0) {
  if (pts_kf.empty() || pts_cur.empty()) return 0.0;

  const int N = pts_cur.size();
  const double r2 = radius * radius;

  // --- pack current points into matrix (2 x N)
  Eigen::Matrix2Xd C(2, N);
  for (int i = 0; i < N; ++i) C.col(i) = pts_cur[i];

  int covered = 0;

  for (const auto& k : pts_kf) {
    // broadcast subtraction (vectorized)
    Eigen::ArrayXd dx = C.row(0).array() - k.x();
    Eigen::ArrayXd dy = C.row(1).array() - k.y();

    // squared distances (SIMD)
    Eigen::ArrayXd d2 = dx.square() + dy.square();

    // check if any point is within radius
    if ((d2 < r2).any()) {
      covered++;
    }
  }

  return static_cast<double>(covered) / pts_kf.size();
}

void drawEllipse(cv::Mat& img,
                 const Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d>& solver,
                 const Eigen::Vector2d& mean, const cv::Scalar& color) {
  const Eigen::Vector2d eigvals = solver.eigenvalues();
  const Eigen::Matrix2d eigvecs = solver.eigenvectors();

  // Validate eigenvalues
  if (!eigvals.allFinite()) {
    return;
  }

  // Clamp tiny negative values caused by numerics
  const double lambda0 = std::max(0.0, eigvals(0));
  const double lambda1 = std::max(0.0, eigvals(1));

  // Semi-axis lengths
  const double axis0 = std::sqrt(lambda0) * 2.0;
  const double axis1 = std::sqrt(lambda1) * 2.0;

  if (!std::isfinite(axis0) || !std::isfinite(axis1)) {
    return;
  }

  // OpenCV requires non-negative integer axes
  cv::Size axes(static_cast<int>(std::round(axis0)),
                static_cast<int>(std::round(axis1)));

  // Largest eigenvector determines orientation
  int idx = lambda0 > lambda1 ? 0 : 1;

  double angle = std::atan2(eigvecs(1, idx), eigvecs(0, idx)) * 180.0 / M_PI;

  if (!std::isfinite(angle)) {
    return;
  }

  const cv::Point p(utils::gtsamPointToCv<int>(mean));

  if (!utils::matContains(img, p)) {
    return;
  }

  cv::ellipse(img, p, axes, angle, 0.0, 360.0, color, 1, cv::LINE_AA);
}

const cv::Scalar KF_COLOR = cv::Scalar(0, 255, 0);
const cv::Scalar CUR_COLOR = cv::Scalar(0, 0, 255);
const cv::Scalar ELLIPSE_KF = cv::Scalar(0, 200, 0);
const cv::Scalar ELLIPSE_CUR = cv::Scalar(0, 0, 200);

const cv::Scalar TEXT_BG = cv::Scalar(50, 50, 50);
const cv::Scalar TEXT_COLOR = cv::Scalar(255, 255, 255);

void drawTextBlock(cv::Mat& img, const std::vector<std::string>& lines,
                   const cv::Point& origin = cv::Point(10, 10)) {
  int x = origin.x;
  int y = origin.y;

  constexpr int line_height = 20;
  constexpr int padding = 5;

  int width = 0;

  for (const auto& line : lines) {
    int baseline = 0;

    cv::Size text_size =
        cv::getTextSize(line, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

    width = std::max(width, text_size.width);
  }

  int height = line_height * static_cast<int>(lines.size());

  // Background rectangle
  cv::rectangle(img, cv::Point(x - padding, y - padding),
                cv::Point(x + width + padding, y + height + padding), TEXT_BG,
                -1);

  // Text
  for (size_t i = 0; i < lines.size(); ++i) {
    cv::putText(img, lines[i],
                cv::Point(x, y + static_cast<int>((i + 1) * line_height - 5)),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv::LINE_AA);
  }
}

bool HybridObjectMotionSmoother::shouldBeKeyframe(Frame::Ptr frame,
                                                  cv::Mat* debug_image) const {
  const auto& object_observations = frame->getObjectObservations();

  // Jesse: not even sure this should happen!
  if (!object_observations.exists(object_id_)) {
    return false;
  }

  std::vector<Eigen::Vector2d> pts_kf;
  std::vector<Eigen::Vector2d> pts_cur;

  size_t num_detections = 0;
  const auto& dynamic_features_lOKF = lOKF_frame_->dynamic_features_;

  for (const auto& feature : frame->usableDynamicIterator(object_id_)) {
    const TrackletId id = feature->trackletId();
    num_detections++;

    if (!dynamic_features_lOKF.exists(id)) continue;

    auto kp_cur = feature->keypoint();
    auto kp_kf = dynamic_features_lOKF.getByTrackletId(id)->keypoint();

    pts_cur.push_back(kp_cur);
    pts_kf.push_back(kp_kf);

    if (debug_image) {
      cv::circle(*debug_image, utils::gtsamPointToCv<int>(kp_cur), 2, CUR_COLOR,
                 -1, cv::LINE_AA);
      cv::circle(*debug_image, utils::gtsamPointToCv<int>(kp_kf), 2, KF_COLOR,
                 -1, cv::LINE_AA);
    }
  }

  const auto computeCovariance = [](const std::vector<Eigen::Vector2d>& pts,
                                    Eigen::Matrix2d& cov,
                                    Eigen::Vector2d& mean) -> void {
    const int N = static_cast<int>(pts.size());

    if (N == 0) {
      cov.setIdentity();
      mean.setZero();
      return;
    }

    // --- stack into matrix (2 x N)
    Eigen::Matrix<double, 2, Eigen::Dynamic> X(2, N);

    for (int i = 0; i < N; ++i) {
      X.col(i) = pts[i];
    }

    // --- mean (vectorized)
    mean = X.rowwise().mean();

    // --- center in one shot
    Eigen::Matrix<double, 2, Eigen::Dynamic> Xc = X.colwise() - mean;

    // --- covariance (fully batched)
    cov = (Xc * Xc.transpose()) / double(N);
  };

  Eigen::Matrix2d cov_kf, cov_cur;
  Eigen::Vector2d mean_kf, mean_cur;

  computeCovariance(pts_kf, cov_kf, mean_kf);
  computeCovariance(pts_cur, cov_cur, mean_cur);

  // --- eigenvalues
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver_kf(cov_kf);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver_cur(cov_cur);

  const Eigen::Vector2d eig_kf = solver_kf.eigenvalues();
  const Eigen::Vector2d eig_cur = solver_cur.eigenvalues();

  // --- scale ratio (0..1-ish)
  const double scale_ratio = std::sqrt((eig_cur.sum()) / (eig_kf.sum() + 1e-9));

  // --- shape score (0..1)
  // Eigen guarantees ascending order
  const double kf_min = eig_kf(0);
  const double kf_max = eig_kf(1);

  const double cur_min = eig_cur(0);
  const double cur_max = eig_cur(1);

  const double ratio_kf = kf_min / (kf_max + 1e-9);
  const double ratio_cur = cur_min / (cur_max + 1e-9);

  const double shape_score =
      std::min(ratio_cur / (ratio_kf + 1e-9), ratio_kf / (ratio_cur + 1e-9));

  double coverage = coverageKeyframeSupportSIMD(pts_kf, pts_cur);

  if (debug_image) {
    drawEllipse(*debug_image, solver_cur, mean_cur, ELLIPSE_CUR);
    drawEllipse(*debug_image, solver_kf, mean_kf, ELLIPSE_KF);

    std::vector<std::string> lines = {
        "Object Keyframe metrics", "scale: " + cv::format("%.2f", scale_ratio),
        "shape: " + cv::format("%.2f", shape_score),
        "coverage: " + cv::format("%.2f", coverage)};
    drawTextBlock(*debug_image, lines);
  }

  bool distribution_degraded = shape_score < 0.3;  // structure collapsed

  bool coverage_lost = coverage < 0.3;  // drift outside expected region

  // 1.0 (ish) means no scale change
  // > 1 is increase in size, < 1 is decrease in size.
  // scale change is the delta in change
  double scale_change = std::abs(scale_ratio - 1.0);

  bool large_scale_change = scale_change > 0.8;

  bool need_new_keyframe =
      distribution_degraded || coverage_lost || large_scale_change;

  // need at least some detections to be a good keyframe
  if (num_detections < 10) {
    need_new_keyframe = false;
  }

  // but also need at least some initial points
  if (all_m_L_points_.size() < 10) {
    need_new_keyframe = false;
  }

  LOG(INFO) << "KF stats j=" << object_id_ << " cov " << coverage
            << " scale_change " << scale_change << " shape_score "
            << shape_score;

  //  kf_decision_logger_ << frame->getTimestamp() << frame->getFrameId()
  //                     << need_new_keyframe << coverage << shape_score
  //                     << scale_ratio << frames_since_lkf << keyFrameId();

  return need_new_keyframe;
}

// bool HybridObjectMotionSmoother::shouldBeKeyframe(Frame::Ptr frame) const {
//   // must at least have two frames!
//   const FrameId frames_since_lkf = frame->getFrameId() - keyFrameId();
//   if (frames_since_lkf < 2) {
//     return false;
//   }

//   const auto& cam_params = frame->getCamera()->getParams();

//   const auto& object_observations = frame->getObjectObservations();

//   // Jesse: not even sure this should happen!
//   if (!object_observations.exists(object_id_)) {
//     return false;
//   }

//   const ObjectDetection& obj_det = object_observations.at(object_id_);

//   // Downscale factor (same as before)
//   const int scale = 10;

//   const int rows = cam_params.ImageHeight() / scale;
//   const int cols = cam_params.ImageWidth() / scale;

//   const gtsam::Matrix33 K_inv = cam_params.getCameraMatrixEigen().inverse();

//   // --- Resize object mask to working resolution
//   cv::Mat object_mask_resized;
//   cv::resize(obj_det.mask, object_mask_resized, cv::Size(cols, rows), 0, 0,
//              cv::INTER_NEAREST);

//   // Ensure binary
//   cv::threshold(object_mask_resized, object_mask_resized, 1, 255,
//                 cv::THRESH_BINARY);

//   // --- Masks
//   cv::Mat matches = cv::Mat::zeros(rows, cols, CV_8UC1);
//   cv::Mat detections = cv::Mat::zeros(rows, cols, CV_8UC1);

//   const FeatureContainer& dynamic_features_lOKF =
//       lOKF_frame_->dynamic_features_;

//   std::vector<double> displacements;
//   std::vector<double> parallax_angles;

//   int num_detections = 0;
//   int num_matches = 0;

//   auto feature_itr = frame->usableDynamicIterator(object_id_);

//   const double radius = 3.0;  // fixed radius (better for objects)

//   TrackletIds tracklets;
//   for (const auto& feature : feature_itr) {
//     const TrackletId tracklet_id = feature->trackletId();
//     tracklets.push_back(tracklet_id);

//     const Keypoint& kp = feature->keypoint();

//     cv::Point2f pt = utils::gtsamPointToCv(kp) * (1.0 / scale);

//     // --- Only consider points inside object mask
//     if (pt.x < 0 || pt.x >= cols || pt.y < 0 || pt.y >= rows) continue;

//     // --- detections
//     cv::circle(detections, pt, int(radius), cv::Scalar(255), cv::FILLED);
//     num_detections++;

//     // --- matches (tracked from last object KF)
//     if (dynamic_features_lOKF.exists(tracklet_id)) {
//       cv::circle(matches, pt, int(radius), cv::Scalar(255), cv::FILLED);
//       num_matches++;

//       const auto& kp_kf =
//           dynamic_features_lOKF.getByTrackletId(tracklet_id)->keypoint();
//       cv::Point2f pt_kf = utils::gtsamPointToCv(kp_kf) * (1.0 / scale);

//       gtsam::Vector3 kp_kf_bearing =
//           gtsam::Vector3(K_inv * gtsam::Vector3(kp_kf(0), kp_kf(1), 1.0));
//       kp_kf_bearing.normalize();

//       gtsam::Vector3 kp_k_bearing =
//           gtsam::Vector3(K_inv * gtsam::Vector3(kp(0), kp(1), 1.0));
//       kp_k_bearing.normalize();

//       // NOTE: this does not account for the motion of the camera
//       // ie. motion of the camera will cause parallax
//       // but here we're working with assumption that any
//       // change in view makes a good keyframe
//       double cos_angle = kp_kf_bearing.dot(kp_k_bearing);
//       cos_angle = std::clamp(cos_angle, -1.0, 1.0);

//       double angle = std::acos(cos_angle);
//       parallax_angles.push_back(angle);

//       displacements.push_back(cv::norm(pt - pt_kf));
//     }
//   }

//   double median_parallax = calculateMedian(parallax_angles);

//   double object_area = double(cv::countNonZero(object_mask_resized));
//   double match_area = double(cv::countNonZero(matches &
//   object_mask_resized));
//   // How much of the visible object is explained by KF tracks
//   double coverage = (object_area > 0.0) ? match_area / object_area : 0.0;

//   double repr_error = reprojectionError(frame, tracklets);
//   double graph_error =
//   this->getFactors().error(this->getLinearizationPoint());

//   LOG(INFO) << "STATS object j=" << object_id_ << " coverage: " << coverage
//             << " median parallax: " << median_parallax
//             << " frames since kf:" << frames_since_lkf;

//   // const double coverage_thresh = 0.3;  // spatial redundancy
//   const double coverage_thresh = 0.3;  // spatial redundancy
//   double parallax_thresh = 0.05;
//   FrameId min_frames_dt = 15;

//   bool is_keyframe = false;
//   if (coverage < coverage_thresh) {
//     is_keyframe = true;
//   }

//   if (median_parallax > parallax_thresh) {
//     is_keyframe = true;
//   }

//   if (frames_since_lkf > min_frames_dt) {
//     is_keyframe = true;
//   }

//   // need at least some detections to be a good keyframe
//   if (num_detections < 10) {
//     is_keyframe = false;
//   }

//   // kf_decision_logger_ << frame->getTimestamp() << frame->getFrameId()
//   //                     << is_keyframe << coverage << median_parallax
//   //                     << frames_since_lkf << keyFrameId() << repr_error
//   //                     << graph_error;

//   return is_keyframe;
// }

bool HybridObjectMotionSmoother::resetWithNewKeyedMotion(
    const gtsam::Pose3& L_KF, Frame::Ptr frame, const TrackletIds& tracklets) {
  if (VLOG_IS_ON(10)) {
    const std::string current_frame =
        active_frame_ids_.empty() ? "None" : std::to_string(frameId());
    const std::string current_KF =
        active_frame_ids_.empty() ? "None" : std::to_string(keyFrameId());
    VLOG(10) << "Creating new KeyMotion "
             << info_string(frame->getFrameId(), object_id_)
             << " current k=" << current_frame << " KF=" << current_KF;
  }

  // isam_ = gtsam::ISAM2(DefaultISAM2Params());

  dyno::ISAM2 isam_copy = isam_;
  isam_ = dyno::ISAM2(DefaultISAM2Params());
  smoother_interface_ = SmootherInterface(&isam_);

  // update fixed trajectory using current kf state
  // must do this before temporal/keyframe data-structures are reset
  // relies on state_since_lKF_ to fill trajectory values
  // trajectory includes keyframe!
  PoseWithMotionTrajectory trajectory_till_lKF;
  if (frozen_trajectory_.empty()) {
    // if the trajectory is currently emppty, get the full trajectory
    // which will include the keyframe as the first frame of the trajectory.
    trajectory_till_lKF = std::move(localTrajectory());
  } else {
    // TODO: what if the trajectory is broken!!
    const bool include_kf_in_local_traj =
        !frozen_trajectory_.exists(keyFrameId());
    trajectory_till_lKF =
        std::move(localTrajectoryImpl(include_kf_in_local_traj));
  }
  frozen_trajectory_.insert(trajectory_till_lKF);

  // do before we clear all the *_since_lKF so the virtual function can access
  // this stuff if necessary
  this->onNewKeyFrameMotion(isam_copy, L_KF);

  active_frame_ids_.clear();
  active_timestamps_.clear();
  state_since_lKF_.clear();
  tracking_keyframes_.clear();

  lOKF_frame_ = frame;
  tracking_keyframes_.push_back(frame);

  smoother_state_.clear();

  // clear internal timestamp mapping so as to not confuse the Fixed Lag
  timestampKeyMap_.clear();
  keyTimestampMap_.clear();

  keyframe_ids_.push_back(frame->getFrameId());
  keyframe_range_.startNewActiveRange(frame->getFrameId(), L_KF);

  // update and add points at initial frame corresponding with an identity
  // motion allow the update function to insert new frames and timestamps given
  // the frame
  return updateFromInitialMotion(gtsam::Pose3::Identity(), frame, tracklets)
      .solver_okay;
}

bool HybridObjectMotionSmoother::setNewKeyframe(Frame::Ptr frame) {
  lOKF_frame_ = frame;
  tracking_keyframes_.push_back(frame);
  return true;
}

bool HybridObjectMotionSmoother::update(const gtsam::Pose3& H_W_km1_k_predict,
                                        Frame::Ptr frame,
                                        const TrackletIds& tracklets) {
  CHECK(!active_frame_ids_.empty())
      << "HybridObjectMotionSmoother::update "
      << " cannot be called without first creating a valid MotionFrame!";

  gtsam::Pose3 H_W_KF_km1 = gtsam::Pose3::Identity();
  if (state_since_lKF_.exists(ObjectMotionSymbol(object_id_, frameId()))) {
    H_W_KF_km1 = keyFrameMotionImpl(frameId(), state_since_lKF_);
  }

  frame_ids_.push_back(frame->getFrameId());
  // propogate initial guess
  const gtsam::Pose3 H_W_KF_k = H_W_km1_k_predict * H_W_KF_km1;
  return updateFromInitialMotion(H_W_KF_k, frame, tracklets).solver_okay;
}

PoseWithMotionTrajectory HybridObjectMotionSmoother::localTrajectoryImpl(
    bool include_keyframe) const {
  CHECK_EQ(active_frame_ids_.size(), active_timestamps_.size());

  if (active_frame_ids_.empty()) {
    return PoseWithMotionTrajectory{};
  }

  auto kf_data = keyframe_range_.find(frameId());
  const auto [frame_KF_id, L_W_KF] = *kf_data;

  // NOTE: this is only true becuase keyframe id returns the frame at the start
  // of the currently active frame ids (ie. frame_KF_id) but not the id of
  // lOKF_frame_ which we are tracking against!!
  CHECK_EQ(frame_KF_id, keyFrameId());

  // build trajectory from best state estimate since last KF
  PoseWithMotionTrajectory local_trajectory;
  for (size_t i = 0; i < active_frame_ids_.size(); i++) {
    const FrameId frame_id = active_frame_ids_.at(i);
    const Timestamp timestamp = active_timestamps_.at(i);

    // sanity check that all frames are part of the same KF range
    CHECK(kf_data->contains(frame_id));

    const gtsam::Symbol H_key_k = ObjectMotionSymbol(object_id_, frame_id);
    // TODO: bit of a hack - in the case that the isam2 solver fails
    // e.g with ILS, the values will not be added to the smoother
    //  and therefore will not appear in state_since_lKF_
    //  currently just skip!!
    if (!state_since_lKF_.exists(H_key_k)) {
      continue;
    }
    CHECK(state_since_lKF_.exists(H_key_k)) << DynosamKeyFormatter(H_key_k);

    const gtsam::Pose3 H_W_KF_k =
        keyFrameMotionImpl(frame_id, state_since_lKF_);

    Motion3ReferenceFrame f2f_motion;
    if (i == 0) {
      CHECK_EQ(frame_id, frame_KF_id);

      // skip this frame if requested
      if (!include_keyframe) {
        continue;
      }

      f2f_motion =
          Motion3ReferenceFrame(H_W_KF_k, Motion3ReferenceFrame::Style::F2F,
                                ReferenceFrame::GLOBAL, frame_id, frame_id);
    } else {
      CHECK_GT(frame_id, 0u);
      FrameId frame_id_km1 = frame_id - 1u;
      // sanity check that the previous frame is frame id -1 (ie. we're
      // consecutive!)
      CHECK_EQ(frame_id_km1, active_frame_ids_.at(i - 1));

      const gtsam::Symbol H_key_km1 =
          ObjectMotionSymbol(object_id_, frame_id - 1u);
      CHECK(state_since_lKF_.exists(H_key_km1));
      const gtsam::Pose3 H_W_KF_km1 =
          keyFrameMotionImpl(frame_id - 1u, state_since_lKF_);
      const gtsam::Pose3 H_W_km1_k = H_W_KF_k * H_W_KF_km1.inverse();

      f2f_motion = Motion3ReferenceFrame(
          H_W_km1_k, Motion3ReferenceFrame::Style::F2F, ReferenceFrame::GLOBAL,
          frame_id - 1u, frame_id);
    }

    const gtsam::Pose3 L_W_k = H_W_KF_k * L_W_KF;

    local_trajectory.insert(frame_id, timestamp, {L_W_k, f2f_motion});
  }

  return local_trajectory;
}

HybridObjectMotionSmoother::Result
HybridObjectMotionSmoother::updateFromInitialMotion(
    const gtsam::Pose3& H_W_KF_k_initial, Frame::Ptr frame,
    const TrackletIds& tracklets) {
  const FrameId frame_id = frame->getFrameId();
  const Timestamp timestamp = frame->getTimestamp();

  // update temporal data-structure immediately so frameId() and keyFrameId()
  // functions work
  active_frame_ids_.push_back(frame_id);
  active_timestamps_.push_back(timestamp);

  camera_poses_.insert2(frame_id, frame->getPose());

  gtsam::Values smoother_state;
  auto result = this->updateFromInitialMotionImpl(
      smoother_state, H_W_KF_k_initial, frame, tracklets);

  smoother_state_ = std::move(smoother_state);
  state_since_lKF_.insert_or_assign(smoother_state_);

  all_states_.insert_or_assign(smoother_state_);

  // add potentially new or updated points to set of all (accumulated) object
  // points in L
  const std::map<gtsam::Key, gtsam::Point3> points_in_state =
      getObjectPointsFromState(smoother_state_);
  for (const auto& [key, m_L] : points_in_state) {
    all_m_L_points_.insert_or_assign(key, m_L);
  }

  return result;
}

HybridObjectMotionSmoother::Result HybridObjectMotionSmoother::updateSmoother(
    const gtsam::NonlinearFactorGraph& newFactors,
    const gtsam::Values& newTheta, const KeyTimestampMap& timestamps,
    const dyno::ISAM2UpdateParams& update_params,
    const gtsam::KeyVector& additional_keys_marginalize) {
  gtsam::FastVector<size_t> removedFactors;
  boost::optional<gtsam::FastMap<gtsam::Key, int>> constrainedKeys = {};

  Result result;
  // Update the Timestamps associated with the factor keys
  updateKeyTimestampMap(timestamps);

  // Get current timestamp
  double current_timestamp = getCurrentTimestamp();
  // LOG(INFO) << "Current timestamp: " << current_timestamp;

  // Find the set of variables to be marginalized out
  // LOG(INFO) << "Findig keys before " << current_timestamp - smootherLag_;
  gtsam::KeyVector marginalizableKeysOriginal =
      findKeysBefore(current_timestamp - smootherLag_);

  gtsam::KeyVector marginalizableKeys = marginalizableKeysOriginal;
  marginalizableKeys.insert(marginalizableKeys.end(),
                            additional_keys_marginalize.begin(),
                            additional_keys_marginalize.end());

  result.marginalized_keys = marginalizableKeys;

  std::cout << "Gets to marginalize due to filter: ";
  for (const auto& key : marginalizableKeys) {
    std::cout << gtsam::DefaultKeyFormatter(key) << " ";

    CHECK(newTheta.exists(key) || isam_.valueExists(key))
        << DynosamKeyFormatter(key);
  }
  std::cout << std::endl;

  // newTheta.print("New theta", DynosamKeyFormatter);

  // the problem is that this will include only include variables
  // in theta
  // if we're not careful removing all factors on a single variable
  // without marginalizsing it will cause it to disappear from
  // all internal value data-structures (ie theta_, delta_ etc)
  // but NOT from the bayes tree
  // if (marginalizableKeys.size() > 0) {
  //   constrainedKeys.emplace();
  //   for (const auto& key : isam_.getLinearizationPoint().keys()) {
  //     constrainedKeys->operator[](key) = 1;
  //   }
  //   for (auto key : marginalizableKeys) {
  //     constrainedKeys->operator[](key) = 0;
  //   }
  // }

  // Force iSAM2 to put the marginalizable variables at the beginning
  createOrderingConstraints(marginalizableKeys, constrainedKeys);

  // if(constrainedKeys) {
  //   for (const auto& [key, group] : *constrainedKeys) {
  //     std::cout << gtsam::DefaultKeyFormatter(key) << " " << group << " ";
  //   }
  //   std::cout << std::endl;
  // }

  std::unordered_set<gtsam::Key> additionalKeys =
      BayesTreeMarginalizationHelper<
          dyno::ISAM2>::gatherAdditionalKeysToReEliminate(isam_,
                                                          marginalizableKeys);

  // std::cout << "Gets to additionalKeys due to filter: ";
  // for (const auto& key : additionalKeys) {
  //   std::cout << DynosamKeyFormatter(key) << " ";
  // }
  // std::cout << std::endl;
  // newFactors.print("New factors ");

  gtsam::KeyList additionalMarkedKeys(additionalKeys.begin(),
                                      additionalKeys.end());
  result.additional_keys_reeliminate = additionalMarkedKeys;

  dyno::ISAM2UpdateParams mutable_update_params = update_params;
  if (!mutable_update_params.extraReelimKeys) {
    mutable_update_params.extraReelimKeys = gtsam::KeyList{};
  }
  mutable_update_params.extraReelimKeys->insert(
      mutable_update_params.extraReelimKeys->begin(),
      additionalMarkedKeys.begin(), additionalMarkedKeys.end());

  if (constrainedKeys) {
    mutable_update_params.constrainedKeys.emplace(*constrainedKeys);
  }

  utils::ChronoTimingStats update_timer(logger_prefix_ + ".isam_update", 10);

  auto error_hooks = getDefaultILSErrorHandlingHooks();
  error_hooks.identifier = "hybrid_smoother_j" + std::to_string(object_id_);

  // std::cout << "Keys before: ";
  // for (const auto& key : isam_.getLinearizationPoint().keys()) {
  //   std::cout << DynosamKeyFormatter(key) << " ";
  // }
  // std::cout << std::endl;

  smoother_interface_.setMaxExtraIterations(0);
  result.solver_okay = smoother_interface_.optimize(
      &isamResult_,
      [&](const SmootherInterface::Smoother&,
          SmootherInterface::UpdateArguments& update_arguments) {
        update_arguments.new_values = newTheta;
        update_arguments.new_factors = newFactors;
        update_arguments.update_params = mutable_update_params;
      },
      error_hooks);

  result.update_time_ms = update_timer.stop();
  result.isam_result = isamResult_;

  // Marginalize out any needed variables
  if (marginalizableKeys.size() > 0) {
    gtsam::FastList<gtsam::Key> leafKeys(marginalizableKeys.begin(),
                                         marginalizableKeys.end());
    utils::ChronoTimingStats marginalize_timer(
        logger_prefix_ + ".marginalize_leaves", 10);
    isam_.marginalizeLeaves(leafKeys);
    result.marginalize_time_ms = marginalize_timer.stop();
  }

  // Remove marginalized keys from the KeyTimestampMap
  // TODO: will break with extra marginalize keys becuase they are not
  // in the timestamp-key data-structures
  // remove only keys that were temporally marginalized
  eraseKeyTimestampMap(marginalizableKeysOriginal);

  return result;
}

std::map<gtsam::Key, gtsam::Point3>
HybridObjectMotionSmoother::getObjectPointsFromState(
    const gtsam::Values& values) const {
  return values.extract<gtsam::Point3>(
      gtsam::Symbol::ChrTest(kDynamicLandmarkSymbolChar));
}

std::map<gtsam::Key, gtsam::Pose3>
HybridObjectMotionSmoother::getObjectMotionsFromState(
    const gtsam::Values& values) const {
  return values.extract<gtsam::Pose3>(Symbol::ChrTest(kObjectMotionSymbolChar));
}

void HybridObjectMotionSmoother::eraseKeysBefore(double timestamp) {
  TimestampKeyMap::iterator end = timestampKeyMap_.lower_bound(timestamp);
  TimestampKeyMap::iterator iter = timestampKeyMap_.begin();
  while (iter != end) {
    keyTimestampMap_.erase(iter->second);
    timestampKeyMap_.erase(iter++);
  }
}

/* ************************************************************************* */
void HybridObjectMotionSmoother::createOrderingConstraints(
    const gtsam::KeyVector& marginalizableKeys,
    boost::optional<gtsam::FastMap<gtsam::Key, int>>& constrainedKeys) const {
  if (marginalizableKeys.size() > 0) {
    constrainedKeys = gtsam::FastMap<gtsam::Key, int>();
    // Generate ordering constraints so that the marginalizable variables will
    // be eliminated first Set all variables to Group1
    for (const TimestampKeyMap::value_type& timestamp_key : timestampKeyMap_) {
      constrainedKeys->operator[](timestamp_key.second) = 1;
    }
    // Set marginalizable variables to Group0
    for (gtsam::Key key : marginalizableKeys) {
      constrainedKeys->operator[](key) = 0;
    }
  }
}

bool HybridObjectMotionSmoother::takeBackendUpdate(
    PoseChangeUpdateComplete& backend_update) {
  const std::lock_guard<std::mutex> lock(backend_update_mutex_);
  if (!has_backend_update_) {
    return false;
  }

  backend_update = std::move(backend_update_);
  has_backend_update_ = false;
  return true;
}

HybridObjectMotionOnlySmoother::Result
HybridObjectMotionOnlySmoother::updateFromInitialMotionImpl(
    gtsam::Values& smoother_state, const gtsam::Pose3& H_W_KF_k_initial,
    Frame::Ptr frame, const TrackletIds& tracklets) {
  utils::ChronoTimingStats t("hm_smoother.update", 10);
  const auto frame_id = frameId();
  const auto timestamp = this->timestamp();

  const double frame_as_double = static_cast<double>(frame_id);

  const gtsam::Symbol H_key_k = ObjectMotionSymbol(object_id_, frame_id);
  // fixed camera pose
  const gtsam::Pose3 X_W_k = frame->getPose();
  // get current keyframe pose
  const gtsam::Pose3 L_KF = keyFramePose();

  const gtsam::FastMap<TrackletId, gtsam::Point3> all_object_points =
      this->getObjectPoints();

  gtsam::Values new_values;
  gtsam::NonlinearFactorGraph new_factors;
  gtsam::FastMap<gtsam::FactorIndex, gtsam::KeySet> newly_affected_keys;

  KeyTimestampMap timestamps;
  // add motions
  // TODO: bug when we marginalize!!!!
  // Asking to remove variables from the variable index that are not unused
  // also somehow related to running multiple isam update iterations
  // timestamps[H_key_k] = frame_as_double;
  gtsam::Values initial_values = state_since_lKF_;
  initial_values.insert(H_key_k, H_W_KF_k_initial);

  // we must at least have the first motion as this is passed back to the Solver
  new_values.insert(H_key_k, H_W_KF_k_initial);

  PoseChangeUpdateComplete backend_update;

  const gtsam::NonlinearFactorGraph& factors_in_smoother = getFactors();

  gtsam::SharedNoiseModel stereo_noise_model =
      gtsam::noiseModel::Isotropic::Sigma(3u, 2.0);
  stereo_noise_model =
      factor_graph_tools::robustifyHuber(0.001, stereo_noise_model);

  auto stereo_non_robust_noise_model =
      gtsam::noiseModel::Isotropic::Sigma(3u, 2.0);

  // for debug stats
  size_t num_tracks_used = 0;
  size_t avg_feature_age = 0;

  size_t points_in_previous_kf = 0;
  size_t num_new_points = 0;
  // object_motion_to_tracklets_.insert2(H_key_k, TrackletIds{});

  gtsam::FactorIndices factors_to_delete;
  gtsam::KeyVector keys_that_should_be_deleted;
  gtsam::KeyVector additional_keys_to_marginalize;

  // const bool close_to_keyframe = frameId() <= (keyFrameId() + 2);
  const FrameId tracking_kf = lOKF_frame_->getFrameId();
  LOG(INFO) << "Tracking kf= " << tracking_kf << "kf id=" << keyFrameId();
  // TODO: currently keyframeId() is the first frame in the active frame set
  //  ie the frame of L_KF which may not be lOKF_frame_
  const bool close_to_keyframe = frameId() <= (tracking_kf + 2);

  utils::ChronoTimingStats update_timer("hm_smoother.update.tracklets", 10);
  for (const TrackletId& tracklet_id : tracklets) {
    Feature::Ptr feature = frame->at(tracklet_id);
    CHECK(feature);

    auto [stereo_keypoint_status, stereo_measurement] =
        rgbd_camera_->getStereo(feature);
    if (!stereo_keypoint_status) {
      continue;
    }

    double disparity = stereo_measurement.uL() - stereo_measurement.uR();
    if (disparity < 0.5) {
      feature->markOutlier();
      continue;
    }

    const gtsam::Symbol m_key(PointSymbol(tracklet_id));

    auto& z_per_frame = stereo_measurements_[tracklet_id];
    z_per_frame.insert2(frame_id, stereo_measurement);

    // totally new point for this keyframe
    // test only add new points for keyframe!
    if (!point_state_.exists(tracklet_id)) {
      // TODO: enforce real-time ;)
      //  if we do this we seem to loose many many points!
      // accept new keypoints only in or around a new keyframe
      // if (!close_to_keyframe) {
      //   continue;
      // }

      const gtsam::Point3 m_X_k = frame->backProjectToCamera(tracklet_id);
      Landmark m_L_init = HybridObjectMotion::projectToObject3(
          X_W_k, H_W_KF_k_initial, L_KF, m_X_k);
      point_state_.insert2(tracklet_id,
                           std::make_pair(PointState::InState, m_L_init));

      auto factor = boost::make_shared<StereoHybridMotionFactor2>(
          stereo_measurement, L_KF, X_W_k, stereo_noise_model,
          stereo_calibration_, H_key_k, m_key, false /*throw ceirality*/
      );
      num_new_points++;
    }

    num_tracks_used++;
    avg_feature_age += feature->age();
  }
  update_timer.stop();

  if (num_tracks_used == 0) {
    HybridObjectMotionSmoother::Result result;
    result.solver_okay = false;
    return result;
  }

  auto checkAndAddFactor =
      [&](TrackletId tracklet_id,
          FrameId frame_id) -> gtsam::NonlinearFactor::shared_ptr {
    const gtsam::Symbol m_key(PointSymbol(tracklet_id));

    // check seen at keyframe id, k-2, k-2, k
    const auto& z_per_frame = stereo_measurements_.at(tracklet_id);

    if (!z_per_frame.exists(frame_id)) {
      return nullptr;
    }

    // TODO: getCameraPose should keep track of the frame::ptr
    //  in this way the frontend will update the pose
    //  and we always get the latest camera pose estimate
    const auto X_W = this->getCameraPose(frame_id);
    const gtsam::Symbol H_key = ObjectMotionSymbol(object_id_, frame_id);

    // sanity check that L_KF is correct for frame_id
    auto kf_data = keyframe_range_.find(frame_id);
    CHECK(kf_data);
    CHECK_EQ(keyFrameId(), kf_data->dataPair().first);

    auto factor = boost::make_shared<StereoHybridMotionFactor2>(
        z_per_frame.at(frame_id), L_KF, X_W, stereo_noise_model,
        stereo_calibration_, H_key, m_key, false /*throw ceirality*/
    );

    return factor;
  };

  gtsam::FastMap<FrameId, gtsam::NonlinearFactorGraph> local_graphs;
  gtsam::FastMap<FrameId, TrackletIds> landmarks_to_add;

  std::set<FrameId> frame_ids_to_try;

  // try get last N tracking keyframes to include in the optimizer
  static constexpr size_t N_tracking_frames = 3;
  // Determine how many elements we can safely grab
  size_t count = std::min(tracking_keyframes_.size(), N_tracking_frames);
  for (auto it = tracking_keyframes_.end() - count;
       it != tracking_keyframes_.end(); ++it) {
    frame_ids_to_try.insert((*it)->getFrameId());
  }

  // NOTE: really need this keyframe to exist in the values so we can add our
  // prior
  //  this is SUPER important for problem stability (likely some kind of weird
  //  gague freedom) as without it we converge to an awful solution rely on the
  //  Solver to make keyframes at adaquate frames such that we have good overlap
  //  between k and KF
  frame_ids_to_try.insert(keyFrameId());
  // frame_ids_to_try.insert(tracking_kf);
  frame_ids_to_try.insert(frame_id - 2u);
  frame_ids_to_try.insert(frame_id - 1u);
  frame_ids_to_try.insert(frame_id);

  utils::ChronoTimingStats timer1("hm_smoother.update.points", 10);
  for (const auto& [tracklet_id, point_w_state] : point_state_) {
    gtsam::FastMap<FrameId, gtsam::NonlinearFactor::shared_ptr> observations;
    for (FrameId frame_id : frame_ids_to_try) {
      // only include frames >= current keyframe
      // prevents using different reference pose (ie. L_KF) between factors
      if (frame_id < keyFrameId()) {
        continue;
      }

      auto factor = checkAndAddFactor(tracklet_id, frame_id);
      if (factor) {
        observations[frame_id] = factor;
      }
    }

    // seen at at least three requested frames
    // make sure our graph is well-connected and represents motion well!
    if (observations.size() > 3) {
      for (const auto& [frame_id, factor] : observations) {
        local_graphs[frame_id] += factor;
        landmarks_to_add[frame_id].push_back(tracklet_id);
      }
    }
  }
  timer1.stop();

  utils::ChronoTimingStats timer2("hm_smoother.update.build_graph", 10);
  for (const auto& [frame_id, local_graph] : local_graphs) {
    const TrackletIds& tracklets = landmarks_to_add.at(frame_id);
    const gtsam::Symbol H_key = ObjectMotionSymbol(object_id_, frame_id);

    LOG(INFO) << "Num measurements= " << local_graph.size()
              << " k=" << frame_id;

    if (local_graph.size() > 3 && tracklets.size() > 3) {
      CHECK(initial_values.exists(H_key_k)) << "Missing H at " << frame_id;

      // LOG(INFO) << "Addoing";

      // right now only to account for that the H at this frame must always be
      // added
      if (!new_values.exists(H_key)) {
        // LOG(INFO) << "Adding key: " << DynosamKeyFormatter(H_key);
        new_values.insert(H_key, initial_values.at<gtsam::Pose3>(H_key));
      }
      new_factors += local_graph;

      for (TrackletId i : tracklets) {
        const gtsam::Symbol m_key(PointSymbol(i));

        if (!new_values.exists(m_key)) {
          const Landmark lmk = point_state_.at(i).second;
          new_values.insert(m_key, lmk);
        }
      }

      if (frame_id == keyFrameId() && new_values.exists(H_key)) {
        LOG(INFO) << "Added prior factor";

        gtsam::SharedNoiseModel identity_motion_model =
            gtsam::noiseModel::Isotropic::Sigma(6u, 0.00001);

        new_factors.addPrior<gtsam::Pose3>(H_key, gtsam::Pose3::Identity(),
                                           identity_motion_model);
      }
    }
  }

  if (frame_id > 2) {
    const gtsam::Symbol H_key_km1 =
        ObjectMotionSymbol(object_id_, frame_id - 1u);
    const gtsam::Symbol H_key_km2 =
        ObjectMotionSymbol(object_id_, frame_id - 2u);

    // TODO: params
    gtsam::Vector6 sigmas;
    sigmas << 0.2, 0.2, 0.2, 0.1, 0.1, 0.1;
    gtsam::SharedNoiseModel smoothing_motion_model =
        gtsam::noiseModel::Isotropic::Sigmas(sigmas);

    // TODO: ALL motions should use the same L_KF_
    //  if L_KF_ is only updated when we reset internal ISAM then no problem!
    // if (isam_.valueExists(H_key_km1) && isam_.valueExists(H_key_km2)) {
    if (new_values.exists(H_key_km1) && new_values.exists(H_key_km2) &&
        new_values.exists(H_key_k)) {
      // auto smoothing_factor = boost::make_shared<HybridSmoothingFactor>(
      //     H_key_km2, H_key_km1, H_key_k, L_KF, smoothing_motion_model);
      auto smoothing_factor = boost::make_shared<HybridSmoothingFactor>(
          H_key_km2, H_key_km1, H_key_k, L_KF, smoothing_motion_model);

      new_factors += smoothing_factor;
      smoothing_factors_.insert2(frame_id, smoothing_factor);
    }
  }
  timer2.stop();

  LOG(INFO) << "Starting OPt";
  gtsam::LevenbergMarquardtParams opt_params;
  // for speed
  opt_params.setMaxIterations(4);

  // dyno::NonlinearOptimizer<gtsam::LevenbergMarquardtOptimizer> solver(
  //     new_factors, new_values, opt_params);

  // NonlinearOptimizerSummary summary;
  // NonlinearOptimizerOptions options;

  // gtsam::Values optimised_values = new_values;
  // {
  //   utils::ChronoTimingStats update_timer(logger_prefix_ + ".LM_solve", 2);
  //   CHECK(solver.solve(optimised_values, options, &summary));
  // }

  // VLOG(10) << "Initial error: " << summary.initial_error << " final error "
  //          << summary.final_error << " time[s] "
  //          << summary.cumulative_time_in_seconds
  //          << " #iterations= " << summary.numIterations();

  static CsvWriterToFile timing_logger(
      getOutputFilePath("motion_solve_LM_timing_compare.csv"),
      CsvHeader("type", "num_values", "num_factors", "timing[ms]"));

  auto logger_prefix = logger_prefix_;
  auto solveDense =
      [&opt_params, &logger_prefix](
          const gtsam::Values& values,
          const gtsam::NonlinearFactorGraph& graph) -> gtsam::Values {
    using BaseSolver = gtsam::LevenbergMarquardtOptimizer;
    using DenseSolver = DenseNonlinearSolver<BaseSolver>;

    dyno::NonlinearOptimizer<DenseSolver> solver(graph, values, opt_params);
    NonlinearOptimizerSummary summary;
    NonlinearOptimizerOptions options;

    gtsam::Values optimised_values = values;
    utils::ChronoTimingStats update_timer(logger_prefix + ".LM_solve_dense", 2);
    CHECK(solver.solve(optimised_values, options, &summary));

    auto timing_ms = update_timer.stop();

    timing_logger << "dense" << values.size() << graph.size() << timing_ms;
    return optimised_values;
  };

  auto solveSparse =
      [&opt_params, &logger_prefix](
          const gtsam::Values& values,
          const gtsam::NonlinearFactorGraph& graph) -> gtsam::Values {
    // solving with GN defs faster ;)
    using BaseSolver = gtsam::LevenbergMarquardtOptimizer;

    dyno::NonlinearOptimizer<BaseSolver> solver(graph, values, opt_params);
    NonlinearOptimizerSummary summary;
    NonlinearOptimizerOptions options;

    gtsam::Values optimised_values = values;
    utils::ChronoTimingStats update_timer(logger_prefix + ".LM_solve_sparse",
                                          2);
    CHECK(solver.solve(optimised_values, options, &summary));

    auto timing_ms = update_timer.stop();

    timing_logger << "sparse" << values.size() << graph.size() << timing_ms;
    return optimised_values;
  };

  auto optimised_values = solveSparse(new_values, new_factors);

  HybridObjectMotionSmoother::Result result;
  result.solver_okay = true;
  // dyno::ISAM2UpdateParams update_params;
  // update_params.newAffectedKeys = std::move(newly_affected_keys);
  // update_params.removeFactorIndices = std::move(factors_to_delete);

  // LOG(INFO) << "Starting update";
  // HybridObjectMotionSmoother::Result result =
  //     this->updateSmoother(new_factors, new_values, timestamps,
  //     update_params,
  //                          additional_keys_to_marginalize);

  // if (!result.solver_okay) {
  //   return result;
  // }

  // const auto& isam_result = result.isam_result;
  // const gtsam::KeySet& unused_keys = isam_result.unusedKeys;
  // // if we've done out bookeeping right then each point that has become a
  // // batch factor will have had all its factors removed and therefore will be
  // // implicitly deleted by isam2 (since no other factors mark it)
  // // we can sanity check this by checking that all new batch factor points
  // // are marked as unused and that the point is no longer in the state
  // for (auto key : keys_that_should_be_deleted) {
  //   CHECK(!isam_.valueExists(key));
  //   CHECK(unused_keys.exists(key));
  // }

  // std::stringstream ss;
  // for (auto key : isam_result.markedKeys) {
  //   ss << DynosamKeyFormatter(key) << " ";
  // }
  // LOG(INFO) << "Marked keys: " << ss.str();
  //  ss.clear();

  // //  std::stringstream ss;
  //  for(auto key : isam_result.observedKeys) {
  //   ss << DynosamKeyFormatter(key) << " ";
  //  }
  //  LOG(INFO) << "Observed keys: " << ss.str();

  // auto active_motion_estimates = optimised_values.extract<gtsam::Pose3>(
  //     gtsam::Symbol::ChrTest(kObjectMotionSymbolChar));
  // for (const auto& [key, value] : active_motion_estimates) {
  //   smoother_state.insert(key, value);
  // }
  // smoother_state = new_values;

  utils::ChronoTimingStats timer3("hm_smoother.update.recover", 10);
  smoother_state = optimised_values;
  for (auto& [tracklet_id, point_state_pair] : point_state_) {
    const gtsam::Symbol m_key(PointSymbol(tracklet_id));
    const PointState& point_state = point_state_pair.first;
    // may be in point state but never made it to estimator! TODO: this is not
    // correct logic!
    if (smoother_state.exists(m_key))
      point_state_pair.second = smoother_state.at<gtsam::Point3>(m_key);
  }
  timer3.stop();
  return result;
}

// HybridObjectMotionOnlySmoother::Result
// HybridObjectMotionOnlySmoother::updateFromInitialMotionImpl(
//     gtsam::Values& smoother_state, const gtsam::Pose3& H_W_KF_k_initial,
//     Frame::Ptr frame, const TrackletIds& tracklets) {
//   const auto frame_id = frameId();
//   const auto timestamp = this->timestamp();

//   const double frame_as_double = static_cast<double>(frame_id);

//   const gtsam::Symbol H_key_k = ObjectMotionSymbol(object_id_, frame_id);
//   // fixed camera pose
//   const gtsam::Pose3 X_W_k = frame->getPose();
//   // get current keyframe pose
//   const gtsam::Pose3 L_KF = keyFramePose();

//   const gtsam::FastMap<TrackletId, gtsam::Point3> all_object_points =
//       this->getObjectPoints();

//   gtsam::Values new_values;
//   gtsam::NonlinearFactorGraph new_factors;
//   gtsam::FastMap<gtsam::FactorIndex, gtsam::KeySet> newly_affected_keys;

//   KeyTimestampMap timestamps;
//   // add motions
//   // TODO: bug when we marginalize!!!!
//   // Asking to remove variables from the variable index that are not unused
//   // also somehow related to running multiple isam update iterations
//   timestamps[H_key_k] = frame_as_double;

//   new_values.insert(H_key_k, H_W_KF_k_initial);

//   PoseChangeUpdateComplete backend_update;
//   if (takeBackendUpdate(backend_update)) {
//     const PoseChangeUpdateComplete::Object& object_update =
//         backend_update.objects.at(object_id_);
//     const auto& optimized_trajectory = object_update.trajectory;
//     const auto& optimized_camera_trajectory =
//     backend_update.camera.trajectory;

//     // TODO: actually trajectory up to current kf
//     // better name is "frozen" trajectory and maybe active trajectory (ie.
//     // local)
//     //  const auto& trajectory_upto_lKF = frozen_trajectory_;

//     // in the case that lkast_okf  < current keyframe
//     // we dont have any optimized estimates that current overlap with the
//     // active
//     // optimisation but we can still use the latest pose update the keyframe
//     // pose

//     LOG(WARNING) << "j=" << object_id_ << " has update k=" << frame_id
//                  << " current okf=" << keyFrameId();

//     gtsam::Pose3 L_KF_updated;
//     FrameId last_okf_optimized = optimized_trajectory.maxFrame();

//     // update all current fractors current in estimate with new camera pose
//     // this is basically like doing batch every step since we relinearize all
//     // but this is just for testing to see how much of an issue this makes
//     // TODO: only do if we actually have updates! (or maybe significant
//     // updates!)
//     // for (auto& [_, factors] : structured_factors_) {
//     //   for (auto& m_factor : factors) {
//     //     CHECK(m_factor);
//     //     const gtsam::Key H_key_k = m_factor->key1();
//     //     LOG(INFO) << DynosamKeyFormatter(H_key_k);

//     //     ObjectId recovered_object_id;
//     //     FrameId recovered_frame_id;
//     //     CHECK(reconstructMotionInfo(H_key_k, recovered_object_id,
//     //                                 recovered_frame_id));
//     //     CHECK(optimized_camera_trajectory.exists(recovered_frame_id));

//     //     // // assume that the factor will get re-lineairized but
//     //     m_factor->cameraPose(
//     //         optimized_camera_trajectory.at(recovered_frame_id));

//     //     // gtsam::FactorIndex slot;
//     //     // CHECK(smoother_interface_.safeGetFactorIndex(m_factor, slot));
//     //     // newly_affected_keys[static_cast<gtsam::FactorIndex>(slot)] =
//     //     // {H_key_k, m_factor->key2()};
//     //   }
//     // }

//     // for (auto& [_, factor] : batch_factor_map_) {
//     //   CHECK(factor);
//     //   gtsam::FactorIndex slot;
//     //   CHECK(smoother_interface_.safeGetFactorIndex(factor, slot));

//     //   // gtsam::KeySet affected_keys(factor->keys().begin(),
//     //   // factor->keys().end());
//     //   // newly_affected_keys[static_cast<gtsam::FactorIndex>(slot)] =
//     //   // affected_keys;

//     //   for (auto& m_factor : factor->factors_) {
//     //     const gtsam::Key H_key_k = m_factor.key1();
//     //     LOG(INFO) << DynosamKeyFormatter(H_key_k);

//     //     ObjectId recovered_object_id;
//     //     FrameId recovered_frame_id;
//     //     CHECK(reconstructMotionInfo(H_key_k, recovered_object_id,
//     //                                 recovered_frame_id));
//     //     CHECK(optimized_camera_trajectory.exists(recovered_frame_id));

//     //     // assume that the factor will get re-lineairized but
//     //
//     m_factor.cameraPose(optimized_camera_trajectory.at(recovered_frame_id));
//     //   }
//     // }
//   }
//   const gtsam::NonlinearFactorGraph& factors_in_smoother = getFactors();

//   gtsam::SharedNoiseModel stereo_noise_model =
//       gtsam::noiseModel::Isotropic::Sigma(3u, 2.0);
//   stereo_noise_model =
//       factor_graph_tools::robustifyHuber(0.001, stereo_noise_model);

//   auto stereo_non_robust_noise_model =
//       gtsam::noiseModel::Isotropic::Sigma(3u, 2.0);

//   // for debug stats
//   size_t num_tracks_used = 0;
//   size_t avg_feature_age = 0;

//   size_t points_in_previous_kf = 0;
//   size_t num_new_points = 0;
//   // object_motion_to_tracklets_.insert2(H_key_k, TrackletIds{});

//   gtsam::FactorIndices factors_to_delete;
//   gtsam::KeyVector keys_that_should_be_deleted;
//   gtsam::KeyVector additional_keys_to_marginalize;

//   // const bool close_to_keyframe = frameId() <= (keyFrameId() + 2);
//   const FrameId tracking_kf = lOKF_frame_->getFrameId();
//   //TODO: currently keyframeId() is the first frame in the active frame set
//   // ie the frame of L_KF which may not be lOKF_frame_
//   const bool close_to_keyframe = frameId() <= (tracking_kf + 2);

//   for (const TrackletId& tracklet_id : tracklets) {
//     Feature::Ptr feature = frame->at(tracklet_id);
//     CHECK(feature);

//     auto [stereo_keypoint_status, stereo_measurement] =
//         rgbd_camera_->getStereo(feature);
//     if (!stereo_keypoint_status) {
//       continue;
//     }

//     double disparity = stereo_measurement.uL() - stereo_measurement.uR();
//     if (disparity < 0.5) {
//       feature->markOutlier();
//       continue;
//     }

//     const gtsam::Symbol m_key(PointSymbol(tracklet_id));

//     auto& z_per_frame = stereo_measurements_[tracklet_id];
//     z_per_frame.insert2(frame_id, stereo_measurement);

//     // totally new point for this keyframe
//     // test only add new points for keyframe!
//     if (!point_state_.exists(tracklet_id)) {
//       // TODO: enforce real-time ;)
//       //  if we do this we seem to loose many many points!
//       // accept new keypoints only in or around a new keyframe
//       if (!close_to_keyframe) {
//         continue;
//       }

//       const gtsam::Point3 m_X_k = frame->backProjectToCamera(tracklet_id);
//       Landmark m_L_init = HybridObjectMotion::projectToObject3(
//           X_W_k, H_W_KF_k_initial, L_KF, m_X_k);
//       point_state_.insert2(tracklet_id,
//                            std::make_pair(PointState::InState, m_L_init));

//       new_values.insert(m_key, m_L_init);

//       auto factor = boost::make_shared<StereoHybridMotionFactor2>(
//           stereo_measurement, L_KF, X_W_k, stereo_noise_model,
//           stereo_calibration_, H_key_k, m_key, false /*throw ceirality*/
//       );
//       num_new_points++;

//       structured_factors_.insert2(tracklet_id, {factor});
//       new_factors += factor;
//     } else {
//       // point does exist
//       const PointState point_state = point_state_.at(tracklet_id).first;
//       const Landmark m_L_point = point_state_.at(tracklet_id).second;
//       if (point_state == PointState::InState) {
//         CHECK(structured_factors_.exists(tracklet_id));
//         CHECK(isam_.valueExists(m_key));

//         std::vector<StereoHybridMotionFactor2::shared_ptr>&
//             measurement_factors = structured_factors_.at(tracklet_id);

//         // add the new factor for this measurement (so we capture the
//         // measurement at this frane) and ensure the variable is not unused
//         auto factor = boost::make_shared<StereoHybridMotionFactor2>(
//             stereo_measurement, L_KF, X_W_k, stereo_noise_model,
//             stereo_calibration_, H_key_k, m_key, false /*throw ceirality*/
//         );
//         // new_factors += factor;

//         measurement_factors.push_back(factor);

//         // convert to batch factor
//         // we have 2 observations so add the new one to the measurement
//         // factors so for the smart factors we have 3 in total (this will be
//         // added next) add the new factor so it is not unused and mark as
//         // marginalized!
//         if (measurement_factors.size() > 3) {
//           // TBH these match factors probably not so good becuas we
//           // re-linearize
//           // every time!
//           // auto batch_factor =
//           //     boost::make_shared<BatchStereoHybridMotionFactor3>(
//           //         m_L_point, L_KF, stereo_noise_model,
//           stereo_calibration_,
//           //         true /* use hessian factor*/);
//           auto smart_factor = boost::make_shared<StereoSmartFactor>(
//               L_KF, m_L_point, stereo_non_robust_noise_model,
//               stereo_calibration_);

//           // TODO: now that we actually try marginalisation
//           //  dont remove factors just marginalize point and all factors
//           //  on new measurements!
//           for (StereoHybridMotionFactor2::shared_ptr m_factor :
//                measurement_factors) {
//             const gtsam::StereoPoint2& measurement = m_factor->measured();
//             const gtsam::Key H_key_k = m_factor->key1();
//             const gtsam::Pose3 X_W_k = m_factor->cameraPose();

//             smart_factor->add(measurement, H_key_k, X_W_k);
//             // batch_factor->add(measurement, X_W_k, H_key_k);

//             // gtsam::FactorIndex slot;
//             // CHECK(smoother_interface_.safeGetFactorIndex(m_factor, slot));
//             // factors_to_delete.push_back(slot);
//           }
//           // new_factors += batch_factor;
//           // doubling up some information here!
//           // NOTE: this was needed to stop some awful bug when using isam2!
//           new_factors += smart_factor;
//           // batch_factor_map_.insert2(tracklet_id, batch_factor);
//           structureless_factors_.insert2(tracklet_id, {smart_factor});

//           // keys_that_should_be_deleted.push_back(m_key);
//           // just marginalize and dont add factor on old measurements
//           // store factor just as current way to recover measurements for new
//           // factors
//           additional_keys_to_marginalize.push_back(m_key);
//           // LOG(INFO) << "Marginalize " << tracklet_id;
//           point_state_.at(tracklet_id).first = PointState::Marginalized;

//           structured_factors_.erase(tracklet_id);
//         } else {
//           // auto factor = boost::make_shared<StereoHybridMotionFactor2>(
//           //     stereo_measurement, L_KF, X_W_k, stereo_noise_model,
//           //     stereo_calibration_, H_key_k, m_key, false /*throw
//           ceirality*/
//           // );

//           // measurement_factors.push_back(factor);
//           new_factors += factor;
//         }

//       }
//       // PointState::Marginalized
//       else {
//         CHECK(!isam_.valueExists(m_key));
//         // CHECK(batch_factor_map_.exists(tracklet_id));
//         CHECK(structureless_factors_.exists(tracklet_id));
//         CHECK(!structured_factors_.exists(tracklet_id));

//         auto last_smart_motion_factor =
//             structureless_factors_.at(tracklet_id).back();

//         auto last_camera_poses = last_smart_motion_factor->cameraPoses();
//         auto last_measurement_factors =
//             last_smart_motion_factor->measurementFactors();

//         // CHECK_EQ(last_camera_poses.size(), 3);
//         auto smart_factor = boost::make_shared<StereoSmartFactor>(
//             L_KF, m_L_point, stereo_non_robust_noise_model,
//             stereo_calibration_);

//         // connect last two frames with first frame
//         // TODO: double check we're not using the same frame for measurements
//         const auto& measurements_i = stereo_measurements_.at(tracklet_id);
//         auto it = measurements_i.begin();
//         FrameId frame_id_i = it->first;
//         const auto z_i = it->second;
//         // FrameId kf_id = keyFrameId();
//         gtsam::Pose3 X_W_i = getCameraPose(frame_id_i);
//         // CHECK_LT()
//         // CHECK(stereo_measurements_.at(tracklet_id).exists(kf_id));
//         // const auto z_kf = stereo_measurements_.at(tracklet_id).at(kf_id);

//         auto H_key_i = ObjectMotionSymbol(object_id_, frame_id_i);

//         // take last three poses
//         CHECK_EQ(last_camera_poses.size(), 4);

//         // smart_factor->add(z_i, H_key_i, X_W_i);
//         smart_factor->add(last_measurement_factors.at(1).measured(),
//                           last_smart_motion_factor->keys().at(1),
//                           last_camera_poses.at(1));
//         smart_factor->add(last_measurement_factors.at(2).measured(),
//                           last_smart_motion_factor->keys().at(2),
//                           last_camera_poses.at(2));
//         smart_factor->add(last_measurement_factors.at(3).measured(),
//                           last_smart_motion_factor->keys().at(3),
//                           last_camera_poses.at(3));
//         smart_factor->add(stereo_measurement, H_key_k, X_W_k);

//         structureless_factors_.at(tracklet_id).push_back(smart_factor);
//         new_factors += smart_factor;

//         // auto batch_factor = batch_factor_map_.at(tracklet_id);
//         // gtsam::FactorIndex current_slot;
//         // CHECK(
//         //     smoother_interface_.safeGetFactorIndex(batch_factor,
//         //     current_slot));

//         // newly_affected_keys.insert2(current_slot, {H_key_k});
//         // {
//         //   // test!
//         //   const auto factors_in_smoother = getFactors();
//         //   CHECK_LT(current_slot, factors_in_smoother.size());

//         //   auto factor_in_smoother = factors_in_smoother.at(current_slot);
//         //   CHECK_EQ(batch_factor, factor_in_smoother);
//         // }

//         // batch_factor->add(stereo_measurement, X_W_k, H_key_k);
//       }
//     }

//     num_tracks_used++;
//     avg_feature_age += feature->age();
//   }

//   if (num_tracks_used == 0) {
//     HybridObjectMotionSmoother::Result result;
//     result.solver_okay = false;
//     return result;
//   }

//   avg_feature_age /= num_tracks_used;

//   LOG(INFO) << "Num tracks used: " << num_tracks_used << " num_new_points "
//             << num_new_points;
//   LOG(INFO) << "Num new factors: " << new_factors.size();

//   // we must also remove all tracks that are lost (if in state, remove all
//   // factors that touch these points) so that we dont include information
//   // arguably these factors should probably never have been added but oh well
//   auto points_in_state =
//       getObjectPointsFromState(smoother_interface_.getLinearizationPoint());
//   for (const auto& [key, _] : points_in_state) {
//     gtsam::Symbol sym(key);
//     TrackletId tracklet_id = static_cast<TrackletId>(sym.index());
//     if (!frame->exists(tracklet_id)) {
//       // std::stringstream ss;
//       // ss << "Tracklet dropped: " << tracklet_id << " seen at k= ";
//       // for(const auto& [frame_id, _] :
//       stereo_measurements_.at(tracklet_id)) {
//       //   ss << frame_id << " ";
//       // }
//       // LOG(INFO) << ss.str();
//       // assume tracklet dropped and therefore remove factors
//       CHECK(point_state_.exists(tracklet_id));
//       CHECK_EQ(point_state_.at(tracklet_id).first, PointState::InState);
//       CHECK(structured_factors_.exists(tracklet_id));

//       // current implementation is such that the last measurement factor
//       // wont actually be in the graph? If we make it to marginalizing out
//       the
//       // pont?
//       const std::vector<StereoHybridMotionFactor2::shared_ptr>&
//           measurement_factors = structured_factors_.at(tracklet_id);
//       for (StereoHybridMotionFactor2::shared_ptr m_factor :
//            measurement_factors) {
//         gtsam::FactorIndex slot;
//         CHECK(smoother_interface_.safeGetFactorIndex(m_factor, slot));
//         factors_to_delete.push_back(slot);
//       }
//       structured_factors_.erase(tracklet_id);
//       keys_that_should_be_deleted.push_back(key);
//       // assume we will never see this point again as tracklets are not
//       // reassociated and id's only increase
//       point_state_.erase(tracklet_id);
//     }
//   }

//   if (frameId() == keyFrameId()) {
//     gtsam::SharedNoiseModel identity_motion_model =
//         gtsam::noiseModel::Isotropic::Sigma(6u, 0.00001);

//     new_factors.addPrior<gtsam::Pose3>(H_key_k, gtsam::Pose3::Identity(),
//                                        identity_motion_model);
//   }

//   if (frame_id > 2) {
//     const gtsam::Symbol H_key_km1 =
//         ObjectMotionSymbol(object_id_, frame_id - 1u);
//     const gtsam::Symbol H_key_km2 =
//         ObjectMotionSymbol(object_id_, frame_id - 2u);

//     // TODO: params
//     gtsam::Vector6 sigmas;
//     sigmas << 0.2, 0.2, 0.2, 0.1, 0.1, 0.1;
//     gtsam::SharedNoiseModel smoothing_motion_model =
//         gtsam::noiseModel::Isotropic::Sigmas(sigmas);

//     // TODO: ALL motions should use the same L_KF_
//     //  if L_KF_ is only updated when we reset internal ISAM then no problem!
//     if (isam_.valueExists(H_key_km1) && isam_.valueExists(H_key_km2)) {
//       auto smoothing_factor = boost::make_shared<HybridSmoothingFactor>(
//           H_key_km2, H_key_km1, H_key_k, L_KF, smoothing_motion_model);

//       new_factors += smoothing_factor;
//       smoothing_factors_.insert2(frame_id, smoothing_factor);
//     }
//   }

//   dyno::ISAM2UpdateParams update_params;
//   update_params.newAffectedKeys = std::move(newly_affected_keys);
//   update_params.removeFactorIndices = std::move(factors_to_delete);

//   LOG(INFO) << "Starting update";
//   HybridObjectMotionSmoother::Result result =
//       this->updateSmoother(new_factors, new_values, timestamps,
//       update_params,
//                            additional_keys_to_marginalize);

//   if (!result.solver_okay) {
//     return result;
//   }

//   const auto& isam_result = result.isam_result;
//   const gtsam::KeySet& unused_keys = isam_result.unusedKeys;
//   // if we've done out bookeeping right then each point that has become a
//   // batch factor will have had all its factors removed and therefore will be
//   // implicitly deleted by isam2 (since no other factors mark it)
//   // we can sanity check this by checking that all new batch factor points
//   // are marked as unused and that the point is no longer in the state
//   for (auto key : keys_that_should_be_deleted) {
//     CHECK(!isam_.valueExists(key));
//     CHECK(unused_keys.exists(key));
//   }

//   // std::stringstream ss;
//   // for (auto key : isam_result.markedKeys) {
//   //   ss << DynosamKeyFormatter(key) << " ";
//   // }
//   // LOG(INFO) << "Marked keys: " << ss.str();
//   //  ss.clear();

//   // //  std::stringstream ss;
//   //  for(auto key : isam_result.observedKeys) {
//   //   ss << DynosamKeyFormatter(key) << " ";
//   //  }
//   //  LOG(INFO) << "Observed keys: " << ss.str();

//   // smoother_state = calculateEstimate();
//   // for (auto& [tracklet_id, point_state_pair] : point_state_) {
//   //   const gtsam::Symbol m_key(PointSymbol(tracklet_id));
//   //   const PointState& point_state = point_state_pair.first;

//   //   if (point_state == PointState::InState) {
//   //     CHECK(smoother_state.exists(m_key));
//   //     // update variable
//   //     point_state_pair.second = smoother_state.at<gtsam::Point3>(m_key);
//   //   } else {
//   //     // removed
//   //     CHECK(!smoother_state.exists(m_key));
//   //     const Landmark& m_L_point = point_state_pair.second;
//   //     smoother_state.insert(m_key, m_L_point);
//   //   }
//   // }

//   // only add points once they are marginalized?
//   // this means the backend will only get initial point estimates once
//   // they are refined
//   // and by using the m_l_init value in the backend we wait for these initial
//   // points even though the map is getting measurements filled make sure that
//   // the point is seen across enough keyframes otherwise it will enever end
//   // up
//   // int eh map
//   // smoother_state = calculateEstimate();
//   gtsam::Values active_state = calculateEstimate();

//   auto active_motion_estimates = active_state.extract<gtsam::Pose3>(
//       gtsam::Symbol::ChrTest(kObjectMotionSymbolChar));
//   for (const auto& [key, value] : active_motion_estimates) {
//     smoother_state.insert(key, value);
//   }

//   // in this case we may NEVER add points if we keyframe often
//   // ie more often than required to refine a point!
//   for (auto& [tracklet_id, point_state_pair] : point_state_) {
//     const gtsam::Symbol m_key(PointSymbol(tracklet_id));
//     const PointState& point_state = point_state_pair.first;

//     if (point_state == PointState::InState) {
//       // CHECK(smoother_state.exists(m_key));
//       CHECK(active_state.exists(m_key));
//       // update variable
//       point_state_pair.second = active_state.at<gtsam::Point3>(m_key);
//     } else {
//       // removed
//       CHECK(!smoother_state.exists(m_key));
//       const Landmark& m_L_point = point_state_pair.second;
//       // add points once they are removed
//       smoother_state.insert(m_key, m_L_point);
//     }
//   }

//   // TODO: debug
//   return result;
// }

gtsam::Pose3 HybridObjectMotionOnlySmoother::keyFrameMotionImpl(
    FrameId frame_id, const gtsam::Values& values) const {
  const gtsam::Symbol H_key_k = ObjectMotionSymbol(object_id_, frame_id);
  CHECK(values.exists(H_key_k));

  return values.at<gtsam::Pose3>(H_key_k);
}

void HybridObjectMotionOnlySmoother::onNewKeyFrameMotion(
    const dyno::ISAM2& smoother_before_reset, const gtsam::Pose3 new_L_KF) {
  // mo_factor_map_.clear();
  // mo_factor_to_tracklet_id_.clear();
  // trackletid_to_frame_ids_.clear();
  // object_motion_to_tracklets_.clear();

  // batch_factor_map_.clear();
  structureless_factors_.clear();

  // awaiting_measurements_.clear();
  // m_L_points_.clear();

  // isam is cleared so we need to clear the structured factors
  smoothing_factors_.clear();
  structured_factors_.clear();
  point_state_.clear();
}

HybridObjectMotionSmartSmoother::Result
HybridObjectMotionSmartSmoother::updateFromInitialMotionImpl(
    gtsam::Values& smoother_state, const gtsam::Pose3& H_W_KF_k_initial,
    Frame::Ptr frame, const TrackletIds& tracklets) {
  const auto frame_id = frameId();
  const double frame_as_double = static_cast<double>(frame_id);

  const gtsam::Symbol H_key_k = ObjectMotionSymbol(object_id_, frame_id);
  // fixed camera pose
  const gtsam::Pose3 X_W_k = frame->getPose();
  // get current keyframe pose
  const gtsam::Pose3 L_KF = keyFramePose();

  gtsam::Values new_values;
  gtsam::NonlinearFactorGraph new_factors;

  KeyTimestampMap timestamps;
  // add motions
  timestamps[H_key_k] = frame_as_double;

  const gtsam::Pose3 G_W = X_W_k.inverse() * H_W_KF_k_initial * L_KF;
  new_values.insert(H_key_k, G_W.inverse());

  gtsam::SharedNoiseModel stereo_noise_model =
      gtsam::noiseModel::Isotropic::Sigma(3u, 2.0);
  // stereo_noise_model =
  //     factor_graph_tools::robustifyHuber(0.01, stereo_noise_model);
  CHECK(stereo_noise_model);

  // for debug stats
  size_t num_tracks_used = 0;
  size_t avg_feature_age = 0;

  gtsam::FastMap<gtsam::FactorIndex, gtsam::KeySet> newly_affected_keys;

  for (const TrackletId& tracklet_id : tracklets) {
    const Feature::Ptr feature = frame->at(tracklet_id);
    CHECK(feature);

    const gtsam::Symbol m_key(PointSymbol(tracklet_id));

    // bool is_new = false;
    // if variable is removed (ie due to marginalization)!
    // this is re-initalizing it!!! Is this what we want
    // to do!!?
    // if (!isam_.valueExists(m_key)) {
    //   const gtsam::Point3 m_X_k = frame->backProjectToCamera(tracklet_id);
    //   Landmark m_L_init = HybridObjectMotion::projectToObject3(
    //       X_W_k, H_W_KF_k_initial, L_KF, m_X_k);

    //   new_values.insert(m_key, m_L_init);
    //   is_new = true;
    // }
    const auto [stereo_keypoint_status, stereo_measurement] =
        rgbd_camera_->getStereo(feature);
    if (!stereo_keypoint_status) {
      continue;
    }

    if (!factor_map_.exists(tracklet_id)) {
      const gtsam::Point3 m_X_k = frame->backProjectToCamera(tracklet_id);
      Landmark m_L_init = HybridObjectMotion::projectToObject3(
          X_W_k, H_W_KF_k_initial, L_KF, m_X_k);

      gtsam::SmartStereoProjectionParams smart_factor_params;
      // for some reason JACOBIAN_SVD more stable (we get const char*
      // exception!?) smart_factor_params.linearizationMode =
      //   gtsam::LinearizationMode::JACOBIAN_SVD;
      smart_factor_params.setLinearizationMode(gtsam::HESSIAN);
      smart_factor_params.setDegeneracyMode(gtsam::ZERO_ON_DEGENERACY);
      smart_factor_params.setDynamicOutlierRejectionThreshold(8.0);
      smart_factor_params.setRetriangulationThreshold(1.0e-3);

      // totalReprojectionError
      auto factor = boost::make_shared<gtsam::SmartStereoProjectionPoseFactor>(
          stereo_noise_model, smart_factor_params);
      // hacky way to force the result to get set!
      // factor->totalReprojectionError({}, m_L_init);
      // factor->point()

      // must happen before updating new factors
      const Slot starting_slot = new_factors.size();
      factor_map_.insert2(tracklet_id, std::make_pair(factor, starting_slot));
      factor_to_tracklet_id_.insert2(factor, tracklet_id);
      // newly_affected_keys.insert2(starting_slot, {H_key_k});

      // only add when new factors!
      new_factors += factor;
    } else {
      Slot current_slot = factor_map_.at(tracklet_id).second;
      // factor_map_.at(tracklet_id).first->print("SS", DynosamKeyFormatter);
      newly_affected_keys.insert2(static_cast<gtsam::FactorIndex>(current_slot),
                                  {H_key_k});

      {
        // test!
        const auto factors_in_smoother = getFactors();
        CHECK_LT(current_slot, factors_in_smoother.size());

        auto this_factor = factor_map_.at(tracklet_id).first;
        auto factor_in_smoother = factors_in_smoother.at(current_slot);
        CHECK(this_factor->equals(*factor_in_smoother));
      }
    }

    auto factor = factor_map_.at(tracklet_id).first;
    factor->add(stereo_measurement, H_key_k, stereo_calibration_);

    num_tracks_used++;
    avg_feature_age += feature->age();

    // timestamps[m_key] = frame_as_double;

    // auto factor = boost::make_shared<StereoHybridMotionFactor2>(
    //     stereo_measurement, L_KF, X_W_k, stereo_noise_model,
    //     stereo_calibration_, H_key_k, m_key, true /*throw ceirality*/
    // );
    // CHECK(factor);
  }

  if (num_tracks_used == 0) {
    HybridObjectMotionSmoother::Result result;
    result.solver_okay = false;
    return result;
  }

  avg_feature_age /= num_tracks_used;

  if (frameId() == keyFrameId()) {
    gtsam::SharedNoiseModel identity_motion_model =
        gtsam::noiseModel::Isotropic::Sigma(6u, 0.0001);

    // TODO: add prior on this first motion to make it identity!
    // H_W is identity on the first motion so equation reduces
    const gtsam::Pose3 G_W_I = X_W_k.inverse() * L_KF;
    new_factors.addPrior<gtsam::Pose3>(H_key_k, G_W_I.inverse(),
                                       identity_motion_model);
  }

  if (frame_id > 2) {
    const gtsam::Symbol H_key_km1 =
        ObjectMotionSymbol(object_id_, frame_id - 1u);
    const gtsam::Symbol H_key_km2 =
        ObjectMotionSymbol(object_id_, frame_id - 2u);

    // TODO: params
    gtsam::SharedNoiseModel smoothing_motion_model =
        gtsam::noiseModel::Isotropic::Sigma(6u, 0.1);

    // TODO: ALL motions should use the same L_KF_
    //  if L_KF_ is only updated when we reset internal ISAM then no problem!
    // smoother is different now as we estimate for G_w!
    // if (isam_.valueExists(H_key_km1) && isam_.valueExists(H_key_km2)) {
    //   VLOG(10) << "Adding smoothing factor "
    //            << info_string(frame_id, object_id_);
    //   new_factors.emplace_shared<HybridSmoothingFactor>(
    //       H_key_km2, H_key_km1, H_key_k, L_KF, smoothing_motion_model);
    // }
  }

  dyno::ISAM2UpdateParams update_params;
  update_params.newAffectedKeys = std::move(newly_affected_keys);

  HybridObjectMotionSmoother::Result result = this->updateSmoother(
      new_factors, new_values, KeyTimestampMap{}, update_params);

  if (!result.solver_okay) {
    return result;
  }

  // update smoother slots
  const auto& isam_result = result.isam_result;
  const gtsam::FactorIndices& new_factor_indicies =
      isam_result.newFactorsIndices;

  if (isam_result.errorBefore && isam_result.errorAfter) {
    LOG(INFO) << "ISAM error - before: " << isam_result.getErrorBefore()
              << " after: " << isam_result.getErrorAfter();
  }

  const auto factors_in_smoother = getFactors();
  for (size_t i = 0; i < new_factors.size(); i++) {
    gtsam::FactorIndex new_index = new_factor_indicies.at(i);
    auto nonlinear_factor = new_factors.at(i);
    CHECK_EQ(nonlinear_factor, factors_in_smoother.at(new_index));

    auto smart_factor =
        boost::dynamic_pointer_cast<gtsam::SmartStereoProjectionPoseFactor>(
            nonlinear_factor);
    if (smart_factor) {
      CHECK(factor_to_tracklet_id_.exists(smart_factor));
      TrackletId tracklet_for_factor = factor_to_tracklet_id_.at(smart_factor);

      // update slot!
      factor_map_.at(tracklet_for_factor).second = static_cast<Slot>(new_index);

      CHECK_EQ(factor_map_.at(tracklet_for_factor).first, smart_factor);
    }
  }

  // deal with keys that were marginalized
  const gtsam::KeyVector& marginalized_keys = result.marginalized_keys;
  const gtsam::KeyVector& keys_in_smoother = getLinearizationPoint().keys();

  // should be calculateEstimate!
  // smoother_state_.insert_or_assign(new_values);
  smoother_state = calculateEstimate();

  // HACK! (fill smoother states with points!!!)
  for (const auto& [tracklet_id, factor_pair] : factor_map_) {
    const gtsam::Symbol m_key(PointSymbol(tracklet_id));

    const Slot slot = factor_pair.second;
    const auto smart_factor = factor_pair.first;

    gtsam::TriangulationResult triangulation_result = smart_factor->point();

    if (triangulation_result) {
      smoother_state.insert(m_key, *triangulation_result);
    }
  }

  DebugResult debug_result;
  debug_result.result = result;

  // TODO: debug flag
  // debug_result.smoother_stats.fill(&isam_);
  debug_result.object_id = object_id_;
  debug_result.frame_id = frame_id;
  debug_result.timestamp = timestamp();
  debug_result.frame_id_KF = keyFrameId();

  debug_result.num_tracks = num_tracks_used;
  debug_result.average_feature_age = avg_feature_age;

  debug_result.num_landmarks_in_smoother =
      getObjectPointsFromSmootherState().size();
  debug_result.num_motions_in_smoother =
      getObjectMotionsFromSmootherState().size();

  debug_results_.push_back(std::move(debug_result));

  return result;
}

gtsam::Pose3 HybridObjectMotionSmartSmoother::keyFrameMotionImpl(
    FrameId frame_id, const gtsam::Values& values) const {
  const gtsam::Symbol H_key_k = ObjectMotionSymbol(object_id_, frame_id);
  CHECK(values.exists(H_key_k));

  auto kf_data = keyframe_range_.find(frame_id);
  const auto [_, L_W_KF] = *kf_data;

  const gtsam::Pose3 G_W_KF_k = values.at<gtsam::Pose3>(H_key_k).inverse();
  const gtsam::Pose3 H_W_KF_k =
      getCameraPose(frame_id) * G_W_KF_k * L_W_KF.inverse();
  return H_W_KF_k;
}

void HybridObjectMotionSmartSmoother::onNewKeyFrameMotion(
    const dyno::ISAM2& smoother_before_reset, const gtsam::Pose3 new_L_KF) {
  factor_map_.clear();
  factor_to_tracklet_id_.clear();
}

HybridObjectMotionFullSmoother::Result
HybridObjectMotionFullSmoother::updateFromInitialMotionImpl(
    gtsam::Values& smoother_state, const gtsam::Pose3& H_W_KF_k_initial,
    Frame::Ptr frame, const TrackletIds& tracklets) {
  const auto frame_id = frameId();
  const double frame_as_double = static_cast<double>(frame_id);

  const gtsam::Symbol H_key_k = ObjectMotionSymbol(object_id_, frame_id);
  // fixed camera pose
  const gtsam::Pose3 X_W_k = frame->getPose();
  // get current keyframe pose
  const gtsam::Pose3 L_KF = keyFramePose();

  gtsam::Values new_values;
  gtsam::NonlinearFactorGraph new_factors;

  // is keyframe this frame
  const bool is_keyframe = frame_id == keyFrameId();
  // was keyframe one frame ago
  const bool is_keyframe2 = frame_id - 1u == keyFrameId();
  const bool try_cross_KF_smoothing =
      frame_id > 2 && (is_keyframe || is_keyframe2);
  // for now lets just warm start the smoother with the last
  // hack to way to determine if KF
  // should also add on frame_id - 1 == keyframe id
  if (false) {
    const FrameId frame_km1 = frame_id - 1u;
    const FrameId frame_km2 = frame_id - 2u;

    const gtsam::Symbol H_key_km1(ObjectMotionSymbol(object_id_, frame_km1));
    const gtsam::Symbol H_key_km2(ObjectMotionSymbol(object_id_, frame_km2));

    if (all_states_.exists(H_key_km1) && all_states_.exists(H_key_km2)) {
      LOG(INFO) << "On KF - found previous two motion states";

      // All states should be updated to contain the latest state estimate
      gtsam::Pose3 H_W_KF_km2 = all_states_.at<gtsam::Pose3>(H_key_km2);
      gtsam::Pose3 H_W_KF_km1 = all_states_.at<gtsam::Pose3>(H_key_km1);

      gtsam::SharedNoiseModel motion_prior =
          gtsam::noiseModel::Isotropic::Sigma(6u, 0.001);

      // get keyframe pose for previous motions and it SHOULD be different
      auto kf_data_km2 = keyframe_range_.find(frame_km2);
      CHECK(kf_data_km2);
      const auto [KF_km2, LKF_km2] = *kf_data_km2;

      auto kf_data_km1 = keyframe_range_.find(frame_km1);
      CHECK(kf_data_km1);
      const auto [KF_km1, LKF_km1] = *kf_data_km1;

      LOG(INFO) << "KF km2 " << KF_km2;
      LOG(INFO) << "KF km1 " << KF_km1;

      // // previous motions should come from the same keyframe pose (I guess,
      // // unless somehow tracking bad!?)
      // CHECK_EQ(KF_km2, KF_km1);

      // TODO: params
      gtsam::SharedNoiseModel smoothing_motion_model =
          gtsam::noiseModel::Isotropic::Sigma(6u, 0.1);

      auto smoothing_factor = boost::make_shared<HybridSmoothingFactor2>(
          H_key_km2, H_key_km1, H_key_k, LKF_km2, LKF_km1, L_KF,
          smoothing_motion_model);
      new_factors += smoothing_factor;

      // only the latest motion will be in the smoother
      // therefore add both previous motions so we can connect to them!
      if (is_keyframe) {
        CHECK(!isam_.valueExists(H_key_km2));
        CHECK(!isam_.valueExists(H_key_km1));

        new_values.insert(H_key_km1, H_W_KF_km1);
        new_values.insert(H_key_km2, H_W_KF_km2);

        //  add prior or marginal covariance?
        // These shouldn't really change though....
        new_factors.addPrior<gtsam::Pose3>(H_key_km2, H_W_KF_km2, motion_prior);

        new_factors.addPrior<gtsam::Pose3>(H_key_km1, H_W_KF_km1, motion_prior);
      }
    }
  }

  KeyTimestampMap timestamps;
  // add motions
  timestamps[H_key_k] = frame_as_double;
  new_values.insert(H_key_k, H_W_KF_k_initial);

  gtsam::SharedNoiseModel stereo_noise_model =
      gtsam::noiseModel::Isotropic::Sigma(3u, 2.0);
  stereo_noise_model =
      factor_graph_tools::robustifyHuber(0.01, stereo_noise_model);
  CHECK(stereo_noise_model);

  // for debug stats
  size_t num_tracks_used = 0;
  size_t avg_feature_age = 0;

  for (const TrackletId& tracklet_id : tracklets) {
    const Feature::Ptr feature = frame->at(tracklet_id);
    CHECK(feature);

    const gtsam::Symbol m_key(PointSymbol(tracklet_id));

    auto [stereo_keypoint_status, stereo_measurement] =
        rgbd_camera_->getStereo(feature);

    if (!stereo_keypoint_status) {
      continue;
    }

    double disparity = stereo_measurement.uL() - stereo_measurement.uR();
    if (disparity < 0.5) {
      continue;
    }

    // stereo_measurement = utils::perturbWithNoise(stereo_measurement, 1.5);

    bool is_new = false;
    // if variable is removed (ie due to marginalization)!
    // this is re-initalizing it!!! Is this what we want
    // to do!!?
    if (!isam_.valueExists(m_key)) {
      const gtsam::Point3 m_X_k = frame->backProjectToCamera(tracklet_id);

      // gtsam::Point3 m_W_K_noisy = utils::perturbWithNoise(m_X_k, 0.1);

      Landmark m_L_init = HybridObjectMotion::projectToObject3(
          X_W_k, H_W_KF_k_initial, L_KF, m_X_k);

      new_values.insert(m_key, m_L_init);
      is_new = true;
    }

    num_tracks_used++;
    avg_feature_age += feature->age();

    timestamps[m_key] = frame_as_double;

    auto factor = boost::make_shared<StereoHybridMotionFactor2>(
        stereo_measurement, L_KF, X_W_k, stereo_noise_model,
        stereo_calibration_, H_key_k, m_key, true /*throw ceirality*/
    );

    CHECK(factor);

    new_factors += factor;
  }

  if (num_tracks_used == 0) {
    HybridObjectMotionSmoother::Result result;
    result.solver_okay = false;
    return result;
  }

  avg_feature_age /= num_tracks_used;

  if (frameId() == keyFrameId()) {
    gtsam::SharedNoiseModel identity_motion_model =
        gtsam::noiseModel::Isotropic::Sigma(6u, 0.01);

    // TODO: add prior on this first motion to make it identity!
    new_factors.addPrior<gtsam::Pose3>(H_key_k, gtsam::Pose3::Identity(),
                                       identity_motion_model);
  }

  const FrameId frame_km1 = frame_id - 1u;
  const FrameId frame_km2 = frame_id - 2u;
  // AH now when we add previous smoothing factors H_key_km1/H_key_km2 will
  // be in isam immediately. Therfore this smoother factor is wrong as the
  /// keyframe poses will be wrong!

  // only add this smoother variant if all the motions added were part of the
  // same keyframe range
  // TODO: better logic here as we could check which motions were added as part
  // of the KF range or bring the cross smoothing logic here too and use the
  // keyframe range lookup explicity rather than relying on frame subtraction
  // logic! need the frame_id >= 2 so the substraction does not result in a
  // large size_t value
  // if (false) {
  if (frame_id > 2 && frame_km2 >= keyFrameId()) {
    // sanity check that all motions use the same keyframe poses
    auto kf_data_km2 = keyframe_range_.find(frame_km2);
    CHECK(kf_data_km2);
    const auto [KF_km2, LKF_km2] = *kf_data_km2;
    auto kf_data_km1 = keyframe_range_.find(frame_km1);
    CHECK(kf_data_km1);
    const auto [KF_km1, LKF_km1] = *kf_data_km1;
    CHECK_EQ(KF_km1, keyFrameId());
    CHECK_EQ(KF_km2, keyFrameId());

    const gtsam::Symbol H_key_km1 = ObjectMotionSymbol(object_id_, frame_km1);
    const gtsam::Symbol H_key_km2 = ObjectMotionSymbol(object_id_, frame_km2);

    // TODO: params
    gtsam::SharedNoiseModel smoothing_motion_model =
        gtsam::noiseModel::Isotropic::Sigma(6u, 0.1);

    // TODO: ALL motions should use the same L_KF_
    //  if L_KF_ is only updated when we reset internal ISAM then no problem!
    if (isam_.valueExists(H_key_km1) && isam_.valueExists(H_key_km2)) {
      VLOG(10) << "Adding smoothing factor "
               << info_string(frame_id, object_id_);
      // new_factors.emplace_shared<HybridSmoothingFactor>(
      //     H_key_km2, H_key_km1, H_key_k, L_KF, smoothing_motion_model);
    }
  }

  auto result = this->updateSmoother(new_factors, new_values, timestamps,
                                     ISAM2UpdateParams{});

  smoother_state = calculateEstimate();

  DebugResult debug_result;
  debug_result.result = result;

  // TODO: debug flag
  debug_result.smoother_stats.fill(&isam_);
  debug_result.object_id = object_id_;
  debug_result.frame_id = frame_id;
  debug_result.timestamp = timestamp();
  debug_result.frame_id_KF = keyFrameId();

  debug_result.num_tracks = num_tracks_used;
  debug_result.average_feature_age = avg_feature_age;

  debug_result.num_landmarks_in_smoother =
      getObjectPointsFromSmootherState().size();
  debug_result.num_motions_in_smoother =
      getObjectMotionsFromSmootherState().size();

  debug_results_.push_back(std::move(debug_result));

  return result;
}

gtsam::Pose3 HybridObjectMotionFullSmoother::keyFrameMotionImpl(
    FrameId frame_id, const gtsam::Values& values) const {
  const gtsam::Symbol H_key_k = ObjectMotionSymbol(object_id_, frame_id);
  CHECK(values.exists(H_key_k));

  return values.at<gtsam::Pose3>(H_key_k);
}

void HybridObjectMotionFullSmoother::onNewKeyFrameMotion(
    const dyno::ISAM2& smoother_before_reset, const gtsam::Pose3 new_L_KF) {}

///// SmartSolver

// void to_json(json& j, const HybridObjectMotionSmoother::Result& result) {
//   j["marginalized_keys"] = result.marginalized_keys;
//   j["additional_keys_reeliminate"] = result.additional_keys_reeliminate;
//   j["update_time_ms"] = result.update_time_ms;
//   j["marginalize_time_ms"] = result.marginalize_time_ms;
//   j["isam_result"] = result.isam_result;
// }

// void to_json(json& j, const HybridObjectMotionSmoother::DebugResult& result)
// {
//   j["smoother_result"] = result.result;
//   j["smoother_stats"] = result.smoother_stats;
//   j["object_id"] = result.object_id;
//   j["frame_id"] = result.frame_id;
//   j["timestamp"] = result.timestamp;
//   j["average_feature_age"] = result.average_feature_age;
//   j["num_tracks"] = result.num_tracks;
//   j["frame_id_KF"] = result.frame_id_KF;
//   j["num_landmarks_in_smoother"] = result.num_landmarks_in_smoother;
//   j["num_motions_in_smoother"] = result.num_motions_in_smoother;
// }

}  // namespace dyno
