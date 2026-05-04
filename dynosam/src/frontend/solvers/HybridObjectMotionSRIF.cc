#include "dynosam/frontend/solvers/HybridObjectMotionSRIF.hpp"

#include "dynosam/factors/HybridFormulationFactors.hpp"
#include "dynosam_common/logger/Logger.hpp"
#include "dynosam_common/utils/TimingStats.hpp"

namespace dyno {

/**
 * @brief Compute the 2-norm condition number of the SRIF matrix R.
 *
 * R must be the upper-triangular matrix obtained after QR decomposition
 * of the *whitened* Jacobian stack:
 *
 *     H = Q R
 *
 * where H is already whitened by measurement covariance.
 *
 * The condition number is:
 *
 *     cond(R) = sigma_max / sigma_min
 *
 * where sigma are the singular values of R.
 *
 * Interpretation guidelines (double precision):
 *
 *   cond(R) < 1e3      → Very healthy system
 *   1e4 – 1e6          → Mild degeneracy
 *   1e7 – 1e9          → Serious warning
 *   > 1e10             → Numerically dangerous
 *
 * Why it matters:
 *   - Large cond(R) means the information ellipsoid is highly stretched.
 *   - Small singular values indicate weakly constrained directions.
 *   - cond(R) ≈ ∞ implies rank deficiency (unobservable directions).
 *
 * In SLAM / SRIF systems, large cond(R) often indicates:
 *   - Gauge freedom not fixed
 *   - Poor motion excitation (e.g., pure rotation)
 *   - Underconstrained object states
 *   - Marginalization corruption
 *   - Incorrect Jacobians
 *
 * Important:
 *   - R must come from whitened Jacobians.
 *   - cond(R) == cond(H)
 *   - Using SRIF avoids squaring the condition number
 *     (since cond(H^T H) = cond(H)^2).
 *
 * @param R Upper triangular SRIF matrix
 * @return Condition number (2-norm)
 */
double conditionNumberR(const Eigen::MatrixXd& R) {
  if (R.rows() == 0 || R.cols() == 0)
    throw std::invalid_argument("R must be non-empty.");

  // Compute singular values of R
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      R, Eigen::ComputeThinU | Eigen::ComputeThinV);

  const auto& singular_values = svd.singularValues();

  const double sigma_max = singular_values(0);
  const double sigma_min = singular_values(singular_values.size() - 1);

  // Handle rank-deficient case
  if (sigma_min <= std::numeric_limits<double>::epsilon())
    return std::numeric_limits<double>::infinity();

  return sigma_max / sigma_min;
}

HybridObjectMotionSRIF::HybridObjectMotionSRIF(
    const ObjectId object_id, const gtsam::Pose3& initial_state_H,
    const gtsam::Pose3& L_e, const FrameId& frame_id_KF,
    const Timestamp& timestamp_KF, const gtsam::Matrix66& initial_P,
    const gtsam::Matrix66& Q, const gtsam::Matrix33& R, Camera::Ptr camera,
    double huber_k)
    : object_id_(object_id),
      H_linearization_point_(initial_state_H),
      Q_(Q),
      R_noise_(R),
      R_inv_(R.inverse()),
      initial_P_(initial_P),
      huber_k_(huber_k),
      logger_(CsvHeader("timestamp", "frame_id", "residual", "initial_error",
                        "weighted_error", "new_features", "existing_tracks",
                        "cond_R")) {
  // set camera
  rgbd_camera_ = CHECK_NOTNULL(camera)->safeGetRGBDCamera();
  CHECK(rgbd_camera_);
  stereo_calibration_ = rgbd_camera_->getFakeStereoCalib();

  resetState(L_e, frame_id_KF, timestamp_KF);
}

HybridObjectMotionSRIF::~HybridObjectMotionSRIF() {
  const std::string file_out =
      "hybrid_srif_j_" + std::to_string(object_id_) + ".csv";
  OfstreamWrapper::WriteOutCsvWriter(logger_, file_out);
}

const gtsam::Pose3& HybridObjectMotionSRIF::getKeyFramePose() const {
  return L_KF_;
}
const gtsam::Pose3& HybridObjectMotionSRIF::lastCameraPose() const {
  return X_K_;
}
FrameId HybridObjectMotionSRIF::getKeyFrameId() const { return frame_id_KF_; }
FrameId HybridObjectMotionSRIF::getFrameId() const { return frame_id_; }

Timestamp HybridObjectMotionSRIF::getKeyFrameTimestamp() const {
  return timestamp_KF_;
}
Timestamp HybridObjectMotionSRIF::getTimestamp() const { return timestamp_; }

const gtsam::FastMap<TrackletId, gtsam::Point3>&
HybridObjectMotionSRIF::getCurrentLinearizedPoints() const {
  return m_linearized_;
}

gtsam::Pose3 HybridObjectMotionSRIF::getPose() const {
  return getKeyFramedMotion() * getKeyFramePose();
}

void HybridObjectMotionSRIF::predictAndUpdate(
    const gtsam::Pose3& H_w_km1_k_predict, Frame::Ptr frame,
    const TrackletIds& tracklets, const int num_irls_iterations) {
  utils::ChronoTimingStats timer("hybrid_object_motion_srif.predictAndUpdate");
  predict(H_w_km1_k_predict);

  const FrameId frame_id_before_update = frame_id_;
  update(frame, tracklets, num_irls_iterations);
  const FrameId frame_id_after_update = frame_id_;
}

/**
 * @brief Recovers the state perturbation delta_w by solving R * delta_w = d.
 */
gtsam::Vector6 HybridObjectMotionSRIF::getStatePerturbation() const {
  // R is upper triangular, so this is a fast back-substitution
  return R_info_.triangularView<Eigen::Upper>().solve(d_info_);
}

const gtsam::Pose3& HybridObjectMotionSRIF::getCurrentLinearization() const {
  return H_linearization_point_;
}

gtsam::Pose3 HybridObjectMotionSRIF::getKeyFramedMotion() const {
  return H_linearization_point_.retract(getStatePerturbation());
}

Motion3ReferenceFrame HybridObjectMotionSRIF::getKeyFramedMotionReference()
    const {
  return Motion3ReferenceFrame(
      getKeyFramedMotion(), Motion3ReferenceFrame::Style::KF,
      ReferenceFrame::GLOBAL, getKeyFrameId(), getFrameId());
}

gtsam::Pose3 HybridObjectMotionSRIF::getF2FMotion() const {
  gtsam::Pose3 H_W_e_k = H_linearization_point_.retract(getStatePerturbation());
  gtsam::Pose3 H_W_e_km1 = previous_H_;
  return H_W_e_k * H_W_e_km1.inverse();
}

gtsam::Matrix66 HybridObjectMotionSRIF::getCovariance() const {
  gtsam::Matrix66 Lambda = R_info_.transpose() * R_info_;
  return Lambda.inverse();
}

gtsam::Matrix66 HybridObjectMotionSRIF::getInformationMatrix() const {
  return R_info_.transpose() * R_info_;
}

void HybridObjectMotionSRIF::predict(const gtsam::Pose3& H_W_km1_k) {
  // 1. Get current mean state and covariance
  gtsam::Vector6 delta_w = getStatePerturbation();
  gtsam::Pose3 H_current_mean = H_linearization_point_.retract(delta_w);
  gtsam::Matrix66 P_current = getCovariance();  // Slow step

  // call before updating the previous_H_
  // previous motion -> assume constant!
  // gtsam::Pose3 H_W_km1_k = getF2FMotion();

  // gtsam::Pose3 L_km1 = previous_H_ * L_KF_;
  // gtsam::Pose3 L_k = H_current_mean * L_KF_;

  // //calculate motion relative to object
  // gtsam::Pose3 local_motion = L_km1.inverse() * L_k;
  // // forward predict pose using constant velocity model
  // gtsam::Pose3 predicted_pose = L_k * local_motion;
  // // convert pose back to motion
  // gtsam::Pose3 predicted_motion = predicted_pose * L_KF_.inverse();

  previous_H_ = H_current_mean;

  // 2. Perform EKF prediction (add process noise)
  // P_k = P_{k-1} + Q
  gtsam::Matrix66 P_predicted = P_current + Q_;

  // 3. Re-linearize: Set new linearization point to the current mean
  H_linearization_point_ = H_W_km1_k * H_current_mean;

  // 4. Recalculate R and d based on new P and new linearization point
  // The new perturbation is 0 relative to the new linearization point.
  d_info_ = gtsam::Vector6::Zero();
  gtsam::Matrix66 Lambda_predicted = P_predicted.inverse();
  R_info_ =
      Lambda_predicted.llt().matrixU();  // R_info_ = L^T where L*L^T = Lambda
}

HybridObjectMotionSRIFResult HybridObjectMotionSRIF::update(
    Frame::Ptr frame, const TrackletIds& tracklets,
    const int num_irls_iterations) {
  const size_t num_measurements = tracklets.size();
  const size_t total_rows = num_measurements * ZDim;

  const gtsam::Pose3& X_W_k = frame->getPose();
  const gtsam::Pose3 X_W_k_inv = X_W_k.inverse();
  X_K_ = X_W_k;
  frame_id_ = frame->getFrameId();
  timestamp_ = frame->getTimestamp();

  return HybridObjectMotionSRIFResult{};

  // 1. Calculate Jacobians (H) and Linearized Residuals (y_lin)
  // These are calculated ONCE at the linearization point and are fixed
  // for all IRLS iterations.
  gtsam::Matrix H_stacked = gtsam::Matrix ::Zero(total_rows, StateDim);
  gtsam::Vector y_linearized = gtsam::Vector::Zero(total_rows);

  const gtsam::Pose3 A = X_W_k_inv * H_linearization_point_;
  // const gtsam::Pose3 A = X_W_k_inv * H_linearization_point_;
  // G will project the point from the object frame into the camera frame
  const gtsam::Pose3 G_w = A * L_KF_;
  const gtsam::Matrix6 J_correction = -A.AdjointMap();

  gtsam::StereoCamera gtsam_stereo_camera(G_w.inverse(), stereo_calibration_);
  // whitened error
  gtsam::Vector initial_error = gtsam::Vector::Zero(num_measurements);
  gtsam::Vector re_weighted_error = gtsam::Vector::Zero(num_measurements);

  size_t num_new_features = 0u, num_tracked_features = 0u,
         num_good_measurements = 0u;
  double average_residual = 0;

  for (size_t i = 0; i < num_measurements; ++i) {
    const TrackletId& tracklet_id = tracklets.at(i);
    const Feature::Ptr feature = frame->at(tracklet_id);
    CHECK(feature);

    if (!m_linearized_.exists(tracklet_id)) {
      // this is initalising from a predicted motion (bad!) Should use previous
      // linearization (ie k-1)
      const gtsam::Point3 m_X_k = frame->backProjectToCamera(tracklet_id);
      Landmark m_init = HybridObjectMotion::projectToObject3(
          X_W_k, getKeyFramedMotion(), L_KF_, m_X_k);
      m_linearized_.insert2(tracklet_id, m_init);
      // VLOG(10) << "Initalising new point i=" << tracklet_id << " Le " <<
      // L_KF_;
      num_new_features++;
    } else {
      num_tracked_features++;
    }

    gtsam::Point3 m_L = m_linearized_.at(tracklet_id);

    const auto [stereo_keypoint_status, stereo_measurement] =
        rgbd_camera_->getStereo(feature);
    if (!stereo_keypoint_status) {
      continue;
    }

    const auto& z_obs = stereo_measurement;

    // const Point2& z_obs = feature->keypoint();
    gtsam::Matrix36 J_pi;  // 2x6
    gtsam::StereoPoint2 z_pred;
    try {
      // Project using the *linearization point*
      // z_pred = camera.project(m_L, J_pi);
      z_pred = gtsam_stereo_camera.project2(m_L, J_pi);
    } catch (const gtsam::StereoCheiralityException& e) {
      LOG(WARNING) << "Warning: Point " << i << " behind camera. Skipping.";
      continue;  // Skip this measurement
    }

    // Calculate linearized residual y_lin = z - h(x_nom)
    gtsam::Vector3 y_i_lin = (z_obs - z_pred).vector();
    Eigen::Matrix<double, ZDim, StateDim> H_ekf_i = J_pi * J_correction;

    size_t row_idx = i * ZDim;

    H_stacked.block<ZDim, StateDim>(row_idx, 0) = H_ekf_i;
    y_linearized.segment<ZDim>(row_idx) = y_i_lin;

    double error_sq = y_i_lin.transpose() * R_inv_ * y_i_lin;
    double error = std::sqrt(error_sq);
    initial_error(i) = error;

    average_residual += y_i_lin.norm();
    num_good_measurements++;
  }

  average_residual /= (double)num_good_measurements;

  VLOG(30) << "Feature stats. New features: " << num_new_features << "/"
           << num_measurements << " Tracked features " << num_tracked_features
           << "/" << num_measurements;

  // 2. Store the prior information
  gtsam::Matrix6 R_info_prior = R_info_;
  gtsam::Vector6 d_info_prior = d_info_;

  // 3. --- Start Iteratively Reweighted Least Squares (IRLS) Loop ---
  // cout << "\n--- SRIF Robust Update (IRLS) Started ---" << endl;
  for (int iter = 0; iter < num_irls_iterations; ++iter) {
    // 3a. Get current state estimate (from previous iteration)
    const gtsam::Vector6 delta_w =
        R_info_.triangularView<Eigen::Upper>().solve(d_info_);
    const gtsam::Pose3 H_current_mean = H_linearization_point_.retract(delta_w);
    // Need to validate this:
    //  We intentionally do not recalculate the Jacobian block (H_stacked).
    //  to solve for the single best perturbation (delta_w) relative to the
    //  single linearization point (W_linearization_point_) that we had at the
    //  start of the update.
    const gtsam::Pose3 A = X_W_k_inv * H_current_mean;
    const gtsam::Pose3 G_w = A * L_KF_;
    gtsam::StereoCamera gtsam_stereo_camera_current(G_w.inverse(),
                                                    stereo_calibration_);

    gtsam::Vector weights = gtsam::Vector::Ones(num_measurements);

    for (size_t i = 0; i < num_measurements; ++i) {
      const TrackletId& tracklet_id = tracklets.at(i);
      const Feature::Ptr feature = frame->at(tracklet_id);
      CHECK(feature);

      const auto [stereo_keypoint_status, stereo_measurement] =
          rgbd_camera_->getStereo(feature);
      if (!stereo_keypoint_status) {
        weights(i) = 0.0;  // Skip point
        continue;
      }

      const auto& z_obs = stereo_measurement;

      const gtsam::Point3& m_L = m_linearized_.at(tracklet_id);

      gtsam::StereoPoint2 z_pred_current;
      try {
        z_pred_current = gtsam_stereo_camera_current.project(m_L);
      } catch (const gtsam::StereoCheiralityException& e) {
        LOG(WARNING) << "Warning: Point " << i << " behind camera. Skipping.";
        weights(i) = 0.0;  // Skip point
        continue;
      }

      // Calculate non-linear residual
      gtsam::Vector3 y_nonlinear = (z_obs - z_pred_current).vector();

      // //CHECK we cannot do this with gtsam's noise models (we can...)
      // // Calculate Mahalanobis-like distance (whitened error)
      double error_sq = y_nonlinear.transpose() * R_inv_ * y_nonlinear;
      double error = std::sqrt(error_sq);

      re_weighted_error(i) = error;

      // Calculate Huber weight w(e) = min(1, delta / |e|)
      weights(i) = (error <= huber_k_) ? 1.0 : huber_k_ / error;
      if (/*weights(i) < 0.99 &&*/ iter == num_irls_iterations - 1) {
        VLOG(50) << "  [Meas " << i << "] Final Weight: " << weights(i)
                 << " (Error: " << error << ")";
      }
    }

    // 3c. Construct the giant matrix A_qr for this iteration
    gtsam::Matrix A_qr =
        gtsam::Matrix::Zero(total_rows + StateDim, StateDim + 1);

    for (size_t i = 0; i < num_measurements; ++i) {
      double w_i = weights(i);
      if (w_i < 1e-6) continue;  // Skip 0-weighted points

      // R_robust = R / w_i  (Note: w is |e|^-1, so R_robust = R * |e|/delta)
      // R_robust_inv = R_inv * w_i
      // We need W_i such that W_i^T * W_i = R_robust_inv
      // W_i = sqrt(w_i) * R_inv_sqrt

      gtsam::Matrix33 R_robust_i = R_noise_ / w_i;
      gtsam::Matrix33 L_robust_i = R_robust_i.llt().matrixL();
      gtsam::Matrix33 R_robust_inv_sqrt = L_robust_i.inverse();

      size_t row_idx = i * ZDim;

      A_qr.block<ZDim, StateDim>(row_idx, 0) =
          R_robust_inv_sqrt * H_stacked.block<3, StateDim>(row_idx, 0);
      A_qr.block<ZDim, 1>(row_idx, StateDim) =
          R_robust_inv_sqrt * y_linearized.segment<3>(row_idx);
    }

    // 3d. Fill the prior part of the QR matrix
    // This is *always* the original prior
    A_qr.block<StateDim, StateDim>(total_rows, 0) = R_info_prior;
    A_qr.block<StateDim, 1>(total_rows, StateDim) = d_info_prior;

    // 3e. Perform QR decomposition
    Eigen::HouseholderQR<gtsam::Matrix> qr(A_qr);
    gtsam::Matrix R_full = qr.matrixQR().triangularView<Eigen::Upper>();

    // 3f. Extract the new R_info and d_info for the *next* iteration
    R_info_ = R_full.block<StateDim, StateDim>(0, 0);
    d_info_ = R_full.block<StateDim, 1>(0, StateDim);
  }

  // (Optional) Enforce R is upper-triangular
  R_info_ = R_info_.triangularView<Eigen::Upper>();

  HybridObjectMotionSRIFResult result;
  result.error = initial_error.norm();
  result.reweighted_error = re_weighted_error.norm();

  logger_ << timestamp_ << frame_id_ << average_residual << result.error
          << result.reweighted_error << num_new_features << num_tracked_features
          << conditionNumberR(R_info_);

  return result;
}

void HybridObjectMotionSRIF::resetState(const gtsam::Pose3& L_e,
                                        FrameId frame_id_KF,
                                        Timestamp timestamp_KF) {
  // 2. Initialize SRIF State (R, d) from EKF State (W, P)
  // W_linearization_point_ is set to initial_state_W
  // The initial perturbation is 0, so the initial d_info_ is 0.
  d_info_ = gtsam::Vector6::Zero();

  // Calculate initial Information Matrix Lambda = P^-1
  gtsam::Matrix66 Lambda_initial = initial_P_.inverse();

  // Calculate R_info_ from Cholesky decomposition: Lambda = R^T * R
  // We use LLT (L*L^T) and take the upper-triangular factor of L^T.
  // Or, more robustly, U^T*U from a U^T*U decomposition.
  // Eigen's LLT computes L (lower) such that A = L * L^T.
  // We need U (upper) such that A = U^T * U.
  // A.llt().matrixU() gives U where A = L*L^T (U is L^T). No.
  // A.llt().matrixL() gives L where A = L*L^T.
  // Let's use Eigen's LDLT and reconstruct.
  // A more direct way:
  R_info_ = Lambda_initial.llt().matrixU();  // L^T

  // will this give us discontinuinities betweek keyframes?
  //  reset other variables
  previous_H_ = gtsam::Pose3::Identity();
  L_KF_ = L_e;
  frame_id_KF_ = frame_id_KF;
  frame_id_ = frame_id_KF;
  timestamp_KF_ = timestamp_KF;
  timestamp_ = timestamp_KF;
  X_K_ = gtsam::Pose3::Identity();

  m_linearized_.clear();

  // should always be identity since the deviation from L_e = I when frame = e
  // the initial H should not be parsed into the constructor by this logic as
  // well!
  H_linearization_point_ = gtsam::Pose3::Identity();
}

// void HybridObjectMotionSRIF::relinearize() {
//   // not really relineaize? Just setting delta and R to zero and updating
//   H_lin
//   // but h_lin gets updated in predict anyway?
//   H_linearization_point_ = getBestEstimate();
//   previous_H_ = gtsam::Pose3::Identity();
//   d_info_ = gtsam::Vector6::Zero();

//   gtsam::Matrix66 Lambda_initial = initial_P_.inverse();
//   R_info_ = Lambda_initial.llt().matrixU();  // L^T
// }

/////////////////////////////////////////
//////////////////////////////////////////
///////////////////////////////////

FullHybridObjectMotionSRIF::FullHybridObjectMotionSRIF(
    const ObjectId object_id, const gtsam::Pose3& initial_state_H,
    const gtsam::Pose3& L_e, const FrameId& frame_id_KF,
    const Timestamp& timestamp_KF, const gtsam::Matrix66& initial_P,
    const gtsam::Matrix66& Q, const gtsam::Matrix33& R, Camera::Ptr camera,
    double huber_k)
    : HybridObjectMotionSolverImpl(object_id, camera),
      H_linearization_point_(initial_state_H),
      Q_(Q),
      R_noise_(R),
      R_inv_(R.inverse()),
      initial_P_(initial_P),
      huber_k_(huber_k),
      logger_(CsvHeader("timestamp", "frame_id", "residual", "initial_error",
                        "weighted_error", "new_features", "existing_tracks",
                        "cond_R")) {
  resetState(L_e, frame_id_KF, timestamp_KF);
}

FullHybridObjectMotionSRIF::~FullHybridObjectMotionSRIF() {
  // const std::string file_out =
  //     "full_hybrid_srif_j_" + std::to_string(object_id_) + ".csv";
  // OfstreamWrapper::WriteOutCsvWriter(logger_, file_out);
}

gtsam::Pose3 FullHybridObjectMotionSRIF::keyFramePose() const { return L_KF_; }
gtsam::Pose3 FullHybridObjectMotionSRIF::lastCameraPose() const { return X_K_; }
FrameId FullHybridObjectMotionSRIF::keyFrameId() const { return frame_id_KF_; }
FrameId FullHybridObjectMotionSRIF::frameId() const { return frame_id_; }

// Timestamp FullHybridObjectMotionSRIF::getKeyFrameTimestamp() const {
//   return timestamp_KF_;
// }
Timestamp FullHybridObjectMotionSRIF::timestamp() const { return timestamp_; }

gtsam::FastMap<TrackletId, gtsam::Point3>
FullHybridObjectMotionSRIF::getObjectPoints() const {
  return m_linearized_;
}

bool FullHybridObjectMotionSRIF::update(const gtsam::Pose3& H_w_km1_k_predict,
                                        Frame::Ptr frame,
                                        const TrackletIds& tracklets) {
  utils::ChronoTimingStats timer("hybrid_object_motion_srif.update");
  predict(H_w_km1_k_predict);

  // updateSchur(frame, tracklets);
  update(frame, tracklets, 1);
  // TODO: always assume successful solve for now!
  return true;
}

/**
 * @brief Recovers the state perturbation delta_w by solving R * delta_w = d.
 */
gtsam::Vector6 FullHybridObjectMotionSRIF::getStatePerturbation() const {
  // We solve the normal equations: Lambda * delta = eta
  // Using LLT is fast and numerically stable for information matrices
  gtsam::Vector delta = Lambda_.llt().solve(eta_);

  // If the landmarks are many, and you only need the pose part,
  // you could technically use a Schur Complement here, but
  // for now, a full solve is fine and correct.
  return delta.head<6>();
}

const gtsam::Pose3& FullHybridObjectMotionSRIF::getCurrentLinearization()
    const {
  return H_linearization_point_;
}

gtsam::Pose3 FullHybridObjectMotionSRIF::keyFrameMotion() const {
  return H_linearization_point_;
}

Motion3ReferenceFrame FullHybridObjectMotionSRIF::frameToFrameMotionReference()
    const {
  // gtsam::Pose3 H_W_e_k =
  // H_linearization_point_.retract(getStatePerturbation());
  gtsam::Pose3 H_W_KF_k = H_linearization_point_;
  gtsam::Pose3 H_W_km1_k = H_W_KF_k * H_W_KF_km1_->inverse();

  return Motion3ReferenceFrame(H_W_km1_k, Motion3ReferenceFrame::Style::F2F,
                               ReferenceFrame::GLOBAL, H_W_KF_km1_.to(),
                               frameId());
}

gtsam::Matrix66 FullHybridObjectMotionSRIF::getCovariance() const {
  // gtsam::Matrix66 Lambda = R_info_.transpose() * R_info_;
  // return Lambda.inverse();
}

gtsam::Matrix66 FullHybridObjectMotionSRIF::getInformationMatrix() const {
  // return R_info_.transpose() * R_info_;
}

void FullHybridObjectMotionSRIF::predict(const gtsam::Pose3& H_W_km1_k) {
  utils::ChronoTimingStats timer("full_hybrid_object_motion_srif.predict");

  H_W_KF_km1_ = this->keyFrameMotionReference();

  // gtsam::Vector delta = Lambda_.llt().solve(eta_);
  // LOG(INFO) << H_linearization_point_;
  // H_linearization_point_ = H_W_km1_k *
  // H_linearization_point_.retract(delta.head<6>()); LOG(INFO) <<
  // H_linearization_point_;
  H_linearization_point_ = H_W_km1_k * H_linearization_point_;

  // 2. Add Process Noise (The EIF way)
  // We only invert the 6x6 pose block of the information matrix
  // currently not increasing uncertainty on pose!!!
  // TODO: filter goes crazy as a result of the this covariance update in
  // information space
  // is wrong implemntation or is poorly tuned?
  gtsam::Matrix66 P_pose = Lambda_.block<6, 6>(0, 0).inverse();
  // P_pose += Q_;
  // Lambda_.block<6, 6>(0, 0) = P_pose.inverse();

  // 4. CRITICAL: Add a small "Floor" of information to keep the matrix positive
  // definite
  Lambda_.block<6, 6>(0, 0) += gtsam::Matrix66::Identity() * 1e-6;

  // 3. Reset eta because we shifted the linearization point
  eta_.setZero();
}

HybridObjectMotionSRIFResult FullHybridObjectMotionSRIF::updateSchur(
    Frame::Ptr frame, const TrackletIds& tracklets) {
  const gtsam::Pose3& X_W_k = frame->getPose();
  const gtsam::Pose3 X_W_k_inv = X_W_k.inverse();
  X_K_ = X_W_k;
  frame_id_ = frame->getFrameId();
  timestamp_ = frame->getTimestamp();

  utils::ChronoTimingStats timer_update("hybrid_srif.update_total");

  // 1. Identify visibility and new features
  std::set<TrackletId> visible_set(tracklets.begin(), tracklets.end());
  TrackletIds to_marginalize;
  for (const auto& [tid, slot] : tracklet_to_slot_) {
    if (visible_set.find(tid) == visible_set.end()) {
      to_marginalize.push_back(tid);
    }
  }

  for (const auto& tid : tracklets) {
    if (m_linearized_.find(tid) == m_linearized_.end()) {
      const gtsam::Point3 m_X_k = frame->backProjectToCamera(tid);
      m_linearized_[tid] = HybridObjectMotion::projectToObject3(
          X_W_k, keyFrameMotion(), L_KF_, m_X_k);
      tracklet_to_slot_[tid] = 6 + (tracklet_to_slot_.size() * 3);
    }
  }

  const size_t current_dim = 6 + (tracklet_to_slot_.size() * 3);

  LOG(INFO) << "Vector dimensions " << current_dim;
  LOG(INFO) << "Num points: " << tracklet_to_slot_.size();

  // Resize Lambda_ and eta_ instead of R and d
  if (Lambda_.rows() < current_dim) {
    size_t old_dim = Lambda_.rows();
    Lambda_.conservativeResize(current_dim, current_dim);
    // Fill new blocks with zeros and then add the landmark priors to the
    // diagonal
    Lambda_.block(0, old_dim, old_dim, current_dim - old_dim).setZero();
    Lambda_.block(old_dim, 0, current_dim - old_dim, current_dim).setZero();
    for (size_t i = old_dim; i < current_dim; i += 3) {
      // TODO: should be initalise by uncertainty on H too!
      Lambda_.block<3, 3>(i, i) =
          gtsam::Matrix33::Identity() * 100;  // 0.1m sigma -> 1/0.01 info
    }

    eta_.conservativeResize(current_dim);
    eta_.segment(old_dim, current_dim - old_dim).setZero();
  }

  // --- Accumulation ---
  // No R^T*R here! We just work on the existing Lambda_
  const gtsam::Pose3 A = X_W_k_inv * H_linearization_point_;
  const gtsam::Matrix66 J_corr = -A.AdjointMap();

  // const gtsam::Pose3 A = X_W_k_inv * H_linearization_point_;
  // G will project the point from the object frame into the camera frame
  const gtsam::Pose3 G_w = A * L_KF_;
  gtsam::StereoCamera stereo_cam(G_w.inverse(), stereo_calibration_);

  // --- 1. Accumulators ---
  // We copy the PRIOR pose info. We will add current frame info to this.
  gtsam::Matrix66 Lambda_pose_reduced = Lambda_.block<6, 6>(0, 0);
  gtsam::Vector6 eta_pose_reduced = eta_.head<6>();

  struct PointData {
    int slot;
    TrackletId tid;
    gtsam::Matrix33 L_mm;
    gtsam::Matrix63 L_hm;
    gtsam::Vector3 e_m;
  };
  std::vector<PointData> point_registry;

  double average_residual = 0;
  double good_tracks = 0;

  // --- 2. The Reduction Loop ---
  for (const auto& tid : tracklets) {
    int slot = tracklet_to_slot_[tid];

    // --- Linearization ---
    gtsam::Matrix36 Jh;
    gtsam::Matrix33 Jm;
    gtsam::StereoPoint2 z_pred;
    try {
      z_pred = stereo_cam.project2(m_linearized_[tid], Jh, Jm);
    } catch (...) {
      continue;
    }  // Skip points behind camera

    gtsam::Vector3 res =
        (rgbd_camera_->getStereo(frame->at(tid)).second - z_pred).vector();

    // Robust Weighting
    double error = std::sqrt(res.transpose() * R_inv_ * res);
    double w = (error <= huber_k_) ? 1.0 : huber_k_ / error;
    if (w < 1e-6) continue;

    average_residual += res.norm();
    good_tracks++;

    gtsam::Matrix33 info_weight = w * R_inv_;
    gtsam::Matrix36 Jh_c = Jh * J_corr;

    // Step A: Calculate current frame blocks
    gtsam::Matrix33 current_Lmm = Jm.transpose() * info_weight * Jm;
    gtsam::Matrix63 current_Lhm = Jh_c.transpose() * info_weight * Jm;
    gtsam::Matrix66 current_Lhh = Jh_c.transpose() * info_weight * Jh_c;
    gtsam::Vector3 current_em = Jm.transpose() * info_weight * res;
    gtsam::Vector6 current_eh = Jh_c.transpose() * info_weight * res;

    // Step B: Merge with PERSISTENT history
    gtsam::Matrix33 L_mm_total = current_Lmm + Lambda_.block<3, 3>(slot, slot);
    gtsam::Matrix63 L_hm_total = current_Lhm + Lambda_.block<6, 3>(0, slot);
    gtsam::Vector3 e_m_total = current_em + eta_.segment<3>(slot);

    // Step C: Schur Reduction
    // This is the part that must be exact.
    gtsam::Matrix33 invLmm = L_mm_total.inverse();

    // update the pose system: Add the direct info, subtract the marginalized
    // info
    Lambda_pose_reduced +=
        current_Lhh - (L_hm_total * invLmm * L_hm_total.transpose());
    eta_pose_reduced += current_eh - (L_hm_total * invLmm * e_m_total);

    // Register for Bake-in
    point_registry.push_back({slot, tid, L_mm_total, L_hm_total, e_m_total});
  }

  // --- 3. Atomic Solve ---
  // This delta_h is now numerically identical to the full solve
  gtsam::Vector6 delta_h = Lambda_pose_reduced.llt().solve(eta_pose_reduced);

  // --- 4. Back-Substitution and Global Sync ---
  for (const auto& p : point_registry) {
    gtsam::Matrix33 invLmm = p.L_mm.inverse();
    gtsam::Vector3 delta_m = invLmm * (p.e_m - p.L_hm.transpose() * delta_h);

    // A. Update Mean
    m_linearized_[p.tid] += delta_m;

    // B. Bake Persistent Knowledge (Persistence!)
    Lambda_.block<3, 3>(p.slot, p.slot) = p.L_mm;
    Lambda_.block<6, 3>(0, p.slot) = p.L_hm;
    Lambda_.block<3, 6>(p.slot, 0) = p.L_hm.transpose();
  }

  // Update Pose Mean
  H_linearization_point_ = H_linearization_point_.retract(delta_h);

  const gtsam::Pose3 L_W_k = this->pose();
  const Motion3ReferenceFrame H_W_km1_k = frameToFrameMotionReference();

  trajectory_.insert(frame_id_, timestamp_, {L_W_k, H_W_km1_k});

  // C. Final Global Sync
  Lambda_.block<6, 6>(0, 0) = Lambda_pose_reduced;

  if (!to_marginalize.empty()) {
    // Solve via Cholesky
    LOG(INFO) << "Marginalizing points" << container_to_string(to_marginalize);
    utils::ChronoTimingStats timer_LLT(
        "full_hybrid_object_motion_srif.update.marginalize");
    marginalizeInInformationForm(to_marginalize);
  }

  // must happen after marginalize
  eta_.setZero();  // Residuals consumed

  return HybridObjectMotionSRIFResult{};
}

HybridObjectMotionSRIFResult FullHybridObjectMotionSRIF::update(
    Frame::Ptr frame, const TrackletIds& tracklets,
    const int num_irls_iterations) {
  const gtsam::Pose3& X_W_k = frame->getPose();
  const gtsam::Pose3 X_W_k_inv = X_W_k.inverse();
  X_K_ = X_W_k;
  frame_id_ = frame->getFrameId();
  timestamp_ = frame->getTimestamp();

  utils::ChronoTimingStats timer_update("hybrid_srif.update_total");

  // 1. Identify visibility and new features
  std::set<TrackletId> visible_set(tracklets.begin(), tracklets.end());
  TrackletIds to_marginalize;
  for (const auto& [tid, slot] : tracklet_to_slot_) {
    if (visible_set.find(tid) == visible_set.end()) {
      to_marginalize.push_back(tid);
    }
  }

  for (const auto& tid : tracklets) {
    if (m_linearized_.find(tid) == m_linearized_.end()) {
      const gtsam::Point3 m_X_k = frame->backProjectToCamera(tid);
      m_linearized_[tid] = HybridObjectMotion::projectToObject3(
          X_W_k, keyFrameMotion(), L_KF_, m_X_k);
      tracklet_to_slot_[tid] = 6 + (tracklet_to_slot_.size() * 3);
    }
  }

  const size_t current_dim = 6 + (tracklet_to_slot_.size() * 3);

  LOG(INFO) << "Vector dimensions " << current_dim;
  LOG(INFO) << "Num points: " << tracklet_to_slot_.size();

  // Resize Lambda_ and eta_ instead of R and d
  if (Lambda_.rows() < current_dim) {
    size_t old_dim = Lambda_.rows();
    Lambda_.conservativeResize(current_dim, current_dim);
    // Fill new blocks with zeros and then add the landmark priors to the
    // diagonal
    Lambda_.block(0, old_dim, old_dim, current_dim - old_dim).setZero();
    Lambda_.block(old_dim, 0, current_dim - old_dim, current_dim).setZero();
    for (size_t i = old_dim; i < current_dim; i += 3) {
      // TODO: should be initalise by uncertainty on H too!
      Lambda_.block<3, 3>(i, i) =
          gtsam::Matrix33::Identity() * 100;  // 0.1m sigma -> 1/0.01 info
    }

    eta_.conservativeResize(current_dim);
    eta_.segment(old_dim, current_dim - old_dim).setZero();
  }

  // --- Accumulation ---
  // No R^T*R here! We just work on the existing Lambda_
  const gtsam::Pose3 A = X_W_k_inv * H_linearization_point_;
  const gtsam::Matrix66 J_corr = -A.AdjointMap();

  // const gtsam::Pose3 A = X_W_k_inv * H_linearization_point_;
  // G will project the point from the object frame into the camera frame
  const gtsam::Pose3 G_w = A * L_KF_;
  gtsam::StereoCamera stereo_cam(G_w.inverse(), stereo_calibration_);

  utils::ChronoTimingStats timer_irls(
      "full_hybrid_object_motion_srif.update.irls");

  // 1. Initialize our reduced Pose-only system
  // We start with the persistent Pose information (the prior)
  gtsam::Matrix66 Lambda_pose = Lambda_.block<6, 6>(0, 0);
  gtsam::Vector6 eta_pose = eta_.head<6>();

  // Temporary storage to avoid re-calculating during back-substitution
  // struct PointUpdateBlock {
  //   TrackletId tid;
  //   int slot;
  //   gtsam::Matrix33 inv_Lambda_mm;
  //   gtsam::Matrix36 Lambda_mh;
  //   gtsam::Vector3 eta_m;
  // };
  // std::vector<PointUpdateBlock> point_blocks;
  // point_blocks.reserve(tracklets.size());

  // double average_residual = 0;
  // double good_tracks = 0;

  // // 2. Accumulation Loop (The "Reduce" Phase)
  // for (const auto& tid : tracklets) {
  //   if (tracklet_to_slot_.find(tid) == tracklet_to_slot_.end()) continue;
  //   int slot = tracklet_to_slot_[tid];

  //   // --- Linearization ---
  //   gtsam::Matrix36 Jh;
  //   gtsam::Matrix33 Jm;
  //   gtsam::StereoPoint2 z_pred;
  //   try {
  //     z_pred = stereo_cam.project2(m_linearized_[tid], Jh, Jm);
  //   } catch (...) {
  //     continue;
  //   }  // Skip points behind camera

  //   gtsam::Vector3 res =
  //       (rgbd_camera_->getStereo(frame->at(tid)).second - z_pred).vector();

  //   // Robust Weighting
  //   double error = std::sqrt(res.transpose() * R_inv_ * res);
  //   double w = (error <= huber_k_) ? 1.0 : huber_k_ / error;
  //   if (w < 1e-6) continue;

  //   average_residual += res.norm();
  //   good_tracks++;

  //   gtsam::Matrix33 info_weight = w * R_inv_;
  //   gtsam::Matrix36 Jh_c = Jh * J_corr;

  //   // --- Local Information Blocks ---
  //   // Combine current measurement with persistent history
  //   gtsam::Matrix33 Lambda_mm =
  //       (Jm.transpose() * info_weight * Jm) + Lambda_.block<3, 3>(slot,
  //       slot);
  //   gtsam::Matrix33 inv_Lmm = Lambda_mm.inverse();

  //   // Combine current coupling with persistent history
  //   gtsam::Matrix63 Lambda_hm =
  //       (Jh_c.transpose() * info_weight * Jm) + Lambda_.block<6, 3>(0, slot);

  //   // Combine current residual with persistent history
  //   gtsam::Vector3 eta_m =
  //       (Jm.transpose() * info_weight * res) + eta_.segment<3>(slot);

  //   // --- THE SCHUR REDUCTION ---
  //   Lambda_pose += (Jh_c.transpose() * info_weight * Jh_c) -
  //                  (Lambda_hm * inv_Lmm * Lambda_hm.transpose());
  //   eta_pose +=
  //       (Jh_c.transpose() * info_weight * res) - (Lambda_hm * inv_Lmm *
  //       eta_m);

  //   // --- STEP 2.5: BAKE INTO PERSISTENT LAMBDA ---
  //   // This ensures the next frame starts with this frame's knowledge
  //   Lambda_.block<3, 3>(slot, slot) = Lambda_mm;
  //   Lambda_.block<6, 3>(0, slot) = Lambda_hm;
  //   Lambda_.block<3, 6>(slot, 0) = Lambda_hm.transpose();

  //   // Store for back-substitution
  //   point_blocks.push_back({tid, slot, inv_Lmm, Lambda_hm.transpose(),
  //   eta_m});
  // }

  // average_residual /= good_tracks;
  // LOG(INFO) << "Residual " << average_residual;

  // // 3. Solve for Pose (Fast 6x6)
  // gtsam::Vector6 delta_pose = Lambda_pose.llt().solve(eta_pose);
  // // if (!delta_pose.allFinite()) return {999.0, 0};

  // // 4. Back-Substitution (The "Update Points" Phase)
  // for (const auto& pb : point_blocks) {
  //   // Formula: delta_m = inv(Lambda_mm) * (eta_m - Lambda_mh * delta_pose)
  //   gtsam::Vector3 delta_m =
  //       pb.inv_Lambda_mm * (pb.eta_m - pb.Lambda_mh * delta_pose);

  //   // Update landmark linearization point
  //   m_linearized_[pb.tid] += delta_m;
  // }

  // // 5. Update Pose linearization point
  // H_linearization_point_ = H_linearization_point_.retract(delta_pose);

  // // Update the persistent Pose-Pose block as well
  // Lambda_.block<6, 6>(0, 0) = Lambda_pose;

  // const gtsam::Pose3 L_W_k = this->pose();
  // const Motion3ReferenceFrame H_W_km1_k = frameToFrameMotionReference();

  // trajectory_.insert(frame_id_, timestamp_, {L_W_k, H_W_km1_k});

  // // 6. Final "Cleanup"
  // // Since we are now at the new linearization point, the 'residual' info is
  // // consumed
  // eta_.setZero();

  // return {0.0, point_blocks.size()};

  // // NOT actually implemented!
  // TODO: try instead with actual hybrid factor. Use robust cost here
  // and then get whitened system
  for (int iter = 0; iter < num_irls_iterations; ++iter) {
    double average_residual = 0;
    double num_good_points = 0;
    for (const auto& tid : tracklets) {
      int slot = tracklet_to_slot_.at(tid);
      // gtsam::Matrix36 Jh;
      // gtsam::Matrix33 Jm;

      // gtsam::StereoPoint2 z_pred = stereo_cam.project2(m_linearized_.at(tid),
      // Jh, Jm);

      gtsam::Matrix36 Jh;
      gtsam::Matrix33 Jm;
      gtsam::StereoPoint2 z_pred;
      try {
        z_pred = stereo_cam.project2(m_linearized_[tid], Jh, Jm);
      } catch (...) {
        continue;
        // TODO: should immedialy remove as this is effecting the linearization
        // point!!
      }  // Skip points behind camera

      gtsam::Vector3 res =
          (rgbd_camera_->getStereo(frame->at(tid)).second - z_pred).vector();

      double error_sq = res.transpose() * R_inv_ * res;
      double error = std::sqrt(error_sq);

      // Calculate Huber weight w(e) = min(1, delta / |e|)
      double w = (error <= huber_k_) ? 1.0 : huber_k_ / error;

      // 2. Outlier Rejection: If weight is too small, ignore this measurement
      // This prevents the "Near-Zero" information from messing up the matrix
      // inversion
      if (w < 1e-6) {
        continue;
      }

      average_residual += res.norm();
      num_good_points++;

      // because R is meant to be from sigmas
      gtsam::Matrix33 Omega = w * R_inv_.inverse();
      gtsam::Matrix36 Jh_c = Jh * J_corr;

      // Add information directly
      // Update Pose-Pose block (6x6)
      Lambda_.block<6, 6>(0, 0) += Jh_c.transpose() * Omega * Jh_c;
      // Update Point-Point block (3x3)
      Lambda_.block<3, 3>(slot, slot) += Jm.transpose() * Omega * Jm;
      // Update Pose-Point coupling (6x3)
      gtsam::Matrix63 J_pose_point = Jh_c.transpose() * Omega * Jm;
      Lambda_.block<6, 3>(0, slot) += J_pose_point;
      Lambda_.block<3, 6>(slot, 0) += J_pose_point.transpose();

      eta_.head<6>() += Jh_c.transpose() * Omega * res;
      eta_.segment<3>(slot) += Jm.transpose() * Omega * res;
    }

    average_residual /= num_good_points;
    LOG(INFO) << "Residual " << average_residual;

    // always solve at the start to help with initalisation
    //  if(frame_id_ % 4 == 0 || frame_id_ == frame_id_KF_ || frame_id_ - 1u ==
    //  frame_id_KF_) {
    if (true) {
      // Solve via Cholesky
      utils::ChronoTimingStats timer_LLT(
          "full_hybrid_object_motion_srif.update.cholesky");
      Eigen::LLT<gtsam::Matrix> llt(Lambda_);
      gtsam::Vector delta = llt.solve(eta_);

      // Update means
      H_linearization_point_ = H_linearization_point_.retract(delta.head<6>());
      for (auto& [tid, slot] : tracklet_to_slot_) {
        const auto d_i = delta.segment<3>(slot);
        m_linearized_[tid] += d_i;
      }

      // 6. After retraction, your eta_ for the NEXT iteration must be 0
      // because the "error" has been moved into the linearization point.
      // However, the Lambda_ (Information) from this frame must be SAVED
      // only AFTER the final iteration.

      // In a pure Information Filter, we don't need to convert back to R.
      // We just reset eta for the next re-linearization or next frame.
      const gtsam::Pose3 L_W_k = this->pose();
      const Motion3ReferenceFrame H_W_km1_k = frameToFrameMotionReference();

      trajectory_.insert(frame_id_, timestamp_, {L_W_k, H_W_km1_k});
    }
  }

  // 3. Marginalize (Now just a slice operation)
  if (!to_marginalize.empty()) {
    // Solve via Cholesky
    LOG(INFO) << "Marginalizing points" << container_to_string(to_marginalize);
    utils::ChronoTimingStats timer_LLT(
        "full_hybrid_object_motion_srif.update.marginalize");
    marginalizeInInformationForm(to_marginalize);
  }

  eta_.setZero();

  return HybridObjectMotionSRIFResult{};

  // utils::ChronoTimingStats timer("full_hybrid_object_motion_srif.update");

  // // Update state mapping for any new landmarks
  // for (const auto& tid : tracklets) {
  //   if (m_linearized_.find(tid) == m_linearized_.end()) {
  //     const gtsam::Point3 m_X_k = frame->backProjectToCamera(tid);
  //     m_linearized_[tid] = HybridObjectMotion::projectToObject3(
  //         frame->getPose(), getKeyFramedMotion(), L_KF_, m_X_k);

  //     // Add new slot in state vector: Pose(6) + 3*N
  //     int new_slot = 6 + (tracklet_to_slot_.size() * 3);
  //     tracklet_to_slot_[tid] = new_slot;
  //   }
  // }

  // const size_t current_dim = 6 + (tracklet_to_slot_.size() * 3);
  // const size_t n_obs = tracklets.size();

  // // Resize R_info_ and d_info_ if new points were added (padding with weak
  // // priors)
  // if (R_info_.rows() < current_dim) {
  //   size_t old_dim = R_info_.rows();
  //   R_info_.conservativeResize(current_dim, current_dim);
  //   R_info_.block(0, old_dim, old_dim, current_dim - old_dim).setZero();
  //   R_info_.block(old_dim, 0, current_dim - old_dim, current_dim).setZero();
  //   // mmm no construct uncertainty based on projection of pose, motion and
  //   // sensor model! Landmark Prior (e.g., 0.1m uncertainty)
  //   for (size_t i = old_dim; i < current_dim; i += 3)
  //     R_info_.block<3, 3>(i, i) = gtsam::Matrix33::Identity() * (1.0 / 0.1);

  //   d_info_.conservativeResize(current_dim);
  //   d_info_.segment(old_dim, current_dim - old_dim).setZero();
  // }

  // const gtsam::Pose3 A = frame->getPose().inverse() * H_linearization_point_;
  // const gtsam::Matrix6 J_correction = -A.AdjointMap();
  // gtsam::StereoCamera stereo_cam((A * L_KF_).inverse(), stereo_calibration_);

  // // 1. Convert Square-Root Prior to Information form
  // // This is the starting point for this frame's optimization
  //  utils::ChronoTimingStats timer_lambda(
  //     "full_hybrid_object_motion_srif.update.make_lambda");
  //   timer_lambda.stop();

  // gtsam::Matrix Lambda = R_info_.transpose() * R_info_;
  // gtsam::Vector eta = R_info_.transpose() * d_info_;

  // utils::ChronoTimingStats timer_irls(
  //     "full_hybrid_object_motion_srif.update.irls");

  // // IRLS Loop
  // for (int iter = 0; iter < num_irls_iterations; ++iter) {
  //   // Note: If you have multiple IRLS iterations, you must reset Lambda/eta
  //   // to the ORIGINAL frame prior at the start of each loop because Jh/Jm
  //   change.

  //   // and we solve at each iteration now rather than outside the loop>?

  //   for (size_t i = 0; i < n_obs; ++i) {
  //       TrackletId tid = tracklets[i];
  //       int slot = tracklet_to_slot_[tid];

  //       gtsam::Matrix36 Jh;
  //       gtsam::Matrix33 Jm;
  //       gtsam::StereoPoint2 z_pred =
  //       stereo_cam.project2(m_linearized_.at(tid), Jh, Jm); gtsam::Vector3
  //       res = (rgbd_camera_->getStereo(frame->at(tid)).second -
  //       z_pred).vector();

  //       // 2. Compute Information Contribution (H^T * Omega * H)
  //       // Using R_inv_ (assuming it's the inverse noise covariance)
  //       gtsam::Matrix36 Jh_c = Jh * J_correction;
  //       gtsam::Matrix33 Omega = R_inv_; // Or apply robust weight here:
  //       R_inv_ * w

  //       // Update Pose-Pose block (6x6)
  //       Lambda.block<6, 6>(0, 0) += Jh_c.transpose() * Omega * Jh_c;

  //       // Update Point-Point block (3x3)
  //       Lambda.block<3, 3>(slot, slot) += Jm.transpose() * Omega * Jm;

  //       // Update Pose-Point coupling (6x3)
  //       gtsam::Matrix63 J_pose_point = Jh_c.transpose() * Omega * Jm;
  //       Lambda.block<6, 3>(0, slot) += J_pose_point;
  //       Lambda.block<3, 6>(slot, 0) += J_pose_point.transpose();

  //       // Update Information Vector (eta)
  //       eta.head<6>() += Jh_c.transpose() * Omega * res;
  //       eta.segment<3>(slot) += Jm.transpose() * Omega * res;
  //   }

  //   // 3. Fast Solve via Cholesky
  //   utils::ChronoTimingStats timer_LLT(
  //     "full_hybrid_object_motion_srif.update.LLT");
  //   Eigen::LLT<gtsam::Matrix> llt(Lambda);
  //   gtsam::Vector delta = llt.solve(eta);
  //   timer_LLT.stop();

  //   // 4. Update Mean
  //   H_linearization_point_ = H_linearization_point_.retract(delta.head<6>());
  //   for (auto const& [tid, slot] : tracklet_to_slot_) {
  //       m_linearized_[tid] += delta.segment<3>(slot);
  //   }

  //   utils::ChronoTimingStats timer_revert(
  //     "full_hybrid_object_motion_srif.update.revert");
  //   // 5. Transform back to Square-Root form for the next frame
  //   // This extracts the new R and d from the Cholesky factor
  //   R_info_ = llt.matrixU();
  //   // d = (R^T)^-1 * eta
  //   d_info_ = R_info_.transpose().triangularView<Eigen::Lower>().solve(eta);
  // }

  // d_info_.setZero(); // Re-linearized state means delta is now zero

  // timer_irls.stop();

  // // IRLS Loop
  // utils::ChronoTimingStats timer_irls(
  //     "full_hybrid_object_motion_srif.update.irls");

  // for (int iter = 0; iter < num_irls_iterations; ++iter) {
  //   gtsam::Matrix A_qr =
  //       gtsam::Matrix::Zero(n_obs * 3 + current_dim, current_dim + 1);

  //   for (size_t i = 0; i < n_obs; ++i) {
  //     TrackletId tid = tracklets[i];
  //     int slot = tracklet_to_slot_[tid];

  //     gtsam::Matrix36 Jh;
  //     gtsam::Matrix33 Jm;
  //     gtsam::StereoPoint2 z_pred =
  //         stereo_cam.project2(m_linearized_.at(tid), Jh, Jm);
  //     gtsam::Vector3 res =
  //         (rgbd_camera_->getStereo(frame->at(tid)).second - z_pred).vector();

  //     // Robust Weighting
  //     double err = std::sqrt(res.transpose() * R_inv_ * res);
  //     // double w = (err <= huber_k_) ? 1.0 : huber_k_ / err;
  //     double w = 1.0;

  //     // if (w_i < 1e-6) continue;  // Skip 0-weighted points

  //     gtsam::Matrix33 R_robust = R_noise_ / w;
  //     gtsam::Matrix33 L_robust = R_robust.llt().matrixL();
  //     gtsam::Matrix33 Whiten = L_robust.inverse();
  //     // gtsam::Matrix33 Whiten = (R_noise_ / w).llt().matrixL().inverse();

  //     A_qr.block<3, 6>(i * 3, 0) = Whiten * Jh * J_correction;
  //     A_qr.block<3, 3>(i * 3, slot) = Whiten * Jm;
  //     A_qr.block<3, 1>(i * 3, current_dim) = Whiten * res;
  //   }

  //   // Add Prior
  //   A_qr.block(n_obs * 3, 0, current_dim, current_dim) = R_info_;
  //   A_qr.block(n_obs * 3, current_dim, current_dim, 1) = d_info_;

  //   utils::ChronoTimingStats timer_QR(
  //     "full_hybrid_object_motion_srif.update.QR");
  //   Eigen::HouseholderQR<gtsam::Matrix> qr(A_qr);
  //   gtsam::Matrix R_full = qr.matrixQR().triangularView<Eigen::Upper>();
  //   timer_QR.stop();

  //   R_info_ = R_full.block(0, 0, current_dim, current_dim);
  //   d_info_ = R_full.block(0, current_dim, current_dim, 1);
  // }

  // timer_irls.stop();

  // Finalize Estimate
  // utils::ChronoTimingStats timer_calc_est(
  //     "full_hybrid_object_motion_srif.update.calc_estimate");
  // gtsam::Vector delta =
  // R_info_.triangularView<Eigen::Upper>().solve(d_info_);
  // H_linearization_point_ = H_linearization_point_.retract(delta.head<6>());
  // for (auto const& [tid, slot] : tracklet_to_slot_) {
  //   m_linearized_[tid] += delta.segment<3>(slot);
  // }
  // timer_calc_est.stop();
  // d_info_.setZero();  // Reset d after retraction

  // return HybridObjectMotionSRIFResult{0.0, 0.0};
}

bool FullHybridObjectMotionSRIF::createNewKeyedMotion(
    const gtsam::Pose3& L_KF, Frame::Ptr frame, const TrackletIds& tracklets) {
  X_W_KF_ = frame->getPose();
  resetState(L_KF, frame->getFrameId(), frame->getTimestamp());
  update(gtsam::Pose3::Identity(), frame, tracklets);
  // TODO: for now!
  return true;
}

void FullHybridObjectMotionSRIF::removeOutlierTracklets(
    const TrackletIds& to_remove) {
  if (to_remove.empty()) return;

  // 1. Filter out IDs that aren't actually in our state
  TrackletIds valid_removals;
  for (const auto& tid : to_remove) {
    if (tracklet_to_slot_.count(tid)) {
      valid_removals.push_back(tid);
    }
  }
  if (valid_removals.empty()) return;

  // 2. Identify tracklets to keep
  std::vector<TrackletId> tracklets_to_keep;
  for (const auto& [tid, slot] : tracklet_to_slot_) {
    // If not in the removal list, keep it
    if (std::find(valid_removals.begin(), valid_removals.end(), tid) ==
        valid_removals.end()) {
      tracklets_to_keep.push_back(tid);
    }
  }

  // 3. Rebuild the system (Contiguous Memory)
  const size_t new_dim = 6 + (tracklets_to_keep.size() * 3);
  gtsam::Matrix new_Lambda = gtsam::Matrix::Zero(new_dim, new_dim);
  gtsam::Vector new_eta = gtsam::Vector::Zero(new_dim);

  // Copy Pose block (Crucial: We DON'T apply Schur math to the 6x6)
  new_Lambda.block<6, 6>(0, 0) = Lambda_.block<6, 6>(0, 0);
  new_eta.head<6>() = eta_.head<6>();

  gtsam::FastMap<TrackletId, int> new_tracklet_to_slot;
  int current_slot = 6;

  for (const auto& tid : tracklets_to_keep) {
    const int old_slot = tracklet_to_slot_.at(tid);

    // Copy Landmark data to new contiguous slots
    new_Lambda.block<3, 3>(current_slot, current_slot) =
        Lambda_.block<3, 3>(old_slot, old_slot);
    new_Lambda.block<6, 3>(0, current_slot) = Lambda_.block<6, 3>(0, old_slot);
    new_Lambda.block<3, 6>(current_slot, 0) = Lambda_.block<3, 6>(old_slot, 0);
    new_eta.segment<3>(current_slot) = eta_.segment<3>(old_slot);

    new_tracklet_to_slot[tid] = current_slot;
    current_slot += 3;
  }

  // 4. Update State and Clear Maps
  Lambda_ = new_Lambda;
  eta_ = new_eta;
  tracklet_to_slot_ = new_tracklet_to_slot;

  for (const auto& tid : valid_removals) {
    m_linearized_.erase(tid);
  }
}

void FullHybridObjectMotionSRIF::marginalizeInInformationForm(
    const TrackletIds& to_remove) {
  if (to_remove.empty()) return;

  // 1. Identify which tracklets are actually in the state
  TrackletIds valid_removals;
  for (const auto& tid : to_remove) {
    if (tracklet_to_slot_.count(tid)) {
      valid_removals.push_back(tid);
    }
  }
  if (valid_removals.empty()) return;

  // 2. Perform Schur Complement for each landmark to be removed
  // We apply the math to the current Lambda_ and eta_ directly
  for (const auto& tid : valid_removals) {
    const int m_idx = tracklet_to_slot_.at(tid);

    // Extract relevant blocks
    // Lambda_hm: Pose-Landmark coupling (6x3)
    // Lambda_mm: Landmark-Landmark information (3x3)
    const gtsam::Matrix63 Lambda_hm = Lambda_.block<6, 3>(0, m_idx);
    const gtsam::Matrix33 Lambda_mm = Lambda_.block<3, 3>(m_idx, m_idx);
    const gtsam::Matrix36 Lambda_mh = Lambda_.block<3, 6>(m_idx, 0);

    // Precompute Lambda_hm * inv(Lambda_mm) to avoid double inversion
    // Using a fixed-size solver for speed
    const gtsam::Matrix63 H_invM = Lambda_hm * Lambda_mm.inverse();

    // Preserve information in the Pose block
    Lambda_.block<6, 6>(0, 0) -= H_invM * Lambda_mh;
    eta_.head<6>() -= H_invM * eta_.segment<3>(m_idx);
  }

  // 3. Rebuild the matrix and vector to remove the "holes"
  // We create a "Keep" list of landmarks that were NOT in the removal list
  std::vector<TrackletId> tracklets_to_keep;
  for (const auto& [tid, slot] : tracklet_to_slot_) {
    if (std::find(valid_removals.begin(), valid_removals.end(), tid) ==
        valid_removals.end()) {
      tracklets_to_keep.push_back(tid);
    }
  }

  const size_t new_dim = 6 + (tracklets_to_keep.size() * 3);
  gtsam::Matrix new_Lambda = gtsam::Matrix::Zero(new_dim, new_dim);
  gtsam::Vector new_eta = gtsam::Vector::Zero(new_dim);

  // Copy Pose block
  new_Lambda.block<6, 6>(0, 0) = Lambda_.block<6, 6>(0, 0);
  new_eta.head<6>() = eta_.head<6>();

  // Copy Landmark blocks and update mapping
  gtsam::FastMap<TrackletId, int> new_tracklet_to_slot;
  int current_slot = 6;

  for (const auto& tid : tracklets_to_keep) {
    const int old_slot = tracklet_to_slot_.at(tid);

    // Diagonal block
    new_Lambda.block<3, 3>(current_slot, current_slot) =
        Lambda_.block<3, 3>(old_slot, old_slot);

    // Coupling blocks
    new_Lambda.block<6, 3>(0, current_slot) = Lambda_.block<6, 3>(0, old_slot);
    new_Lambda.block<3, 6>(current_slot, 0) = Lambda_.block<3, 6>(old_slot, 0);

    // Vector block
    new_eta.segment<3>(current_slot) = eta_.segment<3>(old_slot);

    new_tracklet_to_slot[tid] = current_slot;
    current_slot += 3;
  }

  // 4. Update state variables and cleanup
  Lambda_ = new_Lambda;
  eta_ = new_eta;
  tracklet_to_slot_ = new_tracklet_to_slot;

  for (const auto& tid : valid_removals) {
    m_linearized_.erase(tid);
  }

  // Safety: Ensure Lambda remains positive definite
  Lambda_.block<6, 6>(0, 0) += gtsam::Matrix66::Identity() * 1e-9;
}

void FullHybridObjectMotionSRIF::resetState(const gtsam::Pose3& L_e,
                                            FrameId frame_id_KF,
                                            Timestamp timestamp_KF) {
  // 2. Initialize SRIF State (R, d) from EKF State (W, P)
  // W_linearization_point_ is set to initial_state_W
  // The initial perturbation is 0, so the initial d_info_ is 0.
  eta_ = gtsam::Vector6::Zero();

  // Calculate initial Information Matrix Lambda = P^-1
  Lambda_ = initial_P_.inverse();

  // In the Constructor or Reset
  // Add a strong diagonal prior to the 6x6 Pose block
  // Sigma_pose = 0.01m/rad -> Info = 10,000
  Lambda_.block<6, 6>(0, 0) = gtsam::Matrix66::Identity() * 1e4;
  // Lambda_.block<6,6>(0,0) = gtsam::Matrix66::Identity() * 1e4;

  // will this give us discontinuinities betweek keyframes?
  //  reset other variables
  L_KF_ = L_e;
  frame_id_KF_ = frame_id_KF;
  frame_id_ = frame_id_KF;
  timestamp_KF_ = timestamp_KF;
  timestamp_ = timestamp_KF;
  X_K_ = gtsam::Pose3::Identity();

  H_W_KF_km1_ = Motion3ReferenceFrame(
      gtsam::Pose3::Identity(), MotionRepresentationStyle::KF,
      ReferenceFrame::GLOBAL, frame_id_KF_, frame_id_);

  m_linearized_.clear();
  tracklet_to_slot_.clear();

  // should always be identity since the deviation from L_e = I when frame = e
  // the initial H should not be parsed into the constructor by this logic as
  // well!
  H_linearization_point_ = gtsam::Pose3::Identity();
}

}  // namespace dyno
