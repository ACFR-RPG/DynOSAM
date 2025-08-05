/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris
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

#pragma once

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/StereoPoint2.h>
#include <gtsam/linear/NoiseModel.h>

#include "dynosam/common/Exceptions.hpp"
#include "dynosam/common/Types.hpp"
#include "dynosam/utils/GtsamUtils.hpp"

namespace dyno {

template <typename M>
struct measurement_traits {
  // Template for derivation
  static constexpr bool has_point = false;
  static constexpr bool has_keypoint = false;
};

/**
 * @brief Defines a single visual measurement with tracking label, object id,
 * frame id and measurement.
 *
 * @tparam T
 */
template <typename T>
struct VisualMeasurementStatus : public TrackedValueStatus<T> {
  using Base = TrackedValueStatus<T>;
  using Base::Value;
  //! dimension of the measurement. Must have gtsam::dimension traits
  constexpr static int dim = gtsam::traits<T>::dimension;

  /// @brief Default constructor for IO
  VisualMeasurementStatus() = default;

  /**
   * @brief Construct a new Visual Measurement Status from the Base type
   *
   * @param base TrackedValueStatus<T>
   */
  VisualMeasurementStatus(const Base& base)
      : VisualMeasurementStatus(base.value(), base.frameId(), base.trackletId(),
                                base.objectId(), base.referenceFrame()) {}

  /**
   * @brief Construct a new Visual Measurement Status object
   *
   * @param m const T&
   * @param frame_id FrameId
   * @param tracklet_id TrackletId
   * @param label ObjectId
   * @param reference_frame ReferenceFrame
   */
  VisualMeasurementStatus(const T& m, FrameId frame_id, TrackletId tracklet_id,
                          ObjectId label, ReferenceFrame reference_frame)
      : Base(m, frame_id, tracklet_id, label, reference_frame) {}

  /**
   * @brief Constructs a static Visual Measurement Status.
   *
   * @param m const T&
   * @param frame_id FrameId
   * @param tracklet_id TrackletId
   * @param reference_frame ReferenceFrame
   * @return VisualMeasurementStatus
   */
  inline static VisualMeasurementStatus Static(const T& m, FrameId frame_id,
                                               TrackletId tracklet_id,
                                               ReferenceFrame reference_frame) {
    return VisualMeasurementStatus(m, frame_id, tracklet_id, background_label,
                                   reference_frame);
  }

  /**
   * @brief Constructs a dynamic Visual Measurement Status.
   *
   * @param m const T&
   * @param frame_id FrameId
   * @param tracklet_id TrackletId
   * @param reference_frame ReferenceFrame
   * @return VisualMeasurementStatus
   */
  inline static VisualMeasurementStatus Dynamic(
      const T& m, FrameId frame_id, TrackletId tracklet_id, ObjectId label,
      ReferenceFrame reference_frame) {
    CHECK(label != background_label);
    return VisualMeasurementStatus(m, frame_id, tracklet_id, label,
                                   reference_frame);
  }

  /**
   * @brief Constructs a static Visual Measurement Status in the local frame.
   *
   * @param m const T&
   * @param frame_id FrameId
   * @param tracklet_id TrackletId
   * @return VisualMeasurementStatus
   */
  inline static VisualMeasurementStatus StaticInLocal(const T& m,
                                                      FrameId frame_id,
                                                      TrackletId tracklet_id) {
    return VisualMeasurementStatus(m, frame_id, tracklet_id, background_label,
                                   ReferenceFrame::LOCAL);
  }

  /**
   * @brief Constructs a dynamic Visual Measurement Status in the local frame.
   *
   * @param m const T&
   * @param frame_id FrameId
   * @param tracklet_id TrackletId
   * @return VisualMeasurementStatus
   */
  inline static VisualMeasurementStatus DynamicInLocal(const T& m,
                                                       FrameId frame_id,
                                                       TrackletId tracklet_id,
                                                       ObjectId label) {
    CHECK(label != background_label);
    return VisualMeasurementStatus(m, frame_id, tracklet_id, label,
                                   ReferenceFrame::LOCAL);
  }

  /**
   * @brief Constructs a static Visual Measurement Status in the global frame.
   *
   * @param m const T&
   * @param frame_id FrameId
   * @param tracklet_id TrackletId
   * @return VisualMeasurementStatus
   */
  inline static VisualMeasurementStatus StaticInGlobal(const T& m,
                                                       FrameId frame_id,
                                                       TrackletId tracklet_id) {
    return VisualMeasurementStatus(m, frame_id, tracklet_id, background_label,
                                   ReferenceFrame::GLOBAL);
  }

  /**
   * @brief Constructs a dynamic Visual Measurement Status in the global frame.
   *
   * @param m const T&
   * @param frame_id FrameId
   * @param tracklet_id TrackletId
   * @return VisualMeasurementStatus
   */
  inline static VisualMeasurementStatus DynamicInGLobal(const T& m,
                                                        FrameId frame_id,
                                                        TrackletId tracklet_id,
                                                        ObjectId label) {
    CHECK(label != background_label);
    return VisualMeasurementStatus(m, frame_id, tracklet_id, label,
                                   ReferenceFrame::GLOBAL);
  }
};

/**
 * @brief Defines a measurement/value with associated covariance matrix.
 *
 * Depending on the context, this model can either represent the uncertainty on
 * a measurement or the marginal covariance of an estimate. In the latter
 * context, the name "Measurement"WithCovariance is misleading (as it would not
 * be a measurement), so to all state-estimation people I profusely apologise.
 *
 * The covariance can be optionally provided.
 *
 * We define the dimensions of this type and provide an equals function to make
 * it compatible with gtsam::traits<T>::dimension and gtsam::traits<T>::Equals.
 *
 *
 *
 * @tparam T measurement type
 * @tparam D dimension of the measurement. Used to construct the covariance
 * matrix.
 */
template <typename T, int D = gtsam::traits<T>::dimension>
class MeasurementWithCovariance {
 public:
  using This = MeasurementWithCovariance<T, D>;
  //! Need to have dimension to satisfy gtsam::dimemsions
  inline constexpr static auto dimension = D;
  //! D x D covariance matrix
  using Covariance = Eigen::Matrix<double, D, D>;
  //! D x 1 sigma matrix used to construct the covariance matrix
  using Sigmas = Eigen::Matrix<double, D, 1>;
  //! Optional alias
  using Optional = std::optional<This>;

  /**
   * @brief Empty constructor
   *
   */
  MeasurementWithCovariance() = default;

  /**
   * @brief Construct object using a measurement and a gtsam::SharedGaussian,
   * representing the noise model/uncertainty of the measurement.
   *
   * @throw DynosamException If the provided model has dimensions that do not
   * match the templated dimensions D.
   *
   * @param measurement const T&
   * @param model const gtsam::SharedGaussian&
   */
  MeasurementWithCovariance(const T& measurement,
                            const gtsam::SharedGaussian& model)
      : measurement_(measurement), model_(CHECK_NOTNULL(model)) {
    if (static_cast<int>(model->dim()) != This::dimension) {
      DYNO_THROW_MSG(DynosamException)
          << "Invalid gtsam::SharedGaussian model provided to "
          << "MeasurementWithCovariance. Type dimensions are "
          << This::dimension << " but noise models dims are " << model->dim();
    }
  }

  /**
   * @brief Construct object using only a measurement type, indicatating there
   * is no associated noise model/uncertainty. The resulting model will be none
   * and the covariance matrix will be all zeros.
   *
   * We ensure this constructor is explicit to prevent unintended implicit
   * conversions
   *
   * @param measurement const T&
   */
  explicit MeasurementWithCovariance(const T& measurement)
      : measurement_(measurement), model_(nullptr) {}

  explicit MeasurementWithCovariance(const std::pair<T, Covariance>& pair)
      : MeasurementWithCovariance(pair.first, pair.second) {}

  /**
   * @brief Construct object using a measurement type and associated covariance
   * matrix. The covariance matrix is used to construct the underlying sensor
   * noise model using gtsam::noiseModel::Gaussian.
   *
   * @param measurement const T&
   * @param covariance const Covariance&
   * @return MeasurementWithCovariance
   */
  MeasurementWithCovariance(const T& measurement, const Covariance& covariance)
      : measurement_(measurement),
        model_(gtsam::noiseModel::Gaussian::Covariance(covariance, false)) {}

  /**
   * @brief Construct object using a measurement type and associated sigma
   * vector. The sigma's represent the on-diagonal values for the covariance
   * matrix and are used to construct the underlying sensor noise model using
   * gtsam::noiseModel::Gaussian.
   *
   * @param measurement const T&
   * @param covariance const Sigmas&
   * @return MeasurementWithCovariance
   */
  static MeasurementWithCovariance FromSigmas(const T& measurement,
                                              const Sigmas& sigmas) {
    return MeasurementWithCovariance(
        measurement, gtsam::noiseModel::Diagonal::Sigmas(sigmas));
  }

  const T& measurement() const { return measurement_; }
  const gtsam::SharedGaussian& model() const { return model_; }
  bool hasModel() const { return model_ != nullptr; }

  Covariance covariance() const {
    if (hasModel()) {
      return model_->covariance();
    } else {
      return Covariance::Zero();
    }
  }

  /// @brief Very important to have these cast operators so we can use
  /// Base::asType to cast to the internal data.
  operator const T&() const { return measurement(); }
  operator Covariance() const { return covariance(); }

  friend std::ostream& operator<<(std::ostream& os,
                                  const This& measurement_with_cov) {
    os << "m: " << measurement_with_cov.measurement()
       << ", cov: " << measurement_with_cov.covariance()
       << ", has model: " << std::boolalpha << measurement_with_cov.hasModel();
    return os;
  }

  inline void print(const std::string& s = "") const {
    std::cout << s << *this << std::endl;
  }

  inline bool equals(const MeasurementWithCovariance& other,
                     double tol = 1e-8) const {
    return gtsam::traits<T>::Equals(measurement_, other.measurement_, tol) &&
           utils::equateGtsamSharedValues(model_, other.model_, tol);
  }

 private:
  //! Measurement
  T measurement_;
  //! Noise model which can either be used to represet the covariance matrix of
  //! an estimate or the noise model of measurement
  gtsam::SharedGaussian model_{nullptr};
};

/// @brief (gtsam) Pose3 with covariance
typedef MeasurementWithCovariance<gtsam::Pose3> Pose3Measurement;

/// @brief (gtsam) Point3 with covariance
typedef MeasurementWithCovariance<gtsam::Point3> Point3Measurement;

/// @brief Keypoint with covariance
typedef MeasurementWithCovariance<Keypoint> KeypointMeasurement;

/**
 * @brief Struct containing landmark and keypoint measurements (with covariance)
 * Landmark is constructed from the depth measurement and the associated
 * keypoint.
 *
 * Used to construct the final measurement information to be passed to the
 * backend and can be obtained from an RGBD style sensor
 *
 */
struct LandmarkKeypoint {
  LandmarkKeypoint() = default;

  /**
   * @brief Construct a LandmarkKeypoint struct using measurement only values
   * (no noise model/covariance)
   *
   * @param l const Landmark&
   * @param kp  const Keypoint&
   */
  LandmarkKeypoint(const Landmark& l, const Keypoint& kp)
      : landmark(Point3Measurement(l)), keypoint(KeypointMeasurement(kp)) {}

  /**
   * @brief  Construct a LandmarkKeypoint struct using full
   * MeasurementWithCovariance objects (which may or may not contain noise
   * models)/.
   *
   * @param l const MeasurementWithCovariance<Landmark>&
   * @param kp const MeasurementWithCovariance<Keypoint>&
   */
  LandmarkKeypoint(const MeasurementWithCovariance<Landmark>& l,
                   const MeasurementWithCovariance<Keypoint>& kp)
      : landmark(l), keypoint(kp) {}

  //! Landmark measurement with covariance
  MeasurementWithCovariance<Landmark> landmark;
  //! Keypoint measurement with covariance
  MeasurementWithCovariance<Keypoint> keypoint;

  friend std::ostream& operator<<(std::ostream& os,
                                  const LandmarkKeypoint& status) {
    os << "landmark: " << status.landmark << ", keypoint " << status.keypoint;
    return os;
  }
};

/**
 * @brief Keypoint with depth
 *
 */
struct KeypointDepth {
 public:
  KeypointDepth() = default;
  KeypointDepth(const Keypoint& p, const Depth& d) : keypoint(p), depth(d) {}

 public:
  Keypoint keypoint;
  Depth depth;
};

/**
 * @brief Structure to contain all possible visual measurements from a camera
 * setup including monocular, RGBD and stereo measurements with covariance
 * matrices
 *
 */
class CameraMeasurement {
 public:
  // CameraMeasurement() = default;
  CameraMeasurement(const MeasurementWithCovariance<Keypoint>& keypoint);

  CameraMeasurement& keypoint(
      const MeasurementWithCovariance<Keypoint>& keypoint);
  CameraMeasurement& landmark(
      const MeasurementWithCovariance<Landmark>& landmark);
  CameraMeasurement& depth(const MeasurementWithCovariance<Depth>& depth);
  CameraMeasurement& rightKeypoint(
      const MeasurementWithCovariance<Keypoint>& right_keypoint);

  bool monocular() const;
  bool rgbd() const;
  bool stereo() const;

  bool hasLandmark() const;

  const MeasurementWithCovariance<Keypoint>& keypoint() const;
  const MeasurementWithCovariance<Landmark>& landmark() const;
  const MeasurementWithCovariance<Depth>& depth() const;
  const MeasurementWithCovariance<Keypoint>& rightKeypoint() const;

 protected:
  //! 2D keypoint measurement with covariance.
  //! If the measurement is stereo, then this is expected to be the left
  //! keypoint.
  MeasurementWithCovariance<Keypoint> keypoint_;
  //! 3d Landmark measurement in the camera frame with covariance
  MeasurementWithCovariance<Landmark>::Optional landmark_ = {};
  //! Depth measurement associated with an RGBD camera. Usually used to project
  //! a 2D keypoint into a 3D landmark. There should never be a case where we
  //! have depth without a landmark
  MeasurementWithCovariance<Depth>::Optional depth_ = {};
  //! Right 2D keypoint measurement with covariance (associated with the right
  //! frame of a stereo pair).
  MeasurementWithCovariance<Keypoint>::Optional right_keypoint_ = {};
};

/// @brief Alias to a visual measurement with a fixed-sized covariance matrix.
/// @tparam T Measurement type (e.g. 2D keypoint, 3D landmark)
template <typename T>
using VisualMeasurementWithCovStatus =
    VisualMeasurementStatus<MeasurementWithCovariance<T>>;

/// @brief Alias for a landmark measurement with 3x3 covariance.
typedef VisualMeasurementWithCovStatus<Landmark> LandmarkStatus;
/// @brief Alias for a keypoint measurement with 2x2 covariance.
typedef VisualMeasurementWithCovStatus<Keypoint> KeypointStatus;
/// @brief Alias for a depth measurement with scalar covariance.
typedef VisualMeasurementWithCovStatus<Depth> DepthStatus;
/// @brief Alias for a stereo measurement with 3x3 covariance.
typedef VisualMeasurementWithCovStatus<gtsam::StereoPoint2> StereoStatus;

using LandmarkKeypointStatus = TrackedValueStatus<LandmarkKeypoint>;
using CameraMeasurementStatus = TrackedValueStatus<CameraMeasurement>;

/// @brief A vector of LandmarkStatus
typedef GenericTrackedStatusVector<LandmarkStatus> StatusLandmarkVector;
/// @brief A vector of KeypointStatus
typedef GenericTrackedStatusVector<KeypointStatus> StatusKeypointVector;
/// @brief A vector of CameraMeasurementStatus
typedef GenericTrackedStatusVector<CameraMeasurementStatus>
    CameraMeasurementStatusVector;

/**
 * @brief Noise parameters used when constructing the output of each frontend
 * and defines the measurement noise of each visual measurement.
 *
 */
struct VisualNoiseParams {
  //! sigma (standard devation) on each measured keypoint. Pixel units.
  double keypoint_sigma = 2.0;
  //! sigma (standard deviation) on the depth of each measured keypoint. Pixel
  //! meters.
  double depth_sigma = 0.3;
};

template <>
struct measurement_traits<LandmarkKeypoint> {
  static constexpr bool has_point = true;
  static constexpr bool has_keypoint = true;

  static Landmark point(const LandmarkKeypoint& measurement) {
    return measurement.landmark;
  }

  static std::pair<Landmark, gtsam::SharedNoiseModel> pointWithCovariance(
      const LandmarkKeypoint& measurement) {
    return {measurement.landmark.measurement(), measurement.landmark.model()};
  }

  static Keypoint keypoint(const LandmarkKeypoint& measurement) {
    return measurement.keypoint;
  }

  static std::pair<Keypoint, gtsam::SharedNoiseModel> keypointWithCovariance(
      const LandmarkKeypoint& measurement) {
    return {measurement.keypoint.measurement(), measurement.keypoint.model()};
  }
};

}  // namespace dyno

template <typename T, int D>
struct gtsam::traits<dyno::MeasurementWithCovariance<T, D>>
    : public gtsam::Testable<dyno::MeasurementWithCovariance<T, D>> {};

template <typename T, int D>
struct gtsam::traits<const dyno::MeasurementWithCovariance<T, D>>
    : public gtsam::Testable<dyno::MeasurementWithCovariance<T, D>> {};
