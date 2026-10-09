/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the full license information

 * -------------------------------------------------------------------------- */

/**
 * @file    FixedStereoFactor.h
 * @brief   A non-linear stereo factor with a fixed landmark
 */

#pragma once

#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/StereoCamera.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace gtsam {

/**
 * A Generic Stereo Factor with a fixed landmark.
 *
 * The factor is a function only of the pose. The landmark is stored internally
 * and therefore does not appear as an optimization variable.
 *
 * The error is:
 *
 *   e(T, P) = stereoProject(T, P) - measurement
 *
 * with P fixed for the lifetime of the factor.
 *
 * @ingroup slam
 */
template <class POSE>
class GenericFixedStereoFactor : public NoiseModelFactor1<POSE> {
 private:
  // Keep a copy of measurement, landmark, and calibration for I/O.
  StereoPoint2 measured_;                ///< stereo measurement
  Point3 landmark_;                      ///< fixed landmark
  Cal3_S2Stereo::shared_ptr K_;          ///< shared pointer to calibration
  boost::optional<POSE> body_P_sensor_;  ///< pose of sensor in body frame

  // Verbosity handling for Cheirality Exceptions.
  bool throwCheirality_;    ///< If true, rethrows exceptions
  bool verboseCheirality_;  ///< If true, prints cheirality errors

 public:
  using Base = NoiseModelFactor1<POSE>;
  using This = GenericFixedStereoFactor<POSE>;
  using shared_ptr = boost::shared_ptr<This>;
  using CamPose = POSE;

  /**
   * Default constructor.
   */
  GenericFixedStereoFactor()
      : K_(new Cal3_S2Stereo(444, 555, 666, 777, 888, 1.0)),
        throwCheirality_(false),
        verboseCheirality_(false) {}

  /**
   * Constructor.
   *
   * @param measured is the Stereo Point measurement (u_l, u_r, v).
   * @param model is the noise model on the measurement.
   * @param poseKey the pose variable key.
   * @param landmark the fixed 3D landmark.
   * @param K the constant calibration.
   * @param body_P_sensor is the transform from body to sensor frame.
   */
  GenericFixedStereoFactor(const StereoPoint2& measured,
                           const SharedNoiseModel& model, Key poseKey,
                           const Point3& landmark,
                           const Cal3_S2Stereo::shared_ptr& K,
                           boost::optional<POSE> body_P_sensor = boost::none)
      : Base(model, poseKey),
        measured_(measured),
        landmark_(landmark),
        K_(K),
        body_P_sensor_(body_P_sensor),
        throwCheirality_(false),
        verboseCheirality_(false) {}

  /**
   * Constructor with exception-handling flags.
   */
  GenericFixedStereoFactor(const StereoPoint2& measured,
                           const SharedNoiseModel& model, Key poseKey,
                           const Point3& landmark,
                           const Cal3_S2Stereo::shared_ptr& K,
                           bool throwCheirality, bool verboseCheirality,
                           boost::optional<POSE> body_P_sensor = boost::none)
      : Base(model, poseKey),
        measured_(measured),
        landmark_(landmark),
        K_(K),
        body_P_sensor_(body_P_sensor),
        throwCheirality_(throwCheirality),
        verboseCheirality_(verboseCheirality) {}

  /**
   * Virtual destructor.
   */
  ~GenericFixedStereoFactor() override = default;

  /**
   * @return a deep copy of this factor.
   */
  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(new This(*this)));
  }

  /**
   * Print.
   */
  void print(
      const std::string& s = "",
      const KeyFormatter& keyFormatter = DefaultKeyFormatter) const override {
    Base::print(s, keyFormatter);
    measured_.print(s + ".z");
    // landmark_.print(s + ".landmark");

    if (body_P_sensor_) {
      body_P_sensor_->print("  sensor pose in body frame: ");
    }
  }

  /**
   * Equals.
   */
  bool equals(const NonlinearFactor& f, double tol = 1e-9) const override {
    const This* e = dynamic_cast<const This*>(&f);

    return e && Base::equals(f) && measured_.equals(e->measured_, tol) &&
           //  landmark_.equals(e->landmark_, tol) &&
           ((!body_P_sensor_ && !e->body_P_sensor_) ||
            (body_P_sensor_ && e->body_P_sensor_ &&
             body_P_sensor_->equals(*e->body_P_sensor_, tol)));
  }

  /**
   * Stereo reprojection error:
   *
   *   h(T) - z
   *
   * The landmark is fixed and therefore there is no landmark Jacobian.
   */
  Vector evaluateError(const POSE& pose, boost::optional<Matrix&> H1 =
                                             boost::none) const override {
    try {
      if (body_P_sensor_) {
        if (H1) {
          gtsam::Matrix H0;

          StereoCamera stereoCam(pose.compose(*body_P_sensor_, H0), K_);

          StereoPoint2 reprojectionError(stereoCam.project(landmark_, H1) -
                                         measured_);

          // Chain rule:
          //
          // d error / d pose
          //   = d error / d sensorPose * d sensorPose / d pose
          //
          *H1 = *H1 * H0;

          return reprojectionError.vector();
        } else {
          StereoCamera stereoCam(pose.compose(*body_P_sensor_), K_);

          return (stereoCam.project(landmark_) - measured_).vector();
        }
      } else {
        StereoCamera stereoCam(pose, K_);

        return (stereoCam.project(landmark_, H1) - measured_).vector();
      }
    } catch (StereoCheiralityException& e) {
      if (H1) {
        *H1 = Matrix::Zero(3, 6);
      }

      if (verboseCheirality_) {
        std::cout << e.what() << ": Fixed landmark moved behind camera "
                  << DefaultKeyFormatter(this->key()) << std::endl;
      }

      if (throwCheirality_) {
        throw StereoCheiralityException(this->key());
      }
    }

    return Vector3::Constant(2.0 * K_->fx());
  }

  /**
   * @return the measured stereo point.
   */
  const StereoPoint2& measured() const { return measured_; }

  /**
   * @return the fixed landmark.
   */
  const Point3& landmark() const { return landmark_; }

  /**
   * @return the calibration object.
   */
  const Cal3_S2Stereo::shared_ptr calibration() const { return K_; }

  /**
   * @return verbosity flag.
   */
  bool verboseCheirality() const { return verboseCheirality_; }

  /**
   * @return whether cheirality exceptions are thrown.
   */
  bool throwCheirality() const { return throwCheirality_; }

 private:
};

/// Traits.
template <class POSE>
struct traits<GenericFixedStereoFactor<POSE>>
    : public Testable<GenericFixedStereoFactor<POSE>> {};

}  // namespace gtsam
