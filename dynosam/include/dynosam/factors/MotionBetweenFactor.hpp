#pragma once

#include <gtsam/base/Lie.h>
#include <gtsam/base/Testable.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include <boost/concept/assert.hpp>

#include "dynosam_opt/Symbols.hpp"

namespace dyno {

template <class VALUE>
class MotionBetweenFactor : public gtsam::NoiseModelFactorN<VALUE, VALUE> {
  // Check that VALUE type is a testable Lie group
  BOOST_CONCEPT_ASSERT((gtsam::IsTestable<VALUE>));
  BOOST_CONCEPT_ASSERT((gtsam::IsLieGroup<VALUE>));

 public:
  typedef VALUE T;

 private:
  typedef MotionBetweenFactor<VALUE> This;
  typedef gtsam::NoiseModelFactorN<VALUE, VALUE> Base;

  VALUE measured_; /** The measurement */

 public:
  // Provide access to the Matrix& version of evaluateError:
  using Base::evaluateError;

  // shorthand for a smart pointer to a factor
  typedef typename boost::shared_ptr<MotionBetweenFactor> shared_ptr;

  /** default constructor - only use for serialization */
  MotionBetweenFactor() {}

  /** Constructor */
  MotionBetweenFactor(gtsam::Key key1, gtsam::Key key2, const VALUE& measured,
                      const gtsam::SharedNoiseModel& model)
      : Base(model, key1, key2), measured_(measured) {}

  ~MotionBetweenFactor() override {}

  /// @return a deep copy of this factor
  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(new This(*this)));
  }

  /// print with optional string
  void print(const std::string& s = "",
             const gtsam::KeyFormatter& keyFormatter =
                 DynosamKeyFormatter) const override {
    std::cout << s << "MotionBetweenFactor(" << keyFormatter(this->key1())
              << "," << keyFormatter(this->key2()) << ")\n";
    gtsam::traits<T>::Print(measured_, "  measured: ");
    this->noiseModel_->print("  noise model: ");
  }

  /// assert equality up to a tolerance
  bool equals(const gtsam::NonlinearFactor& expected,
              double tol = 1e-9) const override {
    const This* e = dynamic_cast<const This*>(&expected);
    return e != nullptr && Base::equals(*e, tol) &&
           gtsam::traits<T>::Equals(this->measured_, e->measured_, tol);
  }

  /// @}
  /// @name NoiseModelFactorN methods
  /// @{

  /// evaluate error, returns vector of errors size of tangent space
  gtsam::Vector evaluateError(
      const T& Xi, const T& Xj,
      boost::optional<gtsam::Matrix&> H_i = boost::none,
      boost::optional<gtsam::Matrix&> H_j = boost::none) const override {
    using traitsT = gtsam::traits<T>;
    using Jacobian = typename traitsT::ChartJacobian::Jacobian;

    // --- Step 1: Xi^{-1}
    Jacobian H_inv;
    T Xi_inv = traitsT::Inverse(Xi, H_i ? &H_inv : nullptr);

    // --- Step 2: predicted = Xj * Xi^{-1}
    Jacobian H_pred_Xj, H_pred_Xi_inv;
    T predicted = traitsT::Compose(Xj, Xi_inv, H_j ? &H_pred_Xj : nullptr,
                                   H_i ? &H_pred_Xi_inv : nullptr);

    // --- Step 3: chain rule for Xi
    Jacobian H_pred_Xi;
    if (H_i) {
      H_pred_Xi = H_pred_Xi_inv * H_inv;
    }

    // --- Step 4: error = Log( Z^{-1} * predicted )
    Jacobian H_local;
    gtsam::Vector error = traitsT::Local(measured_, predicted, boost::none,
                                         (H_i || H_j) ? &H_local : nullptr);

    // --- Step 5: final Jacobians
    if (H_i) {
      *H_i = H_local * H_pred_Xi;
    }

    if (H_j) {
      *H_j = H_local * H_pred_Xj;
    }

    return error;
  }

  /// return the measurement
  const VALUE& measured() const { return measured_; }

 private:
  // Alignment, see
  // https://eigen.tuxfamily.org/dox/group__TopicStructHavingEigenMembers.html
  inline constexpr static auto NeedsToAlign = (sizeof(VALUE) % 16) == 0;

 public:
  GTSAM_MAKE_ALIGNED_OPERATOR_NEW_IF(NeedsToAlign)
};

}  // namespace dyno
