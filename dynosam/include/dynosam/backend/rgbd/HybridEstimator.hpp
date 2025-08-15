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

#include <gtsam/base/numericalDerivative.h>
#include <gtsam/geometry/triangulation.h>
#include <gtsam/linear/GaussianFactorGraph.h>  //needed for triangulate
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>  //needed for triangulate

#include "dynosam/backend/Accessor.hpp"
#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/backend/Formulation.hpp"
#include "dynosam/common/Map.hpp"
#include "dynosam/common/StructuredContainers.hpp"  //for FrameRange
#include "dynosam/common/Types.hpp"                 //only needed for factors
#include "dynosam/factors/HybridFormulationFactors.hpp"

namespace dyno {

template <size_t DIM, typename MOTION, typename POSE>
struct MotionFactorTraits {
  static constexpr size_t ZDim = DIM;
  using Motion = MOTION;
  using Pose = POSE;

  using Motions = std::vector<Motion, Eigen::aligned_allocator<Motion>>;
  using Poses = std::vector<Pose, Eigen::aligned_allocator<Pose>>;

  //! Measurement dimension
  using Z = Eigen::Matrix<double, ZDim, 1>;
  using ZVector = std::vector<Z, Eigen::aligned_allocator<Z>>;

  //! Motion dimension
  static constexpr size_t HDim =
      static_cast<size_t>(gtsam::traits<Motion>::dimension);
  //! Pose dimension
  static constexpr size_t XDim =
      static_cast<size_t>(gtsam::traits<Pose>::dimension);
  static constexpr size_t HessianDim = HDim + XDim;
  //! G blocks (derivatives wrt motion, H)
  using MatrixGD = Eigen::Matrix<double, ZDim, HDim>;
  //! F blocks (derivatives wrt pose, X)
  using MatrixFD = Eigen::Matrix<double, ZDim, XDim>;
  //! E blocks (derivatives wrt point, m)
  using MatrixED = Eigen::Matrix<double, ZDim, ZDim>;
  //! Combined GF blocks for Schur compliment
  using MatrixGFD = Eigen::Matrix<double, ZDim, HessianDim>;

  using GBlocks = std::vector<MatrixGD, Eigen::aligned_allocator<MatrixGD>>;
  using FBlocks = std::vector<MatrixFD, Eigen::aligned_allocator<MatrixFD>>;
  using EBlocks = std::vector<MatrixED, Eigen::aligned_allocator<MatrixED>>;
  using GFBlocks = std::vector<MatrixGFD, Eigen::aligned_allocator<MatrixGFD>>;
};

struct SmartMotionFactorParams {
  //! Threshold to decide whether to re-triangulate
  double retriangulation_threshold = 1e-5;
  double landmark_distance_threshold = -1;
  double dyanmic_outlier_rejection_threshold = -1;
};

// should be object centric smart factor... :)
template <size_t DIM = 3u, typename MOTION = gtsam::Pose3,
          typename POSE = gtsam::Pose3>
class SmartMotionFactor : public gtsam::NonlinearFactor,
                          MotionFactorTraits<DIM, MOTION, POSE> {
 public:
  using Base = gtsam::NonlinearFactor;
  using This = SmartMotionFactor<DIM, MOTION, POSE>;
  using MotionTraits = MotionFactorTraits<DIM, MOTION, POSE>;
  using shared_ptr = boost::shared_ptr<This>;

  static constexpr size_t ZDim = MotionTraits::ZDim;
  static constexpr size_t HDim = MotionTraits::HDim;
  static constexpr size_t XDim = MotionTraits::XDim;

  using typename MotionTraits::Z;
  using typename MotionTraits::ZVector;

  using typename MotionTraits::Motion;
  using typename MotionTraits::Pose;

  using typename MotionTraits::Motions;
  using typename MotionTraits::Poses;

  using typename MotionTraits::MatrixED;
  using typename MotionTraits::MatrixFD;
  using typename MotionTraits::MatrixGD;
  using typename MotionTraits::MatrixGFD;

  using typename MotionTraits::EBlocks;
  using typename MotionTraits::FBlocks;
  using typename MotionTraits::GBlocks;
  using typename MotionTraits::GFBlocks;

  static constexpr int HessianDim = MotionTraits::HessianDim;

  SmartMotionFactor(const gtsam::Pose3& L_e,
                    const gtsam::SharedNoiseModel& noise_model,
                    const SmartMotionFactorParams& params = {},
                    std::optional<gtsam::Point3> initial_point_l = {})
      : Base(), L_e_(L_e), noise_model_(noise_model), params_(params) {
    if (initial_point_l) {
      result_ = gtsam::TriangulationResult(initial_point_l.value());
    }
  }
  ~SmartMotionFactor() {}

 public:
  /// Return the dimension (number of rows!) of the factor.
  size_t dim() const override { return ZDim * this->measured_.size(); }

  void add(const Z& measured, const gtsam::Key& motion_key,
           const gtsam::Key& pose_key) {
    if (std::find(keys_.begin(), keys_.end(), motion_key) != keys_.end()) {
      throw std::runtime_error(
          "SmartMotionFactor::add: adding duplicate measurement for motion "
          "key.");
    }
    if (std::find(keys_.begin(), keys_.end(), pose_key) != keys_.end()) {
      throw std::runtime_error(
          "SmartMotionFactor::add: adding duplicate measurement for pose key.");
    }

    this->measured_.push_back(measured);

    this->keys_.push_back(motion_key);
    this->keys_.push_back(pose_key);
    this->motion_keys_.push_back(motion_key);
    this->pose_keys_.push_back(pose_key);
  }

  void print(const std::string& s = "",
             const gtsam::KeyFormatter& keyFormatter =
                 DynoLikeKeyFormatter) const override {
    std::cout << s << "SmartMotionFactor\n";
    std::cout << "result:\n" << result_ << std::endl;
    Base::print("", DynoLikeKeyFormatter);
  }

  /// Return the 2D measurements (ZDim, in general).
  const ZVector& measured() const { return measured_; }

  size_t numMeasurements() const { return this->measured_.size(); }

  // these are no longer in Base::key order!! does this matter?!
  Motions motions(const gtsam::Values& values) const {
    Motions motions;
    for (const auto& k : motion_keys_) {
      motions.push_back(values.at<Motion>(k));
    }
    return motions;
  }

  Poses poses(const gtsam::Values& values) const {
    Poses poses;
    for (const auto& k : pose_keys_) {
      poses.push_back(values.at<Pose>(k));
    }
    return poses;
  }

  /**
   * @brief Get the objects embedded frame {^WL_e} associated with this tracklet
   *
   * @return const gtsam::Pose3&
   */
  const gtsam::Pose3& embeddedFrame() const { return L_e_; }

  double totalReprojectionError(
      const Motions& motions, const Poses& poses,
      boost::optional<gtsam::Point3> external_point = {}) const {
    TriangulationResult point;

    // if (external_point) {
    //   point = TriangulationResult(*external_point);
    // } else {
    //   // do triangulation without updating the internal 3d point result
    //   point = triangulatePoint3Internal(motions, poses);
    // }

    if (external_point) {
      result_ = TriangulationResult(*external_point);
    } else {
      result_ = triangulateSafe(motions, poses);
    }

    if (result_) {
      // All good, just use version in base class
      return this->totalReprojectionError(motions, poses, *result_);
      // else if (params_.degeneracyMode == HANDLE_INFINITY) {
      //   // Otherwise, manage the exceptions with rotation-only factors
      //   Unit3 backprojected = cameras.front().backprojectPointAtInfinity(
      //       this->measured_.at(0));
      //   return totalReprojectionError(cameras, backprojected);
    } else {
      // if we don't want to manage the exceptions we discard the factor
      return 0.0;
    }
  }
  /**
   * Calculate the error of the factor.
   * This is the log-likelihood, e.g. \f$ 0.5(h(x)-z)^2/\sigma^2 \f$ in case of
   * Gaussian. In this class, we take the raw prediction error \f$ h(x)-z \f$,
   * ask the noise model to transform it to \f$ (h(x)-z)^2/\sigma^2 \f$, and
   * then multiply by 0.5. Will be used in "error(Values)" function required by
   * NonlinearFactor base class
   */
  double totalReprojectionError(const Motions& motions, const Poses& poses,
                                const gtsam::Point3& point_l) const {
    gtsam::Vector error = whitenedError(motions, poses, point_l);
    return 0.5 * error.dot(error);
  }

  /**
   * Calculate vector of re-projection errors [h(x)-z] = [cameras.project(p) -
   * z], with the noise model applied.
   */
  gtsam::Vector whitenedError(const Motions& motions, const Poses& poses,
                              const gtsam::Point3& point_l) const {
    gtsam::Vector error = unwhitenedError(motions, poses, point_l);
    if (noise_model_) noise_model_->whitenInPlace(error);
    return error;
  }

  gtsam::Vector unwhitenedError(const Motions& motions, const Poses& poses,
                                const gtsam::Point3& point_l,
                                GBlocks* Gs = nullptr, FBlocks* Fs = nullptr,
                                EBlocks* Es = nullptr) const {
    return This::reprojectionError(motions, poses, point_l, Gs, Fs, Es);
  }

  // TODO: clean up and provide better call structure
  // see SmartProjectionFactor where external point can be used to recalculate
  // the internal result?
  gtsam::Vector reprojectionError(const gtsam::Values& values) const {
    return reprojectionError(motions(values), poses(values));
  }

  template <class... OptArgs,
            typename = std::enable_if_t<sizeof...(OptArgs) != 0>>
  gtsam::Vector reprojectionError(const gtsam::Values& values,
                                  OptArgs&&... optArgs) const {
    return reprojectionError(motions(values), poses(values), (&optArgs)...);
  }

  template <class... OptArgs,
            typename = std::enable_if_t<sizeof...(OptArgs) != 0>>
  gtsam::Vector reprojectionError(const Motions& motions, const Poses& poses,
                                  OptArgs&&... optArgs) const {
    return reprojectionError(motions, poses, (&optArgs)...);
  }

  gtsam::Vector reprojectionError(const Motions& motions, const Poses& poses,
                                  GBlocks* Gs = nullptr, FBlocks* Fs = nullptr,
                                  EBlocks* Es = nullptr) const {
    // copy result to avoid using the mutable result_ in a const function that
    // is called by other heavily templated functions
    const auto result = result_;
    if (result) {
      return this->reprojectionError(motions, poses, *result, Gs, Fs, Es);
    } else {
      return Vector::Zero(ZDim * numMeasurements());
    }
  }

  double error(const gtsam::Values& c) const override {
    if (this->active(c)) {
      return totalReprojectionError(motions(c), poses(c));
    } else {  // else of active flag
      return 0.0;
    }
    return 0;
  }

  gtsam::TriangulationResult point() const { return result_; }
  gtsam::TriangulationResult point(const Values& values) const {
    return triangulateSafe(motions(values), poses(values));
  }

  //   std::shared_ptr<gtsam::GaussianFactor> linearize(
  //       const Values& c) const override {
  //     std::vector<gtsam::Matrix> A(size());

  //     Vector b = -unwhitenedError(x, A);
  //     // check(noiseModel_, b.size());

  //     // Whiten the corresponding system now
  //     if (noiseModel_) noiseModel_->WhitenSystem(A, b);

  //     // Fill in terms, needed to create JacobianFactor below
  //     std::vector<std::pair<Key, Matrix>> terms(size());
  //     for (size_t j = 0; j < size(); ++j) {
  //       terms[j].first = keys()[j];
  //       terms[j].second.swap(A[j]);
  //     }

  //     // TODO pass unwhitened + noise model to Gaussian factor
  //     using noiseModel::Constrained;
  //     if (noiseModel_ && noiseModel_->isConstrained())
  //       return GaussianFactor::shared_ptr(new JacobianFactor(
  //           terms, b,
  //           std::static_pointer_cast<Constrained>(noiseModel_)->unit()));
  //     else {
  //       return GaussianFactor::shared_ptr(new JacobianFactor(terms, b));
  //     }
  //   }

  boost::shared_ptr<gtsam::GaussianFactor> linearize(
      const gtsam::Values& c) const override {
    // TODo: when to retriangulate point?
    // linearizeDamped(values)
    return createHessianFactor(motions(c), poses(c));
    // return directQRElimination(motions(c), poses(c));
  }

  gtsam::SymmetricBlockMatrix createReducedMatrix(
      const gtsam::Values& c) const {
    return createReducedMatrix(motions(c), poses(c));
  }

  gtsam::SymmetricBlockMatrix createReducedMatrix(
      const Motions& motions, const Poses& poses, const double lambda = 0.0,
      bool diagonalDamping = false) const {
    // trinagulate safe -> get point which is used when getting the Jacobians
    // TODO: can we use the LOST values etc by making the cam pose: G = X.inv()
    // * H like we do for reprojection RANSAC? no degeneracy as we are 3D?

    GBlocks Gs;
    FBlocks Fs;
    EBlocks Es;
    gtsam::Vector b;

    // compute G, F, E and b blocks
    computeJacobiansWithTriangulatedPoint(motions, poses, Gs, Fs, Es, b);
    // TODO: add back in will fail tests!!!
    whitenJacobians(Gs, Fs, Es, b);

    const size_t m = numMeasurements();
    CHECK_EQ(m, Es.size());

    gtsam::Matrix E;
    EVectorToMatrix(Es, E);

    // gtsam::print(b, "b term Es ");
    // gtsam::print(E, "stacked Es ");

    // whiten Jacobians (how does this differ now we have 3?)
    // check difference between WhitenSystem and Whiten

    // do shuur compliment to get augmented Jacobian
    // no damping
    gtsam::Matrix EtE = E.transpose() * E;
    const Eigen::Matrix<double, N, N> P = (EtE).inverse();

    GFBlocks GFs;
    GFVectorsToGFBlocks(Gs, Fs, GFs);

    gtsam::SymmetricBlockMatrix augmentedHessian =
        SchurComplement(GFs, E, P, b);
    // how on earth to test

    // LOG(INFO) << augmentedHessian.full();
    // gtsam::Matrix view = augmentedHessian.selfadjointView();
    // LOG(INFO) << view;
    // LOG(INFO) << "Done";

    return augmentedHessian;
  }

  boost::shared_ptr<gtsam::GaussianFactor> directQRElimination(
      const Motions& motions, const Poses& poses) const {
    triangulateSafe(motions, poses);

    GBlocks Gs;
    FBlocks Fs;
    EBlocks Es;
    gtsam::Vector b;

    // compute G, F, E and b blocks
    computeJacobiansWithTriangulatedPoint(motions, poses, Gs, Fs, Es, b);

    // gtsam::SharedDiagonal noise_model =
    // std::dynamic_pointer_cast<gtsam::SharedDiagonal>(

    // )

    // gtsam::GaussianFactorGraph gfg;
    // gtsam::Symbol pointKey('p', 0);
    // for (size_t k = 0; k < Gs.size(); ++k) {
    //   gtsam::Key motion_key = this->motion_keys_.at(k);
    //   gtsam::Key pose_key = this->pose_keys_.at(k);

    //   const gtsam::Matrix& E = Es[k];
    //   const gtsam::Matrix& G = Gs[k];
    //   const gtsam::Matrix& F = Fs[k];
    //   gfg.add(pointKey, E, motion_key, G, pose_key, F,
    //       b.segment < ZDim > (ZDim * k), noise_model_);
    // }

    // gtsam::KeyVector variables;
    // variables.push_back(pointKey);
    // const auto [_, fg] = gfg.eliminatePartialSequential(variables,
    // gtsam::EliminateQR);

    // return boost::make_shared<gtsam::JacobianFactor>(*fg);
  }

  boost::shared_ptr<gtsam::RegularHessianFactor<HDim>> createHessianFactor(
      const Motions& motions, const Poses& poses, const double lambda = 0.0,
      bool diagonalDamping = false) const {
    // retriangulateHere!!!!!
    triangulateSafe(motions, poses);

    // if (params_.degeneracyMode == ZERO_ON_DEGENERACY && !result_) {
    // if (result_.outlier()) {
    //   return boost::make_shared<gtsam::RegularHessianFactor<HDim>>(
    //       this->keys_, constructSymmetricBlockMatrix(m));

    // }
    // else {
    gtsam::SymmetricBlockMatrix augmented_hessian =
        createReducedMatrix(motions, poses, lambda, diagonalDamping);
    return boost::make_shared<gtsam::RegularHessianFactor<HDim>>(
        this->keys_, augmented_hessian);
    // }
  }

  //  private:
  void computeJacobiansWithTriangulatedPoint(const Motions& motions,
                                             const Poses& poses, GBlocks& Gs,
                                             FBlocks& Fs, EBlocks& Es,
                                             gtsam::Vector& b) const {
    if (result_) {
      computeJacobians(motions, poses, *result_, Gs, Fs, Es, b);
    } else {
      // TODO: this is in place of the backprojectPointAtInfinity in the regular
      // SmartProjectionFactor... not sure what we sould do here?
      gtsam::Point3 projected = HybridObjectMotion::projectToObject3(
          poses.at(0), motions.at(0), embeddedFrame(), this->measured_.at(0));
      computeJacobians(motions, poses, projected, Gs, Fs, Es, b);
    }
  }

  void computeJacobians(const Motions& motions, const Poses& poses,
                        const gtsam::Point3& point_l, GBlocks& Gs, FBlocks& Fs,
                        EBlocks& Es, gtsam::Vector& b) const {
    b = -unwhitenedError(motions, poses, point_l, &Gs, &Fs, &Es);

    CHECK_EQ(motions.size(), Gs.size());
    CHECK_EQ(poses.size(), Fs.size());
  }

  void whitenJacobians(GBlocks& Gs, FBlocks& Fs, EBlocks& Es,
                       gtsam::Vector& b) const {
    CHECK(noise_model_);

    for (size_t i = 0; i < Gs.size(); i++) Gs[i] = noise_model_->Whiten(Gs[i]);
    for (size_t i = 0; i < Fs.size(); i++) Fs[i] = noise_model_->Whiten(Fs[i]);
    for (size_t i = 0; i < Es.size(); i++) Es[i] = noise_model_->Whiten(Es[i]);

    b = noise_model_->whiten(b);
  }

  // not actually const as modified result_
  const gtsam::TriangulationResult& triangulateSafe(const Motions& motions,
                                                    const Poses& poses) const {
    if (numMeasurements() < 2) {
      result_ = gtsam::TriangulationResult::Degenerate();
    }

    // what if result is {}?
    bool retriangulate = decideIfTriangulate(motions, poses);
    if (retriangulate) {
      // all triangulate safe logic from gtsam::triangulateSafe foes here,
      // including outlier rejection etc...
      result_ = triangulatePoint3Internal(motions, poses);
    }

    return result_;
  }

  // should be private
  // does triangulation anyway given motions points and measurements
  // does not modify any variables
  gtsam::TriangulationResult triangulatePoint3Internal(
      const Motions& motions, const Poses& poses) const {
    if (numMeasurements() < 2) {
      return gtsam::TriangulationResult::Degenerate();
    }

    gtsam::TriangulationResult result = triangulateLinear(motions, poses);
    if (result) {
      // should we always calculate the non-linear result?
      result = triangulateNonlinear(motions, poses, result.value());
      double max_repr_error = 0.0;
      const gtsam::Point3& m_object = *result;
      // Check landmark distance and re-projection errors to avoid outliers
      for (size_t i = 0; i < numMeasurements(); i++) {
        const auto& object_motion = motions.at(i);
        const auto& camera_pose = poses.at(i);

        const gtsam::Point3 m_camera = HybridObjectMotion::projectToCamera3(
            camera_pose, object_motion, embeddedFrame(), m_object);

        if (params_.landmark_distance_threshold > 0 &&
            gtsam::distance3(camera_pose.translation(), m_camera) >
                params_.landmark_distance_threshold) {
          return TriangulationResult::FarPoint();
        }

        const auto& z = measured_.at(i);
        gtsam::Vector3 reprojection_error = m_camera - z;
        max_repr_error = std::max(max_repr_error, reprojection_error.norm());
      }

      if (params_.dyanmic_outlier_rejection_threshold > 0 &&
          max_repr_error > params_.dyanmic_outlier_rejection_threshold) {
        return TriangulationResult::Outlier();
      }
    }
    return result;
  }

  gtsam::Vector reprojectionError(const Motions& motions, const Poses& poses,
                                  const gtsam::Point3& point_l,
                                  GBlocks* Gs = nullptr, FBlocks* Fs = nullptr,
                                  EBlocks* Es = nullptr) const {
    utils::TimingStatsCollector timer("smf_reprojectionError");
    CHECK_EQ(motions.size(), poses.size());
    const auto measurements = measured();
    CHECK_EQ(motions.size(), measurements.size());

    const size_t m = measurements.size();
    // expected z, ie h(x)
    ZVector z;
    z.reserve(m);

    // Allocate derivatives
    if (Es) Es->resize(m);
    if (Fs) Fs->resize(m);
    if (Gs) Gs->resize(m);

    // Project and fill derivatives
    for (size_t i = 0; i < m; i++) {
      MatrixGD Gi;
      MatrixFD Fi;
      MatrixED Ei;

      // given the point in the object frame and associated camera pose and
      // motion, project the point into the camera frame
      auto project = [&](const Motion& motioni, const Pose& posei,
                         const gtsam::Point3& point_l) -> Z {
        const gtsam::Point3 map_point_world = motioni * (L_e_ * point_l);
        return posei.inverse() * map_point_world;
      };

      Gi = gtsam::numericalDerivative31<gtsam::Vector3, gtsam::Pose3,
                                        gtsam::Pose3, gtsam::Point3>(
          project, motions.at(i), poses.at(i), point_l);

      Fi = gtsam::numericalDerivative32<gtsam::Vector3, gtsam::Pose3,
                                        gtsam::Pose3, gtsam::Point3>(
          project, motions.at(i), poses.at(i), point_l);

      Ei = gtsam::numericalDerivative33<gtsam::Vector3, gtsam::Pose3,
                                        gtsam::Pose3, gtsam::Point3>(
          project, motions.at(i), poses.at(i), point_l);

      z.emplace_back(project(motions.at(i), poses.at(i), point_l));
      if (Gs) (*Gs)[i] = Gi;
      if (Fs) (*Fs)[i] = Fi;
      if (Es) (*Es)[i] = Ei;
      // if (E) E->block<ZDim, N>(ZDim * i, 0) = Ei;
    }

    // Fill Error factor
    gtsam::Vector b(ZDim * m);
    CHECK_EQ(m, z.size());
    for (size_t i = 0, row = 0; i < m; i++, row += ZDim) {
      gtsam::Vector bi = gtsam::traits<Z>::Local(measurements[i], z[i]);
      // if (ZDim == 3 && std::isnan(bi(1))) {  // if it is a stereo point and
      // the
      //                                       // right pixel is missing (nan)
      //   bi(1) = 0;
      // }
      b.segment<ZDim>(row) = bi;
    }
    return b;
  }

  bool decideIfTriangulate(const Motions& motions, const Poses& poses) const {
    // Several calls to linearize will be done from the same linearization
    // point, hence it is not needed to re-triangulate. Note that this is not
    // yet "selecting linearization", that will come later, and we only check if
    // the current linearization is the "same" (up to tolerance) w.r.t. the last
    // time we triangulated the point.
    CHECK_EQ(motions.size(), poses.size());
    const auto measurements = measured();
    CHECK_EQ(motions.size(), measurements.size());

    const size_t m = measurements.size();

    // sanity check: this should never fail!
    CHECK_EQ(camera_poses_triangulation_.size(),
             object_motions_triangulation_.size());

    bool retriangulate = false;

    if (!result_) retriangulate = true;

    // Definitely true if we do not have a previous linearization point or the
    // new linearization point includes more poses.
    // Only check against poses since we know motions and poses have the same
    // size and the cached poses/motions should have the same size
    if (camera_poses_triangulation_.empty() ||
        poses.size() != camera_poses_triangulation_.size())
      retriangulate = true;

    // Otherwise, check poses and motions against cache.
    if (!retriangulate) {
      for (size_t i = 0; i < poses.size(); i++) {
        if (!poses[i].equals(camera_poses_triangulation_[i],
                             params_.retriangulation_threshold) ||
            !motions[i].equals(object_motions_triangulation_[i],
                               params_.retriangulation_threshold)) {
          retriangulate =
              true;  // at least two poses are different, hence we retriangulate
          break;
        }
      }
    }

    // Store the current poses used for triangulation if we will re-triangulate.
    if (retriangulate) {
      camera_poses_triangulation_.clear();
      camera_poses_triangulation_.reserve(m);

      object_motions_triangulation_.clear();
      object_motions_triangulation_.reserve(m);
      for (size_t i = 0; i < m; i++) {
        camera_poses_triangulation_.push_back(poses[i]);
        object_motions_triangulation_.push_back(motions[i]);
      }
    }

    return retriangulate;
  }

  // TODO: add in all gtsam::triangulateSafe checks and return a
  // TriangulationResult
  gtsam::TriangulationResult triangulateLinear(const Motions& motions,
                                               const Poses& poses) const {
    utils::TimingStatsCollector timer("smf_triangulateLinear");
    CHECK_EQ(motions.size(), poses.size());
    CHECK_EQ(motions.size(), measured_.size());

    // need at least two measurements to solve and must have at least 1
    // measurement or graph.optimise() will be empty
    if (numMeasurements() < 2) {
      return gtsam::TriangulationResult::Degenerate();
    }

    gtsam::Key point_key(1);

    gtsam::GaussianFactorGraph graph;

    auto diagonal_noise =
        gtsam::noiseModel::Diagonal::Sigmas(noise_model_->sigmas());

    for (size_t i = 0; i < measured_.size(); i++) {
      // This transform is such that A*m = z where m is th m_L and unknown
      gtsam::Pose3 A = HybridObjectMotion::projectToCamera3Transform(
          poses.at(i), motions.at(i), L_e_);
      // linearize the system
      gtsam::Matrix R = A.rotation().matrix();
      gtsam::Vector3 t = A.translation();

      // R * m_L + t = m_X = z
      // R * m_L = z - t
      gtsam::Vector rhs = measured_.at(i) - t;
      graph.add(gtsam::JacobianFactor(point_key, R, rhs, diagonal_noise));
    }

    // solve linear system by Cholesky Factorization
    gtsam::VectorValues result = graph.optimize();
    gtsam::Point3 m_L_est = result.at(point_key);
    return m_L_est;
  }

  gtsam::TriangulationResult triangulateNonlinear(
      const Motions& motions, const Poses& poses,
      const gtsam::Point3& m_L) const {
    utils::TimingStatsCollector timer("smf_triangulateNonlinear");
    CHECK_EQ(motions.size(), poses.size());
    CHECK_EQ(motions.size(), measured_.size());

    gtsam::Key point_key(1);

    // TODO: robust param
    gtsam::NonlinearFactorGraph graph;
    gtsam::Values values;
    values.insert(point_key, m_L);

    for (size_t i = 0; i < measured_.size(); i++) {
      graph.emplace_shared<ObjectPointTriangulationFactor>(
          point_key, poses.at(i), motions.at(i), L_e_, measured_.at(i),
          noise_model_  // should be robust!?
      );
    }

    gtsam::LevenbergMarquardtOptimizer problem(graph, values);
    gtsam::Values optimised_values = problem.optimize();
    return optimised_values.at<gtsam::Point3>(point_key);
  }

  // TODO: should actually check that Es is the size of the measurements
  // Pull into base class to have static and non static versions
  static void EVectorToMatrix(const EBlocks& Es, gtsam::Matrix& E) {
    size_t m = Es.size();

    E.resize(ZDim * m, ZDim);
    for (size_t i = 0; i < m; i++) {
      const MatrixED& Ei = Es.at(i);
      E.block<ZDim, ZDim>(ZDim * i, 0) = Ei;
    }
  }

  static void GFVectorsToGFBlocks(const GBlocks& Gs, const FBlocks& Fs,
                                  GFBlocks& GFs) {
    CHECK_EQ(Gs.size(), Fs.size());

    GFs.resize(Fs.size());

    for (size_t i = 0; i < Fs.size(); i++) {
      const MatrixGD& Gi = Gs.at(i);
      const MatrixFD& Fi = Fs.at(i);
      MatrixGFD GFi;

      // LOG(INFO) << "Gi:\n " << Gi;
      // LOG(INFO) << "Fi:\n " << Fi;
      // gtsam::print(Fi, "Fi ");

      // concat blocks horizontally
      GFi.block(0, 0, ZDim, HDim) = Gi;
      GFi.block(0, HDim, ZDim, XDim) = Fi;
      GFs[i] = GFi;

      // LOG(INFO) << "GFs[i]:\n " << GFs[i];

      // gtsam::print(GFs[i], "GFs[i] ");
    }
  }

  static constexpr int N = 3;  // point size //only used for schur compliment?

  static gtsam::SymmetricBlockMatrix constructSymmetricBlockMatrix(
      size_t m, std::optional<gtsam::Matrix> M = {}) {
    // Create a SymmetricBlockMatrix (augmented hessian, with extra row/column
    // with info vector)
    size_t M1 = HessianDim * m + 1;
    // 2 * m is 2 blocks (H and x) for each measurement
    // dims are filled with HDim since dimensions for H and X are the same!
    std::vector<Eigen::DenseIndex> dims(2 * m +
                                        1);  // this also includes the b term
    // vertical dimensions of ATA
    std::fill(dims.begin(), dims.end() - 1, HDim);
    dims.back() = 1;

    if (M) {
      // check sizes
      if (M->rows() != (Eigen::Index)M1 || M->cols() != (Eigen::Index)M1) {
        std::stringstream ss;
        ss << "Symmetric Block Matrix construction failed: input"
              " augmented hessian has incorrect size ("
           << M->rows() << " x " << M->cols()
           << ")"
              " when it should be a sqaure ("
           << M1 << " x " << M1 << ")";
        throw std::runtime_error(ss.str());
      } else {
        return gtsam::SymmetricBlockMatrix(dims, *M);
      }
    }

    return gtsam::SymmetricBlockMatrix(dims, gtsam::Matrix::Zero(M1, M1));
  }

  // N should be point dimension
  // JDim = HDim = XDim (we assume these are the same size!!!)
  //  static gtsam::SymmetricBlockMatrix SchurComplement(
  //      const GFBlocks& GFs, const gtsam::Matrix& E,
  //      const Eigen::Matrix<double, N, N>& P, const gtsam::Vector& b) {

  //   utils::TimingStatsCollector timer("smf_SchurComplement");
  //     size_t m = GFs.size();

  //   // SymmetricBlockMatrix augmentedHessian(dims, Matrix::Zero(M1, M1));
  //   //make zero-filled block matrix of the right dimensions
  //   SymmetricBlockMatrix augmentedHessian = constructSymmetricBlockMatrix(m);

  //   const size_t last_block_idx = 2*m;

  //   // Blockwise Schur complement
  //   for (size_t i = 0; i < m; ++i) {
  //     const auto& GFi = GFs[i];
  //     const Eigen::Matrix<double, ZDim, HDim>& Gi = GFi.block(0, 0, ZDim,
  //     HDim); const Eigen::Matrix<double, ZDim, XDim>& Fi = GFi.block(0, HDim,
  //     ZDim, XDim); const Eigen::Matrix<double, ZDim, N>& Ei = E.block(ZDim *
  //     i, 0, ZDim, N);

  //     //TODO: should return const ref?
  //     auto A = [&](size_t idx, bool is_G) -> Eigen::Matrix<double, ZDim,
  //     HDim> {
  //       if(is_G) return GFs[idx].block(0, 0, ZDim, HDim);
  //       else return GFs[idx].block(0, HDim, ZDim, XDim);
  //     };

  //     {
  //       //construct linear gradient term
  //       Eigen::Matrix<double, ZDim, 1> inner_sum;
  //       for (size_t j = 0; j < m; ++j) {
  //         const Eigen::Matrix<double, ZDim, N>& Ej = E.block(ZDim * j, 0,
  //         ZDim, N); inner_sum += -Ei * P * Ej.transpose() *
  //         b.segment<ZDim>(ZDim * j);
  //       }
  //       inner_sum += b.segment<ZDim>(ZDim * i);

  //       //block dimension
  //       Eigen::Matrix<double, HDim, 1> linear_term_G = Gi.transpose() *
  //       inner_sum; Eigen::Matrix<double, HDim, 1> linear_term_F =
  //       Fi.transpose() * inner_sum;

  //       const int g_block_idx = 2 * i;
  //       const int f_block_idx = 2 * i + 1;

  //       //2m takes us to the last block
  //       augmentedHessian.setOffDiagonalBlock(
  //         g_block_idx, last_block_idx, linear_term_G);

  //       augmentedHessian.setOffDiagonalBlock(
  //         f_block_idx, last_block_idx, linear_term_F);

  //     }

  //     for (size_t j = i; j < m; ++j) {
  //       const Eigen::Matrix<double, ZDim, N>& Ej = E.block(ZDim * j, 0, ZDim,
  //       N); const Eigen::Matrix<double, ZDim, N>& Ei = E.block(ZDim * i, 0,
  //       ZDim, N);

  //       for (int r_is_G = 0; r_is_G <= 1; ++r_is_G) {
  //         for (int c_is_G = 0; c_is_G <= 1; ++c_is_G) {
  //           bool rowIsG = static_cast<bool>(r_is_G);
  //           bool colIsG = static_cast<bool>(c_is_G);

  //           int rowIndex = 2 * static_cast<int>(i) + r_is_G;  // r_is_G ∈ {0,
  //           1} int colIndex = 2 * static_cast<int>(j) + c_is_G;

  //           Eigen::Matrix<double, ZDim, HDim> Ai, Aj;
  //           Ai = A(i, rowIsG);
  //           Aj = A(j, colIsG);

  //           Eigen::Matrix<double, HDim, HDim> block_value;

  //           if (i == j) {
  //             // diagonal block
  //             block_value = Ai.transpose() * Aj - Ai.transpose() * Ei * P *
  //             Ei.transpose() * Aj;
  //           }
  //           else {
  //             block_value = -Ai.transpose() * Ei * P * Ej.transpose() * Aj;
  //           }

  //           if(rowIndex == colIndex) {
  //             augmentedHessian.setDiagonalBlock(rowIndex, block_value);
  //           }
  //           else {
  //             augmentedHessian.setOffDiagonalBlock(rowIndex, colIndex,
  //             block_value);
  //           }
  //         }
  //       }
  //     }
  //   }
  //   augmentedHessian.diagonalBlock(last_block_idx)(0, 0) += b.squaredNorm();
  //   return augmentedHessian;
  // }

  //   static gtsam::SymmetricBlockMatrix SchurComplement(
  //     const GFBlocks& GFs, const gtsam::Matrix& E,
  //     const Eigen::Matrix<double, N, N>& P, const gtsam::Vector& b) {

  //   utils::TimingStatsCollector timer("smf_SchurComplement");
  //   const size_t m = GFs.size();
  //   const size_t last_block_idx = 2 * m;

  //   SymmetricBlockMatrix augmentedHessian = constructSymmetricBlockMatrix(m);

  //   // Accessor for G or F blocks
  //   auto A = [&](size_t idx, bool is_G) -> Eigen::Ref<const
  //   Eigen::Matrix<double, ZDim, HDim>> {
  //         if(is_G) return GFs[idx].block(0, 0, ZDim, HDim);
  //         else return GFs[idx].block(0, HDim, ZDim, XDim);
  //   };

  //   // Precompute Eᵢ, Eᵢᵀ, bᵢ to avoid recomputation
  //   std::vector<Eigen::Matrix<double, ZDim, N>> E_blocks(m);
  //   std::vector<Eigen::Matrix<double, N, ZDim>> E_transpose(m);
  //   std::vector<Eigen::Matrix<double, ZDim, 1>> b_segments(m);

  //   for (size_t i = 0; i < m; ++i) {
  //     E_blocks[i] = E.block<ZDim, N>(ZDim * i, 0);
  //     E_transpose[i] = E_blocks[i].transpose();
  //     b_segments[i] = b.segment<ZDim>(ZDim * i);
  //   }

  //   for (size_t i = 0; i < m; ++i) {
  //     const auto& Ei = E_blocks[i];
  //     const auto& EiT = E_transpose[i];

  //     // Construct linear term: inner_sum = bᵢ - Eᵢ P ∑ⱼ (Eⱼᵀ bⱼ)
  //     Eigen::Matrix<double, ZDim, 1> inner_sum = b_segments[i];
  //     for (size_t j = 0; j < m; ++j) {
  //       inner_sum.noalias() -= Ei * P * E_transpose[j] * b_segments[j];
  //     }

  //     // Compute linear terms
  //     const auto& Gi = A(i, true);
  //     const auto& Fi = A(i, false);

  //     const Eigen::Matrix<double, HDim, 1> linear_term_G = Gi.transpose() *
  //     inner_sum; const Eigen::Matrix<double, HDim, 1> linear_term_F =
  //     Fi.transpose() * inner_sum;

  //     const int g_block_idx = 2 * static_cast<int>(i);
  //     const int f_block_idx = g_block_idx + 1;

  //     augmentedHessian.setOffDiagonalBlock(g_block_idx, last_block_idx,
  //     linear_term_G); augmentedHessian.setOffDiagonalBlock(f_block_idx,
  //     last_block_idx, linear_term_F);

  //     // Quadratic terms
  //     for (size_t j = i; j < m; ++j) {
  //       const auto& Ej = E_blocks[j];
  //       const auto& EjT = E_transpose[j];

  //       for (int r_is_G = 0; r_is_G <= 1; ++r_is_G) {
  //         for (int c_is_G = 0; c_is_G <= 1; ++c_is_G) {
  //           const int row_index = 2 * static_cast<int>(i) + r_is_G;
  //           const int col_index = 2 * static_cast<int>(j) + c_is_G;

  //           const auto& Ai = A(i, r_is_G);
  //           const auto& Aj = A(j, c_is_G);

  //           Eigen::Matrix<double, HDim, HDim> block_value;

  //           if (i == j) {
  //             block_value.noalias() = Ai.transpose() * Aj;
  //             block_value.noalias() -= Ai.transpose() * Ei * P * EiT * Aj;
  //           } else {
  //             block_value.noalias() = -Ai.transpose() * Ei * P * EjT * Aj;
  //           }

  //           if (row_index == col_index) {
  //             augmentedHessian.setDiagonalBlock(row_index, block_value);
  //           } else {
  //             augmentedHessian.setOffDiagonalBlock(row_index, col_index,
  //             block_value);
  //           }
  //         }
  //       }
  //     }
  //   }

  //   // Final scalar term (bottom right block)
  //   augmentedHessian.diagonalBlock(last_block_idx)(0, 0) += b.squaredNorm();
  //   return augmentedHessian;
  // }

  static gtsam::SymmetricBlockMatrix SchurComplement(
      const GFBlocks& GFs, const gtsam::Matrix& E,
      const Eigen::Matrix<double, N, N>& P, const gtsam::Vector& b) {
    utils::TimingStatsCollector timer("smf_SchurComplement");
    // a single point is observed in m cameras
    size_t m = GFs.size();
    // gtsam::Matrix Et = E.transpose();

    gtsam::Matrix F_block_matrix(m * 3, m * HessianDim);
    F_block_matrix.setZero();

    for (size_t i = 0; i < m; i++) {
      const Eigen::Matrix<double, 3, HessianDim>& GFblock = GFs.at(i);
      F_block_matrix.block<3, HessianDim>(3 * i, HessianDim * i) = GFblock;
    }

    // TODO: F block should be close to diagonal - inverting can be made much
    // faster!!!!
    auto ft_timer =
        std::make_unique<utils::TimingStatsCollector>("smf_F_transpose");
    gtsam::Matrix F = F_block_matrix;
    gtsam::Matrix Ft = F.transpose();
    ft_timer.reset();

    auto gg_timer =
        std::make_unique<utils::TimingStatsCollector>("smf_Gg_calc");
    // gtsam::Matrix g = Ft * (b - E * P * Et * b);
    // gtsam::Matrix G = Ft * F - Ft * E * P * Et * F;

    // gtsam::Matrix EPtF = E.transpose() * F;  // 3×n
    // gtsam::Matrix EPtb = E.transpose() * b;  // 3×1

    // gtsam::Matrix P_EPtF = P * EPtF;         // 3×n
    // gtsam::Vector P_EPtb = P * EPtb;         // 3×1

    // gtsam::Matrix G = Ft * F - Ft * E * P_EPtF;
    // gtsam::Vector g = Ft * b - Ft * E * P_EPtb;

    // Step 1: Compute full QR decomposition of E
    Eigen::HouseholderQR<gtsam::Matrix> qr(E);
    gtsam::Matrix Q = qr.householderQ();  // Q is m x m

    const int nm = E.rows();  // number of residuals
    const int nr = E.cols();  // should be 3

    // Step 2: Extract the nullspace basis N = Q.rightCols(m - r)
    const int null_dim = nm - nr;
    gtsam::Matrix N = Q.rightCols(null_dim);  // m x (m - 3)

    // Step 3: Project F and b into the nullspace
    gtsam::Matrix NF = N.transpose() * F;  // (m-3) x n
    gtsam::Vector Nb = N.transpose() * b;  // (m-3) x 1

    // Step 4: Build reduced system
    gtsam::Matrix G = NF.transpose() * NF;  // n x n
    gtsam::Matrix g = NF.transpose() * Nb;  // n x 1

    gg_timer.reset();

    // size of schur = num measurements * Hessian size + 1
    auto shur_timer =
        std::make_unique<utils::TimingStatsCollector>("smf_schur");
    size_t aug_hessian_size = m * HessianDim + 1;
    gtsam::Matrix schur(aug_hessian_size, aug_hessian_size);

    schur << G, g, g.transpose(), b.squaredNorm();
    shur_timer.reset();

    std::vector<Eigen::DenseIndex> dims(2 * m + 1);  // includes b term
    std::fill(dims.begin(), dims.end() - 1,
              HDim);  // assuming HDim and Xdim are the same size
    dims.back() = 1;

    gtsam::SymmetricBlockMatrix augmented_hessian(dims, schur);
    return augmented_hessian;
  }

 protected:
  /**
   * @brief Nonlinear factor used to refine the support variable m during
   * triangulation
   *
   */
  class ObjectPointTriangulationFactor
      : public gtsam::NoiseModelFactor1<gtsam::Point3> {
   public:
    typedef boost::shared_ptr<ObjectPointTriangulationFactor> shared_ptr;
    typedef ObjectPointTriangulationFactor This;
    typedef gtsam::NoiseModelFactor1<gtsam::Point3> Base;

    ObjectPointTriangulationFactor(gtsam::Key m_key, const gtsam::Pose3& X_k,
                                   const gtsam::Pose3& e_H_k_world,
                                   const gtsam::Pose3& L_e,
                                   const gtsam::Point3& z_k,
                                   gtsam::SharedNoiseModel model)
        : Base(model, m_key),
          X_k_(X_k),
          e_H_k_world_(e_H_k_world),
          L_e_(L_e),
          z_k_(z_k) {}

    gtsam::Vector evaluateError(
        const gtsam::Point3& m_L,
        boost::optional<gtsam::Matrix&> J1 = boost::none) const override {
      if (J1) {
        // error w.r.t to object point
        Eigen::Matrix<double, 3, 3> df_dm =
            gtsam::numericalDerivative11<gtsam::Vector3, gtsam::Point3>(
                std::bind(&HybridObjectMotion::residual, X_k_, e_H_k_world_,
                          std::placeholders::_1, z_k_, L_e_),
                m_L);
        *J1 = df_dm;
      }

      return HybridObjectMotion::residual(X_k_, e_H_k_world_, m_L, z_k_, L_e_);
    }

   private:
    gtsam::Pose3 X_k_;
    gtsam::Pose3 e_H_k_world_;
    gtsam::Pose3 L_e_;
    gtsam::Point3 z_k_;
  };

 protected:
  const gtsam::Pose3 L_e_;
  SmartMotionFactorParams params_;

  // TODO: not used
  //  Cache for Fblocks, to avoid a malloc ever time we re-linearize
  //  mutable FBlocks Fs;
  //  // Cache for Fblocks, to avoid a malloc ever time we re-linearize
  //  mutable GBlocks Gs;

  ZVector measured_;

  gtsam::SharedNoiseModel noise_model_;
  gtsam::KeyVector motion_keys_;
  gtsam::KeyVector pose_keys_;

  // making this mutable seems to break things... :)
  mutable gtsam::TriangulationResult result_;
  mutable Poses camera_poses_triangulation_;  //! current triangulation poses
  mutable Motions
      object_motions_triangulation_;  //! current triangulation object motions
};

using HybridSmartFactor = SmartMotionFactor<3, gtsam::Pose3, gtsam::Pose3>;

struct HybridFormulationProperties {
  inline gtsam::Symbol makeDynamicKey(TrackletId tracklet_id) const {
    return (gtsam::Symbol)DynamicLandmarkSymbol(0u, tracklet_id);
  }
};

using KeyFrameData = MultiFrameRangeData<ObjectId, gtsam::Pose3>;
using KeyFrameRanges = KeyFrameData::FrameRangeDataTVector;
using KeyFrameRange = KeyFrameData::FrameRangeT;

/**
 * @brief Information shared from the HybridFormulation to the HybridAccessor.
 *
 */
struct SharedHybridFormulationData {
  //! Maps an object j with an embedded frame (gtsam::Pose3) for a range of
  //! timesteps k. Each embedded frame L_e is defined such that the range is e
  //! to e + N.
  const KeyFrameData* key_frame_data;
  //! Tracklet Id to the embedded frame (e) the point is represented in (ie.
  //! which timestep, k)
  const gtsam::FastMap<TrackletId, FrameId>* tracklet_id_to_keyframe;
};

class HybridAccessor : public Accessor<MapVision>,
                       public HybridFormulationProperties {
 public:
  DYNO_POINTER_TYPEDEFS(HybridAccessor)

  HybridAccessor(
      const SharedFormulationData& shared_data, MapVision::Ptr map,
      const SharedHybridFormulationData& shared_hybrid_formulation_data)
      : Accessor<MapVision>(shared_data, map),
        shared_hybrid_formulation_data_(shared_hybrid_formulation_data) {}
  virtual ~HybridAccessor() {}

  StateQuery<gtsam::Pose3> getSensorPose(FrameId frame_id) const override;
  StateQuery<gtsam::Pose3> getObjectMotion(FrameId frame_id,
                                           ObjectId object_id) const override;
  StateQuery<gtsam::Pose3> getObjectPose(FrameId frame_id,
                                         ObjectId object_id) const override;
  StateQuery<gtsam::Point3> getDynamicLandmark(
      FrameId frame_id, TrackletId tracklet_id) const override;
  // in thie case we can actually propogate all object points ;)
  // if frame id does not exist or the object does not exist at this frame,
  // return an empty vector!!
  // use with, does object exist and does frame exist perhaps...?
  StatusLandmarkVector getDynamicLandmarkEstimates(
      FrameId frame_id, ObjectId object_id) const override;

  std::optional<Motion3ReferenceFrame> getRelativeLocalMotion(
      FrameId frame_id, ObjectId object_id) const;

  /**
   * @brief Get all dynamic object estimates in the local object frame
   *
   * @param object_id
   * @return StatusLandmarkVector
   */
  StatusLandmarkVector getLocalDynamicLandmarkEstimates(
      ObjectId object_id) const;

 private:
  struct DynamicLandmarkQuery {
    StateQuery<gtsam::Point3>* query_m_W = nullptr;
    StateQuery<gtsam::Point3>* query_m_L = nullptr;
    StateQuery<gtsam::Pose3>* query_H_W_e_k = nullptr;
    KeyFrameRange::ConstPtr* frame_range_ptr = nullptr;
  };

  // TODO: dont need this - evenetually put structureless etc into an
  // encapsualted class
  virtual StateQuery<gtsam::Point3> queryPoint(gtsam::Key point_key,
                                               TrackletId tracklet_id) const;

  // should treturn true if and only if all valid queries (ie, non-null queries)
  // were set with valid dataa!!
  bool getDynamicLandmarkImpl(FrameId frame_id, TrackletId tracklet_id,
                              DynamicLandmarkQuery& query) const;

  bool getDynamicLandmarkImpl(FrameId frame_id, TrackletId tracklet_id,
                              StateQuery<gtsam::Point3>* query_m_W,
                              StateQuery<gtsam::Point3>* query_m_L,
                              StateQuery<gtsam::Pose3>* query_H_W_e_k,
                              KeyFrameRange::ConstPtr* frame_range_ptr) const;

 private:
  const SharedHybridFormulationData shared_hybrid_formulation_data_;
};

// TODO: for future proofing with new measurement stuff the formulation (at the
// top level)
//  should be templated on the map and then we use type traits to extract the
//  measurements care about so it can be measurement generic... eventually need
//  way to define (AND CHECK, becuase we cannot assume all types have the same
//  compile-time properties) and get the measurement we are interested in
class HybridFormulation : public Formulation<MapVision>,
                          public HybridFormulationProperties {
 public:
  using Base = Formulation<MapVision>;
  using Base::AccessorTypePointer;
  using Base::MapTraitsType;
  using Base::ObjectUpdateContextType;
  using Base::PointUpdateContextType;

  DYNO_POINTER_TYPEDEFS(HybridFormulation)

  HybridFormulation(const FormulationParams& params, typename Map::Ptr map,
                    const NoiseModels& noise_models, const Sensors& sensors,
                    const FormulationHooks& hooks)
      : Base(params, map, noise_models, sensors, hooks) {}
  virtual ~HybridFormulation() {}

  virtual void dynamicPointUpdateCallback(
      const PointUpdateContextType& context, UpdateObservationResult& result,
      gtsam::Values& new_values,
      gtsam::NonlinearFactorGraph& new_factors) override;
  virtual void objectUpdateContext(
      const ObjectUpdateContextType& context, UpdateObservationResult& result,
      gtsam::Values& new_values,
      gtsam::NonlinearFactorGraph& new_factors) override;

  inline bool isDynamicTrackletInMap(
      const typename MapTraitsType::LandmarkNodePtr& lmk_node) const override {
    const TrackletId tracklet_id = lmk_node->tracklet_id;
    return is_dynamic_tracklet_in_map_.exists(tracklet_id);
  }

  // TODO: functions should be shared with accessor
  bool getObjectKeyFrameHistory(ObjectId object_id,
                                const KeyFrameRanges* ranges) const;

  bool hasObjectKeyFrame(ObjectId object_id, FrameId frame_id) const;
  std::pair<FrameId, gtsam::Pose3> getObjectKeyFrame(ObjectId object_id,
                                                     FrameId frame_id) const;
  // get the estimated motion in the representation used directly by the
  // estimation (ie. not frame-2-frame)
  // TODO: should be in accessor!!!!
  StateQuery<Motion3ReferenceFrame> getEstimatedMotion(ObjectId object_id,
                                                       FrameId frame_id) const;

  std::pair<FrameId, gtsam::Pose3> forceNewKeyFrame(FrameId frame_id,
                                                    ObjectId object_id);
  /**
   * @brief
   *
   * frame_id is not necessary a keyframe but will be used to search for the
   * keyframe in the range
   *
   * @param frame_id
   * @return TrackletIds
   */
  TrackletIds collectPointsAtKeyFrame(ObjectId object_id,
                                      FrameId frame_id) const;

 protected:
  // TODO: make this virtual for now - eventual move structureless etc
  // properties into a class to encapsulate!
  virtual AccessorTypePointer createAccessor(
      const SharedFormulationData& shared_data) const override {
    SharedHybridFormulationData shared_hybrid_data;
    shared_hybrid_data.key_frame_data = &key_frame_data_;
    shared_hybrid_data.tracklet_id_to_keyframe = &all_dynamic_landmarks_;

    return std::make_shared<HybridAccessor>(shared_data, this->map(),
                                            shared_hybrid_data);
  }

  virtual std::string loggerPrefix() const override { return "hybrid"; }

 protected:
  std::pair<FrameId, gtsam::Pose3> getOrConstructL0(ObjectId object_id,
                                                    FrameId frame_id);

  // hacky update solution for now!!
  gtsam::Pose3 computeInitialH(ObjectId object_id, FrameId frame_id,
                               bool* keyframe_updated = nullptr);

  gtsam::Pose3 calculateObjectCentroid(ObjectId object_id,
                                       FrameId frame_id) const;

  // TODO: in the sliding window case the formulation gets reallcoated every
  // time so that L0 map is different, but the values will share the same H
  // (which is now from a different L0)!! make static (hack) for now
  // TODO: bad!! should not be static as this will also get held between
  // estimators!!!?
  // gtsam::FastMap<ObjectId, std::pair<FrameId, gtsam::Pose3>> L0_;
  // gtsam::FastMap<ObjectId, std::vector<KeyFrameRange>> L0_;

  // we need a separate way of tracking if a dynamic tracklet is in the map,
  // since each point is modelled uniquely simply used as an O(1) lookup, the
  // value is not actually used. If the key exists, we assume that the tracklet
  // is in the map
  // tracklet Id associated with reference frame (to track which KeyFrame the
  // point is in!!!)
  gtsam::FastMap<TrackletId, FrameId>
      is_dynamic_tracklet_in_map_;  //! the set of dynamic points that have been
                                    //! added by this updater. We use a separate
                                    //! map containing the tracklets as the keys
                                    //! are non-unique. This indicates values
                                    //! that are currently in the estimator so
                                    //! if values are removed from the
                                    //! esstimator they need to be updated here
                                    //! too!
  //! All tracks on the object and their keyframe and is shared with the
  //! accessor. This does not get cleared when the a new keyframe is added so
  //! that accessor still has access to the meta-data for each tracked point.
  gtsam::FastMap<TrackletId, FrameId> all_dynamic_landmarks_;

  //! set to track if a smoothing factor has been added to the graph so as to
  //! not add it multiple times We use the greatest (ie last) key of the ternary
  //! smoothing factor as the key!
  gtsam::KeySet smoothing_factors_added_;

 protected:
  KeyFrameData key_frame_data_;
};

// additional functionality when solved with the Regular Backend!
class RegularHybridFormulation : public HybridFormulation {
 public:
  using Base = HybridFormulation;

  RegularHybridFormulation(const FormulationParams& params,
                           typename Map::Ptr map,
                           const NoiseModels& noise_models,
                           const Sensors& sensors,
                           const FormulationHooks& hooks)
      : Base(params, map, noise_models, sensors, hooks) {}

  // using previous (postUdpate) check when the last time an object was updated
  // (in the estimator) if more than 1 frame ago, create new keyframe - this
  // will then take affect as this function is called prior to the graph
  // construction!
  void preUpdate(const PreUpdateData& data) override;
  // use post update information to set internal data about when objects were
  // last udpated!!
  virtual void postUpdate(const PostUpdateData& data) override;

 protected:
  struct ObjectUpdateData {
    FrameId frame_id{0};  //! Last time the object was updted in the estimator
    size_t count{0};  //! Number of (total) times the object has been updated.
                      //! If 1, then new

    inline bool isNew() const { return count == 1u; }
  };
  // The last frame
  gtsam::FastMap<ObjectId, ObjectUpdateData> objects_update_data_;
};

}  // namespace dyno
