
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>

#include <Eigen/Core>
#include <chrono>
#include <cmath>
#include <future>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace {

struct PoseError {
  double translation;
  double rotation;
};

PoseError poseError(const gtsam::Pose3& reference,
                    const gtsam::Pose3& estimate) {
  const gtsam::Pose3 error = reference.between(estimate);

  return {error.translation().norm(),
          gtsam::Rot3::Logmap(error.rotation()).norm()};
}

gtsam::Pose3 makePose(double x, double y, double z, double roll, double pitch,
                      double yaw) {
  return gtsam::Pose3(gtsam::Rot3::RzRyRx(roll, pitch, yaw),
                      gtsam::Point3(x, y, z));
}

gtsam::Pose3 convertReferenceFrame(const gtsam::Pose3& L_W_A,
                                   const gtsam::Pose3& L_W_B,
                                   const gtsam::Pose3& L_W_k) {
  const gtsam::Pose3 H_W_B_k = L_W_k * L_W_B.inverse();
  const gtsam::Pose3 H_W_A_B = L_W_B * L_W_A.inverse();
  const gtsam::Pose3 H_W_A_k = H_W_B_k * H_W_A_B;

  return H_W_A_k * L_W_A;
}

void expectValidRotation(const gtsam::Pose3& pose, double tolerance) {
  const Eigen::Matrix3d R = pose.rotation().matrix();

  EXPECT_TRUE(R.allFinite());
  EXPECT_NEAR(R.determinant(), 1.0, tolerance);

  const Eigen::Matrix3d orthogonality_error =
      R.transpose() * R - Eigen::Matrix3d::Identity();

  EXPECT_LT(orthogonality_error.norm(), tolerance);
}

}  // namespace

TEST(PoseNumericalStability, ReferenceFrameConversionPreservesPose) {
  const gtsam::Pose3 L_W_A = makePose(1.2, -0.5, 2.0, 0.1, -0.2, 0.3);

  const gtsam::Pose3 L_W_B = makePose(3.0, 1.5, -0.7, -0.3, 0.15, 0.6);

  const gtsam::Pose3 L_W_k = makePose(-2.0, 4.0, 1.1, 0.2, 0.4, -0.5);

  const gtsam::Pose3 reconstructed = convertReferenceFrame(L_W_A, L_W_B, L_W_k);

  const PoseError error = poseError(L_W_k, reconstructed);

  EXPECT_LT(error.translation, 1e-10);
  EXPECT_LT(error.rotation, 1e-10);

  expectValidRotation(reconstructed, 1e-10);
}

TEST(PoseNumericalStability, RepeatedReferenceFrameConversionDoesNotDrift) {
  //   const gtsam::Pose3 L_W_A =
  //       makePose(1.2, -0.5, 2.0, 0.1, -0.2, 0.3);

  //   const gtsam::Pose3 L_W_B =
  //       makePose(3.0, 1.5, -0.7, -0.3, 0.15, 0.6);

  //   const gtsam::Pose3 original =
  //       makePose(-2.0, 4.0, 1.1, 0.2, 0.4, -0.5);
  const gtsam::Pose3 L_W_A =
      makePose(0.0012, -0.0005, 0.0020, 0.001, -0.002, 0.003);

  const gtsam::Pose3 L_W_B =
      makePose(-0.180, -0.020, -0.305, -0.003, 0.002, 0.001);

  const gtsam::Pose3 original =
      makePose(-0.2006, -0.0193, -0.3050, 0.001, -0.003, 0.002);

  gtsam::Pose3 current = original;

  constexpr size_t kMaxIterations = 100000;

  for (size_t i = 0; i < kMaxIterations; ++i) {
    current = convertReferenceFrame(L_W_A, L_W_B, current);

    if (!current.rotation().matrix().allFinite() ||
        !current.translation().allFinite()) {
      FAIL() << "Non-finite pose at iteration " << i;
    }
  }

  const PoseError error = poseError(original, current);

  EXPECT_LT(error.translation, 1e-8);
  EXPECT_LT(error.rotation, 1e-8);

  expectValidRotation(current, 1e-8);
}

TEST(PoseNumericalStability, CompositionWithInversePreservesIdentity) {
  const gtsam::Pose3 T = makePose(3.0, -2.0, 1.5, 0.4, -0.3, 0.8);

  gtsam::Pose3 result = gtsam::Pose3::Identity();

  constexpr size_t kMaxIterations = 100000;

  for (size_t i = 0; i < kMaxIterations; ++i) {
    result = result * T * T.inverse();

    if (!result.rotation().matrix().allFinite() ||
        !result.translation().allFinite()) {
      FAIL() << "Non-finite pose at iteration " << i;
    }
  }

  const PoseError error = poseError(gtsam::Pose3::Identity(), result);

  EXPECT_LT(error.translation, 1e-8);
  EXPECT_LT(error.rotation, 1e-8);

  expectValidRotation(result, 1e-8);
}

#include <gtest/gtest.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>

#include <Eigen/Core>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

namespace {

using Matrix3 = Eigen::Matrix3d;

struct RotationError {
  double orthogonality;
  double determinant_error;
  double max_element_error;
};

RotationError measureRotationError(const Matrix3& R) {
  const Matrix3 residual = R.transpose() * R - Matrix3::Identity();

  return {residual.norm(), std::abs(R.determinant() - 1.0),
          residual.cwiseAbs().maxCoeff()};
}

void printRotationError(const std::string& label, const Matrix3& R) {
  const RotationError error = measureRotationError(R);

  std::cout << std::scientific << std::setprecision(8) << label
            << " | orthogonality = " << error.orthogonality
            << " | determinant error = " << error.determinant_error
            << " | max element error = " << error.max_element_error << '\n';
}

TEST(RotationNumericalStability, RepeatedEigenMatrixComposition) {
  // Construct the initial rotations using GTSAM.
  const gtsam::Rot3 initial_rotation = gtsam::Rot3::RzRyRx(0.31, -0.27, 0.19);

  const gtsam::Rot3 increment_rotation =
      gtsam::Rot3::RzRyRx(0.0013, -0.0021, 0.0017);

  // Extract their matrices.
  Matrix3 R = initial_rotation.matrix();
  const Matrix3 dR = increment_rotation.matrix();

  std::cout << "\n=== Repeated Eigen matrix composition ===\n";

  printRotationError("Initial", R);
  printRotationError("Increment", dR);

  constexpr int kIterations = 1000000;

  for (int i = 1; i <= kIterations; ++i) {
    // Deliberately perform raw matrix multiplication.
    // No projection back onto SO(3) is performed.
    R = R * dR;

    if (i == 10 || i == 100 || i == 1000 || i == 10000 || i == 100000 ||
        i == kIterations) {
      printRotationError("Iteration " + std::to_string(i), R);
    }
  }

  const RotationError error = measureRotationError(R);

  // This is a diagnostic stress test, not a requirement that
  // every platform produce a particular amount of drift.
  EXPECT_TRUE(R.allFinite());
  EXPECT_TRUE(std::isfinite(error.orthogonality));
  EXPECT_TRUE(std::isfinite(error.determinant_error));

  std::cout << "Final orthogonality error: " << error.orthogonality << '\n';
}

TEST(RotationNumericalStability, RepeatedGtsamRot3Composition) {
  gtsam::Rot3 rotation = gtsam::Rot3::RzRyRx(0.31, -0.27, 0.19);

  const gtsam::Rot3 increment = gtsam::Rot3::RzRyRx(0.0013, -0.0021, 0.0017);

  std::cout << "\n=== Repeated GTSAM Rot3 composition ===\n";

  printRotationError("Initial", rotation.matrix());

  constexpr int kIterations = 1000000;

  for (int i = 1; i <= kIterations; ++i) {
    rotation = rotation.compose(increment);

    if (i == 10 || i == 100 || i == 1000 || i == 10000 || i == 100000 ||
        i == kIterations) {
      printRotationError("Iteration " + std::to_string(i), rotation.matrix());
    }
  }

  const Matrix3 R = rotation.matrix();
  const RotationError error = measureRotationError(R);

  EXPECT_TRUE(R.allFinite());
  EXPECT_TRUE(std::isfinite(error.orthogonality));
  EXPECT_TRUE(std::isfinite(error.determinant_error));

  std::cout << "Final orthogonality error: " << error.orthogonality << '\n';
}

TEST(RotationNumericalStability, RepeatedMatrixQuaternionConversions) {
  gtsam::Rot3 rotation = gtsam::Rot3::RzRyRx(0.31, -0.27, 0.19);

  std::cout << "\n=== Repeated matrix/quaternion conversions ===\n";

  printRotationError("Initial", rotation.matrix());

  constexpr int kIterations = 100000;

  for (int i = 1; i <= kIterations; ++i) {
    // Extract the matrix representation.
    const Matrix3 R = rotation.matrix();

    // Reconstruct Rot3 from the matrix.
    rotation = gtsam::Rot3(R);

    // Explicitly convert through a quaternion.
    const auto q = rotation.toQuaternion();

    rotation = gtsam::Rot3(q);

    if (i == 10 || i == 100 || i == 1000 || i == 10000 || i == kIterations) {
      printRotationError("Iteration " + std::to_string(i), rotation.matrix());
    }
  }

  const Matrix3 R = rotation.matrix();
  const RotationError error = measureRotationError(R);

  EXPECT_TRUE(R.allFinite());
  EXPECT_TRUE(std::isfinite(error.orthogonality));
  EXPECT_TRUE(std::isfinite(error.determinant_error));

  std::cout << "Final orthogonality error: " << error.orthogonality << '\n';
}

TEST(RotationNumericalStability, RepeatedCoordinateFrameConversions) {
  using gtsam::Point3;
  using gtsam::Pose3;
  using gtsam::Rot3;

  // Define fixed coordinate frames:
  // T_AB transforms coordinates from frame B to frame A.
  const Pose3 T_AB(Rot3::RzRyRx(0.03242431, -0.0022124417, 0.00012131419),
                   Point3(1.2, -0.7, 2.4));

  const Pose3 T_BC(Rot3::RzRyRx(-0.131313, 0.434346, 0.1545357637),
                   Point3(-0.4, 1.1, 0.8));

  const Pose3 T_CD(Rot3::RzRyRx(0.099998, -0.1121313136, 0.213153642629),
                   Point3(0.6, -0.3, 1.5));

  // Ground-truth transform between frames A and D.
  const Pose3 T_AD = T_AB.compose(T_BC).compose(T_CD);

  // Start with the ground-truth transform.
  Pose3 current = T_AD;

  std::cout << "\n=== Repeated coordinate frame conversions ===\n";

  printRotationError("Initial", current.rotation().matrix());

  constexpr int kIterations = 100000;

  for (int i = 1; i <= kIterations; ++i) {
    // A -> D -> A
    const Pose3 T_DA = current.inverse();

    // A -> D -> A -> D
    current = T_DA.inverse();

    // Introduce additional coordinate-frame conversions.
    // T_AB: B -> A
    // T_BC: C -> B
    // T_CD: D -> C
    //
    // T_AD = T_AB * T_BC * T_CD
    //
    // Recover the relative transforms using between().
    // const Pose3 recovered_T_BC =
    //     T_AB.between(T_AB.compose(T_BC));

    // const Pose3 recovered_T_CD =
    //     T_AB.compose(T_BC).between(T_AB.compose(T_BC).compose(T_CD));

    const Pose3 recovered_T_BC = T_AB.inverse() * T_AB * T_BC;
    // T_AB.between(T_AB.compose(T_BC));

    const Pose3 recovered_T_CD = (T_AB * T_BC).inverse() * (T_AB * T_BC * T_CD);
    // const Pose3 recovered_T_CD =
    // T_AB.compose(T_BC).between(T_AB.compose(T_BC).compose(T_CD));

    // Reconstruct the transform through the recovered frames.
    // current = T_AB.compose(recovered_T_BC).compose(recovered_T_CD);
    current = T_AB * recovered_T_BC * recovered_T_CD;

    if (i == 10 || i == 100 || i == 1000 || i == 10000 || i == kIterations) {
      printRotationError("Iteration " + std::to_string(i),
                         current.rotation().matrix());

      const Pose3 pose_error = T_AD.between(current);
      const gtsam::Vector6 error = Pose3::Logmap(pose_error);

      std::cout << "  Rotation deviation (rad): " << error.head<3>().norm()
                << "\n  Translation deviation (m): "
                << (current.translation() - T_AD.translation()).norm() << '\n';
    }
  }

  const Matrix3 R = current.rotation().matrix();
  const RotationError error = measureRotationError(R);

  EXPECT_TRUE(R.allFinite());
  EXPECT_TRUE(std::isfinite(error.orthogonality));
  EXPECT_TRUE(std::isfinite(error.determinant_error));

  std::cout << "Final orthogonality error: " << error.orthogonality << '\n';
}

TEST(RotationNumericalStability, RecursiveCoordinateFrameUpdates) {
  using gtsam::Point3;
  using gtsam::Pose3;
  using gtsam::Rot3;
  using gtsam::Vector6;

  // Initial world-frame poses.
  Pose3 T_WA(Rot3::RzRyRx(0.31, -0.27, 0.19), Point3(1.2, -0.7, 2.4));

  Pose3 T_WB(Rot3::RzRyRx(-0.13, 0.22, 0.17), Point3(-0.4, 1.1, 0.8));

  // Small relative motion applied at every iteration.
  const Pose3 delta_A(Rot3::RzRyRx(0.0013, -0.0021, 0.0017),
                      Point3(0.001, -0.002, 0.003));

  const Pose3 delta_B(Rot3::RzRyRx(-0.0011, 0.0018, 0.0023),
                      Point3(-0.002, 0.001, 0.001));

  // Reference states are propagated independently using the
  // same known increments.
  Pose3 reference_T_WA = T_WA;
  Pose3 reference_T_WB = T_WB;

  std::cout << "\n=== Recursive coordinate-frame updates ===\n";

  printRotationError("Initial T_WA", T_WA.rotation().matrix());
  printRotationError("Initial T_WB", T_WB.rotation().matrix());

  constexpr int kIterations = 100000;

  for (int i = 1; i <= kIterations; ++i) {
    // 1. Compute the relative pose using the current states.
    //
    // T_AB = T_WA^{-1} * T_WB
    const Pose3 T_AB = T_WA.between(T_WB);

    // 2. Update both poses.
    //
    // The update is deliberately expressed through composition.
    // The updated values become the inputs to the next iteration.
    T_WA = T_WA.compose(delta_A);
    T_WB = T_WB.compose(delta_B);

    // 3. Recompute the relative transform from the UPDATED states.
    const Pose3 updated_T_AB = T_WA.between(T_WB);

    // 4. Reconstruct T_WB from T_WA and the relative transform.
    //
    // This is an explicit coordinate-frame round trip.
    T_WB = T_WA.compose(updated_T_AB);

    // 5. Independently propagate the reference states.
    reference_T_WA = reference_T_WA.compose(delta_A);
    reference_T_WB = reference_T_WB.compose(delta_B);

    if (i == 10 || i == 100 || i == 1000 || i == 10000 || i == kIterations) {
      std::cout << "\nIteration " << i << '\n';

      printRotationError("T_WA", T_WA.rotation().matrix());
      printRotationError("T_WB", T_WB.rotation().matrix());
      printRotationError("T_AB", updated_T_AB.rotation().matrix());

      // Compare the recursively reconstructed pose against
      // the independently propagated reference.
      const Pose3 pose_error = reference_T_WB.between(T_WB);

      const Vector6 error = Pose3::Logmap(pose_error);

      std::cout << "Reference rotation deviation (rad): "
                << error.head<3>().norm() << '\n';

      std::cout << "Reference translation deviation (m): "
                << (T_WB.translation() - reference_T_WB.translation()).norm()
                << '\n';
    }
  }

  // Check that the resulting rotations remain finite.
  EXPECT_TRUE(T_WA.rotation().matrix().allFinite());
  EXPECT_TRUE(T_WB.rotation().matrix().allFinite());

  const RotationError error_A = measureRotationError(T_WA.rotation().matrix());

  const RotationError error_B = measureRotationError(T_WB.rotation().matrix());

  EXPECT_TRUE(std::isfinite(error_A.orthogonality));
  EXPECT_TRUE(std::isfinite(error_B.orthogonality));

  std::cout << "\nFinal T_WA orthogonality error: " << error_A.orthogonality
            << '\n';

  std::cout << "Final T_WB orthogonality error: " << error_B.orthogonality
            << '\n';
}

// void printRotationError(
//     const std::string& label,
//     const Matrix3& R) {
//   const RotationError error = measureRotationError(R);

//   std::cout << std::scientific << std::setprecision(8)
//             << label
//             << " | orthogonality = " << error.orthogonality
//             << " | determinant error = " << error.determinant_error
//             << " | max element error = " << error.max_element_error
//             << '\n';
// }

struct RecursiveResult {
  gtsam::Pose3 T_WA;
  gtsam::Pose3 T_WB;
};

RecursiveResult runRecursiveTest(bool useLieAlgebra) {
  using gtsam::Point3;
  using gtsam::Pose3;
  using gtsam::Rot3;

  Pose3 T_WA(Rot3::RzRyRx(0.31, -0.27, 0.19), Point3(1.2, -0.7, 2.4));

  Pose3 T_WB(Rot3::RzRyRx(-0.13, 0.22, 0.17), Point3(-0.4, 1.1, 0.8));

  const Pose3 delta_A(Rot3::RzRyRx(0.0013, -0.0021, 0.0017),
                      Point3(0.001, -0.002, 0.003));

  const Pose3 delta_B(Rot3::RzRyRx(-0.0011, 0.0018, 0.0023),
                      Point3(-0.002, 0.001, 0.001));

  std::cout << "\n=== "
            << (useLieAlgebra ? "Lie algebra reconstruction"
                              : "Direct composition reconstruction")
            << " ===\n";

  constexpr int kIterations = 100000;

  for (int i = 1; i <= kIterations; ++i) {
    // Propagate the two world poses.
    T_WA = T_WA.compose(delta_A);
    T_WB = T_WB.compose(delta_B);

    // Compute the relative pose using the current states.
    const Pose3 T_AB = T_WA.between(T_WB);

    if (useLieAlgebra) {
      // Map the relative pose to the Lie algebra se(3).
      const gtsam::Vector6 xi = Pose3::Logmap(T_AB);

      // Reconstruct the relative pose through the exponential map.
      const Pose3 T_AB_reconstructed = Pose3::Expmap(xi);

      // Feed the reconstructed relative pose back into the system.
      T_WB = T_WA.compose(T_AB_reconstructed);
    } else {
      // Directly feed the computed relative pose back into the system.
      T_WB = T_WA.compose(T_AB);
    }

    if (i == 10 || i == 100 || i == 1000 || i == 10000 || i == kIterations) {
      std::cout << "\nIteration " << i << '\n';

      printRotationError("T_WA", T_WA.rotation().matrix());
      printRotationError("T_WB", T_WB.rotation().matrix());
      printRotationError("T_AB", T_AB.rotation().matrix());
    }
  }

  const RotationError error_A = measureRotationError(T_WA.rotation().matrix());

  const RotationError error_B = measureRotationError(T_WB.rotation().matrix());

  EXPECT_TRUE(T_WA.rotation().matrix().allFinite());
  EXPECT_TRUE(T_WB.rotation().matrix().allFinite());

  EXPECT_TRUE(std::isfinite(error_A.orthogonality));
  EXPECT_TRUE(std::isfinite(error_B.orthogonality));

  std::cout << "\nFinal T_WA orthogonality error: " << error_A.orthogonality
            << '\n';

  std::cout << "Final T_WB orthogonality error: " << error_B.orthogonality
            << '\n';

  return {T_WA, T_WB};
}

TEST(RotationNumericalStability, CompareCompositionAndLieAlgebra) {
  const RecursiveResult composition_result = runRecursiveTest(false);

  const RecursiveResult lie_result = runRecursiveTest(true);

  const double composition_error =
      measureRotationError(composition_result.T_WB.rotation().matrix())
          .orthogonality;

  const double lie_error =
      measureRotationError(lie_result.T_WB.rotation().matrix()).orthogonality;

  std::cout << "\n=== Final comparison ===\n"
            << std::scientific << std::setprecision(8)
            << "Composition orthogonality error: " << composition_error << '\n'
            << "Lie algebra orthogonality error: " << lie_error << '\n';

  if (composition_error > 0.0) {
    std::cout << "Lie/composition error ratio: "
              << lie_error / composition_error << '\n';
  }

  // Both paths should produce finite, numerically usable rotations.
  EXPECT_TRUE(std::isfinite(composition_error));
  EXPECT_TRUE(std::isfinite(lie_error));
}

}  // namespace
