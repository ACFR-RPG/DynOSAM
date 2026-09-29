#pragma once

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>

inline gtsam::Vector2 solve2x2(const gtsam::Matrix22& H,
                               const gtsam::Vector2& b) {
  const double a = H(0, 0);
  const double c = H(0, 1);
  const double d = H(1, 1);

  const double det = a * d - c * c;

  const double inv_det = 1.0 / det;

  return gtsam::Vector2((d * b(0) - c * b(1)) * inv_det,
                        (a * b(1) - c * b(0)) * inv_det);
}

inline gtsam::Matrix26 solve2x2Matrix(const gtsam::Matrix22& H,
                                      const gtsam::Matrix26& B) {
  gtsam::Matrix26 result;

  const double a = H(0, 0);
  const double c = H(0, 1);
  const double d = H(1, 1);

  const double inv_det = 1.0 / (a * d - c * c);

  for (int i = 0; i < 6; ++i) {
    const double b0 = B(0, i);
    const double b1 = B(1, i);

    result(0, i) = (d * b0 - c * b1) * inv_det;

    result(1, i) = (a * b1 - c * b0) * inv_det;
  }

  return result;
}

inline gtsam::Matrix66 outerProduct6x2(const gtsam::Matrix26& Jx) {
  gtsam::Matrix66 H;
  H.setZero();

  for (int r = 0; r < 6; ++r) {
    for (int c = r; c < 6; ++c) {
      const double value = Jx(0, r) * Jx(0, c) + Jx(1, r) * Jx(1, c);

      H(r, c) = value;
      H(c, r) = value;
    }
  }

  return H;
}

// ============================================================================
// Jx^T Jf
// ============================================================================

inline gtsam::Matrix62 crossProduct6x2(const gtsam::Matrix26& Jx,
                                       const gtsam::Matrix22& Jf) {
  gtsam::Matrix62 H;

  for (int r = 0; r < 6; ++r) {
    H(r, 0) = Jx(0, r) * Jf(0, 0) + Jx(1, r) * Jf(1, 0);

    H(r, 1) = Jx(0, r) * Jf(0, 1) + Jx(1, r) * Jf(1, 1);
  }

  return H;
}

// ============================================================================
// Jf^T Jf
// ============================================================================

inline gtsam::Matrix22 outerProduct2x2(const gtsam::Matrix22& Jf) {
  gtsam::Matrix22 H;

  H(0, 0) = Jf(0, 0) * Jf(0, 0) + Jf(1, 0) * Jf(1, 0);

  H(0, 1) = Jf(0, 0) * Jf(0, 1) + Jf(1, 0) * Jf(1, 1);

  H(1, 0) = H(0, 1);

  H(1, 1) = Jf(0, 1) * Jf(0, 1) + Jf(1, 1) * Jf(1, 1);

  return H;
}

// ============================================================================
// Jx^T r
// ============================================================================

inline gtsam::Vector6 multiplyTranspose(const gtsam::Matrix26& Jx,
                                        const gtsam::Vector2& r) {
  gtsam::Vector6 result;

  for (int i = 0; i < 6; ++i) {
    result(i) = Jx(0, i) * r(0) + Jx(1, i) * r(1);
  }

  return result;
}

// ============================================================================
// Jf^T r
// ============================================================================

inline gtsam::Vector2 multiplyTranspose(const gtsam::Matrix22& Jf,
                                        const gtsam::Vector2& r) {
  return gtsam::Vector2(Jf(0, 0) * r(0) + Jf(1, 0) * r(1),

                        Jf(0, 1) * r(0) + Jf(1, 1) * r(1));
}
