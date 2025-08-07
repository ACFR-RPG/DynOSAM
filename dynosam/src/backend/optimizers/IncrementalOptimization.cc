/*
 *   Copyright (c) 2025 ACFR-RPG, University of Sydney, Jesse Morris
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

#include "dynosam/backend/optimizers/IncrementalOptimization.hpp"

#include <gtsam/nonlinear/ISAM2Result.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

namespace dyno {

template <>
struct iOptimizationTraits<gtsam::IncrementalFixedLagSmoother> {
  typedef iOptimizationTraits<gtsam::IncrementalFixedLagSmoother> This;
  typedef gtsam::IncrementalFixedLagSmoother Smoother;
  typedef gtsam::ISAM2Result ResultType;

  struct IFLSUpdateArguments : public UpdateArguments {
    std::map<gtsam::Key, double> timestamps;
  };
  typedef IFLSUpdateArguments UpdateArguments;

  using FillArguments = std::function<void(const Smoother&, UpdateArguments&)>;

  static ResultType update(Smoother& smoother,
                           const UpdateArguments& update_arguments) {
    smoother.update(update_arguments.new_factors, update_arguments.new_values,
                    update_arguments.timestamps);
    return smoother.getISAM2Result();
  }

  static ResultType update(Smoother& smoother,
                           const FillArguments& update_arguments_filler) {
    UpdateArguments arguments;
    update_arguments_filler(smoother, arguments);

    return This::update(smoother, arguments);
  }

  static gtsam::NonlinearFactorGraph getFactors(const Smoother& smoother) {
    return smoother.getFactors();
  }

  static gtsam::Values calculateEstimate(const Smoother& smoother) {
    return smoother.calculateEstimate();
  }

  static gtsam::Values getLinearizationPoint(const Smoother& smoother) {
    return smoother.getLinearizationPoint();
  }
};

}  // namespace dyno
