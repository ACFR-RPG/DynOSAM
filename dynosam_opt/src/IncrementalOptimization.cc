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

#include "dynosam_opt/IncrementalOptimization.hpp"
namespace dyno {

ErrorHandlingHooks getDefaultILSErrorHandlingHooks(
    const ErrorHandlingHooks::OnFailedObject& on_failed_object) {
  ErrorHandlingHooks error_hooks;
  error_hooks.handle_ils_exception = [](const gtsam::Values& current_values,
                                        gtsam::Key problematic_key) {
    ErrorHandlingHooks::HandleILSResult ils_handle_result;
    // a little gross that I need to set this up in this function
    gtsam::NonlinearFactorGraph& prior_factors = ils_handle_result.pior_factors;
    auto& failed_on_object = ils_handle_result.failed_objects;

    ApplyFunctionalSymbol afs;
    afs.cameraPose([&prior_factors, &current_values](FrameId,
                                                     const gtsam::Symbol& sym) {
         const gtsam::Key& key = sym;
         gtsam::Pose3 pose = current_values.at<gtsam::Pose3>(key);
         gtsam::Vector6 sigmas;
         sigmas.head<3>().setConstant(0.001);  // rotation
         sigmas.tail<3>().setConstant(0.01);   // translation
         gtsam::SharedNoiseModel noise =
             gtsam::noiseModel::Diagonal::Sigmas(sigmas);
         prior_factors.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
             key, pose, noise);
       })
        .objectMotion(
            [&prior_factors, &current_values, &failed_on_object](
                FrameId k, ObjectId j, const gtsam::LabeledSymbol& sym) {
              const gtsam::Key& key = sym;
              gtsam::Pose3 pose = current_values.at<gtsam::Pose3>(key);
              gtsam::Vector6 sigmas;
              sigmas.head<3>().setConstant(0.001);  // rotation
              sigmas.tail<3>().setConstant(0.01);   // translation
              gtsam::SharedNoiseModel noise =
                  gtsam::noiseModel::Diagonal::Sigmas(sigmas);
              prior_factors.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
                  key, pose, noise);
              failed_on_object.push_back(std::make_pair(k, j));
            })
        .
        operator()(problematic_key);
    return ils_handle_result;
  };

  if (on_failed_object) {
    error_hooks.handle_failed_object = on_failed_object;
  }

  return error_hooks;
}

}  // namespace dyno
