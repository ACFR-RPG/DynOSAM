/*
 *   Copyright (c) 2023 ACFR-RPG, University of Sydney, Jesse Morris
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

#include "dynosam/pipeline/PipelineParams.hpp"

#include <config_utilities/config_utilities.h>
#include <config_utilities/parsing/yaml.h>
#include <gflags/gflags.h>

#include "dynosam_common/utils/YamlParser.hpp"

DEFINE_int32(
    data_provider_type, 0,
    "Which data provider (loader) to use. Associated with specific datasets");

DEFINE_int32(backend_updater_enum, 0,
             "Which BackendType the backend should use and should match an "
             "enum in whatever backend module type is loaded");
DEFINE_string(backend_formulation_plugin_type, "",
              "Exported name of the formulation plugin to load");

// hacky solution to determine if backend_updater_enum or
// backend_formulation_plugin_type should be used as we are still using gflags
// for this!!
DEFINE_bool(load_backend_formulations_as_internal, true,
            "If false, then the backend will be loaded via a plugin.");

namespace dyno {

BackendType backendTypeFromGFlags(
    bool load_backend_formulations_as_internal, int32_t backend_updater_enum,
    const std::string& backend_formulation_plugin_type) {
  if (load_backend_formulations_as_internal) {
    return BackendType(backend_updater_enum);
  } else {
    return BackendType(backend_formulation_plugin_type);
  }
}

void declare_config(DynoParams::PipelineParams& config) {
  using namespace config;

  name("PipelineParams");
  field(config.parallel_run, "parallel_run");
  field(config.prefer_data_provider_camera_params,
        "prefer_data_provider_camera_params");
  field(config.prefer_data_provider_imu_params,
        "prefer_data_provider_imu_params");

  config.data_provider_type = FLAGS_data_provider_type;
}

DynoParams::DynoParams(const std::string& params_folder_path) {
  pipeline_params_ = config::fromYamlFile<PipelineParams>(
      params_folder_path + "PipelineParams.yaml");
  camera_params_ = config::fromYamlFile<CameraParams>(params_folder_path +
                                                      "CameraParams.yaml");
  frontend_params_ = config::fromYamlFile<FrontendParams>(
      params_folder_path + "FrontendParams.yaml");
  imu_params_ =
      config::fromYamlFile<ImuParams>(params_folder_path + "ImuParams.yaml");

  backend_type = backendTypeFromGFlags(
      FLAGS_load_backend_formulations_as_internal, FLAGS_backend_updater_enum,
      FLAGS_backend_formulation_plugin_type);
  LOG(INFO) << "Constructed backend type " << backend_type;
}

void DynoParams::printAllParams(bool print_glog_params) const {
  LOG(INFO) << "Frontend Params: " << config::toString(frontend_params_);
  LOG(INFO) << "Pipeline Params: " << config::toString(pipeline_params_);
  LOG(INFO) << "Camera Params: " << config::toString(camera_params_);
  LOG(INFO) << "IMU Params: " << config::toString(imu_params_);

  // TODO: currently cannot print camera params becuase we use intermediate
  // variables in the loading process!!
  //  LOG(INFO) << "Camera Params: " << config::toString(camera_params_);

  if (print_glog_params) google::ShowUsageWithFlags("");
}

bool DynoParams::incrementalBackend() const {
  return backend_type == BackendType::PARALLEL_HYBRID ||
         backend_params_.optimization_mode ==
             RegularOptimizationType::INCREMENTAL;
}

}  // namespace dyno
