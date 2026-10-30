#include "dynosam2/common/Parameters.hpp"

#include <config_utilities/config_utilities.h>
#include <config_utilities/parsing/yaml.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

DEFINE_bool(use_fullgraph, false, "If the full graph estimator should be used");
DEFINE_bool(estimate_dynamic_objects, true,
            "If dynamic objects should be estimated for");

namespace dyno2 {

DebugParams::DebugParams()
    : use_fullgraph(FLAGS_use_fullgraph),
      estimate_dynamic_objects(FLAGS_estimate_dynamic_objects) {}

void PipelineParameters::loadFromYaml(const std::string& file_path) {
  *this = config::fromYamlFile<PipelineParameters>(file_path);
  LOG(INFO) << config::toString(*this);
}

void declare_config(EstimationParameters& config) {
  using namespace config;
  name("EstimationParameters");

  field(config.num_static_keyframes, "num_static_keyframes");
  field(config.num_dynamic_keyframes, "num_dynamic_keyframes");
  field(config.enforce_realtime, "enforce_realtime");
  field(config.realtime_static_min_iterations,
        "realtime_static_min_iterations");
  field(config.realtime_static_max_iterations,
        "realtime_static_max_iterations");
  field(config.realtime_dynamic_min_iterations,
        "realtime_dynamic_min_iterations");
  field(config.realtime_dynamic_max_iterations,
        "realtime_dynamic_max_iterations");
  field(config.realtime_time_limit, "realtime_time_limit");
}

void declare_config(FrontendParameters& config) {
  using namespace config;
  name("FrontendParameters");
  field(config.prefer_provided_object_detection,
        "prefer_provided_object_detection");
  field(config.detection_model, "detection_model");
}

void declare_config(PipelineParameters& config) {
  using namespace config;
  name("PipelineParameters");

  field(config.frontend, "frontend_parameters");
  field(config.estimator, "estimator_parameters");
}

}  // namespace dyno2
