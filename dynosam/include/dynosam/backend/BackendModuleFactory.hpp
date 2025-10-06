#pragma once

#include "dynosam/backend/BackendModule.hpp"
#include "dynosam/backend/optimizers/IncrementalOptimization.hpp"  // for ErrorHandlingHooks
#include "dynosam/visualizer/VisualizerPipelines.hpp"  //for BackendModuleDisplay

namespace dyno {
/**
 * @brief Wrapper containing the BackendModule, any associated additional
 * visualizer and custom error hooks associated with this backend and (likely)
 * its formulation. Created by the BackendModuleFactory which is the only thing
 * that knows all the details of the particular backend and the formulation
 * used.
 *
 */
struct BackendWrapper {
  BackendModule::Ptr backend;
  BackendModuleDisplay::Ptr backend_viz;
  std::optional<ErrorHandlingHooks> error_hooks;
};

/**
 * @brief Params needed to make a module
 *
 */
struct ModuleParams {
  BackendParams backend_params;
  Sensors sensors;
  ImageDisplayQueue* display_queue;
};

class BackendModuleFactory {
 public:
  DYNO_POINTER_TYPEDEFS(BackendModuleFactory)

  virtual BackendWrapper createModule(const ModuleParams& params) = 0;
};

}  // namespace dyno
