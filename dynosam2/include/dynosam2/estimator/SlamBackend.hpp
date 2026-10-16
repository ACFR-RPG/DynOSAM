#pragma once

#include "dynosam_common/Types.hpp"

namespace dyno2 {

/// @brief Main estimator class, holding all current state information
class SlamBackend {
 public:
  DYNO_POINTER_TYPEDEFS(SlamBackend)
  SlamBackend() = default;
};

};  // namespace dyno2
