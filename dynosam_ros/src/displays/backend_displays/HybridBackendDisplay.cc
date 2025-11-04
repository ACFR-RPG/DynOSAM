#include "dynosam_ros/displays/backend_displays/HybridBackendDisplay.hpp"

#include "dynosam_ros/BackendDisplayPolicyRos.hpp"
#include "dynosam_ros/displays/BackendDisplayRos.hpp"

namespace dyno {

void ParalleHybridModuleDisplay::spin(
    const BackendOutputPacket::ConstPtr& output) {}

void RegularHybridFormulationDisplay::spin(
    const BackendOutputPacket::ConstPtr& output) {}

}  // namespace dyno
