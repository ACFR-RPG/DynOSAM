#include "dynosam_ros/FormulationFactoryPlugin.hpp"

namespace dyno {

FormulationFactoryPluginLoader::FormulationFactoryPluginLoader()
    : loader_("dynosam_ros", "dyno::FormulationFactoryPlugin") {}

}  // namespace dyno
