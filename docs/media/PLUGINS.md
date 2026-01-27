# Formulation Plugin System

The DynoSAM backend relies heavily on the `Formulation` class which implements the construction and bookkeeping assocaited with factor-graph construction.

In most cases, the `Formulation` class is tightly coupled with the `RegularBackendModule` which manages the interaction between the `Map`, `Formulation` and the optimization of the constructed factor graph. Each formulation is abstracted in such a way that (generally) the backend module does not need to know about how each formulation is implemented.

We provide functionality to make custom formulation classes (and their associated displays) and inject them into the `RegularBackendModule` via a _plugin system_. Instead of manually loading each formulation via a factory, a formulation can be registered via _pluginlib_ and loaded dynamically at runtime, allowing custom behaviour to be implemented without changing the DynoSAM source code. This means that formulation can be implemented independantly of the DynoSAM code base.

## Implementation
Instead of writing each formulation as a plugin directly, we dynamically load a single factory which is responsible for creating a single formulation and associated (ROS) display.
Once loaded the formulation will be injected into the `RegularBackendModule`.

Since all formulations depend on a templated MAP type, all plugins must derive from the templated plugin base type `FormulationFactoryPluginT<MAP>`

> NOTE: the templated MAP dependancy is mostly for historical reasons and the MAP should always be `MapVision`.

Say we want to write a custom formulation called `MyCustomFormulation` which can be visualised via the display `MyCustomFormulationDisplay`

We can load this formulation by writing a custom plugin:
```c++
# header or .cc file
#include "dynosam_ros/FormulationFactoryPlugin.hpp"

class MyCustomFormulation : Formulation<MapVision> {};
class MyCustomFormulationDisplay : BackendModuleDisplayRos {};

class MyCustomFormulationPlugin : dyno::FormulationFactoryPluginT<MapVision> {
    FormulationVizWrapper<MapVision> create(
      rclcpp::Node::SharedPtr node,
      const DisplayParams& display_params,
      const FormulationConstructorParams<MapVision>& constructor_params)
      override {

        # create MyCustomFormulation
        # create MyCustomFormulationDisplay

        # return result in FormulationVizWrapper<MapVision> struct
    }
};

```
> NOTE: MyCustomFormulation just needs to derive from the base `Formulation<typename MAP>` class and therefore can inherit from any of the existing formulations (i.e WCME, WCPE, Hybrid) if necessary.

## Dependancies and Exporting

Plugins must be registered exported such that _pluginlib_ can find the library at runtime.
We our system follows the ROS pluginlib structure exactly follow the [tutorial instructions](https://docs.ros.org/en/foxy/Tutorials/Beginner-Client-Libraries/Pluginlib.html) to register and export the plugin:
-  In a `.cc` file, register the plugin:
    ```
    PLUGINLIB_EXPORT_CLASS(MyCustomFormulationPlugin, dyno::FormulationFactoryPlugin)
    ```
- Export the plugin via the plugin declaration XML.

Once exported, re-source the workspace and you should see your plugin listed under `ros2 plugin list --package <your package>`

> NOTE: See [formulation_plugins.hpp](../../dynosam_ros/test/formulation_plugins.hpp) for slightly more complete example.

## Using
We control loading of the backend with three GFLAG params
```
--backend_updater_enum
--backend_formulation_plugin_type
--load_backend_formulations_as_internal
```

If `load_backend_formulations_as_internal` is true, then `backend_updater_enum` will be used as the internal type (i.e non-plugin), represnting a enum which indicates which inbuild formulation to load.

Set `load_backend_formulations_as_internal` to false to load formulations from the list of available plugins.
The plugin type is specified by the string value `backend_formulation_plugin_type` which represents the `lookup_type` (ie. the exported plugin name, e.g. `MyCustomFormulationPlugin`).
