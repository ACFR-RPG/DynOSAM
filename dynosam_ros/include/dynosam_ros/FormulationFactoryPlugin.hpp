#pragma once

#include <optional>
#include <pluginlib/class_loader.hpp>

#include "dynosam/backend/BackendFormulationFactory.hpp"
#include "dynosam_ros/Display-Definitions.hpp"
#include "rclcpp/node.hpp"

namespace dyno {

/**
 * @brief Formulation (Factory) Plugin base class.
 *
 * Instead of loading the plugin directly we dynamic load a factory which
 * is responsible for creating a single formulation and associated display.
 *
 * Once loaded the formulation will be injected into the RegularBackendModule.
 *
 * Since all formulations are depending on a templated MAP type, we use
 * this (non-templated) base class to act as the base interface before
 * dynamically casting to the derived plugin type FormulationFactoryPluginT.
 *
 */
class FormulationFactoryPlugin {
 public:
  FormulationFactoryPlugin() {}
  virtual ~FormulationFactoryPlugin() = default;

  /**
   * @brief Allows extending of the ROS namespace
   *
   * @return std::optional<std::string>
   */
  virtual std::optional<std::string> extendDynosamNamespace() const {
    return {};
  };
};

/**
 * @brief Derived Formulation (Factory) Plugin base class from which all
 * formulation plugins must inherit and includes the pure virtual create
 * function which actually constructs the formulation and associated display.
 *
 * @tparam MAP
 */
template <class MAP>
class FormulationFactoryPluginT : public FormulationFactoryPlugin {
 public:
  FormulationFactoryPluginT() = default;

  virtual FormulationVizWrapper<MAP> create(
      rclcpp::Node::SharedPtr node, const DisplayParams& display_params,
      const FormulationConstructorParams<MAP>& constructor_params) = 0;
};

/**
 * @brief Thrown if the base plugin factory class (FormulationFactoryPlugin) is
 * null
 *
 */
class InvalidFormulationFactoryPlugin : public DynosamException {
 public:
  InvalidFormulationFactoryPlugin(const std::string& formulation_class)
      : DynosamException(
            "Loading requested formulation plugin " + formulation_class +
            " failed! Base class (FormulationFactoryPlugin) is null!") {}
};

class InvalidDerivedFormulationFactoryPlugin : public DynosamException {
 public:
  InvalidDerivedFormulationFactoryPlugin(
      const std::string& derived_factory_name, const std::string& map_name,
      const std::string& formulation_class)
      : DynosamException("Loading requested formulation plugin " +
                         formulation_class +
                         " failed! Specified map type was " + map_name +
                         " but the loaded plugin does not derive from class " +
                         derived_factory_name + "!") {}
};

/**
 * @brief Loads a formulation and display from the dynamically loaded
 * formulation factory.
 *
 *
 */
class FormulationFactoryPluginLoader {
 public:
  FormulationFactoryPluginLoader();

  /**
   * @brief Loads a formulation by dynamically loading the requested factory
   * plugin
   *
   * @tparam MAP type the formulation is expected to be templated on
   * @param lookup_name const std::string& The name of the class to load.
   * @param node rclcpp::Node::SharedPtr Node to be passed to the formulation
   * plgin
   * @param constructor_params const FormulationConstructorParams<MAP> Generic
   * paramters for the formulation.
   * @return FormulationVizWrapper<MAP>
   */
  template <class MAP>
  FormulationVizWrapper<MAP> loadFormulation(
      const std::string& lookup_name, rclcpp::Node::SharedPtr node,
      const DisplayParams& display_params,
      const FormulationConstructorParams<MAP>& constructor_params) {
    CHECK_NOTNULL(node);
    // load base class
    std::shared_ptr<FormulationFactoryPlugin> base_factory =
        loader_.createSharedInstance(lookup_name);
    if (!base_factory) {
      throw InvalidFormulationFactoryPlugin(lookup_name);
    }

    rclcpp::Node::SharedPtr node_for_factory = node;
    std::optional<std::string> sub_namespace =
        base_factory->extendDynosamNamespace();
    if (sub_namespace) {
      VLOG(5) << "Extending namespace of node -> " << sub_namespace.value();
      node_for_factory =
          node_for_factory->create_sub_node(sub_namespace.value());
    }

    // try casting to derived factory class with plugin so we can load the
    // formulation with the correct MAP type
    using DerivedFactory = FormulationFactoryPluginT<MAP>;
    std::shared_ptr<DerivedFactory> derived_factory =
        std::dynamic_pointer_cast<DerivedFactory>(base_factory);

    if (!derived_factory) {
      throw InvalidDerivedFormulationFactoryPlugin(
          type_name<DerivedFactory>(), type_name<MAP>(), lookup_name);
    }

    FormulationVizWrapper<MAP> formulation_wrapper = derived_factory->create(
        node_for_factory, display_params, constructor_params);

    CHECK(formulation_wrapper.formulation);
    LOG(INFO) << "Successfully loaded plugin " << lookup_name
              << " with associated formulation: "
              << formulation_wrapper.formulation->getFullyQualifiedName();

    return formulation_wrapper;
  }

 private:
  pluginlib::ClassLoader<FormulationFactoryPlugin> loader_;
};

}  // namespace dyno
