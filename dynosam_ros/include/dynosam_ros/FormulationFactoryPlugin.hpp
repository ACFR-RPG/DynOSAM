#pragma once

#include <optional>
#include <pluginlib/class_loader.hpp>

#include "dynosam/backend/BackendFormulationFactory.hpp"
#include "rclcpp/node.hpp"

namespace dyno {

class FormulationFactoryPlugin {
 public:
  FormulationFactoryPlugin() {}
  virtual ~FormulationFactoryPlugin() = default;

  virtual std::optional<std::string> extendDynosamNamespace() const {
    return {};
  };
};

template <class MAP>
class FormulationFactoryPluginT : public FormulationFactoryPlugin {
 public:
  FormulationFactoryPluginT() = default;

  virtual FormulationVizWrapper<MAP> create(
      rclcpp::Node::SharedPtr node,
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

class FormulationFactoryPluginLoader {
 public:
  FormulationFactoryPluginLoader();

  template <class MAP>
  FormulationVizWrapper<MAP> loadFormulation(
      const std::string& formulation_class, rclcpp::Node::SharedPtr node,
      const FormulationConstructorParams<MAP>& constructor_params) {
    CHECK_NOTNULL(node);
    // load base class
    std::shared_ptr<FormulationFactoryPlugin> base_factory =
        loader_.createSharedInstance(formulation_class);
    if (!base_factory) {
      throw InvalidFormulationFactoryPlugin(formulation_class);
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
          type_name<DerivedFactory>(), type_name<MAP>(), formulation_class);
    }

    FormulationVizWrapper<MAP> formulation_wrapper =
        derived_factory->create(node_for_factory, constructor_params);

    CHECK(formulation_wrapper.formulation);
    LOG(INFO) << "Successfully loaded plugin " << formulation_class
              << " with associated formulation: "
              << formulation_wrapper.formulation->getFullyQualifiedName();

    return formulation_wrapper;
  }

 private:
  pluginlib::ClassLoader<FormulationFactoryPlugin> loader_;
};

}  // namespace dyno
