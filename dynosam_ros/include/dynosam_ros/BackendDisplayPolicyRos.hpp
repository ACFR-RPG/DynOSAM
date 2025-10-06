#pragma once
#include "dynosam/backend/BackendFactory.hpp"
#include "dynosam/backend/BackendModule.hpp"
#include "dynosam/backend/BackendParams.hpp"
#include "dynosam_ros/Display-Definitions.hpp"
#include "dynosam_ros/displays/BackendDisplayRos.hpp"
#include "rclcpp/node.hpp"
#include "rclcpp/node_options.hpp"

// TODO: for now a simple solution for testing
//  puts BackendModuleDisplayTraits into one file explicitly for SFINAE
//  definitions!
#include "dynosam_ros/displays/backend_impl/HybridBackendDisplay.hpp"

namespace dyno {

// NOTE: if and when plugin system for formulations (only support with Regular?)
//  plugins should return the formulation and error handlers which is then put
//  into the BackendModule Backend type could then be std::variant for existing
//  types and plugin types

template <typename T>
struct BackendModuleDisplayTraits;

template <typename T, typename = void>
struct has_backend_module_display : std::false_type {};

template <typename T>
struct has_backend_module_display<
    T, std::void_t<typename BackendModuleDisplayTraits<T>::type>>
    : std::true_type {};

class BackendModulePolicyRos {
 public:
  BackendModulePolicyRos(const DisplayParams& params,
                         rclcpp::Node::SharedPtr node)
      : params_(params), node_(node) {
    VLOG(10) << "Creating BackendModulePolicyRos";
  }
  // makeViz only enabled if VizOf<T>::type exists
  template <typename T>
  std::shared_ptr<BackendModuleDisplayRos> createDisplay(
      std::shared_ptr<T> module) {
    VLOG(20) << "Attempting to create additional display for module "
             << type_name<T>();
    if constexpr (has_backend_module_display<T>::value) {
      using DisplayT = typename BackendModuleDisplayTraits<T>::type;
      VLOG(10) << "Found additional ROS display " << type_name<DisplayT>()
               << "for module " << type_name<T>();
      return std::make_shared<DisplayT>(params_, node_, module);
    }
    return nullptr;
  }

 private:
  DisplayParams params_;
  rclcpp::Node::SharedPtr node_;
};

// class BackendModuleFactoryRos : public BackendModuleFactory {
// public:
//     BackendModuleFactoryRos(const DisplayParams& params,
//     rclcpp::Node::SharedPtr node)
//         : params_(params), node_(node) {}

//     BackendWrapper createModule(
//       const BackendType& backend_type,
//       const BackendModuleFactory::Params& params) override;

//     void log() {
//         std::cout << "In BackendModuleFactoryRos " << std::endl;
//     }

//     // makeViz only enabled if VizOf<T>::type exists
//     template<typename T>
//     std::shared_ptr<BackendModuleDisplayRos>
//     makeBackendModuleDisplay(std::shared_ptr<T> module) {
//         if constexpr (has_backend_module_display<T>::value) {
//             using DisplayT = typename BackendModuleDisplayTraits<T>::type;
//             return std::make_shared<DisplayT>(params_, node_, module);
//         }
//         return nullptr;
//     }

// private:
//     DisplayParams params_;
//     rclcpp::Node::SharedPtr node_;

// };

// class BackendModuleDisplayFactory {
// public:
//     using Creator = std::function<std::unique_ptr<BackendDisplayRos>(const
//     DisplayParams&, rclcpp::Node::SharedPtr)>;

//     BackendModuleDisplayFactory(const DisplayParams& params,
//     rclcpp::Node::SharedPtr node)
//         : params_(params), node_(node) {}

//     template<typename T>
//     static void registerViz(const std::string& name, Creator c) {
//         type_creators_[std::type_index(typeid(T))] = c;
//         string_creators_[name] = c;
//     }

//     // Create by type
//     template<typename T>
//     std::unique_ptr<BackendDisplayRos> create() const {
//         auto it = type_creators_.find(std::type_index(typeid(T)));
//         if (it != type_creators_.end()) {
//             return (it->second)(params_, node_);
//         }
//         return nullptr;
//     }

//     // Create by string
//     std::unique_ptr<BackendDisplayRos> create(const std::string& name) const
//     {
//         auto it = string_creators_.find(name);
//         if (it != string_creators_.end()) {
//             return (it->second)(params_, node_);
//         }
//         return nullptr;
//     }

// private:
//     DisplayParams params_;
//     rclcpp::Node::SharedPtr node_;

//     static std::unordered_map<std::type_index, Creator> type_creators_;
//     static std::unordered_map<std::string, Creator> string_creators_;
// };

// template<typename Module, typename DisplayT>
// struct BackendModuleDisplayRegistrar {
//     BackendModuleDisplayRegistrar(const std::string& name) {
//         BackendModuleDisplayFactory::template registerViz<Module>(
//             name,
//             std::bind([](const DisplayParams& params, rclcpp::Node::SharedPtr
//             node) {
//                 return std::make_unique<DisplayT>(params, node);
//             }, std::placeholders::_1, std::placeholders::_2)
//         );
//     }
// };

}  // namespace dyno

// #define REGISTER_VIZ(Type, VizType, Name) \
//     static BackendModuleDisplayRegistrar<Type, VizType> reg_##Type(Name)

// #define REGISTER_VIZ_AUTONAME(Type, VizType) \
//     static BackendModuleDisplayRegistrar<Type, VizType> reg_##Type(#Type)
