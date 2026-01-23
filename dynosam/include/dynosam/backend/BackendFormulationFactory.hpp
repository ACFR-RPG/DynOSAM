#pragma once

#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/backend/Formulation.hpp"
#include "dynosam/visualizer/VisualizerPipelines.hpp"  //for BackendModuleDisplay

namespace dyno {

template <typename MAP>
struct FormulationVizWrapper {
  typename Formulation<MAP>::Ptr formulation;
  //! Display associated with the formulation. May be nullptr
  BackendModuleDisplay::Ptr display;

  inline bool hasDisplay() const { return display != nullptr; }
};

template <typename MAP>
class BackendFormulationFactory {
 public:
  using This = BackendFormulationFactory<MAP>;
  DYNO_POINTER_TYPEDEFS(This)

  BackendFormulationFactory(const BackendType& backend_type)
      : backend_type_(backend_type) {}
  virtual ~BackendFormulationFactory() = default;

  virtual FormulationVizWrapper<MAP> createFormulation(
      const FormulationConstructorParams<MAP>& constructor_params) = 0;

  BackendType backendType() const { return backend_type_; }

 protected:
  const BackendType backend_type_;
};

}  // namespace dyno
