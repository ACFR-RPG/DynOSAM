#pragma once

#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/backend/Formulation.hpp"
#include "dynosam/visualizer/VisualizerPipelines.hpp"  //for BackendModuleDisplay

namespace dyno {

// TODO: make not templated on MAP!
struct FormulationVizWrapper {
  Formulation::Ptr formulation;
  //! Display associated with the formulation. May be nullptr
  BackendModuleDisplay::Ptr display;

  inline bool hasDisplay() const { return display != nullptr; }

  // cast upwards to a FormulationT, allowing runtime checks as to the MAP type
  // as all formulations must derive from FormulationT
  template <typename MAP>
  typename FormulationT<MAP>::Ptr asFormulationT() const {
    return std::dynamic_pointer_cast<FormulationT<MAP>>(formulation);
  }

  // cast to a fully defined derived formulation
  template <typename FORMULATION>
  std::shared_ptr<FORMULATION> as() const {
    return std::dynamic_pointer_cast<FORMULATION>(formulation);
  }
};

template <typename MAP>
class BackendFormulationFactory {
 public:
  //! Alias needed in BackendFactory
  using Map = MAP;
  using This = BackendFormulationFactory<MAP>;
  DYNO_POINTER_TYPEDEFS(This)

  BackendFormulationFactory(const BackendType& backend_type)
      : backend_type_(backend_type) {}
  virtual ~BackendFormulationFactory() = default;

  virtual FormulationVizWrapper createFormulation(
      const FormulationParams& formulation_params, std::shared_ptr<MAP> map,
      const NoiseModels& noise_models, const Sensors& sensors,
      const FormulationHooks& formulation_hooks) = 0;

  BackendType backendType() const { return backend_type_; }

 protected:
  const BackendType backend_type_;
};

}  // namespace dyno
