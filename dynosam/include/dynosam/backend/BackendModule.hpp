/*
 *   Copyright (c) 2023 ACFR-RPG, University of Sydney, Jesse Morris
 (jesse.morris@sydney.edu.au)
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a
 copy
 *   of this software and associated documentation files (the "Software"), to
 deal
 *   in the Software without restriction, including without limitation the
 rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:

 *   The above copyright notice and this permission notice shall be included in
 all
 *   copies or substantial portions of the Software.

 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE
 *   SOFTWARE.
 */

#pragma once

#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/backend/BackendInputPacket.hpp"
#include "dynosam/backend/BackendOutputPacket.hpp"
#include "dynosam/backend/BackendParams.hpp"
#include "dynosam/backend/Formulation.hpp"
#include "dynosam/visualizer/Visualizer-Definitions.hpp"  //for ImageDisplayQueueOptional,
#include "dynosam_common/Exceptions.hpp"
#include "dynosam_common/Map.hpp"
#include "dynosam_common/ModuleBase.hpp"
#include "dynosam_common/SharedModuleInfo.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_common/utils/SafeCast.hpp"

// DECLARE_string(updater_suffix);

namespace dyno {

template <typename DERIVED_INPUT_PACKET, typename MEASUREMENT_TYPE,
          typename BASE_INPUT_PACKET = BackendInputPacket>
struct BackendModuleTraits {
  using DerivedPacketType = DERIVED_INPUT_PACKET;
  using DerivedPacketTypeConstPtr = std::shared_ptr<const DerivedPacketType>;

  using BasePacketType = BASE_INPUT_PACKET;
  // BasePacketType is the type that gets passed to the module via the pipeline
  // and must be a base class since we pass data along the pipelines via
  // poniters
  static_assert(std::is_base_of_v<BasePacketType, DerivedPacketType>);

  using MeasurementType = MEASUREMENT_TYPE;
  using MapType = Map<MeasurementType>;
};

/**
 * @brief Base class to actually do processing. Data passed to this module from
 * the frontend
 *
 */
class BackendModule
    : public ModuleBase<BackendInputPacket, BackendOutputPacket>,
      public SharedModuleInterface {
 public:
  DYNO_POINTER_TYPEDEFS(BackendModule)

  using Base = ModuleBase<BackendInputPacket, BackendOutputPacket>;
  using Base::SpinReturn;

  BackendModule(const BackendParams& params, ImageDisplayQueue* display_queue);
  virtual ~BackendModule() = default;

  const BackendParams& getParams() const { return base_params_; }
  const NoiseModels& getNoiseModels() const { return noise_models_; }
  const BackendSpinState& getSpinState() const { return spin_state_; }

 protected:
  // called in ModuleBase immediately before the spin function is called
  virtual void validateInput(
      const BackendInputPacket::ConstPtr& input) const override;
  void setFactorParams(const BackendParams& backend_params);

 protected:
  // Redefine base input since these will be cast up by the BackendModuleType
  // class to a new type which we want to refer to as the input type BaseInput
  // is a ConstPtr to the type defined by BackendInputPacket
  using BaseInputConstPtr = Base::InputConstPtr;
  using BaseInput = Base::Input;

 protected:
  const BackendParams base_params_;
  ImageDisplayQueue* display_queue_{nullptr};  //! Optional display queue

  BackendSpinState
      spin_state_;  //! Spin state of the backend. Updated in the backend module
                    //! base via InputCallback (see BackendModule constructor).
  NoiseModels noise_models_;

 private:
};

template <class MODULE_TRAITS>
class BackendModuleType : public BackendModule {
 public:
  using ModuleTraits = MODULE_TRAITS;
  // A Dervied BackedInputPacket type (e.g. RGBDOutputPacketType)
  using DerivedPacketType = typename ModuleTraits::DerivedPacketType;
  using DerivedPacketTypeConstPtr =
      typename ModuleTraits::DerivedPacketTypeConstPtr;
  using MeasurementType = typename ModuleTraits::MeasurementType;
  using This = BackendModuleType<ModuleTraits>;
  using Base = BackendModule;

  using MapType = typename ModuleTraits::MapType;
  using FormulationType = Formulation<MapType>;

  DYNO_POINTER_TYPEDEFS(This)

  using Base::SpinReturn;
  // Define the input type to the derived input type, defined in the
  // MODULE_TRAITS this is the derived Input packet that is passed to the
  // boostrap/nominal Spin Impl functions that must be implemented in the
  // derived class that does the provessing on this module
  using InputConstPtr = DerivedPacketTypeConstPtr;
  using OutputConstPtr = Base::OutputConstPtr;

  BackendModuleType(const BackendParams& params,
                    ImageDisplayQueue* display_queue)
      : Base(params, display_queue), map_(MapType::create()) {}

  virtual ~BackendModuleType() {}

  inline const typename MapType::Ptr getMap() { return map_; }

 protected:
  virtual SpinReturn boostrapSpinImpl(InputConstPtr input) = 0;
  virtual SpinReturn nominalSpinImpl(InputConstPtr input) = 0;

  typename MapType::Ptr map_;

 private:
  SpinReturn boostrapSpin(Base::BaseInputConstPtr base_input) override {
    return boostrapSpinImpl(attemptCast(base_input));
  }

  SpinReturn nominalSpin(Base::BaseInputConstPtr base_input) override {
    return nominalSpinImpl(attemptCast(base_input));
  }

  DerivedPacketTypeConstPtr attemptCast(Base::BaseInputConstPtr base_input) {
    DerivedPacketTypeConstPtr deriverd_input =
        safeCast<Base::BaseInput, DerivedPacketType>(base_input);
    checkAndThrow((bool)deriverd_input,
                  "Failed to cast " + type_name<Base::BaseInput>() + " to " +
                      type_name<DerivedPacketType>() + " in BackendModuleType");
    return deriverd_input;
  }
};

}  // namespace dyno
