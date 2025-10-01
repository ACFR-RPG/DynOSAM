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
#include "dynosam_common/Types.hpp"

namespace dyno {

/**
 * @brief Defines some basic functions and behaviour for a module which will
 * take and send input to and from a pipeline.
 *
 * A module is intended to the core place where all the processing actually
 * happens. Like the pipeline, we expect INPUT and OUTPUT to have certain
 * typedefs defined, partically
 * ::ConstPtr
 *
 *
 *
 * @tparam INPUT
 * @tparam OUTPUT
 */
template <typename INPUT, typename OUTPUT>
class ModuleBase {
 public:
  using Input = INPUT;
  using Output = OUTPUT;
  using InputConstPtr = typename INPUT::ConstPtr;  //! ConstPtr to an INPUT type
  using OutputConstPtr =
      typename OUTPUT::ConstPtr;  //! ConstPtr to an OUTPUT type
  using This = ModuleBase<INPUT, OUTPUT>;

  DYNO_POINTER_TYPEDEFS(This)

  enum class State {
    Boostrap = 0u,  //! Initalize
    Nominal = 1u    //! Run
  };

  using SpinReturn = std::pair<State, OutputConstPtr>;

  using InputCallback = std::function<void(InputConstPtr)>;
  using OutputCallback = std::function<void(OutputConstPtr)>;

  ModuleBase(const std::string& name)
      : name_(name), module_state_(State::Boostrap) {}
  virtual ~ModuleBase() = default;

  OutputConstPtr spinOnce(InputConstPtr input) {
    CHECK(input);
    validateInput(input);

    SpinReturn spin_return{State::Boostrap, nullptr};

    if (input_callback_) input_callback_(input);

    switch (module_state_) {
      case State::Boostrap: {
        spin_return = boostrapSpin(input);
      } break;
      case State::Nominal: {
        spin_return = nominalSpin(input);
      } break;
      default: {
        LOG(FATAL) << "Unrecognized state in module " << name_;
        break;
      }
    }

    if (output_callback_) output_callback_(spin_return.second);

    module_state_ = spin_return.first;
    return spin_return.second;
  }

  /**
   * @brief Registeres a function callback to be triggered prior to
   * nominal/bootstrapSpin being called.
   *
   * @param input_callback const InputCallback&
   */
  inline void registerInputCallback(const InputCallback& input_callback) {
    input_callback_ = input_callback;
  }

  /**
   * @brief Registeres a function callback to be triggered after
   * nominal/bootstrapSpin is called.
   *
   * @param output_callback const OutputCallback&
   */
  inline void registerOutputCallback(const OutputCallback& output_callback) {
    output_callback_ = output_callback;
  }

 protected:
  /**
   * @brief Checks the state of the input before running any virtual spin
   * (boostrap or nominal) functions.
   *
   * The function is intended to be treated as a "fail hard" case, and should
   * throw an exception if the input is not valid
   *
   * @param input
   */
  virtual void validateInput(const InputConstPtr& input) const = 0;

  virtual SpinReturn boostrapSpin(InputConstPtr input) = 0;
  virtual SpinReturn nominalSpin(InputConstPtr input) = 0;

 private:
  const std::string name_;
  std::atomic<State> module_state_;
  std::atomic<State> previous_module_state_;

  InputCallback input_callback_;
  OutputCallback output_callback_;
};

}  // namespace dyno
