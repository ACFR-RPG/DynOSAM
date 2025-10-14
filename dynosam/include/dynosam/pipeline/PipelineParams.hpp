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

#include <glog/logging.h>

#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>

#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/backend/BackendParams.hpp"
#include "dynosam/frontend/Frontend-Definitions.hpp"
#include "dynosam/frontend/FrontendParams.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_common/utils/Macros.hpp"
#include "dynosam_cv/CameraParams.hpp"

namespace dyno {

enum class RuntimeSensorOptions : std::uint8_t {
  //! Use this sensor, if available
  PreferSensor = 1 << 0,
  //! if we prefer the sensor but data is not available the system will use
  //! other information
  AcceptNoSensor = 1 << 1,
  //! if we prefer a sensor but data is not available, fail hard
  FailOnNoSensor = 1 << 2
};

template <>
struct internal::EnableBitMaskOperators<RuntimeSensorOptions> : std::true_type {
};

using RuntimeSensorFlags = Flags<RuntimeSensorOptions>;

//! Default runtime sensor options are PreferSensor AND AcceptNoSensor
//! This means that we will try an use the most sensor data available but
//! continue if not
constexpr static RuntimeSensorOptions DefaultRuntimeSensorOptions{
    RuntimeSensorOptions::PreferSensor | RuntimeSensorOptions::AcceptNoSensor};

class DynoParams {
 public:
  DynoParams(const std::string& params_folder_path);
  /**
   * @brief For I/O construction
   *
   */
  DynoParams() {}

  void printAllParams(bool print_glog_params = true) const;

  struct PipelineParams {
    int data_provider_type;  // Kitti, VirtualKitti, Online... currently set
                             // with flagfile
    //! If camera params are provided from the dataprovider, use this instead of
    //! the params here. This allows the specific dataset camera params (which
    //! changes per dataset) rather than needing the change the
    //! CameraParams.yaml everytime
    bool prefer_data_provider_camera_params{true};
    bool prefer_data_provider_imu_params{true};
    //! Pipeline level params
    bool parallel_run{true};

    // TODO: not used yet!!!!!
    RuntimeSensorFlags imu_runtime_options{DefaultRuntimeSensorOptions};
    RuntimeSensorFlags stereo_runtime_options{DefaultRuntimeSensorOptions};
  };

  // Quick access functions
  bool parallelRun() const { return pipeline_params_.parallel_run; }
  int dataProviderType() const { return pipeline_params_.data_provider_type; }
  bool preferDataProviderCameraParams() const {
    return pipeline_params_.prefer_data_provider_camera_params;
  }
  bool preferDataProviderImuParams() const {
    return pipeline_params_.prefer_data_provider_imu_params;
  }

  bool incrementalBackend() const;

 public:
  PipelineParams pipeline_params_;
  FrontendParams frontend_params_;
  BackendParams backend_params_;
  CameraParams camera_params_;
  ImuParams imu_params_;

  FrontendType frontend_type_ = FrontendType::kRGBD;
  BackendType backend_type = BackendType::PARALLEL_HYBRID;

 private:
};

void declare_config(DynoParams::PipelineParams& config);
// ! Original code from:
// https://github.com/MIT-SPARK/Kimera-VIO/blob/master/include/kimera-vio/pipeline/PipelineParams.h
// /**
//  * @brief The PipelineParams base class
//  * Sets a common base class for parameters of the pipeline
//  * for easy parsing/printing. All parameters in VIO should inherit from
//  * this class and implement the print/parseYAML virtual functions.
//  */
// class PipelineParams {
//  public:
//   DYNO_POINTER_TYPEDEFS(PipelineParams);
//   explicit PipelineParams(const std::string& name);
//   virtual ~PipelineParams() = default;

//  public:
//   // Parameters of the pipeline must specify how to be parsed.
//   virtual bool parseYAML(const std::string& filepath) = 0;

//   // Parameters of the pipeline must specify how to be printed.
//   virtual void print() const = 0;

//   // Parameters of the pipeline must specify how to be compard, they need
//   // to implement the equals function below.
//   friend bool operator==(const PipelineParams& lhs, const PipelineParams&
//   rhs); friend bool operator!=(const PipelineParams& lhs, const
//   PipelineParams& rhs);

//  protected:
//   // Parameters of the pipeline must specify how to be compared.
//   virtual bool equals(const PipelineParams& obj) const = 0;

//   /** @brief print Helper function to print arguments in a standardized way
//    * @param[out] Stream were the printed output will be written. You can then
//    * log that to cout or glog, i.e: `LOG(INFO) << out.c_str()`;
//    * @param[in] *Even* list of arguments, where we assume the first to be the
//    * name
//    * and the second a value (both can be of any type (with operator <<)).
//    * Note: uneven list of arguments will raise compilation errors.
//    * Usage:
//    * ```
//    * std::stringstream out;
//    * int my_first_param = 2;
//    * bool my_second_param = 2;
//    * print(out, "My first param", my_first_param,
//    *  "My second param", my_second_param);
//    * std::cout << out.c_str() << std::endl;
//    * ```
//    */
//   template <typename... args>
//   void print(std::stringstream& out, args... to_print) const {
//     out.str("");  // clear contents.

//     // Add title.
//     out.width(kTotalWidth);
//     size_t center =
//         (kTotalWidth - name_.size() - 2u) / 2u;  // -2u for ' ' chars
//     out << '\n'
//         << std::string(center, '*').c_str() << ' ' << name_.c_str() << ' '
//         << std::string(center, '*').c_str() << '\n';

//     // Add columns' headers.
//     out.width(kNameWidth);  // Remove hardcoded, need to pre-calc width.
//     out.setf(std::ios::left, std::ios::adjustfield);
//     out << "Name";
//     out.setf(std::ios::right, std::ios::adjustfield);
//     out.width(kValueWidth);
//     out << "Value\n";

//     // Add horizontal separator
//     out.width(kTotalWidth);  // Remove hardcoded, need to pre-calc width.
//     out << std::setfill('-') << "\n";

//     // Reset fill to dots
//     out << std::setfill('.');
//     printImpl(out, to_print...);

//     // Add horizontal separator
//     out.width(kTotalWidth);  // Remove hardcoded, need to pre-calc width.
//     out << std::setfill('-') << "\n";
//   }

//  public:
//   std::string name_;
//   static constexpr size_t kTotalWidth = 60;
//   static constexpr size_t kNameWidth = 40;
//   static constexpr size_t kValueWidth = 20;

//  private:
//   template <typename TName, typename TValue>
//   void printImpl(std::stringstream& out, TName name, TValue value) const {
//     out.width(kNameWidth);
//     out.setf(std::ios::left, std::ios::adjustfield);
//     out << name;
//     out.width(kValueWidth);
//     out.setf(std::ios::right, std::ios::adjustfield);
//     out << value << '\n';
//   }

//   template <typename TName, typename TValue, typename... Args>
//   void printImpl(std::stringstream& out,
//                  TName name,
//                  TValue value,
//                  Args... next) const {
//     out.width(kNameWidth);
//     out.setf(std::ios::left, std::ios::adjustfield);
//     out << name;
//     out.width(kValueWidth);
//     out.setf(std::ios::right, std::ios::adjustfield);
//     out << value << '\n';
//     printImpl(out, next...);
//   }
// };

// inline bool operator==(const PipelineParams& lhs, const PipelineParams& rhs)
// {
//   // Allow to compare only instances of the same dynamic type
//   return typeid(lhs) == typeid(rhs) && lhs.name_ == rhs.name_ &&
//          lhs.equals(rhs);
// }
// inline bool operator!=(const PipelineParams& lhs, const PipelineParams& rhs)
// {
//   return !(lhs == rhs);
// }

// template <class T>
// inline void parsePipelineParams(const std::string& params_path,
//                                 T* pipeline_params) {
//   CHECK_NOTNULL(pipeline_params);
//   static_assert(std::is_base_of<PipelineParams, T>::value,
//                 "T must be a class that derives from PipelineParams.");
//   // Read/define tracker params.
//   if (params_path.empty()) {
//     LOG(WARNING) << "No " << pipeline_params->name_
//                  << " parameters specified, using default.";
//     *pipeline_params = T();  // default params
//   } else {
//     VLOG(100) << "Using user-specified " << pipeline_params->name_
//               << " parameters: " << params_path;
//     pipeline_params->parseYAML(params_path);
//   }
// }

}  // namespace dyno
