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

#include "dynosam_common/Types.hpp"

#include <cxxabi.h>
#include <gflags/gflags.h>

#include <string>

#include "dynosam_common/Flags.hpp"

// common glags used in multiple modules
DEFINE_bool(init_object_pose_from_gt, false,
            "If true, then the viz pose from the frontend/backend will start "
            "from the gt");
DEFINE_bool(save_frontend_json, false,
            "If true, then the output of the frontend will be saved as a "
            "binarized json file");
DEFINE_bool(frontend_from_file, false,
            "If true, the frontend will try and load all the data from a "
            "seralized json file");
DEFINE_bool(
    use_byte_tracker, false,
    "If true, the ByteTracking will be used to associated objects between "
    "frames. Else, the instance label will be used as the object id");

namespace dyno {

template <>
std::string to_string(const ReferenceFrame& reference_frame) {
  switch (reference_frame) {
    case ReferenceFrame::LOCAL:
      return "local";
    case ReferenceFrame::GLOBAL:
      return "global";
    case ReferenceFrame::OBJECT:
      return "object";
    default:
      return "Unknown reference frame";
  }
};

template <>
std::string to_string(const MotionRepresentationStyle& style) {
  switch (style) {
    case MotionRepresentationStyle::F2F:
      return "F2F";
    case MotionRepresentationStyle::KF:
      return "KF";
    default:
      return "Unknown MotionRepresentationStyle";
  }
};

std::string demangle(const char* name) {
  // by default set to the original mangled name
  std::string demangled_name = std::string(name);

  // g++ version of demangle
  char* demangled = nullptr;
  int status = -1;  // some arbitrary value to eliminate the compiler warning
  demangled = abi::__cxa_demangle(name, nullptr, nullptr, &status);

  demangled_name = (status == 0) ? std::string(demangled) : std::string(name);
  std::free(demangled);

  return demangled_name;
}

}  // namespace dyno
