/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris
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

#include "dynosam_common/utils/Histogram.hpp"

namespace dyno {

using json = nlohmann::json;

void to_json(json& j, const Histogram& histogram) {
  // json bin;
  // bin["upper"]
  // j[histogram.name_]

  auto impl_hist = histogram.histogram_;

  std::vector<std::vector<json>> bins(impl_hist.rank());

  for (auto&& x : bh::indexed(impl_hist)) {
    for (size_t i = 0; i < impl_hist.rank(); ++i) {
      json bin;
      bin["lower"] = x.bin(i).lower();
      bin["upper"] = x.bin(i).upper();

      std::stringstream ss;
      ss << *x;

      double value;
      ss >> value;
      bin["count"] = value;

      bins.at(i).push_back(bin);
    }
  }

  j[histogram.name_] = bins;
}

void from_json(const json& j, Histogram& histogram) {}

}  // namespace dyno
