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

#pragma once

#include <boost/format.hpp>     // only needed for printing
#include <boost/histogram.hpp>  // make_histogram, integer, indexed
#include <iostream>
#include <nlohmann/json.hpp>
#include <vector>

namespace dyno {

namespace bh = boost::histogram;

// https://www.boost.org/doc/libs/1_82_0/libs/histogram/doc/html/histogram/getting_started.html
class Histogram {
 public:
  //! Possible axis types wrapped in a variant
  using AxisTypes =
      boost::histogram::axis::variant<boost::histogram::axis::regular<>,
                                      boost::histogram::axis::variable<>,
                                      boost::histogram::axis::integer<> >;
  using Axis = std::vector<AxisTypes>;
  using ImplHistogram = boost::histogram::histogram<Axis>;

  // TODO: does this need to be && to enable perfect forwarning?
  template <typename... Axis>
  Histogram(const boost::histogram::histogram<Axis...>& histogram)
      : histogram_(histogram) {}

  std::string toString() const {
    std::stringstream ss;
    for (auto&& x : bh::indexed(histogram_)) {
      ss << "Bin [";
      for (size_t i = 0; i < histogram_.rank(); ++i) {
        ss << x.bin(i).lower() << ", " << x.bin(i).upper();
        if (i < histogram_.rank() - 1) ss << "], [";
      }
      ss << "]: " << x.get() << "\n";
    }
    return ss.str();
  }

 public:
  ImplHistogram histogram_;
  std::string name_;
};

using json = nlohmann::json;

void to_json(json& j, const Histogram& histogram);

void from_json(const json& j, Histogram& histogram);

}  // namespace dyno
