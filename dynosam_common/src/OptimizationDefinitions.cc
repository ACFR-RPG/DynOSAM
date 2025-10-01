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

#include "dynosam_common/OptimizationDefinitions.hpp"

#include <glog/logging.h>

namespace dyno {

namespace {

static bool checkIfLabeledSymbol(gtsam::Key key) {
  const gtsam::LabeledSymbol asLabeledSymbol(key);
  return (asLabeledSymbol.chr() > 0 && asLabeledSymbol.label() > 0);
}

static bool internalReconstructInfo(gtsam::Key key, SymbolChar expected_chr,
                                    ObjectId& object_label, FrameId& frame_id) {
  // assume motion/pose key is a labelled symbol
  if (!checkIfLabeledSymbol(key)) {
    return false;
  }

  const gtsam::LabeledSymbol as_labeled_symbol(key);
  if (as_labeled_symbol.chr() != expected_chr) {
    return false;
  }

  frame_id = static_cast<FrameId>(as_labeled_symbol.index());

  SymbolChar label = as_labeled_symbol.label();
  object_label = label - '0';
  return true;
}
}  // namespace

SymbolChar DynoChrExtractor(gtsam::Key key) {
  const gtsam::LabeledSymbol asLabeledSymbol(key);
  if (asLabeledSymbol.chr() > 0 && asLabeledSymbol.label() > 0) {
    return asLabeledSymbol.chr();
  }
  const gtsam::Symbol asSymbol(key);
  if (asLabeledSymbol.chr() > 0) {
    return asSymbol.chr();
  } else {
    return InvalidDynoSymbol;
  }
}

bool reconstructMotionInfo(gtsam::Key key, ObjectId& object_label,
                           FrameId& frame_id) {
  return internalReconstructInfo(key, kObjectMotionSymbolChar, object_label,
                                 frame_id);
}

bool reconstructPoseInfo(gtsam::Key key, ObjectId& object_label,
                         FrameId& frame_id) {
  return internalReconstructInfo(key, kObjectPoseSymbolChar, object_label,
                                 frame_id);
}

std::string DynosamKeyFormatterVerbose(gtsam::Key key) {
  const gtsam::LabeledSymbol asLabeledSymbol(key);
  if (asLabeledSymbol.chr() > 0 && asLabeledSymbol.label() > 0) {
    // if used as motion
    if (asLabeledSymbol.chr() == kObjectMotionSymbolChar) {
      ObjectId object_label;
      FrameId frame_id;
      CHECK(reconstructMotionInfo(asLabeledSymbol, object_label, frame_id));

      std::stringstream ss;
      ss << "H: label" << object_label << ", frames: " << frame_id - 1 << " -> "
         << frame_id;
      return ss.str();
    } else if (asLabeledSymbol.chr() == kObjectPoseSymbolChar) {
      ObjectId object_label;
      FrameId frame_id;
      CHECK(reconstructPoseInfo(asLabeledSymbol, object_label, frame_id));

      std::stringstream ss;
      ss << "K: label" << object_label << ", frame: " << frame_id;
      return ss.str();
    }
    return (std::string)asLabeledSymbol;
  }

  const gtsam::Symbol asSymbol(key);
  if (asLabeledSymbol.chr() > 0) {
    if (asLabeledSymbol.chr() == kDynamicLandmarkSymbolChar) {
      const DynamicPointSymbol asDynamicPointSymbol(key);

      FrameId frame_id = asDynamicPointSymbol.frameId();
      TrackletId tracklet_id = asDynamicPointSymbol.trackletId();
      std::stringstream ss;
      ss << kDynamicLandmarkSymbolChar << ": frame " << frame_id
         << ", tracklet " << tracklet_id;
      return ss.str();

    } else {
      return (std::string)asSymbol;
    }

  } else {
    return std::to_string(key);
  }
}

std::string DynosamKeyFormatter(gtsam::Key key) {
  const gtsam::LabeledSymbol asLabeledSymbol(key);
  if (asLabeledSymbol.chr() > 0 && asLabeledSymbol.label() > 0) {
    // if used as motion
    if (asLabeledSymbol.chr() == kObjectMotionSymbolChar ||
        asLabeledSymbol.chr() == kObjectPoseSymbolChar) {
      return (std::string)asLabeledSymbol;
    }
    return (std::string)asLabeledSymbol;
  }

  const gtsam::Symbol asSymbol(key);
  if (asLabeledSymbol.chr() > 0) {
    if (asLabeledSymbol.chr() == kDynamicLandmarkSymbolChar) {
      const DynamicPointSymbol asDynamicPointSymbol(key);
      return (std::string)asDynamicPointSymbol;
    } else {
      return (std::string)asSymbol;
    }

  } else {
    return std::to_string(key);
  }
}

std::uint64_t CantorPairingFunction::pair(const Pair& input) {
  const auto k1 = (input.first);
  const auto k2 = (input.second);
  return ((k1 + k2) * (k1 + k2 + 1) / 2) + k2;
}

CantorPairingFunction::Pair CantorPairingFunction::depair(
    const std::uint64_t z) {
  std::uint64_t w =
      static_cast<std::uint64_t>(floor(((sqrt((z * 8) + 1)) - 1) / 2));
  std::uint64_t t = static_cast<std::uint64_t>((w * (w + 1)) / 2);

  std::uint64_t k2 = z - t;
  std::uint64_t k1 = w - k2;
  return std::make_pair(k1, k2);
}

DynamicPointSymbol::DynamicPointSymbol(const DynamicPointSymbol& key)
    : c_(key.c_),
      j_(key.j_),
      tracklet_id_(key.tracklet_id_),
      frame_id_(key.frame_id_) {}

DynamicPointSymbol::DynamicPointSymbol(unsigned char c, TrackletId tracklet_id,
                                       FrameId frame_id)
    : c_(c),
      j_(constructIndex(tracklet_id, frame_id)),
      tracklet_id_(tracklet_id),
      frame_id_(frame_id) {
  // sanity check
  const auto result = CantorPairingFunction::depair(j_);
  CHECK_EQ(result.first, tracklet_id_);
  CHECK_EQ(result.second, frame_id_);
}

DynamicPointSymbol::DynamicPointSymbol(gtsam::Key key) {
  gtsam::Symbol sym(key);
  const unsigned char c = sym.chr();
  const std::uint64_t index = sym.index();

  c_ = c;
  j_ = index;

  recover(j_, tracklet_id_, frame_id_);
}

gtsam::Key DynamicPointSymbol::key() const { return (gtsam::Key)asSymbol(); }

void DynamicPointSymbol::print(const std::string& s) const {
  std::cout << s << ": " << (std::string)(*this) << std::endl;
}

bool DynamicPointSymbol::equals(const DynamicPointSymbol& expected,
                                double) const {
  return (*this) == expected;  // lazy?
}

DynamicPointSymbol::operator std::string() const {
  char buffer[20];
  snprintf(buffer, 20, "%c%ld-%lu", c_, tracklet_id_,
           static_cast<unsigned long>(frame_id_));
  return std::string(buffer);
}

gtsam::Symbol DynamicPointSymbol::asSymbol() const {
  return gtsam::Symbol(c_, j_);
}

std::uint64_t DynamicPointSymbol::constructIndex(TrackletId tracklet_id,
                                                 FrameId frame_id) {
  if (tracklet_id == -1) {
    throw std::invalid_argument(
        "DynamicPointSymbol cannot be constructed from invalid tracklet id "
        "(-1)");
  }

  return CantorPairingFunction::pair({tracklet_id, frame_id});
}

void DynamicPointSymbol::recover(std::uint64_t z, TrackletId& tracklet_id,
                                 FrameId& frame_id) {
  const auto result = CantorPairingFunction::depair(z);
  tracklet_id = static_cast<TrackletId>(result.first);
  frame_id = static_cast<FrameId>(result.second);
}

};  // namespace dyno
