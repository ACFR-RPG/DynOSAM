#pragma once

#include <gtsam/inference/LabeledSymbol.h>
#include <gtsam/inference/Symbol.h>

#include <map>
#include <unordered_map>

#include "dynosam_common/Types.hpp"

namespace dyno {

using SymbolChar = unsigned char;
static constexpr SymbolChar kPoseSymbolChar = 'X';
static constexpr SymbolChar kVelocitySymbolChar = 'V';
static constexpr SymbolChar kObjectMotionSymbolChar = 'H';
static constexpr SymbolChar kObjectPoseSymbolChar = 'L';
static constexpr SymbolChar kStaticLandmarkSymbolChar = 'l';
static constexpr SymbolChar kDynamicLandmarkSymbolChar = 'm';
static constexpr SymbolChar kImuBiasSymbolChar = 'b';

constexpr static SymbolChar InvalidDynoSymbol = '\0';

std::string DynosamKeyFormatter(gtsam::Key);
std::string DynosamKeyFormatterVerbose(gtsam::Key);

// TODO: not actually sure if this is necessary
// in this sytem we mix Symbol and LabelledSymbol so I just check which one the
// correct cast is and use that label, This will return InvalidDynoSymbol if a
// key cannot be constructed
SymbolChar DynoChrExtractor(gtsam::Key);

bool reconstructMotionInfo(gtsam::Key key, ObjectId& object_label,
                           FrameId& frame_id);
bool reconstructPoseInfo(gtsam::Key key, ObjectId& object_label,
                         FrameId& frame_id);

// implementation of Contor's pairing function:
// http://szudzik.com/ElegantPairing.pdf When x and y are non−negative integers,
// pairing them outputs a single non−negative integer
//  that is uniquely associated with that pair.
struct CantorPairingFunction {
  using Pair = std::pair<std::uint64_t, std::uint64_t>;

  static std::uint64_t pair(const Pair& input);
  static Pair depair(const std::uint64_t z);
};

class DynamicPointSymbol {
 protected:
  unsigned char c_;
  std::uint64_t j_;

  TrackletId tracklet_id_;
  FrameId frame_id_;

 public:
  /** Default constructor */
  DynamicPointSymbol() : c_(0), j_(0), tracklet_id_(-1), frame_id_(0) {}

  /** Copy constructor */
  DynamicPointSymbol(const DynamicPointSymbol& key);
  /** Constructor */
  DynamicPointSymbol(unsigned char c, TrackletId tracklet_id, FrameId frame_id);

  /** Constructor that decodes an integer Key */
  DynamicPointSymbol(gtsam::Key key);

  /** return Key (integer) representation */
  gtsam::Key key() const;

  /** Cast to integer */
  operator gtsam::Key() const { return key(); }

  /// Print
  void print(const std::string& s = "") const;

  /// Check equality
  bool equals(const DynamicPointSymbol& expected, double tol = 0.0) const;

  /** Retrieve key character */
  unsigned char chr() const { return c_; }

  /** Retrieve key index */
  std::uint64_t index() const { return j_; }

  TrackletId trackletId() const { return tracklet_id_; }

  FrameId frameId() const { return frame_id_; }

  /** Create a string from the key */
  operator std::string() const;

  /// Return string representation of the key
  std::string string() const { return std::string(*this); }

  /** Comparison for use in maps */
  bool operator<(const DynamicPointSymbol& comp) const {
    return c_ < comp.c_ || (comp.c_ == c_ && j_ < comp.j_);
  }

  /** Comparison for use in maps */
  bool operator==(const DynamicPointSymbol& comp) const {
    return comp.c_ == c_ && comp.j_ == j_;
  }

  /** Comparison for use in maps */
  bool operator==(gtsam::Key comp) const { return comp == (gtsam::Key)(*this); }

  /** Comparison for use in maps */
  bool operator!=(const DynamicPointSymbol& comp) const {
    return comp.c_ != c_ || comp.j_ != j_;
  }

  /** Comparison for use in maps */
  bool operator!=(gtsam::Key comp) const { return comp != (gtsam::Key)(*this); }

 private:
  gtsam::Symbol asSymbol() const;

  static std::uint64_t constructIndex(TrackletId tracklet_id, FrameId frame_id);
  static void recover(std::uint64_t z, TrackletId& tracklet_id,
                      FrameId& frame_id);
};

inline gtsam::Key H(unsigned char label, std::uint64_t j) {
  return gtsam::LabeledSymbol(kObjectMotionSymbolChar, label, j);
}
inline gtsam::Key L(unsigned char label, std::uint64_t j) {
  return gtsam::LabeledSymbol(kObjectPoseSymbolChar, label, j);
}

inline gtsam::Symbol CameraPoseSymbol(FrameId frame_id) {
  return gtsam::Symbol(kPoseSymbolChar, frame_id);
}
inline gtsam::Symbol StaticLandmarkSymbol(TrackletId tracklet_id) {
  return gtsam::Symbol(kStaticLandmarkSymbolChar, tracklet_id);
}
inline DynamicPointSymbol DynamicLandmarkSymbol(FrameId frame_id,
                                                TrackletId tracklet_id) {
  return DynamicPointSymbol(kDynamicLandmarkSymbolChar, tracklet_id, frame_id);
}
inline gtsam::Key ObjectMotionSymbol(ObjectId object_label, FrameId frame_id) {
  unsigned char label = object_label + '0';
  return H(label, static_cast<std::uint64_t>(frame_id));
}

inline gtsam::Key ObjectPoseSymbol(ObjectId object_label, FrameId frame_id) {
  unsigned char label = object_label + '0';
  return L(label, static_cast<std::uint64_t>(frame_id));
}

/**
 * @brief Helper class that allows functional callbacks to be triggered based on
 * the type of gtsam::Key provided, where the key should refer to a valid type
 * within the DynoSAM ecosystem.
 *
 * These include the symbols defined by CameraPoseSymbol, StaticLandmarkSymbol,
 * DynamicLandmarkSymbol, ObjectMotionSymbol and ObjectPoseSymbol.
 *
 * Callbacks can be registered to the class and then the operator will trigger
 * the callback based on the type, extracting and providing associated
 * meta-data.
 *
 * Not all callbacks for all symbols need to be registered.
 *
 */
class ApplyFunctionalSymbol {
 public:
  using CameraPoseFunc = std::function<void(FrameId, const gtsam::Symbol&)>;
  using ObjectMotionFunc =
      std::function<void(FrameId, ObjectId, const gtsam::LabeledSymbol&)>;
  using ObjectPoseFunc =
      std::function<void(FrameId, ObjectId, const gtsam::LabeledSymbol&)>;
  using StaticLmkFunc = std::function<void(TrackletId, const gtsam::Symbol&)>;
  using DynamicLmkFunc =
      std::function<void(TrackletId, const DynamicPointSymbol&)>;

  ApplyFunctionalSymbol() = default;
  virtual ~ApplyFunctionalSymbol() = default;

  bool operator()(gtsam::Key key) const;

  void reset();

  ApplyFunctionalSymbol& cameraPose(const CameraPoseFunc&);
  ApplyFunctionalSymbol& objectMotion(const ObjectMotionFunc&);
  ApplyFunctionalSymbol& objectPose(const ObjectPoseFunc&);
  ApplyFunctionalSymbol& staticLandmark(const StaticLmkFunc&);
  ApplyFunctionalSymbol& dynamicLandmark(const DynamicLmkFunc&);

 protected:
  CameraPoseFunc pose_func_;
  ObjectMotionFunc object_motion_func_;
  ObjectPoseFunc object_pose_func_;
  StaticLmkFunc static_lmk_func_;
  DynamicLmkFunc dynamic_lmk_func_;
};

}  // namespace dyno
