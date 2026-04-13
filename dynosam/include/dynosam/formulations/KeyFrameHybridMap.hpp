#pragma once

#include <unordered_set>

#include "dynosam_common/SensorModels.hpp"
#include "dynosam_opt/Map.hpp"

namespace dyno {

class LandmarkKFNode;
class ObjectKFNode;
class FrameKFNode;

struct KeyFrameNodeTypes {
  using Measurement = CameraMeasurement;
  using FrameNodeT = FrameKFNode;
  using ObjectNodeT = ObjectKFNode;
  using LandmarkNodeT = LandmarkKFNode;
};

// TODO: what about direct calls to the map such as getFrames etc
// does this break things...?

// need to change base behaviour such that "getSeenFrames" or equivalent
// only returns true or equivalent for frames where isCameraKeyFrame returns
// true this means that we only add measurements for camera keyframes then for
// objects we control which frames we iterative over directyl becuase the
// KeyFrameMap knows about keyframes but the base map class does not!

// Bascially all classes use the landmark getSeenFrames function to determine
// their own observations. Modifying only these functions should work! ;)
class LandmarkKFNode : public LandmarkNodeBase<KeyFrameNodeTypes> {
 public:
  typedef LandmarkNodeBase<KeyFrameNodeTypes> Base;
  DYNO_POINTER_TYPEDEFS(LandmarkKFNode)

  LandmarkKFNode(TrackletId tracklet_id, ObjectId object_id)
      : Base(tracklet_id, object_id) {}

  /* Gets all keyframes */
  const Frames& getSeenFrames() const { return getSeenFramesImpl(); }

  /* Gets all keyframes */
  Frames& getSeenFrames() { return getSeenFramesImpl(); }

  /** Overwritten function to add the frame node and measurement
   * to internal data-structures.
   *
   * Will ALSO add frame_node to the set of keyframes_ if frame_node
   * is a keyframe!
   */
  void add(SharedFrame frame_node, const CameraMeasurement& measurement);

  /* Checks all frames not just keyframes */
  bool seenAtAnyFrame(FrameId frame_id) const {
    return this->frames_.exists(frame_id);
  }

  /* Gets all frames with measurements, not just keyframes */
  const Frames& getAllSeenFrames() const { return this->frames_; }

  /* Gets all frames with measurements, not just keyframes */
  Frames& getAllSeenFrames() { return this->frames_; }

 protected:
 private:
  /** If lmk is static only operate on camera keyframes, else all frames */
  const Frames& getSeenFramesImpl() const {
    // if (this->isStatic()) {
    //   return camera_keyframes_;
    // }
    // else {
    //   return Base::getSeenFrames();
    // }
    return keyframes_;
  }

  /** If lmk is static only operate on camera keyframes, else all frames */
  Frames& getSeenFramesImpl() {
    // if (this->isStatic()) {
    //   return camera_keyframes_;
    // }
    // else {
    //   return Base::getSeenFrames();
    // }
    return keyframes_;
  }

 private:
  //! Make FrameKFNode so frames can update the keyframes_ variable
  //! of each attached landmark when it is marked as a keyframe
  friend class FrameKFNode;
  //! All frames where this landmark is observed and marked as keyframe
  //! Marking as a keyframe depends on the object id (ie. if this lmk is static
  //! then marking frame k as a camera keyframe will add frame k to this set)
  Frames keyframes_;
};

class ObjectKFNode : public ObjectNodeBase<KeyFrameNodeTypes> {
 public:
  typedef ObjectNodeBase<KeyFrameNodeTypes> Base;
  DYNO_POINTER_TYPEDEFS(ObjectKFNode)

  ObjectKFNode(ObjectId object_id) : Base(object_id) {}
};

class FrameKFNode : public FrameNodeBase<KeyFrameNodeTypes> {
 public:
  typedef FrameNodeBase<KeyFrameNodeTypes> Base;
  DYNO_POINTER_TYPEDEFS(FrameKFNode)

  FrameKFNode(FrameId frame_id, Timestamp timestamp)
      : Base(frame_id, timestamp) {}

  bool isAnyKeyFrame() const;
  bool isAnyObjectKeyFrame() const;

  bool setCameraKeyFrame();
  bool setObjectKeyFrame(ObjectId object_id);

  bool isCameraKeyFrame() const;
  bool isObjectKeyFrame(ObjectId object_id) const;

 private:
  // allert all landmarks that this frame is now a keyframe
  // achieved by updating LandmarkKFNode::keyframes_ for each
  // LandmarkKFNode in the input set
  // can be const becuase the set contains (non-const) shared pointers
  void updateLandmarksWithKF(const Base::Landmarks& lmks);

 private:
  bool is_camera_keyframe_{false};
  std::unordered_set<ObjectId> object_keyframes_;
};

class KeyFrameMap : public Map<KeyFrameNodeTypes> {
  struct Private {};

 public:
  using Base = Map<KeyFrameNodeTypes>;
  DYNO_POINTER_TYPEDEFS(KeyFrameMap)

  // Constructor is only usable by this class
  KeyFrameMap(Private) {}

  static std::shared_ptr<KeyFrameMap> create() {
    return std::make_shared<KeyFrameMap>(Private());
  }

  bool setCameraKeyFrame(FrameId frame_id);
  bool setObjectKeyFrame(FrameId frame_id, ObjectId object_id);

  bool isAnyKeyFrame(FrameId frame_id) const;
  bool isCameraKeyFrame(FrameId frame_id) const;
  bool isObjectKeyFrame(FrameId frame_id, ObjectId object_id) const;
};

}  // namespace dyno
