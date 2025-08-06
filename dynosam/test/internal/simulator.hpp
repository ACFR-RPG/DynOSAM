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

#include <gtsam/geometry/Pose3.h>

#include <optional>

#include "dynosam/common/Camera.hpp"
#include "dynosam/common/StructuredContainers.hpp"
#include "dynosam/common/Types.hpp"
#include "dynosam/frontend/RGBDInstance-Definitions.hpp"
#include "dynosam/utils/GtsamUtils.hpp"
#include "helpers.hpp"

namespace dyno_testing {

using namespace dyno;

class ScenarioBodyBase {
 public:
  DYNO_POINTER_TYPEDEFS(ScenarioBodyBase)

  virtual ~ScenarioBodyBase() {}

  virtual gtsam::Pose3 pose(FrameId frame_id) const = 0;  ///< pose at time t
  virtual gtsam::Pose3 motionWorld(
      FrameId frame_id) const = 0;  ///< motion in world frame from t-1 to t
  virtual gtsam::Pose3 motionBody(FrameId frame_id)
      const = 0;  ///< motion local frame from t-1 to t, in ^{t-1}X_{t-1}
  virtual gtsam::Pose3 motionWorldFromInitial(
      FrameId frame_id) const = 0;  ///< motion in world frame from 0 to t

  gtsam::Rot3 rotation(FrameId frame_id) const {
    return this->pose(frame_id).rotation();
  }
  gtsam::Vector3 translation(FrameId frame_id) const {
    return this->pose(frame_id).translation();
  }
};

class ScenarioBodyVisitor : public ScenarioBodyBase {
 public:
  DYNO_POINTER_TYPEDEFS(ScenarioBodyVisitor)
  virtual ~ScenarioBodyVisitor() {}

  virtual gtsam::Pose3 pose(FrameId frame_id) const = 0;  ///< pose at time t
  virtual gtsam::Pose3 motionWorld(
      FrameId frame_id) const = 0;  ///< motion in world frame from t-1 to t

  ///< motion local frame from t-1 to t, in ^{t-1}X_{t-1}
  virtual gtsam::Pose3 motionBody(FrameId frame_id) const override {
    // from t-1 to t
    const gtsam::Pose3 motion_k = motionWorld(frame_id);
    const gtsam::Pose3 pose_k = pose(frame_id);
    // TODO: check
    return pose_k.inverse() * motion_k * pose_k.inverse();
  }
  virtual gtsam::Pose3 motionWorldFromInitial(
      FrameId frame_id) const = 0;  ///< motion in world frame from 0 to t
};

class ScenarioBody : public ScenarioBodyBase {
 public:
  DYNO_POINTER_TYPEDEFS(ScenarioBody)

  ScenarioBody(ScenarioBodyVisitor::UniquePtr body_visitor)
      : body_visitor_(std::move(body_visitor)) {}

  gtsam::Pose3 pose(FrameId frame_id) const override {
    return body_visitor_->pose(frame_id);
  }
  gtsam::Pose3 motionWorld(FrameId frame_id) const override {
    return body_visitor_->motionWorld(frame_id);
  }
  gtsam::Pose3 motionBody(FrameId frame_id) const override {
    return body_visitor_->motionBody(frame_id);
  }
  gtsam::Pose3 motionWorldFromInitial(FrameId frame_id) const override {
    return body_visitor_->motionWorldFromInitial(frame_id);
  }

 protected:
  ScenarioBodyVisitor::UniquePtr body_visitor_;
};

using TrackedPoint = std::pair<TrackletId, gtsam::Point3>;
using TrackedPoints = std::vector<TrackedPoint>;

struct PointsGenerator {
  /**
   * @brief Static function to generate a unique tracklet for any generator. If
   * increment is true, the global tracklet id will be incremented
   *
   * @param increment
   * @return TrackletId
   */
  static TrackletId getTracklet(bool increment = true) {
    static TrackletId global_tracklet = 0;

    auto tracklet_id = global_tracklet;

    if (increment) global_tracklet++;
    return tracklet_id;
  }

  static TrackedPoint generateNewPoint(const gtsam::Point3& mean, double sigma,
                                       int32_t seed = 42) {
    gtsam::Point3 point = dyno::utils::perturbWithNoise(mean, sigma, seed);
    return std::make_pair(PointsGenerator::getTracklet(true), point);
  }
};

/**
 * @brief Base class that knows how to generate points given the
 * ScenarioBodyVisitor for an object
 *
 */
class ObjectPointGeneratorVisitor {
 public:
  DYNO_POINTER_TYPEDEFS(ObjectPointGeneratorVisitor)

  virtual ~ObjectPointGeneratorVisitor() = default;
  virtual TrackedPoints getPointsWorld(
      const ScenarioBodyVisitor::UniquePtr& body_visitor,
      FrameId frame_id) const = 0;
};

class StaticPointGeneratorVisitor {
 public:
  DYNO_POINTER_TYPEDEFS(StaticPointGeneratorVisitor)

  virtual ~StaticPointGeneratorVisitor() = default;
  virtual TrackedPoints getPointsWorld(FrameId frame_id) const = 0;
};

using Range = FrameRange<int>;

class RangesWithEnd : public std::vector<Range::Ptr> {
 public:
  using Base = std::vector<Range>;

  Range::Ptr find(FrameId query) const {
    for (const Range::Ptr& r : *this) {
      if (r->contains(query)) {
        return r;
      }
    }
    return nullptr;
  }

  RangesWithEnd& add(FrameId start, FrameId end) {
    this->push_back(std::make_shared<Range>(start, end, 0, false));
    return *this;
  }
};

struct ObjectBodyParams {
  // FrameId enters_scenario = 0;
  // FrameId leaves_scenario = std::numeric_limits<FrameId>::max();
  RangesWithEnd ranges;

  ObjectBodyParams(
      FrameId enters_scenario = 0,
      FrameId leaves_scenario = std::numeric_limits<FrameId>::max()) {
    addRange(enters_scenario, leaves_scenario);
  }

  ObjectBodyParams& addRange(FrameId enters_scenario, FrameId leaves_scenario) {
    ranges.add(enters_scenario, leaves_scenario);
    return *this;
  }
};

class ObjectBody : public ScenarioBody {
 public:
  DYNO_POINTER_TYPEDEFS(ObjectBody)

  ObjectBody(ScenarioBodyVisitor::UniquePtr body_visitor,
             ObjectPointGeneratorVisitor::UniquePtr points_visitor,
             const ObjectBodyParams& params = ObjectBodyParams())
      : ScenarioBody(std::move(body_visitor)),
        points_visitor_(std::move(points_visitor)),
        params_(params) {}

  virtual bool inFrame(FrameId frame_id) const {
    return (bool)params_.ranges.find(frame_id);
  }
  virtual TrackedPoints getPointsWorld(FrameId frame_id) const {
    return points_visitor_->getPointsWorld(body_visitor_, frame_id);
  };

 protected:
  ObjectPointGeneratorVisitor::UniquePtr points_visitor_;
  ObjectBodyParams params_;
};

// Motion and pose visotors
class ConstantMotionBodyVisitor : public ScenarioBodyVisitor {
 public:
  DYNO_POINTER_TYPEDEFS(ConstantMotionBodyVisitor)
  ConstantMotionBodyVisitor(const gtsam::Pose3& pose_0,
                            const gtsam::Pose3& motion)
      : pose_0_(pose_0), motion_(motion) {}

  virtual gtsam::Pose3 pose(FrameId frame_id) const override {
    // from Pose Changes From a Different Point of View
    return motionWorldFromInitial(frame_id) * pose_0_;
  }

  virtual gtsam::Pose3 motionWorld(FrameId) const override { return motion_; }

  // TODO: I have no idea if this is right for constant motion but whatevs...
  gtsam::Pose3 motionWorldFromInitial(FrameId frame_id) const {
    return gtsam::Pose3::Expmap(frame_id * gtsam::Pose3::Logmap(motion_));
  }

 private:
  const gtsam::Pose3 pose_0_;
  const gtsam::Pose3 motion_;
};

class RandomOverlapObjectPointsVisitor : public ObjectPointGeneratorVisitor {
 public:
  RandomOverlapObjectPointsVisitor(size_t num_points, size_t overlap)
      : num_points_(num_points),
        overlap_(overlap),
        overlap_dist_(2, std::max(2, (int)overlap)) {}

  TrackedPoints getPointsWorld(
      const ScenarioBodyVisitor::UniquePtr& body_visitor,
      FrameId frame_id) const override {
    // minimum n_points per frame
    // get points for frame + 1
    // if size points < num points -> generate N = (num points - len(points))
    // points are generated with 2 <-> O(verlap) as number of points to exist
    std::vector<Point> points_next = getPoints(frame_id + 1);
    // LOG(INFO) << "Points next=" <<points_next.size();
    std::vector<Point> points_current = getPoints(frame_id);
    // LOG(INFO) << "Points current=" <<points_current.size();
    if (points_next.size() < num_points_) {
      auto required_points = num_points_ - points_next.size();
      // LOG(INFO) << "Required points=" << required_points;
      std::vector<Point> points_new = generatePoints(frame_id, required_points);
      points_current.insert(points_current.end(), points_new.begin(),
                            points_new.end());
    }
    // LOG(INFO) << "New Points current=" <<points_current.size();
    // CHECK_EQ(points_current.size(), num_points_);

    TrackedPoints tracked_points(points_current.size());
    std::transform(points_current.begin(), points_current.end(),
                   tracked_points.begin(),
                   [&body_visitor, &frame_id](const Point& p_body) {
                     // LOG(INFO) <<
                     CHECK_NOTNULL(body_visitor);
                     CHECK(p_body.contains(frame_id));
                     const gtsam::Point3 P_world =
                         body_visitor->pose(frame_id) * p_body.P_body_.second;
                     return std::make_pair(p_body.P_body_.first, P_world);
                   });

    return tracked_points;
  }

 private:
  struct Point {
    FrameId starting_frame_;
    FrameId ending_frame_;
    TrackedPoint P_body_;  //! Point in the object body frame

    Point(FrameId starting_frame, FrameId ending_frame,
          const TrackedPoint& P_body)
        : starting_frame_(starting_frame),
          ending_frame_(ending_frame),
          P_body_(P_body) {
      CHECK_GT(ending_frame_, starting_frame_);
    }

    bool contains(FrameId frame_id) const {
      return frame_id >= starting_frame_ && frame_id <= ending_frame_;
    }
  };

  Point generatePoint(FrameId frame) const {
    auto O = overlap_dist_(gen);
    auto ending_frame = frame + O;

    auto tracked_point = PointsGenerator::generateNewPoint(
        gtsam::Point3(0, 0, 0), 0.1, seed_dist_(gen));
    Point p(frame, ending_frame, tracked_point);
    all_points_.push_back(p);
    return p;
  }

  std::vector<Point> getPoints(FrameId frame) const {
    std::vector<Point> points;
    for (const Point& point : all_points_) {
      if (point.contains(frame)) {
        points.push_back(point);
      }
    }
    return points;
  }

  std::vector<Point> generatePoints(FrameId frame, size_t N) const {
    std::vector<Point> points;
    for (size_t i = 0; i < N; i++) {
      points.push_back(generatePoint(frame));
    }

    return points;
  }

  const size_t num_points_;
  const size_t overlap_;

  static std::random_device rd;
  static std::mt19937 gen;

  mutable std::uniform_int_distribution<> overlap_dist_;
  mutable std::uniform_int_distribution<int> seed_dist_{0, 100};

  mutable std::vector<Point> all_points_;
};
inline std::random_device RandomOverlapObjectPointsVisitor::rd;
inline std::mt19937 RandomOverlapObjectPointsVisitor::gen{rd()};

// Points generator visitor
class ConstantObjectPointsVisitor : public ObjectPointGeneratorVisitor {
 public:
  ConstantObjectPointsVisitor(size_t num_points) : num_points_(num_points) {}

  // TODO: this assumes that the points we get from the object are ALWAYS the
  // same and ALWAYS the same order
  //
  TrackedPoints getPointsWorld(
      const ScenarioBodyVisitor::UniquePtr& body_visitor,
      FrameId frame_id) const override {
    if (!is_init) {
      initalisePoints(body_visitor->pose(0));
    }

    TrackedPoints points_world_t;  // points in world frame at time t
    for (const auto& tracked_point : points_world_0_) {
      auto tracklet_id = tracked_point.first;
      auto point = tracked_point.second;
      points_world_t.push_back(std::make_pair(
          tracklet_id, body_visitor->motionWorldFromInitial(frame_id) * point));
    }

    return points_world_t;
  }

 private:
  void initalisePoints(const gtsam::Pose3& P0) const {
    std::mt19937 engine(42);
    std::uniform_real_distribution<double> normal(0.0, 1.0);

    for (size_t i = 0; i < num_points_; i++) {
      // generate around pose0 with a normal distrubution around the translation
      // component
      // gtsam::Point3 p(P0.x() + normal(engine), P0.y() + normal(engine),
      //           P0.z() + normal(engine));

      // points_world_0_.push_back(std::make_pair(PointsGenerator::getTracklet(true),
      // p));
      points_world_0_.push_back(
          PointsGenerator::generateNewPoint(P0.translation(), 1.0));
    }

    is_init = true;
  }

  const size_t num_points_;

  // mutable so can be changed in the initalised poitns function, which is
  // called once
  mutable TrackedPoints points_world_0_;  // points in the world frame at time 0
  mutable bool is_init{false};
};

// I think this only ever means that a point can be seen by a max o
class SimpleStaticPointsGenerator : public StaticPointGeneratorVisitor {
 public:
  SimpleStaticPointsGenerator(size_t num_points_per_frame, size_t overlap)
      : num_points_per_frame_(num_points_per_frame),
        overlap_(overlap),
        has_overlap_(overlap < num_points_per_frame) {}

  TrackedPoints getPointsWorld(FrameId frame_id) const override {
    // expect we always start at zero
    if (frame_id == 0) {
      generateNewPoints(num_points_per_frame_);
      return points_world_0_;
    } else {
      // must have at least this many points after the first (zeroth) frame
      CHECK_GE(points_world_0_.size(), num_points_per_frame_);

      CHECK(has_overlap_) << "not implemented";
      int diff = (int)num_points_per_frame_ - (int)overlap_;
      CHECK(diff > 0);
      generateNewPoints((size_t)diff);

      size_t start_i = frame_id * ((size_t)diff);
      CHECK_GT(start_i, 0);

      size_t end_i = start_i + num_points_per_frame_ - 1;
      CHECK_LT(end_i, points_world_0_.size());

      TrackedPoints points_in_window;
      for (size_t i = start_i; i <= end_i; i++) {
        points_in_window.push_back(points_world_0_.at(i));
      }

      CHECK_EQ(points_in_window.size(), num_points_per_frame_);
      return points_in_window;
    }
  }

 private:
  void generateNewPoints(size_t num_new) const {
    // points can be distributed over this distance
    constexpr double point_distance_sigma = 40;
    for (size_t i = 0; i < num_new; i++) {
      points_world_0_.push_back(PointsGenerator::generateNewPoint(
          gtsam::Point3(0, 0, 0), point_distance_sigma));
    }
  }

  const size_t num_points_per_frame_;
  const size_t overlap_;
  const bool has_overlap_;
  mutable TrackedPoints
      points_world_0_;  // all points in the world frame at time 0. This may be
                        // uppdated overtime within the getPointsWorld
};

class OverlappingStaticPointsGenerator : public StaticPointGeneratorVisitor {
 public:
  using CameraPtr = dyno::Camera::CameraImpl;

  OverlappingStaticPointsGenerator(
      ScenarioBody::Ptr scenario_body,
      dyno::Camera::Ptr dyno_camera = dyno_testing::makeDefaultCameraPtr(),
      size_t min_new_points = 30, size_t max_new_points = 60,
      size_t min_lifetime = 4, size_t max_lifetime = 30)
      : scenario_body_(scenario_body),
        dyno_camera_(dyno_camera),
        min_new_points_(min_new_points),
        max_new_points_(max_new_points),
        min_lifetime_(min_lifetime),
        max_lifetime_(max_lifetime),
        gen_(rd_()) {
    CHECK(scenario_body_);
    if (dyno_camera_) {
      camera_ = dyno_camera_->getImplCamera();
    }
  }

  TrackedPoints getPointsWorld(FrameId frame_id) const override {
    if (frame_id == 0) {
      generateInitialPoints();
    } else {
      updatePointSet(frame_id);
    }

    TrackedPoints visible;
    for (const auto& pt : all_points_) {
      if (pt.contains(frame_id)) {
        visible.emplace_back(pt.P_world_);
      }
    }

    LOG(INFO) << "Generated " << visible.size()
              << " points at frame k=" << frame_id;

    return visible;
  }

 private:
  struct PointWindow {
    FrameId start_;
    FrameId end_;
    TrackedPoint P_world_;

    bool contains(FrameId f) const { return f >= start_ && f <= end_; }
  };

  void generateInitialPoints() const {
    all_points_.clear();

    // Reasonable seeding range for initial points (e.g., 30–60)
    constexpr int initial_min_points = 30;
    constexpr int initial_max_points = 60;

    // const size_t num_initial_points = initial_min_points +
    //                                   (std::rand() % (initial_max_points -
    //                                   initial_min_points + 1));

    std::uniform_int_distribution<int> dist(initial_min_points,
                                            initial_max_points);
    const size_t num_initial_points = dist(gen_);

    // Adds points starting from frame 0, with random lifetime
    addNewPoints(0, num_initial_points);
  }

  void updatePointSet(FrameId frame_id) const {
    // Remove expired points
    all_points_.erase(std::remove_if(all_points_.begin(), all_points_.end(),
                                     [&](const PointWindow& pt) {
                                       return frame_id > pt.end_;
                                     }),
                      all_points_.end());

    // Always retain some overlap from previous frame
    size_t num_overlap = std::count_if(
        all_points_.begin(), all_points_.end(),
        [&](const PointWindow& pt) { return pt.contains(frame_id - 1); });
    CHECK_GE(num_overlap, 0u);  // May be 0 at first

    // Random new point count
    std::uniform_int_distribution<int> dist(min_new_points_, max_new_points_);
    size_t num_new = dist(gen_);
    addNewPoints(frame_id, num_new);
  }

  void addNewPoints(FrameId start_frame, size_t num_new) const {
    constexpr double radius_min = 1.5;
    constexpr double radius_max = 5.0;
    constexpr int max_attempts = 100;
    constexpr double visibility_threshold = 1.0;

    VLOG(10) << "Generating " << num_new << " points at k=" << start_frame;

    for (size_t i = 0; i < num_new; ++i) {
      TrackedPoint p;
      bool success = false;

      std::uniform_int_distribution<int> dist(min_lifetime_, max_lifetime_);
      int lifetime = dist(gen_);

      FrameId end_frame = start_frame + lifetime - 1;

      // LOG(INFO) << "Attempting point with lifetime " << lifetime << " end
      // frame " << end_frame;

      for (int attempt = 0; attempt < max_attempts; ++attempt) {
        Eigen::Vector3d pt;

        if (camera_) {
          pt = scenario_body_->pose(start_frame) *
               samplePointInFrontOfCameraFromIntrinsics(0.2, 45);

          int visible_count = 0;
          int total_count = static_cast<int>(end_frame - start_frame + 1);

          for (FrameId f = start_frame; f <= end_frame; ++f) {
            const gtsam::Pose3 cam_pose = scenario_body_->pose(f);
            if (isVisibleInFrustum(pt, cam_pose)) {
              ++visible_count;
            }
          }

          double visibility = static_cast<double>(total_count) /
                              static_cast<double>(visible_count);
          if (visibility < visibility_threshold) {
            LOG(INFO) << "Visibility " << visibility;
            continue;
          }
        } else {
          pt = samplePointInShell(radius_min, radius_max);
        }

        p = PointsGenerator::generateNewPoint(pt, 0.0);
        all_points_.push_back(PointWindow{start_frame, end_frame, p});
        success = true;
        break;
      }

      CHECK(success) << "Could not sample visible point after many attempts";
    }
  }

  Eigen::Vector3d samplePointInShell(double r_min, double r_max) const {
    const double u = randUniform();
    const double v = randUniform();
    const double theta = 2.0 * M_PI * u;
    const double phi = std::acos(2.0 * v - 1.0);
    const double r = r_min + (r_max - r_min) * randUniform();

    const double x = r * std::sin(phi) * std::cos(theta);
    const double y = r * std::sin(phi) * std::sin(theta);
    const double z = r * std::cos(phi);

    return Eigen::Vector3d(x, y, z);
  }

  Eigen::Vector3d samplePointInFrontOfCameraFromIntrinsics(
      double radius_min, double radius_max) const {
    const auto& cam_params = dyno_camera_->getParams();
    const auto& K = cam_params.getCameraMatrixEigen();

    // // Sample pixel (u, v) in image space
    // int u = std::rand() % cam_params.ImageWidth();
    // int v = std::rand() % cam_params.ImageHeight();

    std::uniform_int_distribution<int> distribution_width(
        1, cam_params.ImageWidth() - 1u);
    std::uniform_int_distribution<int> distribution_height(
        1, cam_params.ImageHeight() - 1u);

    int u = distribution_width(gen_);
    int v = distribution_height(gen_);

    // Sample depth (z) in range [radius_min, radius_max]
    double z = radius_min +
               ((std::rand() / (double)RAND_MAX) * (radius_max - radius_min));

    // // Backproject to 3D
    Keypoint kp(u, v);
    // Eigen::Vector3d pixel_homo(u, v, 1.0);
    // Eigen::Vector3d ray = K.inverse() * pixel_homo;

    // Scale ray to depth z
    gtsam::Point3 p_camera;
    dyno_camera_->backProject(kp, z, &p_camera);
    return p_camera;
  }

  bool isVisibleInFrustum(const Eigen::Vector3d& pt_world,
                          const gtsam::Pose3& cam_pose) const {
    dyno::Camera::CameraImpl cam(cam_pose, camera_->calibration());
    const auto [uv, result] = cam.projectSafe(pt_world);

    if (!result) return false;

    const auto& img_size = dyno_camera_->getParams().imageSize();
    if (uv.x() < 0 || uv.x() >= img_size.width || uv.y() < 0 ||
        uv.y() >= img_size.height)
      return false;

    return true;
  }

  double randUniform() const { return static_cast<double>(rand()) / RAND_MAX; }

  ScenarioBody::Ptr scenario_body_;
  dyno::Camera::Ptr dyno_camera_;
  size_t min_new_points_;
  size_t max_new_points_;
  size_t min_lifetime_;
  size_t max_lifetime_;
  dyno::Camera::CameraImpl* camera_;

  mutable std::random_device rd_{};
  mutable std::mt19937 gen_;

  mutable std::vector<PointWindow> all_points_;
};

class Scenario {
 public:
  Scenario(ScenarioBody::Ptr camera_body,
           StaticPointGeneratorVisitor::Ptr static_points_generator)
      : camera_body_(camera_body),
        static_points_generator_(static_points_generator) {}

  void addObjectBody(ObjectId object_id, ObjectBody::Ptr object_body) {
    CHECK_GT(object_id, background_label);
    object_bodies_.insert2(object_id, object_body);
  }

  gtsam::Pose3 cameraPose(FrameId frame_id) const {
    return camera_body_->pose(frame_id);
  }

  ObjectIds getObjectIds(FrameId frame_id) const {
    ObjectIds object_ids;
    for (const auto& [object_id, obj] : object_bodies_) {
      if (objectInScenario(object_id, frame_id))
        object_ids.push_back(object_id);
    }

    return object_ids;
  }

  bool objectInScenario(ObjectId object_id, FrameId frame_id) const {
    if (object_bodies_.exists(object_id)) {
      const auto& object = object_bodies_.at(object_id);

      return object->inFrame(frame_id);
    }
    return false;
  }

 protected:
  ScenarioBody::Ptr camera_body_;
  StaticPointGeneratorVisitor::Ptr static_points_generator_;
  gtsam::FastMap<ObjectId, ObjectBody::Ptr> object_bodies_;

  mutable ObjectMotionMap object_motions_;
  mutable ObjectMotionMap noisy_object_motions_;
  // ObjectPoseMap object_poses_;
};

class RGBDScenario : public Scenario {
 public:
  // pixel + depth noise constructed independantly (ie gaussian around pixel,
  // gaussian around depth) prior to projection into 3D
  struct RGBDNoiseParams {
    double sigma_pixel = 0;
    double sigma_depth = 0;
  };

  // Depth-dependent anisotropic noise directly in 3d space
  struct Point3NoiseParams {
    double sigma_xy = 0;
    double sigma_z = 0;

    Point3NoiseParams(double sigma_xy_, double sigma_z_)
        : sigma_xy(sigma_xy_), sigma_z(sigma_z_) {}
  };

  struct NaivePoint3dNoiseParams {
    double sigma = 0;

    NaivePoint3dNoiseParams(double sigma_) : sigma(sigma_) {}
  };

  using PointNoise =
      std::variant<RGBDNoiseParams, Point3NoiseParams, NaivePoint3dNoiseParams>;

  struct NoiseParams {
    double H_R_sigma{0.0};
    double H_t_sigma{0.0};

    //! rotation noise on relative camera motion
    double X_R_sigma{0.0};
    //! translation noise on relative camera motion
    double X_t_sigma{0.0};

    PointNoise dynamic_point_noise = NaivePoint3dNoiseParams(0);
    PointNoise static_point_noise = NaivePoint3dNoiseParams(0);
    // double dynamic_point_sigma{0.0};
    // double static_point_sigma{0.0};

    NoiseParams() {}
  };

  struct Params {
    double static_outlier_ratio = 0;   // Must be between 0 and 1
    double dynamic_outlier_ratio = 0;  // Must be between 0 and 1

    Params() {}
  };

  RGBDScenario(ScenarioBody::Ptr camera_body,
               StaticPointGeneratorVisitor::Ptr static_points_generator,
               const NoiseParams& noise_params = NoiseParams(),
               const Params& params = Params())
      : Scenario(camera_body, static_points_generator),
        noise_params_(noise_params),
        params_(params) {}

  // first is gt, second is with noisy
  using Output = std::pair<VisionImuPacket::Ptr, VisionImuPacket::Ptr>;

  Output getOutput(FrameId frame_id) const;

  const GroundTruthPacketMap& getGroundTruths() const { return ground_truths_; }

 private:
  Point3Measurement addNoiseStaticPoint(const Point3Measurement& p_local) const;
  Point3Measurement addNoiseDynamicPoint(
      const Point3Measurement& p_local) const;

  Point3Measurement addNoisePoint(const Point3Measurement& p_local,
                                  const PointNoise& options,
                                  double outlier_ratio) const;

  std::pair<gtsam::Point3, gtsam::SharedGaussian> addAnisotropicNoiseToPoint(
      const gtsam::Point3& p, double sigma_xy, double sigma_z) const;

 private:
  NoiseParams noise_params_;
  Params params_;
  mutable GroundTruthPacketMap ground_truths_;
  mutable gtsam::FastMap<FrameId, gtsam::Pose3> noisy_camera_poses_;

  mutable std::uniform_real_distribution<double> outlier_dist{0.0, 1.0};
};

inline dyno_testing::RGBDScenario makeDefaultScenario() {
  dyno_testing::ScenarioBody::Ptr camera =
      std::make_shared<dyno_testing::ScenarioBody>(
          std::make_unique<dyno_testing::ConstantMotionBodyVisitor>(
              gtsam::Pose3::Identity(),
              // motion only in x
              gtsam::Pose3(gtsam::Rot3::Identity(), gtsam::Point3(0.1, 0, 0))));
  // needs to be at least 3 overlap so we can meet requirements in graph
  // TODO: how can we do 1 point but with lots of overlap (even infinity
  // overlap?)
  dyno_testing::RGBDScenario scenario(
      camera,
      std::make_shared<dyno_testing::SimpleStaticPointsGenerator>(8, 3));

  // add one obect
  const size_t num_points = 3;
  dyno_testing::ObjectBody::Ptr object1 =
      std::make_shared<dyno_testing::ObjectBody>(
          std::make_unique<dyno_testing::ConstantMotionBodyVisitor>(
              gtsam::Pose3(gtsam::Rot3::Identity(), gtsam::Point3(10, 0, 0)),
              // motion only in x
              gtsam::Pose3(gtsam::Rot3::Identity(), gtsam::Point3(0.2, 0, 0))),
          std::make_unique<dyno_testing::ConstantObjectPointsVisitor>(
              num_points));

  dyno_testing::ObjectBody::Ptr object2 =
      std::make_shared<dyno_testing::ObjectBody>(
          std::make_unique<dyno_testing::ConstantMotionBodyVisitor>(
              gtsam::Pose3(gtsam::Rot3::Identity(), gtsam::Point3(10, 0, 0)),
              // motion only in x
              gtsam::Pose3(gtsam::Rot3::Identity(), gtsam::Point3(0.2, 0, 0))),
          std::make_unique<dyno_testing::ConstantObjectPointsVisitor>(
              num_points));

  scenario.addObjectBody(1, object1);
  scenario.addObjectBody(2, object2);
  return scenario;
}

}  // namespace dyno_testing
