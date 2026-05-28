#pragma once

#include "dynosam_common/StructuredContainers.hpp"
#include "dynosam_common/Types.hpp"

namespace dyno {

class TrajectoryEntryAlreadyExists : public DynosamException {
 public:
  TrajectoryEntryAlreadyExists(const FrameId& frame_id)
      : DynosamException("Trajectory entry already exists at k=" +
                         std::to_string(frame_id)) {}
};

template <typename TData>
struct TrajectoryEntry {
  FrameId frame_id;
  Timestamp timestamp;
  //! User defined data
  TData data;
};

template <typename Derived, typename TData>
class TrajectoryBase {
 public:
  using Data = TData;

  using This = TrajectoryBase<Derived, Data>;
  using Entry = TrajectoryEntry<Data>;

  //! Alias for internal trajectory map
  using TrajectoryImpl = gtsam::FastMap<FrameId, Entry>;
  using EntryIterator =
      vector_iterator_base<typename TrajectoryImpl::iterator, Entry>;
  using ConstEntryIterator =
      vector_iterator_base<typename TrajectoryImpl::const_iterator,
                           const Entry>;

  struct Segment {
    const FrameId start_frame{0};
    const FrameId end_frame{0};
    Derived trajectory;

    Segment(FrameId s_frame, FrameId e_frame, const Derived& trajectory_segment)
        : start_frame(s_frame),
          end_frame(e_frame),
          trajectory(trajectory_segment) {}

    static Segment fromTrajectory(const Derived& trajectory_segment) {
      if (trajectory_segment.empty()) {
        return Segment{0, 0, {}};
      }

      return Segment(trajectory_segment.first().frame_id,
                     trajectory_segment.last().frame_id, trajectory_segment);
    }

    bool check() const {
      if (trajectory.first().frame_id != start_frame) {
        return false;
      }
      if (trajectory.last().frame_id != end_frame) {
        return false;
      }

      if (trajectory.empty()) {
        return true;
      }

      // Check increasing trajectory
      auto it = trajectory.begin();
      auto prev_it = it;
      ++it;

      bool is_continuous = true;
      for (; it != trajectory.end(); ++it) {
        // Check continuity
        if (it->frame_id != prev_it->frame_id + 1) {
          is_continuous = false;
        }
        prev_it = it;
      }
      return is_continuous;
    }
  };

  TrajectoryBase() : trajectory_() {}
  virtual ~TrajectoryBase() = default;

  Derived& derived() { return static_cast<Derived&>(*this); }

  const Derived& derived() const { return static_cast<const Derived&>(*this); }

  Derived& insert(FrameId frame_id, Timestamp timestamp, const TData& data) {
    return insert(Entry{frame_id, timestamp, data});
  }

  Derived& insertOrUpdate(FrameId frame_id, Timestamp timestamp,
                          const TData& data) {
    return insertOrUpdate(Entry{frame_id, timestamp, data});
  }

  template <typename DERIVED>
  Derived& insert(const TrajectoryBase<DERIVED, Data>& other) {
    for (const auto& entry : other) {
      this->insert(entry);
    }

    return derived();
  }

  template <typename DERIVED>
  Derived& insertOrUpdate(const TrajectoryBase<DERIVED, Data>& other) {
    for (const auto& entry : other) {
      this->insertOrUpdate(entry);
    }
    return derived();
  }

  bool update(FrameId frame_id, const TData& data) {
    if (exists(frame_id)) {
      trajectory_.at(frame_id).data = data;
      return true;
    }
    return false;
  }

  bool empty() const { return trajectory_.empty(); }
  size_t size() const { return trajectory_.size(); }
  void clear() { trajectory_.clear(); }

  const Entry& get(FrameId frame_id) const { return trajectory_.at(frame_id); }

  Entry& get(FrameId frame_id) { return trajectory_.at(frame_id); }

  const TData& at(FrameId frame_id) const { return get(frame_id).data; }

  TData& at(FrameId frame_id) { return get(frame_id).data; }

  Timestamp timeAt(FrameId frame_id) const { return get(frame_id).timestamp; }

  bool exists(FrameId frame_id) const { return trajectory_.exists(frame_id); }

  const Entry& first() const { return trajectory_.begin()->second; }
  const Entry& last() const { return trajectory_.rbegin()->second; }

  FrameId minFrame() const { return first().frame_id; }
  FrameId maxFrame() const { return last().frame_id; }

  Timestamp startTime() const { return first().timestamp; }
  Timestamp endTime() const { return last().timestamp; }

  /**
   * @brief Slice a range of this trajectory and return a new trajectory.
   *
   * If start greater than end, DynosamException is thrown.
   *
   * If the provided end > the actual max frame, a rnage up to the last frame is
   * provided.
   *
   * Both start and end are optinal, if not proviced defaults to minFrame and
   * maxFrame respectively.
   *
   * @param start std::optional<FrameId>
   * @param end std::optional<FrameId>
   * @return Derived
   */
  Derived range(std::optional<FrameId> start = {},
                std::optional<FrameId> end = {}) const {
    if (empty()) {
      return Derived{};
    }

    if (!start && !end) {
      return derived();
    }

    auto begin_it =
        (start) ? trajectory_.lower_bound(*start) : trajectory_.begin();

    auto last_key = end.value_or(trajectory_.rbegin()->first);
    // clamp to max frame in case user provides on that is greater than the last
    // frame
    last_key = std::min(last_key, maxFrame());

    if (begin_it->first > last_key) {
      throw DynosamException(
          "Cannot construct range query for " + trajectoryName() +
          " provided start " + (start ? std::to_string(*start) : "None") +
          " > " + " provided end " + (end ? std::to_string(*end) : "None"));
    }

    Derived result;
    for (auto it = begin_it; it != trajectory_.end() && it->first <= last_key;
         ++it)
      result.insert(*it);

    return result;
  }

  /**
   * @brief Get the previous entry in the trajectory.
   *
   * Entry may not be frame_id - 1 if trajectory has multiple segments
   *
   * @param previous
   * @param query
   * @return true
   * @return false
   */
  bool getPrevious(Entry& previous, const Entry& query) const {
    // use the internal iterator of trajecrory becuase this->begin() returns a
    // forward only iterator
    auto it = trajectory_.find(query.frame_id);
    if (it == trajectory_.end()) {
      return false;
    }

    if (it == trajectory_.begin()) {
      return false;
    }

    auto prev_it = std::prev(it);
    previous = prev_it->second;
    return true;
  }

  std::optional<Entry> getPrevious(const Entry& query) const {
    Entry previous_entry;
    if (getPrevious(previous_entry, query)) {
      return previous_entry;
    } else {
      return {};
    }
  }

  std::vector<Entry> toVector() const {
    std::vector<Entry> destination;
    destination.reserve(size());

    std::transform(begin(), end(), std::back_inserter(destination),
                   [](auto entry) { return entry; });
    return destination;
  }

  std::vector<Data> toDataVector() const {
    std::vector<Data> destination;
    destination.reserve(size());

    std::transform(begin(), end(), std::back_inserter(destination),
                   [](auto entry) { return entry.data; });
    return destination;
  }

  FrameIds toFrameIds() const {
    FrameIds destination;
    destination.reserve(size());

    std::transform(begin(), end(), std::back_inserter(destination),
                   [](auto entry) { return entry.frame_id; });
    return destination;
  }

  std::vector<Segment> segments() const {
    std::vector<Segment> segments;

    if (trajectory_.empty()) return segments;

    auto it = trajectory_.begin();
    Derived current_segment;
    // Initalise from entry
    current_segment.insert(it->second);

    auto prev_it = it;
    ++it;

    for (; it != trajectory_.end(); ++it) {
      // Check continuity
      if (it->first == prev_it->first + 1) {
        current_segment.insert(*it);
      } else {
        // Break in trajectory
        segments.push_back(Segment::fromTrajectory(current_segment));
        current_segment.clear();
        current_segment.insert(it->second);
      }
      prev_it = it;
    }

    // Push final segment
    segments.push_back(Segment::fromTrajectory(current_segment));
    return segments;
  }

  EntryIterator begin() { return EntryIterator(trajectory_.begin()); }
  EntryIterator end() { return EntryIterator(trajectory_.end()); }

  EntryIterator find(FrameId frame_id) {
    return EntryIterator(trajectory_.find(frame_id));
  }
  ConstEntryIterator find(FrameId frame_id) const {
    return ConstEntryIterator(trajectory_.find(frame_id));
  }

  EntryIterator upperBound(FrameId frame_id) {
    return EntryIterator(trajectory_.upper_bound(frame_id));
  }
  ConstEntryIterator upperBound(FrameId frame_id) const {
    return ConstEntryIterator(trajectory_.upper_bound(frame_id));
  }

  ConstEntryIterator begin() const {
    return ConstEntryIterator(trajectory_.begin());
  }
  ConstEntryIterator end() const {
    return ConstEntryIterator(trajectory_.end());
  }

  ConstEntryIterator cbegin() const {
    return ConstEntryIterator(trajectory_.cbegin());
  }
  ConstEntryIterator cend() const {
    return ConstEntryIterator(trajectory_.cend());
  }

  friend std::ostream& operator<<(std::ostream& os, const This& trajectory) {
    os << trajectory.trajectoryName();

    if (trajectory.empty()) {
      os << " is empty";
    } else {
      os << " size: " << trajectory.size()
         << " start frame: " << trajectory.minFrame()
         << " end frame: " << trajectory.maxFrame();
    }
    return os;
  }

 protected:
  TrajectoryBase(const TrajectoryImpl& trajectory) : trajectory_(trajectory) {}
  Derived& insert(const Entry& entry) {
    if (entry.timestamp < 0) {
      throw DynosamException("Negative timestamp provided to" +
                             trajectoryName() + " at frame id " +
                             std::to_string(entry.frame_id));
    }

    if (exists(entry.frame_id)) {
      throw TrajectoryEntryAlreadyExists(entry.frame_id);
    }

    trajectory_.insert2(entry.frame_id, entry);
    return derived();
  }

  Derived& insertOrUpdate(const Entry& entry) {
    if (exists(entry.frame_id)) {
      CHECK(update(entry.frame_id, entry.data));
      return derived();
    }
    return insert(entry);
  }

  // for inserting from iterator
  Derived& insert(const std::pair<FrameId, Entry>& entry_pair) {
    CHECK_EQ(entry_pair.first, entry_pair.second.frame_id);
    return this->insert(entry_pair.second);
  }

 private:
  std::string trajectoryName() const {
    return "Trajectory<" + type_name<TData>() + ">";
  }

  TrajectoryImpl trajectory_;
};

template <typename Data>
class Trajectory : public TrajectoryBase<Trajectory<Data>, Data> {};

using PoseTrajectory = Trajectory<gtsam::Pose3>;
using PoseTrajectoryEntry = PoseTrajectory::Entry;

using MotionTrajetory = Trajectory<Motion3ReferenceFrame>;

struct PoseWithMotion {
  gtsam::Pose3 pose;
  Motion3ReferenceFrame motion;
};
using PoseWithMotionTrajectory = Trajectory<PoseWithMotion>;
using PoseWithMotionEntry = PoseWithMotionTrajectory::Entry;

class MultiObjectTrajectories
    : public gtsam::FastMap<ObjectId, PoseWithMotionTrajectory> {
 public:
  using Base = gtsam::FastMap<ObjectId, PoseWithMotionTrajectory>;
  using Base::at;
  using Base::Base;  // all the stl map stuff
  using Base::exists;
  using Base::insert2;

  using Entry = PoseWithMotionTrajectory::Entry;
  using EntryMap = gtsam::FastMap<ObjectId, Entry>;

  MultiObjectTrajectories() = default;

  // this is basically copying everything over and is therefore slow
  // only needed for backwards compatability
  MultiObjectTrajectories(const ObjectPoseMap& poses,
                          const ObjectMotionMap& motion,
                          const FrameIdTimestampMap& times);

  void insert(ObjectId object_id, FrameId frame_id, Timestamp timestamp,
              const PoseWithMotion& data);

  bool hasObject(ObjectId object_id) const;
  size_t numObjects() const;
  ObjectIds objectIds() const;

  bool hasFrame(ObjectId object_id, FrameId frame_id) const;
  EntryMap entriesAtFrame(FrameId frame_id) const;
  MultiObjectTrajectories trajectoriesAtFrame(FrameId frame_id) const;

  // Most recent frame for any object
  FrameId lastFrame() const;
  // Most recent timestamp for any object
  Timestamp lastTimestamp() const;

  // Slow - mostly needed for backwards compatability
  // TODO: actually not needed I think!!
  ObjectPoseMap toObjectPoseMap() const;
  ObjectMotionMap toObjectMotionMap() const;

  friend std::ostream& operator<<(std::ostream& os,
                                  const MultiObjectTrajectories& trajectory) {
    os << "MultiObjectTrajectories: (# objects=" << trajectory.size() << ")\n";
    for (const auto& [object_id, traj] : trajectory) {
      os << "Object " << object_id << " " << traj << "\n";
    }
    return os;
  }

 private:
  // only searches by frame and assumes that timestamps in all entries are
  // correct since multiple objects can exist at one frame we assume all
  // timestamps are correct accross entries
  bool getTemporalLastEntry(FrameId& frame_id, Timestamp& timestamp) const;
};

}  // namespace dyno
