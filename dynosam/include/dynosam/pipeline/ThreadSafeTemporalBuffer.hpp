/*
 *   Copyright (c) 2023 Jesse Morris
 *   All rights reserved.
 */

#pragma once

#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <utility>

#include "dynosam_common/Types.hpp"

namespace dyno {

template <typename ValueType,
          typename AllocatorType =
              std::allocator<std::pair<const Timestamp, ValueType> > >
class ThreadsafeTemporalBuffer {
 public:
  typedef std::map<Timestamp, ValueType, std::less<Timestamp>, AllocatorType>
      BufferType;
  using This = ThreadsafeTemporalBuffer<ValueType, AllocatorType>;
  using Type = ValueType;

  // private:
  //   template<typename Iterator>
  //   class InternalIterator {
  //     public:
  //       using iterator = Iterator;
  //       //! naming conventions to match those required by iterator traits
  //       using value_type = typename std::iterator_traits<typename
  //       Iter::pointer>::value_type; using reference_type = typename
  //       std::iterator_traits<typename Iter::pointer>::reference; using
  //       pointer = typename std::iterator_traits<typename
  //       Iter::pointer>::pointer; using difference_type = typename
  //       std::iterator_traits<typename Iter::pointer>::difference_type; using
  //       iterator_category = std::forward_iterator_tag; //i guess? only
  //       forward is defined (++iter) right now

  //       InternalIterator(BufferType& buffer, Iterator it,
  //       std::recursive_mutex* r_mutex)
  //         : buffer_(buffer), it_(it), mutex(CHECK_NOTNULL(r_mutex)) {}

  //     private:
  //       BufferType& buffer_; //! reference to container
  //       mutable Iterator it_;
  //       mutable std::recursive_mutex* mutex
  //   };

  // public:

  // Create buffer of infinite length (buffer_length_seconds = -1)
  ThreadsafeTemporalBuffer();

  // Buffer length in nanoseconds defines after which time old entries get
  // dropped. (buffer_length_seonds == -1: infinite length.). Tested with
  // doubles
  explicit ThreadsafeTemporalBuffer(Timestamp buffer_length_seonds);

  ThreadsafeTemporalBuffer(const ThreadsafeTemporalBuffer& other);

  // TODO:alright write some tests
  ThreadsafeTemporalBuffer& operator=(const ThreadsafeTemporalBuffer& other);

  void addValue(Timestamp timestamp, const ValueType& value);
  void addValue(const Timestamp timestamp, const ValueType& value,
                const bool emit_warning_on_value_overwrite);
  void insert(const ThreadsafeTemporalBuffer& other);

  inline size_t size() const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    return values_.size();
  }

  inline Timestamp bufferLength() const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    return buffer_length_seconds_;
  }

  inline bool empty() const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    return values_.empty();
  }
  void clear() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    values_.clear();
  }

  // Returns false if no value at a given timestamp present.
  bool getValueAtTime(Timestamp timestamp_ns, ValueType* value) const;

  bool deleteValueAtTime(Timestamp timestamp_ns);

  bool getNearestValueToTime(Timestamp timestamp_ns, ValueType* value) const;
  bool getNearestValueToTime(Timestamp timestamp_ns, Timestamp maximum_delta_s,
                             ValueType* value) const;
  bool getNearestValueToTime(Timestamp timestamp, Timestamp maximum_delta_s,
                             ValueType* value,
                             Timestamp* timestamp_at_value_s) const;

  // TODO: retrieve timestamp with these values and add test
  bool getOldestValue(ValueType* value, Timestamp* timestamp_of_value) const;
  bool getNewestValue(ValueType* value, Timestamp* timestamp_of_value) const;
  bool getOldestValue(ValueType* value) const;
  bool getNewestValue(ValueType* value) const;

  // TODO: tests
  Timestamp getOldestTimestamp() const;
  Timestamp getNewestTimestamp() const;

  // These functions return False if there is no valid time.
  bool getValueAtOrBeforeTime(Timestamp timestamp_ns,
                              Timestamp* timestamp_ns_of_value,
                              ValueType* value) const;
  bool getValueAtOrAfterTime(Timestamp timestamp_ns,
                             Timestamp* timestamp_ns_of_value,
                             ValueType* value) const;

  // Get all values between the two specified timestamps excluding the border
  // values.
  // Example: content: 2 3 4 5
  //          getValuesBetweenTimes(2, 5, ...) returns elements at 3, 4.
  // Alternatively, you can ask for the lower bound, such that:
  // Example: content: 2 3 4 5
  //          getValuesBetweenTimes(2, 5, ...) returns elements at 2, 3, 4.
  // by setting the parameter get_lower_bound to true.
  template <typename ValueContainerType>
  bool getValuesBetweenTimes(Timestamp timestamp_lower_s,
                             Timestamp timestamp_higher_s,
                             ValueContainerType* values,
                             const bool get_lower_bound = false) const;

  inline void lockContainer() const { mutex_.lock(); }
  inline void unlockContainer() const { mutex_.unlock(); }

  // The container is exposed so we can iterate over the values in a linear
  // fashion. The container is not locked inside this method so call
  // lockContainer()/unlockContainer() when accessing this.
  const BufferType& buffered_values() const { return values_; }

  inline bool operator==(const ThreadsafeTemporalBuffer& other) const {
    return values_ == other.values_ &&
           buffer_length_seconds_ == other.buffer_length_seconds_;
  }

 protected:
  // Remove items that are older than the buffer length.
  void removeOutdatedItems();

  bool getIteratorAtTimeOrEarlier(
      Timestamp timestamp,
      typename BufferType::const_iterator* it_lower_bound) const;

  BufferType values_;
  Timestamp buffer_length_seconds_;
  mutable std::recursive_mutex mutex_;
};

}  // namespace dyno

#include "dynosam/pipeline/ThreadSafeTemporalBuffer-inl.hpp"
