/*
 *   Copyright (c) 2023 Jesse Morris
 *   All rights reserved.
 */

#pragma once

#include <glog/logging.h>

#include <algorithm>
#include <atomic>
#include <limits>
#include <map>
#include <mutex>

#include "dynosam_common/Types.hpp"

namespace dyno {
template <typename ValueType, typename AllocatorType>
ThreadsafeTemporalBuffer<ValueType, AllocatorType>::ThreadsafeTemporalBuffer()
    : buffer_length_seconds_(-1) {}

template <typename ValueType, typename AllocatorType>
ThreadsafeTemporalBuffer<ValueType, AllocatorType>::ThreadsafeTemporalBuffer(
    Timestamp buffer_length_seconds)
    : buffer_length_seconds_(buffer_length_seconds) {}

template <typename ValueType, typename AllocatorType>
ThreadsafeTemporalBuffer<ValueType, AllocatorType>::ThreadsafeTemporalBuffer(
    const ThreadsafeTemporalBuffer<ValueType, AllocatorType>& other) {
  // Lock both mutexes without deadlock.
  std::lock(mutex_, other.mutex_);

  values_ = other.values_;
  buffer_length_seconds_ = other.buffer_length_seconds_;

  mutex_.unlock();
  other.mutex_.unlock();
}

template <typename ValueType, typename AllocatorType>
ThreadsafeTemporalBuffer<ValueType, AllocatorType>&
ThreadsafeTemporalBuffer<ValueType, AllocatorType>::operator=(
    const ThreadsafeTemporalBuffer<ValueType, AllocatorType>& other) {
  // Guard self assignment
  if (this == &other) return *this;

  // Lock both mutexes without deadlock.
  std::lock(mutex_, other.mutex_);
  // if the values contained are pointers this will not transfter ownship of the
  // pointers and the lvalued buffer will not share a mutex with the rvalued
  // buffer. Hence, the values could be modofied by two different queues in a
  // non thread safe manner the user should have to manage this. This is the
  // same problem as in the custom copy constructor
  this->values_ = other.values_;
  this->buffer_length_seconds_ = other.buffer_length_seconds_;

  mutex_.unlock();
  other.mutex_.unlock();

  return *this;
}

template <typename ValueType, typename AllocatorType>
void ThreadsafeTemporalBuffer<ValueType, AllocatorType>::addValue(
    const Timestamp timestamp, const ValueType& value) {
  constexpr bool kEmitWarningOnValueOverwrite = false;
  addValue(timestamp, value, kEmitWarningOnValueOverwrite);
}

template <typename ValueType, typename AllocatorType>
void ThreadsafeTemporalBuffer<ValueType, AllocatorType>::addValue(
    const Timestamp timestamp, const ValueType& value,
    const bool emit_warning_on_value_overwrite) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  const bool value_overwritten = values_.emplace(timestamp, value).second;
  LOG_IF(WARNING, value_overwritten && emit_warning_on_value_overwrite)
      << "A value in temporal buffer at time " << timestamp
      << " already exists!";
  removeOutdatedItems();
}

template <typename ValueType, typename AllocatorType>
bool ThreadsafeTemporalBuffer<ValueType, AllocatorType>::deleteValueAtTime(
    Timestamp timestamp_ns) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return values_.erase(timestamp_ns) > 0u;
}

template <typename ValueType, typename AllocatorType>
bool ThreadsafeTemporalBuffer<ValueType, AllocatorType>::getOldestValue(
    ValueType* value, Timestamp* timestamp_of_value) const {
  CHECK_NOTNULL(value);
  CHECK_NOTNULL(timestamp_of_value);
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (empty()) {
    return false;
  }
  *value = values_.begin()->second;
  *timestamp_of_value = values_.begin()->first;

  return true;
}

template <typename ValueType, typename AllocatorType>
bool ThreadsafeTemporalBuffer<ValueType, AllocatorType>::getNewestValue(
    ValueType* value, Timestamp* timestamp_of_value) const {
  CHECK_NOTNULL(value);
  CHECK_NOTNULL(timestamp_of_value);
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (empty()) {
    return false;
  }
  *value = values_.rbegin()->second;
  *timestamp_of_value = values_.rbegin()->first;
  return true;
}

template <typename ValueType, typename AllocatorType>
bool ThreadsafeTemporalBuffer<ValueType, AllocatorType>::getOldestValue(
    ValueType* value) const {
  Timestamp t;
  return getOldestValue(value, &t);
}

template <typename ValueType, typename AllocatorType>
bool ThreadsafeTemporalBuffer<ValueType, AllocatorType>::getNewestValue(
    ValueType* value) const {
  Timestamp t;
  return getNewestValue(value, &t);
}

template <typename ValueType, typename AllocatorType>
Timestamp
ThreadsafeTemporalBuffer<ValueType, AllocatorType>::getOldestTimestamp() const {
  ValueType value;
  Timestamp t;
  CHECK(getOldestValue(&value, &t));
  (void)value;
  return t;
}

template <typename ValueType, typename AllocatorType>
Timestamp
ThreadsafeTemporalBuffer<ValueType, AllocatorType>::getNewestTimestamp() const {
  ValueType value;
  Timestamp t;
  CHECK(getNewestValue(&value, &t));
  (void)value;
  return t;
}

template <typename ValueType, typename AllocatorType>
bool ThreadsafeTemporalBuffer<ValueType, AllocatorType>::getValueAtTime(
    Timestamp timestamp, ValueType* value) const {
  CHECK_NOTNULL(value);
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  typename BufferType::const_iterator it = values_.find(timestamp);
  if (it != values_.end()) {
    *value = it->second;
    return true;
  }
  return false;
}

template <typename ValueType, typename AllocatorType>
bool ThreadsafeTemporalBuffer<ValueType, AllocatorType>::getNearestValueToTime(
    Timestamp timestamp, ValueType* value) const {
  CHECK_NOTNULL(value);
  return getNearestValueToTime(timestamp, std::numeric_limits<Timestamp>::max(),
                               value);
}

template <typename ValueType, typename AllocatorType>
bool ThreadsafeTemporalBuffer<ValueType, AllocatorType>::getValueAtOrBeforeTime(
    Timestamp timestamp, Timestamp* timestamp_of_value,
    ValueType* value) const {
  CHECK_NOTNULL(timestamp_of_value);
  CHECK_NOTNULL(value);

  std::lock_guard<std::recursive_mutex> lock(mutex_);
  typename BufferType::const_iterator it_lower_bound;
  const bool has_exact_match =
      getIteratorAtTimeOrEarlier(timestamp, &it_lower_bound);

  if (!has_exact_match) {
    // has_exact_match could be also false if the buffer is empty, check that.
    // Which would make it_lower_bound a nullptr, and raise an excetption when
    // being dereferenced later.
    if (empty() || it_lower_bound == values_.begin()) {
      return false;
    }
    --it_lower_bound;
  }

  *timestamp_of_value = it_lower_bound->first;
  *value = it_lower_bound->second;

  CHECK_LE(*timestamp_of_value, timestamp);
  return true;
}

template <typename ValueType, typename AllocatorType>
bool ThreadsafeTemporalBuffer<ValueType, AllocatorType>::getValueAtOrAfterTime(
    Timestamp timestamp, Timestamp* timestamp_of_value,
    ValueType* value) const {
  CHECK_NOTNULL(timestamp_of_value);
  CHECK_NOTNULL(value);

  std::lock_guard<std::recursive_mutex> lock(mutex_);
  typename BufferType::const_iterator it_lower_bound;
  bool has_exact_match = getIteratorAtTimeOrEarlier(timestamp, &it_lower_bound);

  if (!has_exact_match) {
    // has_exact_match could be also false if the buffer is empty, check that.
    // Which would make it_lower_bound a nullptr, and raise an excetption when
    // being dereferenced later.
    if (empty() || it_lower_bound == values_.end()) {
      return false;
    }
  }
  *timestamp_of_value = it_lower_bound->first;
  *value = it_lower_bound->second;

  CHECK_GE(*timestamp_of_value, timestamp);
  return true;
}

template <typename ValueType, typename AllocatorType>
bool ThreadsafeTemporalBuffer<ValueType, AllocatorType>::
    getIteratorAtTimeOrEarlier(
        Timestamp timestamp,
        typename BufferType::const_iterator* it_lower_bound) const {
  CHECK_NOTNULL(it_lower_bound);
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  // Return false if no values in buffer.
  if (empty()) {
    return false;
  }

  // Returns first element with a time that compares not less to timestamp.
  *it_lower_bound = values_.lower_bound(timestamp);

  if (*it_lower_bound != values_.end() &&
      (*it_lower_bound)->first == timestamp) {
    // It's an exact match.
    return true;
  }
  return false;
}

template <typename ValueType, typename AllocatorType>
template <typename ValueContainerType>
bool ThreadsafeTemporalBuffer<ValueType, AllocatorType>::getValuesBetweenTimes(
    Timestamp timestamp_lower_s, Timestamp timestamp_higher_s,
    ValueContainerType* values, const bool get_lower_bound) const {
  CHECK_NOTNULL(values)->clear();
  CHECK_GT(timestamp_higher_s, timestamp_lower_s);
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  // Early exit if there are too few items.
  if (empty()) {
    return false;
  }

  const Timestamp oldest_timestamp = values_.begin()->first;
  const Timestamp latest_timestamp = values_.rbegin()->first;

  // TODO I don't see why only 100% overlapping query intervals are accepted.
  // Shouldn't it be oldest > higher || latest < lower ?
  // Or at least accept as well when timestamp_lower_s is smaller than
  // oldest_timestamp, such that we only return false if we are asking for
  // future msgs?
  if (oldest_timestamp > timestamp_lower_s ||
      timestamp_higher_s > latest_timestamp) {
    return false;
  }

  typename BufferType::const_iterator it =
      values_.lower_bound(timestamp_lower_s);
  for (; it != values_.end() && it->first < timestamp_higher_s; ++it) {
    CHECK(it != values_.end());

    // lower_bound includes the border so we need to skip them when there are
    // perfect matches, except if the user explicitly asked for it.
    if (it->first == timestamp_lower_s && !get_lower_bound) {
      continue;
    }
    values->emplace_back(it->second);
  }
  return true;
}

template <typename ValueType, typename AllocatorType>
bool ThreadsafeTemporalBuffer<ValueType, AllocatorType>::getNearestValueToTime(
    Timestamp timestamp, Timestamp maximum_delta_ns, ValueType* value,
    Timestamp* timestamp_at_value_s) const {
  CHECK_NOTNULL(timestamp_at_value_s);
  CHECK_NOTNULL(value);
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  if (empty()) {
    return false;
  }

  const typename BufferType::const_iterator it_lower =
      values_.lower_bound(timestamp);

  // Verify if we got an exact match.
  if (it_lower != values_.end() && it_lower->first == timestamp) {
    *value = it_lower->second;
    *timestamp_at_value_s = it_lower->first;
    return true;
  }

  // No exact match found so we need to examine lower and upper
  // bound on the timestamp.
  const typename BufferType::const_iterator it_upper =
      values_.upper_bound(timestamp);

  // If the lower bound points out of the array, we have to return the last
  // element.
  if (it_lower == values_.end()) {
    typename BufferType::const_iterator it_last = std::prev(values_.end());
    Timestamp delta_ns = std::abs(it_last->first - timestamp);
    if (delta_ns <= maximum_delta_ns) {
      *value = it_last->second;
      *timestamp_at_value_s = it_last->first;
      return true;
    } else {
      return false;
    }
  }

  // If the lower bound points to begin() and no exact match was found, we
  // have to return the first element.
  if (it_lower == values_.begin()) {
    typename BufferType::const_iterator it_first = values_.begin();
    Timestamp delta_ns = std::abs(it_first->first - timestamp);
    if (delta_ns <= maximum_delta_ns) {
      *value = it_first->second;
      *timestamp_at_value_s = it_first->first;
      return true;
    } else {
      return false;
    }
  }

  // Both iterators are within range so need to find out which of them is
  // closer to the timestamp.
  typename BufferType::const_iterator it_before = std::prev(it_lower);
  Timestamp delta_before_ns = std::abs(it_before->first - timestamp);
  Timestamp delta_after_ns = std::abs(it_upper->first - timestamp);
  if (delta_before_ns < delta_after_ns) {
    if (delta_before_ns <= maximum_delta_ns) {
      *value = it_before->second;
      *timestamp_at_value_s = it_before->first;
      return true;
    }
  } else {
    if (delta_after_ns <= maximum_delta_ns) {
      *value = it_upper->second;
      *timestamp_at_value_s = it_upper->first;
      return true;
    }
  }

  return false;
}

template <typename ValueType, typename AllocatorType>
bool ThreadsafeTemporalBuffer<ValueType, AllocatorType>::getNearestValueToTime(
    Timestamp timestamp, Timestamp maximum_delta_ns, ValueType* value) const {
  Timestamp timestamp_at_value_s;
  return getNearestValueToTime(timestamp, maximum_delta_ns, value,
                               &timestamp_at_value_s);
}

template <typename ValueType, typename AllocatorType>
void ThreadsafeTemporalBuffer<ValueType, AllocatorType>::removeOutdatedItems() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (empty() || buffer_length_seconds_ <= 0) {
    return;
  }

  const Timestamp newest_timestamp_s = values_.rbegin()->first;
  const Timestamp buffer_threshold_s =
      newest_timestamp_s - buffer_length_seconds_;

  if (values_.begin()->first < buffer_threshold_s) {
    typename BufferType::const_iterator it =
        values_.lower_bound(buffer_threshold_s);

    CHECK(it != values_.end());
    CHECK(it != values_.begin());
    values_.erase(values_.begin(), it);
  }
}

template <typename ValueType, typename AllocatorType>
void ThreadsafeTemporalBuffer<ValueType, AllocatorType>::insert(
    const ThreadsafeTemporalBuffer& other) {
  values_.insert(other.buffered_values().begin(),
                 other.buffered_values().end());
}

}  // namespace dyno
