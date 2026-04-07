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

#include <glog/logging.h>
#include <gtsam/base/FastDefaultAllocator.h>

#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "dynosam_common/Types.hpp"
#include "dynosam_common/utils/Numerical.hpp"  //for hash pair

template <class T1, class T2>
using TemplatedPair = std::pair<T1, T2>;

template <class T1, class T2>
struct std::hash<TemplatedPair<T1, T2>> {
  using TP = TemplatedPair<T1, T2>;

  inline std::size_t operator()(const TP& k) const { return dyno::hashPair(k); }
};

namespace dyno {

/**
 * @brief Internal iterator type allowing iteration over a map directly as if it
 * were a vector. This type satisfies constraints for a filter_iterator_base as
 * well as an std::iterator
 *
 * @tparam MapIterator The internal iterator to use.
 * @tparam MappedType The type we are iterating over.
 */
template <typename MapIterator, typename MappedType>
struct vector_iterator_base {
  using iterator_type = MapIterator;
  using iterator_category = std::forward_iterator_tag;
  using value_type = MappedType;
  using reference = value_type&;
  using const_reference = const value_type&;
  using pointer = value_type*;
  using const_pointer = const value_type*;
  // assume correct type
  using difference_type = std::ptrdiff_t;

  iterator_type it_;
  vector_iterator_base(iterator_type it) : it_(it) {}

  reference operator*() { return it_->second; }
  const_reference operator*() const { return it_->second; }

  pointer operator->() { return &it_->second; }
  const_pointer operator->() const { return &it_->second; }

  bool operator==(const vector_iterator_base& other) const {
    return it_ == other.it_;
  }
  bool operator!=(const vector_iterator_base& other) const {
    return it_ != other.it_;
  }

  bool operator==(const iterator_type& other) const { return it_ == other; }
  bool operator!=(const iterator_type& other) const { return it_ != other; }

  vector_iterator_base& operator++() {
    ++it_;
    return *this;
  }
};

namespace internal {

template <typename Iterator, typename Predicate,
          typename = std::enable_if_t<std::is_base_of_v<
              std::forward_iterator_tag,
              typename std::iterator_traits<Iterator>::iterator_category>>>
class FilterIterator {
 public:
  using iterator_category = std::forward_iterator_tag;
  // using value_type        = typename
  // std::iterator_traits<Iterator>::value_type; using difference_type   =
  // typename std::iterator_traits<Iterator>::difference_type; using pointer =
  // typename std::iterator_traits<Iterator>::pointer; using reference         =
  // typename std::iterator_traits<Iterator>::reference;
  using value_type = typename Iterator::value_type;
  using difference_type = typename Iterator::difference_type;
  using pointer = typename Iterator::pointer;
  using reference = typename Iterator::reference;

  using const_reference = const value_type;
  using const_pointer = const pointer;

  FilterIterator() = default;

  FilterIterator(Iterator current, Iterator end, Predicate* pred)
      : current_(current), end_(end), pred_(pred) {
    satisfy();
  }

  reference operator*() { return *current_; }
  const_reference operator*() const { return *current_; }

  pointer operator->() { return std::addressof(*current_); }
  const_pointer operator->() const { return std::addressof(*current_); }

  FilterIterator& operator++() {
    ++current_;
    satisfy();
    return *this;
  }

  FilterIterator operator++(int) {
    FilterIterator tmp = *this;
    ++(*this);
    return tmp;
  }

  friend bool operator==(const FilterIterator& a, const FilterIterator& b) {
    return a.current_ == b.current_;
  }

  friend bool operator!=(const FilterIterator& a, const FilterIterator& b) {
    return !(a == b);
  }

 private:
  void satisfy() {
    while (current_ != end_ && !std::invoke(*pred_, *current_)) {
      ++current_;
    }
  }

  Iterator current_{};
  Iterator end_{};
  Predicate* pred_{nullptr};
};

/**
 * @brief Requirements and intent:
 *
 * Range:
 *  - Must be a forward-iterable range.
 *  - std::begin(range) and std::end(range) must be valid.
 *  - The iterator returned must model at least ForwardIterator.
 *  - The Range object must outlive the FilterView.
 *
 * Predicate:
 *  - Callable with the range's reference type:
 *        bool pred(reference)
 *  - Must return a value convertible to bool.
 *  - Must not invalidate the underlying range.
 *  - Should be pure (no mutation or internal state changes that
 *    break multi-pass forward iterator semantics).
 *
 * Intention:
 *  FilterView provides a lazy, non-owning filtered view over an
 *  existing range. Elements are skipped during iteration if the
 *  predicate returns false. No elements are copied or stored.
 *
 * @tparam Range
 * @tparam Predicate
 */
template <typename Range, typename Predicate>
class FilterView {
 public:
  using iterator =
      FilterIterator<decltype(std::begin(std::declval<Range&>())), Predicate>;

  using const_iterator =
      FilterIterator<decltype(std::begin(std::declval<const Range&>())),
                     Predicate>;

  template <typename... Args>
  FilterView(Range& r, Args&&... args)
      : range_(r), pred_(std::forward<Args>(args)...) {}

  FilterView(Range& r, Predicate p) : range_(r), pred_(std::move(p)) {}

  iterator begin() {
    return iterator(std::begin(range_), std::end(range_), &pred_);
  }

  iterator end() {
    return iterator(std::end(range_), std::end(range_), &pred_);
  }

  const_iterator begin() const {
    return const_iterator(std::begin(range_), std::end(range_), &pred_);
  }

  const_iterator end() const {
    return const_iterator(std::end(range_), std::end(range_), &pred_);
  }

 private:
  Range& range_;
  Predicate pred_;
};

}  // namespace internal

struct FrameRangeBase {
  FrameId start;
  FrameId end;
  //! indicates that this is the current keyframe range for the latest frame id
  //! and therefore the end value is not valid
  bool is_active = false;

  FrameRangeBase() {}
  FrameRangeBase(FrameId start_, FrameId end_, bool is_active_ = false)
      : start(start_), end(end_), is_active(is_active_) {}

  bool contains(FrameId frame_id) const;
  bool operator<(const FrameRangeBase& rhs) const { return start < rhs.start; }
};

template <typename T>
struct FrameRange : public FrameRangeBase {
  using This = FrameRange<T>;
  DYNO_POINTER_TYPEDEFS(This)

  FrameRange() {}
  FrameRange(FrameId start_, FrameId end_, const T& data_,
             bool is_active_ = false)
      : FrameRangeBase(start_, end_, is_active_), data(data_) {}

  T data;
  std::pair<FrameId, T> dataPair() const { return {start, data}; }
};

template <typename T>
class FrameRangeData {
 public:
  using This = FrameRangeData<T>;
  using FrameRangeT = FrameRange<T>;
  using FrameRangeTVector = std::set<typename FrameRangeT::Ptr>;

  const typename FrameRangeT::ConstPtr find(FrameId frame_id) const {
    typename FrameRangeT::Ptr active_range = getActiveRange();
    if (!active_range) {
      return nullptr;
    }

    // sanity check
    CHECK(active_range->is_active);
    if (active_range->contains(frame_id)) {
      return active_range;
    } else {
      CHECK_GE(ranges.size(), 1u);
      // iterate over ranges
      for (const typename FrameRangeT::Ptr& range : ranges) {
        if (range->contains(frame_id)) {
          return range;
        }
      }
    }
    return nullptr;
  }

  const typename FrameRangeT::ConstPtr startNewActiveRange(FrameId frame_id,
                                                           const T& data) {
    typename FrameRangeT::Ptr old_active_range = getActiveRange();
    auto new_range = std::make_shared<FrameRangeT>();
    new_range->start = frame_id;
    // dont set end (yet) but make active
    new_range->is_active = true;
    new_range->data = data;

    return updateRanges(new_range, old_active_range);
  }

  auto begin() { return ranges.begin(); }
  auto end() { return ranges.end(); }

  auto begin() const { return ranges.begin(); }
  auto end() const { return ranges.end(); }

  typename FrameRangeT::Ptr getActiveRange() const {
    if (!active_range) {
      // no range means no data
      CHECK_EQ(ranges.size(), 0u);
      return nullptr;
    }
    CHECK(active_range->is_active);
    return active_range;
  }

 private:
  typename FrameRangeT::Ptr updateRanges(
      typename FrameRangeT::Ptr new_range,
      typename FrameRangeT::Ptr old_active_range = nullptr) {
    CHECK_NOTNULL(new_range);
    if (old_active_range) {
      // modify existing range so that the end is the start of the next (new
      // range)
      old_active_range->end = new_range->start;
      old_active_range->is_active = false;
    }

    // ranges.push_back(new_range);
    ranges.insert(new_range);
    // set new active range
    active_range = new_range;
    return new_range;
  }

 private:
  FrameRangeTVector ranges;
  typename FrameRangeT::Ptr active_range;
};

template <typename V, typename T>
class MultiFrameRangeData {
 public:
  using This = MultiFrameRangeData<V, T>;
  using FrameRangeT = FrameRange<T>;
  using FrameRangeDataTVector = FrameRangeData<T>;

  MultiFrameRangeData() {}

  const typename FrameRangeT::ConstPtr find(V query, FrameId frame_id) const {
    if (!ranges.exists(query)) {
      return nullptr;
    }

    return ranges.at(query).find(frame_id);
  }
  const typename FrameRangeT::ConstPtr startNewActiveRange(V query,
                                                           FrameId frame_id,
                                                           const T& data) {
    if (!ranges.exists(query)) {
      ranges.insert2(query, FrameRangeDataTVector{});
    }

    return ranges.at(query).startNewActiveRange(frame_id, data);
  }

  bool exists(V query) const { return ranges.exists(query); }

  const FrameRangeDataTVector& at(V query) const {
    if (!ranges.exists(query)) {
      DYNO_THROW_MSG(std::out_of_range) << "Query " << query << " out of range";
    }

    return ranges.at(query);
  }

 private:
  gtsam::FastMap<V, FrameRangeDataTVector> ranges;
};

/**
 * FastUnorderedMap is a thin wrapper around std::unordered that uses the boost
 * fast_pool_allocator instead of the default STL allocator.  This is just a
 * convenience to avoid having lengthy types in the code.  Through timing,
 * we've seen that the fast_pool_allocator can lead to speedups of several
 * percent.
 * @ingroup base
 */
template <typename KEY, typename VALUE>
class FastUnorderedMap
    : public std::unordered_map<KEY, VALUE, std::hash<KEY>, std::equal_to<KEY>,
                                typename gtsam::internal::FastDefaultAllocator<
                                    std::pair<const KEY, VALUE>>::type> {
 public:
  typedef std::unordered_map<KEY, VALUE, std::hash<KEY>, std::equal_to<KEY>,
                             typename gtsam::internal::FastDefaultAllocator<
                                 std::pair<const KEY, VALUE>>::type>
      Base;

  /** Default constructor */
  FastUnorderedMap() {}

  /** Constructor from a range, passes through to base class */
  template <typename INPUTITERATOR>
  explicit FastUnorderedMap(INPUTITERATOR first, INPUTITERATOR last)
      : Base(first, last) {}

  /** Copy constructor from another FastUnorderedMap */
  FastUnorderedMap(const FastUnorderedMap<KEY, VALUE>& x) : Base(x) {}

  /** Copy constructor from the base map class */
  FastUnorderedMap(const Base& x) : Base(x) {}

  /** Move constructor */
  FastUnorderedMap(FastUnorderedMap&& x) noexcept = default;

  /** Copy assignment */
  FastUnorderedMap& operator=(const FastUnorderedMap& x) = default;

  /** Move assignment */
  FastUnorderedMap& operator=(FastUnorderedMap&& x) noexcept = default;

  /** Conversion to a standard STL container */
  operator std::unordered_map<KEY, VALUE>() const {
    return std::unordered_map<KEY, VALUE>(this->begin(), this->end());
  }

  /** Handy 'insert' function for Matlab wrapper */
  bool insert2(const KEY& key, const VALUE& val) {
    return Base::insert(std::make_pair(key, val)).second;
  }

  /** Handy 'exists' function */
  bool exists(const KEY& e) const { return this->find(e) != this->end(); }

 private:
};

/**
 * FastSet is a thin wrapper around std::set that uses the boost
 * fast_pool_allocator instead of the default STL allocator.  This is just a
 * convenience to avoid having lengthy types in the code.  Through timing,
 * we've seen that the fast_pool_allocator can lead to speedups of several %.
 * @ingroup base
 */
template <typename VALUE, typename Compare = std::less<VALUE>>
class FastSet
    : public std::set<
          VALUE, Compare,
          typename gtsam::internal::FastDefaultAllocator<VALUE>::type> {
 public:
  typedef std::set<VALUE, Compare,
                   typename gtsam::internal::FastDefaultAllocator<VALUE>::type>
      Base;

  using Base::Base;  // Inherit the set constructors

  FastSet() = default;  ///< Default constructor

  /** Constructor from a iterable container, passes through to base class */
  template <typename INPUTCONTAINER>
  explicit FastSet(const INPUTCONTAINER& container)
      : Base(container.begin(), container.end()) {}

  /** Copy constructor from another FastSet */
  FastSet(const FastSet<VALUE>& x) : Base(x) {}

  /** Copy constructor from the base set class */
  FastSet(const Base& x) : Base(x) {}

  FastSet& operator=(const FastSet& other) = default;

#ifdef GTSAM_ALLOCATOR_BOOSTPOOL
  /** Copy constructor from a standard STL container */
  FastSet(const std::set<VALUE>& x) {
    // This if statement works around a bug in boost pool allocator and/or
    // STL vector where if the size is zero, the pool allocator will allocate
    // huge amounts of memory.
    if (x.size() > 0) Base::insert(x.begin(), x.end());
  }
#endif

  /** Conversion to a standard STL container */
  operator std::set<VALUE>() const {
    return std::set<VALUE>(this->begin(), this->end());
  }

  /** Handy 'exists' function */
  bool exists(const VALUE& e) const { return this->find(e) != this->end(); }

  /** Check for equality within tolerance to implement Testable */
  bool equals(const FastSet<VALUE>& other, double tol = 1e-9) const {
    typename Base::const_iterator it1 = this->begin(), it2 = other.begin();
    while (it1 != this->end()) {
      if (it2 == other.end() || !traits<VALUE>::Equals(*it2, *it2, tol))
        return false;
      ++it1;
      ++it2;
    }
    return true;
  }

  /** insert another set: handy for MATLAB access */
  void merge(const FastSet& other) { Base::insert(other.begin(), other.end()); }

  const VALUE& front() const { return *this->cbegin(); }
  const VALUE& back() const { return *this->crbegin(); }
  VALUE& front() { return *this->begin(); }
  VALUE& back() { return *this->rbegin(); }

 private:
};

}  // namespace dyno

// allow convenience tuple like getters for FrameRange
namespace std {
template <typename T>
struct tuple_size<dyno::FrameRange<T>> : std::integral_constant<size_t, 2> {};

template <typename T>
struct tuple_element<0, dyno::FrameRange<T>> {
  using type = dyno::FrameId;
};

template <typename T>
struct tuple_element<1, dyno::FrameRange<T>> {
  using type = T;
};
}  // namespace std

namespace dyno {
template <size_t N, typename T>
auto get(const FrameRange<T>& p) {
  if constexpr (N == 0)
    return p.dataPair().first;
  else if constexpr (N == 1)
    return p.dataPair().second;
}
}  // namespace dyno
