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

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <exception>
#include <iterator>
#include <vector>

#include "dynosam_common/StructuredContainers.hpp"
#include "dynosam_common/Types.hpp"

using namespace dyno;

using IntFilterIterator = internal::filter_iterator<std::vector<int>>;

/// @brief make definition for testing
template <>
struct std::iterator_traits<IntFilterIterator>
    : public dyno::internal::filter_iterator_detail<
          IntFilterIterator::pointer> {};

// //iterator is the pointer type... this is weird naming but is what the c++
// standard does template<typename _Iterator> struct _filter_iterator_detail {
//     //! naming conventions to match those required by iterator traits
//     using value_type = typename std::iterator_traits<_Iterator>::value_type;
//     using reference = typename std::iterator_traits<_Iterator>::reference;
//     using pointer = typename std::iterator_traits<_Iterator>::pointer;
//     using difference_type = typename
//     std::iterator_traits<_Iterator>::difference_type; using iterator_category
//     = std::forward_iterator_tag; //i guess? only forward is defined (++iter)
//     right now
// };

// //in this case iter is the actual iterator (so _Container::iterator or
// _container::const_iterator) template<typename _Iter, typename _Container>
// struct _filter_iterator : public _filter_iterator_detail<typename
// _Iter::pointer> {

//     using BaseDetail = _filter_iterator_detail<typename _Iter::pointer>;
//     using iterator = _Iter;
//     using typename BaseDetail::value_type;
//     using typename BaseDetail::reference;
//     using typename BaseDetail::pointer;
//     using typename BaseDetail::difference_type;
//     using typename BaseDetail::iterator_category;

//     _filter_iterator(_Container& container,_Iter it) : container_(container),
//     it_(it) {}

//     _Container& container_;
//     iterator it_;

//     reference operator*() { return *it_; }
//     reference operator->() { return *it_; }

//     bool operator==(const _filter_iterator& other) const {
//         return it_ == other.it_;
//     }
//     bool operator!=(const _filter_iterator& other) const { return it_ !=
//     other.it_; }

//     bool operator==(const iterator& other) const {
//         return it_ == other;
//     }
//     bool operator!=(const iterator& other) const { return it_ != other; }

//     _filter_iterator& operator++() {
//         // do {
//         //     ++it_;
//         // }
//         // while(is_invalid());
//         ++it_;
//         return *this;
//     }

//     //allows the iterator to be used as a enhanced for loop
//     _filter_iterator begin() { return _filter_iterator(container_,
//     container_.begin()); } _filter_iterator end() { return
//     _filter_iterator(container_, container_.end()); }

//     const _filter_iterator begin() const { return
//     _filter_iterator(container_, container_.begin()); } const
//     _filter_iterator end() const { return _filter_iterator(container_,
//     container_.end()); }

// };

// TEST(FilterIterator, basic) {

//     using T = std::shared_ptr<int>;

//     std::vector<T> v;
//     _filter_iterator<std::vector<T>::iterator, std::vector<T>> fi(v,
//     v.begin());

//     for(std::shared_ptr<int> i : fi) {}

//     const _filter_iterator<std::vector<T>::const_iterator, std::vector<T>>
//     fi_c(v, v.cbegin()); for(std::shared_ptr<int> i : fi_c) {}
// }

TEST(FilterIterator, testNormalIteration) {
  std::vector<int> v = {1, 2, 3, 4, 5};
  internal::filter_iterator<std::vector<int>> v_iter(
      v, [](const int&) -> bool { return true; });

  EXPECT_EQ(*v_iter, 1);
  ++v_iter;
  EXPECT_EQ(*v_iter, 2);
  ++v_iter;
  EXPECT_EQ(*v_iter, 3);
  ++v_iter;
  EXPECT_EQ(*v_iter, 4);
  ++v_iter;
  EXPECT_EQ(*v_iter, 5);
  ++v_iter;
  EXPECT_EQ(v_iter, v.end());
}

TEST(FilterIterator, testConditionalIteratorWithValidStartingIndex) {
  // start with valid element at v(0)
  std::vector<int> v = {2, 3, 4, 5};
  // true on even numbers
  internal::filter_iterator<std::vector<int>> v_iter(
      v, [](const int& v) -> bool { return v % 2 == 0; });

  EXPECT_EQ(*v_iter, 2);
  ++v_iter;
  EXPECT_EQ(*v_iter, 4);
  ++v_iter;
  EXPECT_EQ(v_iter, v.end());
}

TEST(FilterIterator, testConditionalIteratorAsLoop) {
  // start with valid element at v(0)
  std::vector<int> v = {2, 3, 4, 5};
  // true on even numbers
  internal::filter_iterator<std::vector<int>> v_iter(
      v, [](const int& v) -> bool { return v % 2 == 0; });

  int index = 0;
  for (const int& i : v_iter) {
    if (index == 0) {
      EXPECT_EQ(i, 2);
    } else if (index == 1) {
      EXPECT_EQ(i, 4);
    } else {
      FAIL() << "Should not get here";
    }

    index++;
  }

  EXPECT_EQ(index, 2);  // 2 iterations only!!!
}

TEST(FilterIterator, testConditionalIteratorWithInvalidStartingIndex) {
  // start with valid element at v(0)
  std::vector<int> v = {1, 2, 3, 4, 5, 6};
  // true on even numbers
  internal::filter_iterator<std::vector<int>> v_iter(
      v, [](const int& v) -> bool { return v % 2 == 0; });

  EXPECT_EQ(*v_iter, 2);
  ++v_iter;
  EXPECT_EQ(*v_iter, 4);
  ++v_iter;
  EXPECT_EQ(*v_iter, 6);
  EXPECT_NE(v_iter, v.end());
  ++v_iter;
  EXPECT_EQ(v_iter, v.end());
}

TEST(FilterIterator, testStdDistance) {
  {
    std::vector<int> v = {1, 2, 3, 4, 5, 6};
    // true on even numbers
    internal::filter_iterator<std::vector<int>> v_iter(
        v, [](const int& v) -> bool { return v % 2 == 0; });
    EXPECT_EQ(std::distance(v_iter.begin(), v_iter.end()), 3);
  }

  {
    // start with valid element at v(0)
    std::vector<int> v = {1, 2, 3};
    // true on even numbers
    internal::filter_iterator<std::vector<int>> v_iter(
        v, [](const int& v) -> bool { return v % 2 == 0; });
    EXPECT_EQ(std::distance(v_iter.begin(), v_iter.end()), 1);
  }

  {
    // start with valid element at v(0)
    std::vector<int> v = {2, 2, 2};
    // true on even numbers
    internal::filter_iterator<std::vector<int>> v_iter(
        v, [](const int& v) -> bool { return v % 2 == 0; });
    EXPECT_EQ(std::distance(v_iter.begin(), v_iter.end()), 3);
  }
}
