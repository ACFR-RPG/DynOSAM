/*
 *   Copyright (c) 2025 ACFR-RPG, University of Sydney, Jesse Morris
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

#include <type_traits>

#include "dynosam_common/Types.hpp"

namespace dyno {

namespace internal {

template <typename T>
struct EnableBitMaskOperators : std::false_type {};

template <typename E>
constexpr std::enable_if_t<EnableBitMaskOperators<E>::value, E> operator|(
    E lhs, E rhs) {
  using U = std::underlying_type_t<E>;
  return static_cast<E>(static_cast<U>(lhs) | static_cast<U>(rhs));
}

template <typename E>
constexpr std::enable_if_t<EnableBitMaskOperators<E>::value, E> operator&(
    E lhs, E rhs) {
  using U = std::underlying_type_t<E>;
  return static_cast<E>(static_cast<U>(lhs) & static_cast<U>(rhs));
}

template <typename E>
constexpr std::enable_if_t<EnableBitMaskOperators<E>::value, E> operator~(E e) {
  using U = std::underlying_type_t<E>;
  return static_cast<E>(~static_cast<U>(e));
}

}  // namespace internal

// bring into dyno namespace
using internal::operator|;
using internal::operator&;
using internal::operator~;

}  // namespace dyno
