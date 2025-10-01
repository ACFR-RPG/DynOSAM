/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris
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
#include <variant>

namespace dyno {
namespace internal {

template <typename T>
struct is_variant : std::false_type {};

template <typename... Variants>
struct is_variant<std::variant<Variants...>> : std::true_type {};

// main lookup logic of looking up a type in a list.
// https://www.appsloveworld.com/cplus/100/22/how-do-i-check-if-an-stdvariant-can-hold-a-certain-type
template <typename T, typename... Variants>
struct is_one_of : public std::false_type {};

template <typename T, typename FrontVariant, typename... RestVariants>
struct is_one_of<T, FrontVariant, RestVariants...>
    : public std::conditional<std::is_same<T, FrontVariant>::value,
                              std::true_type,
                              is_one_of<T, RestVariants...>>::type {};

// convenience wrapper for std::variant<>.
template <typename T, typename Variants>
struct is_variant_member : public std::false_type {};

template <typename T, typename... Variants>
struct is_variant_member<T, std::variant<Variants...>>
    : public is_one_of<T, Variants...> {};

}  // namespace internal

template <typename T>
inline constexpr bool is_variant_v = internal::is_variant<T>::value;

template <typename T, typename... Variants>
inline constexpr bool is_variant_member_v =
    internal::is_variant_member<T, Variants...>::value;

}  // namespace dyno
