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

#include <cstddef>
#include <iostream>
#include <tuple>
#include <type_traits>
#include <utility>

namespace dyno {

// template <class TupType, size_t... I>
// void print(const TupType& _tup, std::index_sequence<I...>)
// {
//   std::cout << "(";
//   (..., (std::cout << (I == 0 ? "" : ", ") << std::get<I>(_tup)));
//   std::cout << ")\n";
// }

// template <class... T>
// void print(const std::tuple<T...>& _tup)
// {
//   print(_tup, std::make_index_sequence<sizeof...(T)>());
// }

// https://stackoverflow.com/questions/687490/how-do-i-expand-a-tuple-into-variadic-template-functions-arguments
namespace internal {
template <std::size_t N, typename... Ts>
using select_nth = std::tuple_element_t<N, std::tuple<Ts...>>;

template <class Function, std::size_t... Is>
auto index_apply_impl(Function f, std::index_sequence<Is...>) {
  return f(std::integral_constant<std::size_t, Is>{}...);
}

template <std::size_t N, class Function>
auto index_apply(Function f) {
  return index_apply_impl(f, std::make_index_sequence<N>{});
}

template <class Function, std::size_t... Is>
void for_each_apply_impl(Function f, std::index_sequence<Is...>) {
  (void)std::initializer_list<int>{
      (f(std::integral_constant<std::size_t, Is>{}), void(), 0)...};
}

template <std::size_t N, class Function>
void for_each_apply(Function f) {
  for_each_apply_impl(f, std::make_index_sequence<N>{});
}

template <std::size_t N, class Function>
void select_apply(std::size_t i, Function f) {
  for_each_apply<N>([&](auto&& Is) {
    if (Is == i) f(std::forward<decltype(Is)>(Is));
  });
}

template <size_t N>
struct Apply {
  template <typename F, typename T, typename... A>
  static inline auto apply(F&& f, T&& t, A&&... a) {
    return Apply<N - 1>::apply(::std::forward<F>(f), ::std::forward<T>(t),
                               ::std::get<N - 1>(::std::forward<T>(t)),
                               ::std::forward<A>(a)...);
  }
};

template <>
struct Apply<0> {
  template <typename F, typename T, typename... A>
  static inline auto apply(F&& f, T&&, A&&... a) {
    return ::std::forward<F>(f)(::std::forward<A>(a)...);
  }
};

template <typename F, typename T>
inline auto apply(F&& f, T&& t) {
  return Apply<::std::tuple_size<::std::decay_t<T>>::value>::apply(
      ::std::forward<F>(f), ::std::forward<T>(t));
}

using std::forward;  // You can change this if you like unreadable code or care
                     // hugely about namespace pollution.

template <size_t N>
struct ApplyMember {
  template <typename C, typename F, typename T, typename... A>
  static inline auto apply(C&& c, F&& f, T&& t, A&&... a)
      -> decltype(ApplyMember<N - 1>::apply(forward<C>(c), forward<F>(f),
                                            forward<T>(t),
                                            std::get<N - 1>(forward<T>(t)),
                                            forward<A>(a)...)) {
    return ApplyMember<N - 1>::apply(
        forward<C>(c), forward<F>(f), forward<T>(t),
        std::get<N - 1>(forward<T>(t)), forward<A>(a)...);
  }
};

template <>
struct ApplyMember<0> {
  template <typename C, typename F, typename T, typename... A>
  static inline auto apply(C&& c, F&& f, T&&, A&&... a)
      -> decltype((forward<C>(c)->*forward<F>(f))(forward<A>(a)...)) {
    return (forward<C>(c)->*forward<F>(f))(forward<A>(a)...);
  }
};

// C is the class, F is the member function, T is the tuple.
template <typename C, typename F, typename T>
inline auto apply(C&& c, F&& f, T&& t)
    -> decltype(ApplyMember<std::tuple_size<typename std::decay<T>::type>::
                                value>::apply(forward<C>(c), forward<F>(f),
                                              forward<T>(t))) {
  return ApplyMember<std::tuple_size<typename std::decay<T>::type>::value>::
      apply(forward<C>(c), forward<F>(f), forward<T>(t));
}

template <typename U, typename... T>
constexpr bool contains(std::tuple<T...>) {
  return (std::is_same_v<U, T> || ...);
}

template <typename U, typename Tuple>
constexpr inline bool tuple_contains_type = contains<U>(std::declval<Tuple>());

// Finding a type in a tuple - given a tuple, find the index of a specific type
// https://devblogs.microsoft.com/oldnewthing/20200629-00/?p=103910

template <typename T, typename Tuple>
struct tuple_element_index_helper;

template <typename T>
struct tuple_element_index_helper<T, std::tuple<>> {
  static constexpr std::size_t value = 0;
};

template <typename T, typename... Rest>
struct tuple_element_index_helper<T, std::tuple<T, Rest...>> {
  static constexpr std::size_t value = 0;
  using RestTuple = std::tuple<Rest...>;
  static_assert(tuple_element_index_helper<T, RestTuple>::value ==
                    std::tuple_size_v<RestTuple>,
                "type appears more than once in tuple");
};

template <typename T, typename First, typename... Rest>
struct tuple_element_index_helper<T, std::tuple<First, Rest...>> {
  using RestTuple = std::tuple<Rest...>;
  static constexpr std::size_t value =
      1 + tuple_element_index_helper<T, RestTuple>::value;
};

template <typename T, typename Tuple>
struct tuple_element_index {
  static constexpr std::size_t value =
      tuple_element_index_helper<T, Tuple>::value;
  static_assert(value < std::tuple_size_v<Tuple>,
                "type does not appear in tuple");
};

/**
 * @brief Actual templated class to determine the index of a type in a tuple
 * It asks the helper to do the work and validates that the resulting value is
 * less than the size of the tuple, meaning that the type was found. If not,
 * then we complain with a compile-time assertion.
 *
 * @tparam T
 * @tparam Tuple
 */
template <typename T, typename Tuple>
inline constexpr std::size_t tuple_element_index_v =
    tuple_element_index<T, Tuple>::value;

}  // namespace internal
}  // namespace dyno
