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

#include <memory>

namespace dyno {

// Safely downcast and deal with errors. This works on raw types (not pointers)
// NOTE: this will create a copy
template <class Base, class Derived>
inline Derived safeCast(const Base& base) {
  try {
    return dynamic_cast<const Derived&>(base);
  } catch (const std::bad_cast& e) {
    LOG(ERROR) << "Seems that you are casting an object that is not "
                  "the one you expected!";
    LOG(FATAL) << e.what();
  } catch (...) {
    LOG(FATAL) << "Exception caught when dynamic casting.";
  }
}

// Safely downcast shared pointers. Will not create copies of underlying data,
// but returns a new pointer.
template <typename Base, typename Derived>
inline std::shared_ptr<Derived> safeCast(std::shared_ptr<Base> base_ptr) {
  CHECK(base_ptr);
  try {
    return std::dynamic_pointer_cast<Derived>(base_ptr);
  } catch (const std::bad_cast& e) {
    LOG(ERROR) << "Seems that you are casting an object that is not "
                  "the one you expected!";
    LOG(FATAL) << e.what();
  } catch (...) {
    LOG(FATAL) << "Exception caught when dynamic casting.";
  }
}

// new pointer
template <typename Base, typename Derived>
inline std::shared_ptr<const Derived> safeCast(
    std::shared_ptr<const Base> base_ptr) {
  CHECK(base_ptr);
  std::shared_ptr<Base> base = std::const_pointer_cast<Base>(base_ptr);
  return safeCast<Base, Derived>(base);
}

// Safely downcast unique pointers.
// NOTE: pass by rvalue because this prevents the copy of the pointer and
// is faster. All others don't transfer ownership (aren't using move types)
// and so don't need to pass by rvalue.
template <typename Base, typename Derived>
inline std::unique_ptr<Derived> safeCast(std::unique_ptr<Base>&& base_ptr) {
  std::unique_ptr<Derived> derived_ptr;
  Derived* tmp = nullptr;
  try {
    tmp = dynamic_cast<Derived*>(base_ptr.get());
  } catch (const std::bad_cast& e) {
    LOG(ERROR) << "Seems that you are casting an object that is not "
                  "the one you expected!";
    LOG(FATAL) << e.what();
  } catch (...) {
    LOG(FATAL) << "Exception caught when dynamic casting.";
  }
  if (!tmp) return nullptr;
  base_ptr.release();
  derived_ptr.reset(tmp);
  return derived_ptr;
}

}  // namespace dyno
