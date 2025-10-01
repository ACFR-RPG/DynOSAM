/*
 *   Copyright (c) 2023 Jesse Morris (jesse.morris@sydney.edu.au)
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

#include <memory>
#include <optional>

// These macros were inspired mainly on Maplab's macros
// https://github.com/ethz-asl/maplab

#define DYNO_POINTER_TYPEDEFS(TypeName)                    \
  typedef std::shared_ptr<TypeName> Ptr;                   \
  typedef std::shared_ptr<const TypeName> ConstPtr;        \
  typedef std::unique_ptr<TypeName> UniquePtr;             \
  typedef std::unique_ptr<const TypeName> ConstUniquePtr;  \
  typedef std::weak_ptr<TypeName> WeakPtr;                 \
  typedef std::weak_ptr<const TypeName> WeakConstPtr;      \
  typedef std::optional<TypeName> Optional;                \
  typedef std::optional<const TypeName> ConstOptional;     \
  typedef std::optional<const TypeName&> ConstRefOptional; \
  typedef std::optional<TypeName&> RefOptional;

#define DYNO_DELETE_COPY_CONSTRUCTORS(TypeName) \
  TypeName(const TypeName&) = delete;           \
  void operator=(const TypeName&) = delete;
