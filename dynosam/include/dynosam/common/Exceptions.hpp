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

#include <functional>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace dyno {

struct DynosamException : public std::runtime_error {
  DynosamException(const std::string& what) : std::runtime_error(what) {}
};

// /**
//  * @brief A stream like object that throws an exception on destruction
//  * Allows for easy input stream operators<< to set exception messages at
//  runtime before
//  * actually throwing the exception. The exception is thrown when the
//  destructor of the object
//  * is called.
//  *
//  * e.g ExceptionStream d<SomeException>...
//  * d << "Error message becuase d < " << 10;
//  *
//  * A ExcpetionStream without a template will not throw an exception at
//  destruction
//  *
//  * The exception must take a const std::string& as argument.
//  *
//  * Needs big three definitions due to std::stringstream which is
//  non-copyable.
//  * Unable to make pointer from Create as this will cause destrector to be
//  called and the excpetion thrown.
//  *
//  *
//  */
// class ExceptionStream {

// public:
//     template<typename Exception>
//     static ExceptionStream Create(const std::string& prefix = "")
//     noexcept(false) {
//       ExceptionStream e;
//       e.throw_exception_ = [&e, &prefix]() {
//         if(std::uncaught_exceptions() == e.uncaught_count_) //unchanged
//           throw Exception(prefix + e.ss_.str());
//       };
//       return e;
//     }

//     static ExceptionStream Create() noexcept {
//       ExceptionStream e;
//       e.throw_exception_ = []() {};
//       return e;
//     }

//     /**
//      * @brief Destroy the object and throw the contained excpetion with
//      message.
//      * It must specify noexcept(false) so that the exception throw can be
//      caught with a try/catch
//      * statement.
//      *
//      * Reference:
//      https://stackoverflow.com/questions/130117/if-you-shouldnt-throw-exceptions-in-a-destructor-how-do-you-handle-errors-in-i
//      *
//      */
//     ~ExceptionStream() noexcept(false);
//     ExceptionStream(const ExceptionStream& other);

//     ExceptionStream& operator=(const ExceptionStream& other);

//     template<typename T>
//     ExceptionStream& operator<<(const T& t){
//         ss_ << t;
//         return *this;
//     }

// private:
//   ExceptionStream() = default;
//   void swap(ExceptionStream& other);

// private:
//     std::stringstream ss_;
//     std::function<void()> throw_exception_;

//     int uncaught_count_ = std::uncaught_exceptions();

// };

// Internal helper that builds and throws the exception immediately
namespace {
// Helper class for streaming exception messages
template <typename ExceptionType>
class ThrowStreamHelper {
 public:
  std::ostringstream& stream() { return stream_; }

  // Convert to exception when thrown
  // this is bad practice but following GLOG example in LogMessage
  // see their destructor for more (better) cpp_exception management
  // Good article on why exceptions in a destructor is bad:
  // https://akrzemi1.wordpress.com/2011/09/21/destructors-that-throw/
  ~ThrowStreamHelper() noexcept(false) { throw ExceptionType(stream_.str()); }

 private:
  std::ostringstream stream_;
};
}  // namespace

template <typename Exception = DynosamException>
inline void checkAndThrow(bool condition) {
  if (!condition) {
    throw Exception("");
  }
}

template <typename Exception = DynosamException>
inline void checkAndThrow(bool condition, const std::string& error_message) {
  if (!condition) {
    throw Exception(error_message);
  }
}

template <typename Exception, typename... EArgs>
inline void checkAndThrow(bool condition, EArgs&&... eargs) {
  if (!condition) {
    throw Exception(std::forward<EArgs>(eargs)...);
  }
}

}  // namespace dyno

// Macro to throw with streaming syntax, auto-wrapped
#define DYNO_THROW_MSG(exception_type) \
  ThrowStreamHelper<exception_type>().stream()
