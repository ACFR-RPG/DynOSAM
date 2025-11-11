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

#include <glog/logging.h>

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <tuple>
#include <typeinfo>
#include <vector>

#include "dynosam_common/Exceptions.hpp"
#include "dynosam_common/utils/Macros.hpp"

namespace dyno {

struct InvalidCsvHeaderException : public DynosamException {
  InvalidCsvHeaderException(const std::string& what) : DynosamException(what) {}
};

// cannot be constructed with num args < 1
// TODO: missing a , between header definitions e.g "h1" "h2" and not "h1","h2"
// works and gives weird behavioru
class CsvHeader : public std::vector<std::string> {
 public:
  template <typename... HeaderArgs,
            typename = std::enable_if_t<
                (std::is_convertible<HeaderArgs, std::string>::value && ...)>>
  CsvHeader(const HeaderArgs&... args) {
    std::vector<std::string>* result = this;
    std::apply(
        [result](const auto&... elems) {
          result->reserve(sizeof...(elems));
          (result->push_back(std::string(elems)), ...);
        },
        std::forward_as_tuple(args...));
  }

  std::string toString(const std::string& seperator = " ") const {
    std::stringstream ss;
    size_t i = 0;
    for (; i < size() - 1u; i++) {
      ss << this->at(i);
      ss << seperator;
    }
    ss << this->at(i);
    return ss.str();
  }

  /**
   * @brief Finds the index of a column via its header name. If the query header
   * is not in the CsvHeader, -1 is returned.
   *
   * e.g. if a header is defined CsvHeader header("size", "length", "width")
   * header.index("size") == 0, header.index("length") == 1 and
   * header.index("SLAM") == -1
   *
   * @param column_header
   * @return int
   */
  int index(const std::string& column_header) const {
    auto it = std::find(this->begin(), this->end(), column_header);
    if (it != this->end()) {
      return std::distance(this->begin(), it);
    } else {
      return -1;  // Element not found
    }
  }
};

// csv classes modified from // Modified from
// https://stackoverflow.com/questions/1120140/how-can-i-read-and-parse-csv-files-in-c
class CsvReader {
 public:
  DYNO_POINTER_TYPEDEFS(CsvReader)

  static const char DefaultDelimiter = ',';

  /**
   * @brief Class containg a single row of a csv file after parsing.
   *
   * Each column can be queried via the column index
   * TODO: no checks that rows match the header length, all rows have the same
   * length...
   *
   */
  class Row {
   public:
    Row() : delimiter_(CsvReader::DefaultDelimiter) {}
    Row(const CsvHeader& header)
        : header_(header), delimiter_(CsvReader::DefaultDelimiter) {}
    Row(const std::string& line, const CsvHeader& header, const char delimiter);

    // TODO: should specalise instead if T == std::string as we can just return
    // the result of
    //  operator[], very inefficient to copy into string stream and then back
    //  out
    template <typename T = std::string>
    T at(std::size_t index) const {
      if (index >= this->size()) {
        throw std::out_of_range(std::to_string(index) + " >= size (" +
                                std::to_string(this->size()) + ")");
      }
      std::string value = this->operator[](index);

      std::stringstream ss;
      ss << value;

      T out_value;
      ss >> out_value;
      return out_value;
    }

    Row& operator<<(std::istream& str) {
      auto row = CsvReader::Row::FromStream(str, header(), delimieter());
      std::swap(*this, row);
      return *this;
    }

    std::string operator[](std::size_t index) const;

    inline std::size_t size() const {
      if (pos_data_.empty()) return 0;
      return pos_data_.size() - 1;
    }

    inline bool usableHeader() const { return !header_.empty(); }

    static Row FromStream(std::istream& istream, const CsvHeader& header,
                          const char delimiter = DefaultDelimiter);

    inline const CsvHeader& header() const { return header_; }
    inline char delimieter() const { return delimiter_; }

   private:
    std::string line_;
    CsvHeader header_;
    char delimiter_;
    std::vector<int> pos_data_;
  };

 protected:
  // struct ReadData {
  //     std::istream* input_stream;
  //     CsvHeader header;

  //     ReadData(std::istream* stream, const CsvHeader& csv_header =
  //     CsvHeader()) : input_stream(stream), header(csv_header) {}
  // };

  template <typename ROW_TYPE>
  class RowIterator {
   public:
    using RowType = ROW_TYPE;
    using This = RowIterator<RowType>;

    //! naming conventions to match those required by iterator traits
    using value_type = RowType;
    using reference_type = RowType&;
    using pointer = RowType*;
    using difference_type = std::size_t;
    using iterator_category = std::input_iterator_tag;

    RowIterator(std::istream* input_stream)
        : input_stream_(CHECK_NOTNULL(input_stream)->good() ? input_stream
                                                            : nullptr) {
      ++(*this);
    }

    RowIterator() : input_stream_(nullptr) {}

    // Pre Increment
    RowIterator& operator++() {
      if (input_stream_) {
        // read new row
        row_ << *input_stream_;
        // if end of file
        if (!(*input_stream_)) {
          input_stream_ = nullptr;
        }
      }
      return *this;
    }
    // Post increment
    RowIterator operator++(int) {
      RowIterator tmp(*this);
      ++(*this);
      return tmp;
    }

    value_type const& operator*() const { return row_; }
    value_type const* operator->() const { return &row_; }

    bool operator==(RowIterator const& rhs) {
      return ((this == &rhs) || ((this->input_stream_ == nullptr) &&
                                 (rhs.input_stream_ == nullptr)));
    }
    bool operator!=(RowIterator const& rhs) { return !((*this) == rhs); }

   private:
   private:
    std::istream* input_stream_;
    // a non const value type
    // is requited to be non_const so that we can update the row with new data
    // during the pre-increment
    typename std::remove_const<value_type>::type row_;
  };

 public:
  using iterator = RowIterator<Row>;
  using const_iterator = RowIterator<const Row>;

  // NOTE: takes reference to istream (not ownership)
  // so stream cannot go out of scope!!!
  // TODO: no way to parse the header down to the row!!!!
  // TODO: if header is provided should check and skip the header so the user
  // does not have to manually increment the file
  CsvReader(std::istream& stream) : infile_(stream) { CHECK(infile_.good()); }

  iterator begin() { return iterator(&infile_); }
  iterator end() { return iterator(); }

  const_iterator begin() const { return const_iterator(&infile_); }
  const_iterator end() const { return const_iterator(); }

  const_iterator cbegin() const { return const_iterator(&infile_); }
  const_iterator cend() const { return const_iterator(); }

 private:
  std::istream& infile_;
};

// Modified from:
// https://github.com/al-eax/CSVWriter/blob/master/include/CSVWriter.h
class CsvWriter {
 public:
  DYNO_POINTER_TYPEDEFS(CsvWriter)

  CsvWriter(const CsvHeader& header, const std::string& seperator = ",");
  ~CsvWriter();

  CsvWriter& add(const char* str);
  CsvWriter& add(char* str);
  CsvWriter& add(const std::string& str);

  template <typename T>
  CsvWriter& add(const T& str) {
    // if autoNewRow is enabled, check if we need a line break
    if (this->value_count_ == this->column_number_) {
      this->newRow();
    }

    if (value_count_ > 0) this->ss_ << this->seperator_;
    this->ss_ << str;
    this->value_count_++;

    return *this;
  }

  template <typename T>
  CsvWriter& operator<<(const T& t) {
    return this->add(t);
  }

  void operator+=(CsvWriter& csv) { this->ss_ << std::endl << csv; }

  std::string toString() const { return ss_.str(); }

  friend std::ostream& operator<<(std::ostream& os, const CsvWriter& csv) {
    return os << csv.toString();
  }

  CsvWriter& newRow();

  bool write(const std::string& filename) const;
  bool write(const std::string& filename, bool append) const;
  bool write(std::ostream& stream) const;
  void resetContent();

 protected:
  const CsvHeader header_;
  const size_t column_number_;
  const std::string seperator_;
  size_t value_count_;
  std::stringstream ss_;
};

}  // namespace dyno
