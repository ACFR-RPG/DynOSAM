/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris (jesse.morris@sydney.edu.au)
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a copy
 *   of this software and associated documentation files (the "Software"), to deal
 *   in the Software without restriction, including without limitation the rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:

 *   The above copyright notice and this permission notice shall be included in all
 *   copies or substantial portions of the Software.

 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *   SOFTWARE.
 */

#include "dynosam/utils/CsvWriter.hpp"
#include <glog/logging.h>

namespace dyno {

CsvWriter::CsvWriter(const CsvHeader& header, const std::string& seperator)
    :   header_(header),
        column_number_(header.size()),
        seperator_(seperator),
        value_count_(0),
        ss_() {

    checkAndThrow<InvalidCsvHeaderException>((column_number_ > 0u), "CsvHeader cannot be empty! (size of 0)");
}

CsvWriter::~CsvWriter() { resetContent();}

CsvWriter& CsvWriter::add(const char *str){
    return this->add(std::string(str));
}

CsvWriter& CsvWriter::add(char *str){
    return this->add(std::string(str));
}

CsvWriter& CsvWriter::add(const std::string& str){
    std::string str_cpy = str;
    //if " character was found, escape it
    size_t position = str_cpy.find("\"",0);
    bool foundQuotationMarks = position != std::string::npos;
    while(position != std::string::npos){
        str_cpy.insert(position,"\"");
        position = str_cpy.find("\"",position + 2);
    }
    if(foundQuotationMarks){
        str_cpy = "\"" + str_cpy + "\"";
    } else if(str_cpy.find(this->seperator_) != std::string::npos){
        //if seperator_ was found and string was not escapted before, surround string with "
        str_cpy = "\"" + str_cpy + "\"";
    }
    return this->add<std::string>(str_cpy);
}

CsvWriter& CsvWriter::newRow() {
    ss_ << std::endl;
    value_count_ = 0;
    return *this;
}

bool CsvWriter::write(const std::string& filename) const {
    return write(filename,false);
}

bool CsvWriter::write(const std::string& filename, bool append) const{
    std::ofstream file;
    bool appendNewLine = false;
    if (append) {
        //check if last char of the file is newline
        std::ifstream fin;
        fin.open(filename);
        if (fin.is_open()) {
            fin.seekg(-1, std::ios_base::end); //go to end of file
            int lastChar = fin.peek();
            if (lastChar != -1 && lastChar != '\n') //if file is not empry and last char is not new line char
                appendNewLine = true;
        }
        file.open(filename.c_str(), std::ios::out | std::ios::app);
    }
    else {
        file.open(filename.c_str(), std::ios::out | std::ios::trunc);
    }
    if(!file.is_open())
        return false;
    if(append && appendNewLine)
        file << std::endl;

    return write(file);
}

bool CsvWriter::write(std::ostream& stream) const {
    if(!stream) {
        LOG(ERROR) << "Failed to write with CsVWriter as stream was not good!";
        return false;
    }

    stream << header_.toString(seperator_);
    stream << std::endl;
    stream << toString();
    return stream.good();
}


void CsvWriter::resetContent(){
    const static std::stringstream initial;
    ss_.str(std::string());
    ss_.clear();
    ss_.copyfmt(initial);
}

}
