/*
 *   Copyright (c) 2023 ACFR-RPG, University of Sydney, Jesse Morris (jesse.morris@sydney.edu.au)
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

#pragma once

#include <dynosam/pipeline/PipelineManager.hpp>
#include <dynosam/utils/Statistics.hpp>

#include "rclcpp/node.hpp"
#include "rclcpp/node_options.hpp"

namespace dyno {

class DynoPipelineManagerRos : public rclcpp::Node {

public:
    explicit DynoPipelineManagerRos(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
    ~DynoPipelineManagerRos() = default;

    bool spinOnce() {
        RCLCPP_INFO_STREAM_THROTTLE(this->get_logger(), *this->get_clock(), 2000, getStats());

        return CHECK_NOTNULL(pipeline_)->spin();
    }

    std::string getStats() const {
        return utils::Statistics::Print();
    }

private:
    std::string getParamsPath();
    std::string getDatasetPath();

    /**
     * @brief Retrieves a std::string param (under param_name) which is expected to be a file path
     *
     * @param param_name
     * @param default_path
     * @param description
     * @return std::string
     */
    std::string searchForPathWithParams(const std::string& param_name, const std::string& default_path, const std::string& description = "");

private:
    DynoPipelineManager::UniquePtr pipeline_;

};

}
