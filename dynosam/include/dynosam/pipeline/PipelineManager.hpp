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

#include "dynosam/dataprovider/DataProviderModule.hpp"
#include "dynosam/dataprovider/DataProvider.hpp"
#include "dynosam/frontend/FrontendPipeline.hpp"
#include "dynosam/visualizer/VisualizerPipelines.hpp"
#include "dynosam/utils/Spinner.hpp"
#include "dynosam/common/Types.hpp"

namespace dyno {

class DynoPipelineManager {

public:
    DYNO_POINTER_TYPEDEFS(DynoPipelineManager)

    //why are some unique and some shared?? silly silly
    DynoPipelineManager(DataProvider::UniquePtr data_loader, FrontendModule::Ptr frontend_module, FrontendDisplay::Ptr frontend_display);
    ~DynoPipelineManager();

    void spin(bool parallel_run = true);

private:
    FrontendPipeline::UniquePtr frontend_pipeline_;
    FrontendPipeline::InputQueue frontend_input_queue_;
    FrontendPipeline::OutputQueue frontend_output_queue_;

    //Display and Viz
    FrontendVizPipeline::UniquePtr frontend_viz_pipeline_;

    //Data-provider pointers
    DataProviderModule::UniquePtr data_provider_module_;
    DataProvider::UniquePtr data_loader_;

    //Threaded spinners
    Spinner::UniquePtr data_provider_spinner_;
    Spinner::UniquePtr frontend_pipeline_spinner_;
    Spinner::UniquePtr frontend_viz_pipeline_spinner_;


};


} //dyno
