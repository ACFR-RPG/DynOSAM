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

#include "dynosam/common/Types.hpp"
#include "dynosam/common/Camera.hpp"
#include "dynosam/utils/OpenCVUtils.hpp"
#include "dynosam/utils/GtsamUtils.hpp"
#include "dynosam/frontend/FrontendParams.hpp"
#include "dynosam/frontend/vision/Frame.hpp"
#include "dynosam/frontend/vision/Vision-Definitions.hpp"
#include "dynosam/visualizer/Visualizer-Definitions.hpp"

#include <opencv4/opencv2/opencv.hpp>
#include <glog/logging.h>

namespace dyno {


namespace vision_tools {


// void disparityToDepth(const FrontendParams& params, const cv::Mat& disparity, cv::Mat& depth);

//does not do any undistortion etc on the image pairs -> simply looks to see which tracklet ids's are in both frames
//iteratoes over the current features and checks to see if the feature is in the previous feature set
//previous features can probably just be a FeatureCOntainer but i guess we want to check that it is valid too, via the filter iterator?
void getCorrespondences(FeaturePairs& correspondences, const FeatureFilterIterator& previous_features, const FeatureFilterIterator& current_features);

//unique object labels as present in a semantic/motion segmented image -> does not include background label
ObjectIds getObjectLabels(const cv::Mat& image);

std::vector<std::vector<int> > trackDynamic(const FrontendParams& params, const Frame& previous_frame, Frame::Ptr current_frame);

/**
 * @brief From a instance/semantic mask type img, construct the bounding box for the mask of object_id
 *
 * If object_id is not present in the mask (ie.e no pixe values with this mask),
 * or no valid contours could be constructed, false is returned.
 *
 *
 *
 * @param mask const cv::Mat&
 * @param object_id ObjectId
 * @param rect cv::Rect& The calcualted bounding box
 * @return true
 * @return false
 */
bool findObjectBoundingBox(const cv::Mat& mask, ObjectId object_id, cv::Rect& rect);


} //vision_tools


void determineOutlierIds(const TrackletIds& inliers, const TrackletIds& tracklets, TrackletIds& outliers);



} //dyno


#include "dynosam/frontend/vision/VisionTools-inl.hpp"
