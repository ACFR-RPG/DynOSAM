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

/**
 * This file is part of VDO-SLAM.
 *
 * Copyright (C) 2019-2020 Jun Zhang <jun doc zhang2 at anu dot edu doc au> (The
 *Australian National University) For more information see
 *<https://github.com/halajun/VDO_SLAM>
 *
 **/

#pragma once

#include <list>
#include <opencv4/opencv2/opencv.hpp>
#include <vector>

#include "dynosam_common/utils/Macros.hpp"

namespace dyno {
class ExtractorNode {
 public:
  ExtractorNode() : bNoMore(false) {}

  void DivideNode(ExtractorNode& n1, ExtractorNode& n2, ExtractorNode& n3,
                  ExtractorNode& n4);

  std::vector<cv::KeyPoint> vKeys;
  cv::Point2i UL, UR, BL, BR;
  std::list<ExtractorNode>::iterator lit;
  bool bNoMore;
};

class ORBextractor {
 public:
  DYNO_POINTER_TYPEDEFS(ORBextractor)
  enum { HARRIS_SCORE = 0, FAST_SCORE = 1 };

  ORBextractor(int nfeatures, float scaleFactor, int nlevels, int iniThFAST,
               int minThFAST);

  ~ORBextractor() {}

  // Compute the ORB features and descriptors on an image.
  // ORB are dispersed on the image using an octree.
  // Mask is ignored in the current implementation.
  void operator()(cv::InputArray image, cv::InputArray mask,
                  std::vector<cv::KeyPoint>& keypoints,
                  cv::OutputArray descriptors);

  int inline GetLevels() { return nlevels; }

  float inline GetScaleFactor() { return scaleFactor; }

  std::vector<float> inline GetScaleFactors() { return mvScaleFactor; }

  std::vector<float> inline GetInverseScaleFactors() {
    return mvInvScaleFactor;
  }

  std::vector<float> inline GetScaleSigmaSquares() { return mvLevelSigma2; }

  std::vector<float> inline GetInverseScaleSigmaSquares() {
    return mvInvLevelSigma2;
  }

  std::vector<cv::Mat> mvImagePyramid;

 protected:
  void ComputePyramid(cv::Mat image);
  void ComputeKeyPointsOctTree(
      std::vector<std::vector<cv::KeyPoint> >& allKeypoints);
  std::vector<cv::KeyPoint> DistributeOctTree(
      const std::vector<cv::KeyPoint>& vToDistributeKeys, const int& minX,
      const int& maxX, const int& minY, const int& maxY, const int& nFeatures,
      const int& level);

  void ComputeKeyPointsOld(
      std::vector<std::vector<cv::KeyPoint> >& allKeypoints);
  std::vector<cv::Point> pattern;

  int nfeatures;
  double scaleFactor;
  int nlevels;
  int iniThFAST;
  int minThFAST;

  std::vector<int> mnFeaturesPerLevel;

  std::vector<int> umax;

  std::vector<float> mvScaleFactor;
  std::vector<float> mvInvScaleFactor;
  std::vector<float> mvLevelSigma2;
  std::vector<float> mvInvLevelSigma2;
};

}  // namespace dyno
