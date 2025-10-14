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

#include <opencv4/opencv2/opencv.hpp>

#include "dynosam/frontend/vision/Frame.hpp"
#include "dynosam/frontend/vision/Vision-Definitions.hpp"
#include "dynosam/visualizer/Visualizer-Definitions.hpp"
#include "dynosam_common/Cuda.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_common/utils/GtsamUtils.hpp"
#include "dynosam_common/utils/Histogram.hpp"
#include "dynosam_common/utils/OpenCVUtils.hpp"
#include "dynosam_cv/Camera.hpp"
#include "dynosam_cv/ImageContainer.hpp"

#ifdef DYNO_CUDA_OPENCV_ENABLED
#include <opencv2/cudaoptflow.hpp>
#endif

namespace dyno {

class FrontendParams;

namespace vision_tools {

// class KLTWrapper {

// public:
//     LKWrapper(const cv::Size& win_size = cv::Size(21,21),
//               int max_level = 3,
//               const cv::TermCriteria& criteria =
//               cv::TermCriteria(cv::TermCriteria::COUNT +
//               cv::TermCriteria::EPS, 30, 0.01), int flags = 0, double
//               min_eig_threshold = 1e-4);

//     template<typename ImgType>
//     void calc(const ImgType& prev_img, const ImgType& next_img,
//               const std::vector<cv::Point2f>& prev_pts,
//               std::vector<cv::Point2f>& next_pts,
//               std::vector<uchar>& status,
//               std::vector<float>& err,
//               bool return_gpu = false)
//     {
//         if constexpr (std::is_same_v<ImgType, cv::Mat>) {
//             if()

//         }
//         else if constexpr (std::is_same_v<ImgType, cv::cuda::GpuMat>) {

//         }
//         else {
//             static_assert(False, "Unsupported image type");
//         }
//     }

// private:
//     cv::Size win_size_;
//     int max_level_;
//     cv::TermCriteria criteria_;
//     int flags_;
//     double min_eig_threshold_;
//     bool use_cuda_;
//     bool first_run_;

//     #ifdef DYNO_CUDA_OPENCV_ENABLED
//     cv::Ptr<cv::cuda::OpticalFlowPyrLK> lk_cuda_;
//     cv::cuda::GpuMat d_prev_, d_next_;
//     cv::cuda::GpuMat d_next_pts_, d_status_, d_err_;
//     #endif
// };

// void disparityToDepth(const FrontendParams& params, const cv::Mat& disparity,
// cv::Mat& depth);

// does not do any undistortion etc on the image pairs -> simply looks to see
// which tracklet ids's are in both frames iteratoes over the current features
// and checks to see if the feature is in the previous feature set previous
// features can probably just be a FeatureCOntainer but i guess we want to check
// that it is valid too, via the filter iterator?
void getCorrespondences(FeaturePairs& correspondences,
                        const FeatureFilterIterator& previous_features,
                        const FeatureFilterIterator& current_features);

// unique object labels as present in a semantic/motion segmented image -> does
// not include background label

/**
 * @brief Finds the unique object labels (j) as present in the instance
 * segmentation image. Does not include the background label.
 *
 * @param image const cv::Mat&
 * @return ObjectIds
 */
ObjectIds getObjectLabels(const cv::Mat& image);

/**
 * @brief Shrinks all found object masks by a given ammount.
 *
 * @param mask const cv::Mat& input object mask with pixel values 0, 1... j
 * @param shrunk_mask cv::Mat& output object mask with shrink masks.
 * @param erosion_size int erosion size
 */
void shrinkMask(const cv::Mat& mask, cv::Mat& shrunk_mask, int erosion_size);

/**
 * @brief From a instance/semantic mask type img, construct the bounding box for
 * the mask of object_id
 *
 * If object_id is not present in the mask (ie.e no pixe values with this mask),
 * or no valid contours could be constructed, false is returned.
 *
 *
 *
 * @param mask const cv::Mat&
 * @param object_id ObjectId
 * @param detected_rect cv::Rect& The calcualted bounding box
 * @param detected_contours std::vector<std::vector<cv::Point>>& The calcualted
 * contours used to find the bounding box
 * @return true
 * @return false
 */
bool findObjectBoundingBox(
    const cv::Mat& mask, ObjectId object_id, cv::Rect& detected_rect,
    std::vector<std::vector<cv::Point>>& detected_contours);

/**
 * @brief From a instance/semantic mask type img, construct the bounding box for
 * the mask of object_id.
 *
 * Differes only in output variables.
 *
 * @param mask
 * @param object_id
 * @param detected_rect
 * @return true
 * @return false
 */
bool findObjectBoundingBox(const cv::Mat& mask, ObjectId object_id,
                           cv::Rect& detected_rect);

/**
 * @brief From a instance/semantic mask type img, construct the bounding box for
 * the mask of object_id.
 *
 * Differes only in output variables.
 *
 * @param mask
 * @param object_id
 * @param detected_contours
 * @return true
 * @return false
 */
bool findObjectBoundingBox(
    const cv::Mat& mask, ObjectId object_id,
    std::vector<std::vector<cv::Point>>& detected_contours);

struct ObjectBoundaryMaskResult {
  //! Modified boundary mask for each object (dilated and eroded) representing
  //! the allowing feature detection pixels for each object
  cv::Mat boundary_mask;
  //! If the boundary mask should treated as a feature detection mask. See
  //! computeObjectMaskBoundaryMask for more details.
  bool is_feature_detection_mask;
  //! Same boundary mask will be set but with the object mask boarders set to
  //! the object id (j) instead of 0/255. The background will be 0. This makes
  //! it easier to do object level association.
  cv::Mat labelled_boundary_mask;
  //! Object labels (j) detected on the imput mask
  ObjectIds objects_detected;
  //! Bounding boxes of objects as detected on the original mask. In order of
  //! objects_detected.
  std::vector<cv::Rect> object_bounding_boxes;
  //! Bounding boxes 'inner' layer of the boundary mask; this roughly indicates
  //! the inner area of the object which features can be detected on. In order
  //! of objects_detected.
  std::vector<cv::Rect> inner_boarder_object_bounding_boxes;
};

/**
 * @brief From  an input instance/semantic mask type img construct a image mask
 * (binary) that has True values around each object in the image, with a certain
 * thickness.
 *
 * This is used to mark all points intside the mask as invalid and should not be
 * tracked. Like any mask used for an feature detector the output mask is a
 * 8-bit integer matrix with non-zero values in the region of interest.
 *
 * When use_as_feature_detection_mask is True, all pixels inside the thicc
 * boarder are set to zero, and all other values are 255. If this argument is
 * False, the opposite is true (pixels inside the boarder are 255 and others are
 * set to 0)
 */
void computeObjectMaskBoundaryMask(ObjectBoundaryMaskResult& result,
                                   const cv::Mat& mask, int thickness,
                                   bool use_as_feature_detection_mask = true);

void relabelMasks(const cv::Mat& mask, cv::Mat& relabelled_mask,
                  const ObjectIds& old_labels, const ObjectIds& new_labels);

/**
 * @brief Constructs a map of histograms (static features: 0, one for each
 * dynamic object) containing the tracklet length at this frame.
 *
 * @param frame
 * @param bins
 * @return gtsam::FastMap<ObjectId, Histogram>
 */
gtsam::FastMap<ObjectId, Histogram> makeTrackletLengthHistorgram(
    const Frame::Ptr frame,
    const std::vector<size_t>& bins = {0, 1, 2, 3, 5, 7, 10, 15, 25, 40, 60,
                                       std::numeric_limits<size_t>::max()});

/**
 * @brief Projects a dense depth map to 3D coordinates in the local frame.
 *
 * Returns a (H x W x 3) image of type CV_64F
 *
 * @param depth_image
 * @param K
 * @return cv::Mat
 */
cv::Mat depthTo3D(const ImageWrapper<ImageType::Depth>& depth_image,
                  const cv::Mat& K);

/**
 * @brief Saves a re-projected 3D point cloud with labelled points to file as a
 * cv::Mat.
 *
 * The depth map is projected into the camera coordinate frame and each pixel is
 * labelled with the tracking label from the mask.
 *
 * The final image is saved as a CV_64F (double) of W X H X 4 where channels
 * [0:3] are the xyz values of the point and [3] is the tracking label
 * (0=background, 1....N object track label).
 *
 * Used for reconstructing testing.
 *
 * Write to a folder called project_mask in the OutputFolder defined by
 * getOutputFilePath.
 *
 * @param depth_image NOTE: can be any mask (motion, semantic...)
 * @param mask_image
 * @param frame_id
 */
void writeOutProjectMaskAndDepthMap(
    const ImageWrapper<ImageType::Depth>& depth_image,
    const ImageWrapper<ImageType::SemanticMask>& mask_image,
    const Camera& camera, FrameId frame_id);

/**
 * @brief From a feature with valid depth, backproject to a 3D point in the
 * camera frame with the associated covariance (3x3) matrix.
 *
 * The measurement covariance matrix for the pixel measurement is obtanied
 * directly as a 2x2 diagonal matrix from the pixel sigma and the measurement
 * covariance for the depth is modelled as a quadratic cost where sigma = depth
 * * depth_sgima^2.
 *
 * The final covariance matrix is in the diagonal form [x,y,z]
 *
 * @param feature
 * @param camera
 * @param pixel_sigma
 * @param depth_sigma
 * @return std::pair<gtsam::Vector3, gtsam::Matrix3>
 */
std::pair<gtsam::Vector3, gtsam::Matrix33> backProjectAndCovariance(
    const Feature& feature, const Camera& camera, double pixel_sigma,
    double depth_sigma);

// /**
//  * @brief Fully rectifies all images in the ImageContainerSubset
//  *
//  * Quite slow :)
//  *
//  * @tparam ImageTypes
//  * @param images
//  * @param undistorted_images
//  * @param undistorted
//  * @return ImageContainerSubset<ImageTypes...>
//  */
// template<typename... ImageTypes>
// void rectifyImages(const ImageContainerSubset<ImageTypes...>& images,
// ImageContainerSubset<ImageTypes...>& undistorted_images, const
// UndistorterRectifier& undistorter) {
//     //do deep copy (the underlying ImageWrapper class has a deep copy in its
//     copy constructor so
//     //the image data is cloned)
//     using ImageSet = ImageContainerSubset<ImageTypes...>;
//     undistorted_images = ImageSet(images);

//     static constexpr size_t N = ImageSet::N;
//     for (size_t i = 0; i < N; i++) {
//         internal::select_apply<N>(i, [&](auto I){
//             using ImageType = typename ImageSet::ImageTypeStruct<I>;

//             //get reference to image and modify in place
//             cv::Mat& distorted_image = undistorted_images.template
//             get<ImageType>();
//             undistorter.undistortRectifyImage(distorted_image,
//             distorted_image);

//         });
//     }
// }

}  // namespace vision_tools

/**
 * @brief From a set a tracklets and a subset (of the tracklet) inliers,
 * calculate the remaining subset of outliers.
 *
 * In set notation this is:
 * inliers \cap outliers = \empty (indicating that inliers and outliers are
 * disjoint) inliers \cup outliers = tracklets (indicating that every element in
 * tracklets is either in inliers or outliers)
 *
 * Inliers must be a complete subset of tracklets for this function to work and
 * inliers.size() < tracklets.size(); if these cases do not hold the behaviour
 * is undefined!
 *
 * @param inliers const TrackletIds&
 * @param tracklets  const TrackletIds&
 * @param outliers TrackletIds&
 */
void determineOutlierIds(const TrackletIds& inliers,
                         const TrackletIds& tracklets, TrackletIds& outliers);

}  // namespace dyno

#include "dynosam/frontend/vision/VisionTools-inl.hpp"
