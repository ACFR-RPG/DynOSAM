#pragma once

#include <glog/logging.h>
#include <gtsam/geometry/StereoCamera.h>
#include <gtsam/geometry/triangulation.h>

#include <opencv4/opencv2/core/mat.hpp>
#include <opencv4/opencv2/core/types.hpp>
#include <opencv4/opencv2/opencv.hpp>

#include "dynosam_common/Types.hpp"

#define CHECK_MAT_TYPES(mat1, mat2)                                            \
  using namespace dyno::utils;                                                 \
  CHECK_EQ(mat1.type(), mat2.type())                                           \
      << "Matricies should be of the same type ( "                             \
      << cvTypeToString(mat1.type()) << " vs. " << cvTypeToString(mat2.type()) \
      << ")."

namespace dyno {

/* Partial specalisation of dyno::to_string for all types cv::Rect_ */
template <typename V>
std::string to_string(const cv::Rect_<V>& t) {
  return "[x=" + std::to_string(t.x) + " y=" + std::to_string(t.y) +
         " h=" + std::to_string(t.height) + " w=" + std::to_string(t.width) +
         "]";
}

/// triangulateSafe: extensive checking of the outcome
/// a modifed version that lets use use triangulateLOST
/// while still on gtsam 4.2
template <class CAMERA>
gtsam::TriangulationResult triangulateSafe(
    const gtsam::CameraSet<CAMERA>& cameras,
    const typename CAMERA::MeasurementVector& measured,
    const gtsam::TriangulationParameters& params, const bool useLOST = false);

// NOTE: returns the inlier/outlier mask not H
cv::Mat findHomography(const std::vector<cv::Point2f>& previous,
                       const std::vector<cv::Point2f>& current,
                       double repr_threshold = 3.0, const int max_iters = 1000);

// NOTE: returns the inlier/outlier mask not E
cv::Mat findEssential(const std::vector<cv::Point2f>& previous,
                      const std::vector<cv::Point2f>& current, const cv::Mat& K,
                      double threshold = 3.0, const int max_iters = 1000);

void outlierRejectEssential(const std::vector<cv::Point2f>& previous,
                            const std::vector<cv::Point2f>& current,
                            const TrackletIds& tracklet_ids, const cv::Mat& K,
                            std::vector<cv::Point2f>& verified_previous,
                            std::vector<cv::Point2f>& verified_current,
                            TrackletIds& verified_tracklet_ids);

void outlierRejectEssential(const std::vector<cv::Point2f>& previous,
                            const std::vector<cv::Point2f>& current,
                            const cv::Mat& K,
                            std::vector<cv::Point2f>& verified_previous,
                            std::vector<cv::Point2f>& verified_current);

void outlierRejectHomography(const std::vector<cv::Point2f>& previous,
                             const std::vector<cv::Point2f>& current,
                             const TrackletIds& tracklet_ids,
                             std::vector<cv::Point2f>& verified_previous,
                             std::vector<cv::Point2f>& verified_current,
                             TrackletIds& verified_tracklet_ids);

void outlierRejectHomography(const std::vector<cv::Point2f>& previous,
                             const std::vector<cv::Point2f>& current,
                             std::vector<cv::Point2f>& verified_previous,
                             std::vector<cv::Point2f>& verified_current);

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
 * @brief Get the Object Bounding Boxes object given a mask and a set of object
 * ids.
 *
 * Super optimized function to run very quickly as discovering bounding boxes
 * using regular opencv functions can take up to 5-10ms depending on image size
 * and this is often a bottle-neck.
 *
 *
 * @param mask
 * @param object_ids
 * @param bounding_boxes
 */
void getObjectBoundingBoxes(const cv::Mat& mask, const ObjectIds& object_ids,
                            std::vector<cv::Rect>& bounding_boxes);

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

void relabelMasks(const cv::Mat& mask, cv::Mat& relabelled_mask,
                  const ObjectIds& old_labels, const ObjectIds& new_labels);

template <typename T>
class ParallelOpenCVMat : public cv::ParallelLoopBody {
 public:
  using This = ParallelOpenCVMat<T>;

  ParallelOpenCVMat(cv::Mat& img_ref) : img_(img_ref) {}

  int dataSize() const { return img_.rows * img_.cols; }

  void operator()(const cv::Range& range) const CV_OVERRIDE {
    const T* derived = static_cast<const T*>(this);
    CHECK_NOTNULL(derived);
    for (int r = range.start; r < range.end; r++) {
      int i = r / img_.cols;
      int j = r % img_.cols;
      derived->call(img_, i, j);
    }
  }

  This& run() {
    cv::parallel_for_(cv::Range(0, dataSize()), *this);
    return *this;
  }

  ParallelOpenCVMat& operator=(const ParallelOpenCVMat&) { return *this; };

 protected:
  virtual void call(cv::Mat& img_ref, int i, int j) const = 0;

 private:
  cv::Mat& img_;
};

class FunctionalParallelOpenCVMat
    : public ParallelOpenCVMat<FunctionalParallelOpenCVMat> {
 public:
  using Call = std::function<void(cv::Mat&, int, int)>;
  using Base = ParallelOpenCVMat<FunctionalParallelOpenCVMat>;

  FunctionalParallelOpenCVMat(cv::Mat& img_ref, const Call& call)
      : Base(img_ref), call_(call) {}

 private:
  // allow the base class to access the derived call
  friend Base;

  void call(cv::Mat& img_ref, int i, int j) const override {
    call_(img_ref, i, j);
  }

 private:
  mutable Call call_;
};

namespace utils {

// TODO: this function is really if opencv AVAILABLE, not just enabled (ie. it
// is enabled AND can find a GPU!)
bool opencvCudaAvailable(std::string* message = nullptr);

double calculateIoU(const cv::Rect& a, const cv::Rect& b);

bool cvSizeEqual(const cv::Size& a, const cv::Size& b);
bool cvSizeEqual(const cv::Mat& a, const cv::Mat& b);

template <typename T>
inline float distance(const cv::Point_<T>& pt1, const cv::Point_<T>& pt2) {
  const float dx = static_cast<float>(pt1.x) - static_cast<float>(pt2.x);
  const float dy = static_cast<float>(pt1.y) - static_cast<float>(pt2.y);
  return std::sqrt(dx * dx + dy * dy);
}

/**
 * @brief Returns true of the point is inside bounds of the cv::Mat.
 *
 * @param x int. x coordinate to check (cols)
 * @param y int. y coordinate to check (rows)
 * @return true
 * @return false
 */
inline bool matContains(const cv::Mat& mat, int x, int y) {
  return x >= 0 && y >= 0 && x < mat.cols && y < mat.rows;
}

/**
 * @brief Returns true of the point is inside bounds of the cv::Mat.
 *
 * @tparam T Point template to work with any cv::Point type
 * @param mat const cv::Mat&
 * @param point const cv::Point_<T>&
 * @return true
 * @return false
 */
template <typename T>
inline bool matContains(const cv::Mat& mat, const cv::Point_<T>& point) {
  const auto x = static_cast<int>(point.x);
  const auto y = static_cast<int>(point.y);
  return matContains(mat, x, y);
}

std::string cvTypeToString(int type);
std::string cvTypeToString(const cv::Mat& mat);

// Compares two opencv matrices up to a certain tolerance error.
bool compareCvMatsUpToTol(const cv::Mat& mat1, const cv::Mat& mat2,
                          const double& tol = 1e-7);

cv::Mat concatenateImagesHorizontally(const cv::Mat& left_img,
                                      const cv::Mat& right_img);

cv::Mat concatenateImagesVertically(const cv::Mat& top_img,
                                    const cv::Mat& bottom_img);

void flowToRgb(const cv::Mat& flow, cv::Mat& rgb);

/**
 * @brief Given an image that operates like a mask (ie. is a single channel
 * image where each pixel value correspondes to a label of some type) draws the
 * mask over an input image.
 *
 * The value at each pixel (labels) are treated like object labels which are
 * used to generate a unique colour for the mask. The background label is used
 * to indicate which pixel value should be ignored - this could be the
 * background or simply unknown pixel values.
 *
 *
 * Alpha is used to blend with mask ontop of the rgb image.
 *
 * @param mask const cv::Mat&
 * @param rgb_input const cv::Mat&
 * @param output cv::Mat&
 * @param alpha float default value is 0.7
 * @param background_label int default value is 0
 */
void labelMaskToRGB(const cv::Mat& mask, const cv::Mat& rgb_input,
                    cv::Mat& output, float alpha = 0.7,
                    int background_label = 0);

/**
 * @brief Given an image that operates like a mask (ie. is a single channel
 * image where each pixel value correspondes to a label of some type) draws the
 * mask over an input image.
 *
 * The value at each pixel (labels) are treated like object labels which are
 * used to generate a unique colour for the mask. The background label is used
 * to indicate which pixel value should be ignored - this could be the
 * background or simply unknown pixel values
 *
 * @param mask
 * @param background_label
 * @param rgb
 * @return cv::Mat
 */
cv::Mat labelMaskToRGB(const cv::Mat& mask, int background_label,
                       const cv::Mat& rgb);

/**
 * @brief Same as labelMaskToRGB(mask, background label) but instead draws the
 * mask over a black image so only the mask values are shown
 *
 * @param mask
 * @param background_label
 * @return cv::Mat
 */
cv::Mat labelMaskToRGB(const cv::Mat& mask, int background_label);

/**
 * @brief Allows visualization of disparity image (which is
 * usually outputed as CV_16S or CV_32F from stereo depth reconstruction).
 *
 * @param src cv::InputArray
 * @param dst cv::OutputArray
 * @param unknown_disparity
 */
void getDisparityVis(cv::InputArray src, cv::OutputArray dst,
                     int unknown_disparity = 16320);

// TODO: uh, this was const cv::Mat& and things were still being modified...
// how...?
/**
 * @brief Draws a coloured bounding box and associated label onto a RGB image.
 *
 * Used to draw object bounding boxes and their labels.
 *
 * @param image cv::Mat& rgb image to modify
 * @param label const std::string& label to draw in the top-right corner of the
 * bounding box
 * @param colour const cv::Scalar& colour to draw the bounding box with
 * @param bounding_box const cv::Rect& bounding box to draw.
 * @param bb_thickness const int& Line thickness for the boundong box
 */
void drawLabeledBoundingBox(cv::Mat& image, const std::string& label,
                            const cv::Scalar& colour,
                            const cv::Rect& bounding_box,
                            const int& bb_thickness = 2);

/**
 * @brief Projects SE(3) poses onto the image as an RGB axes.
 * Input poses are expected to be in the camera (local) frame
 *
 * @param image cv::Mat&
 * @param K const cv::Mat& 3x3 calibration matrix
 * @param D const cv::Mat& distortion coeeffs
 * @param poses_c const std::vector<gtsam::Pose3>& Poses in the camera (local)
 * frame to be projected
 * @param scale float The size of the axis in 2D after projection (default: 0.2)
 */
void drawObjectPoseAxes(cv::Mat& image, const cv::Mat& K, const cv::Mat& D,
                        const std::vector<gtsam::Pose3>& poses_c,
                        float scale = 0.2);

/**
 * I have absolutely no idea why but OpenCV seemds to have removed support for
 * the read/write optical flow functions in 4.x I have taken this implementation
 * from OpenCV 3.4
 * (https://github.com/opencv/opencv_contrib/blob/3.4/modules/optflow/src/optical_flow_io.cpp)
 */

/**
 * @brief The function readOpticalFlow loads a flow field from a file and
 * returns it as a single matrix. Resulting Mat has a type CV_32FC2 -
 * floating-point, 2-channel. First channel corresponds to the flow in the
 * horizontal direction (u), second - vertical (v).
 *
 * @param path
 * @return cv::Mat
 */
cv::Mat readOpticalFlow(const std::string& path);

/**
 * @brief The function stores a flow field in a file, returns true on success,
 * false otherwise. The flow field must be a 2-channel, floating-point matrix
 * (CV_32FC2). First channel corresponds to the flow in the horizontal direction
 * (u), second - vertical (v).
 *
 * @param path
 * @param flow
 * @return true
 * @return false
 */
bool writeOpticalFlow(const std::string& path, const cv::Mat& flow);

}  // namespace utils
}  // namespace dyno

#include <yaml-cpp/yaml.h>

namespace YAML {

template <typename T>
struct convert<cv::Size_<T>> {
  static Node encode(const cv::Size_<T>& size) {
    Node node;
    node["width"] = size.width;
    node["height"] = size.height;
    return node;
  }

  static bool decode(const Node& node, cv::Size_<T>& size) {
    if (!node.IsMap()) {
      return false;
    }

    size.width = node["width"].as<T>();
    size.height = node["height"].as<T>();
    return true;
  }
};

}  // namespace YAML

#include "dynosam_cv/CvUtils-inl.hpp"
