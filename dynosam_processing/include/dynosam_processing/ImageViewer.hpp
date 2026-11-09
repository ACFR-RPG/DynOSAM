#pragma once

#include <opencv4/opencv2/core.hpp>
#include <vector>

#include "dynosam_common/utils/Macros.hpp"
#include "dynosam_processing/ThreadSafeQueue.hpp"

namespace dyno {

struct ImageToDisplay {
  ImageToDisplay(const std::string& name, const cv::Mat& image)
      : name_(name), image_(image) {}

  std::string name_;
  cv::Mat image_;
};

/// @brief Alias to a vector of ImageToDisplay types
typedef std::vector<ImageToDisplay> ImageToDisplayVector;

/// @brief Base class to actually display a set of images
class ImageDisplay {
 public:
  DYNO_POINTER_TYPEDEFS(ImageDisplay)
  virtual ~ImageDisplay() = default;

  virtual void display(const ImageToDisplayVector& images) = 0;
};

class ImageDisplayerBase {
 public:
  void addDisplay(ImageDisplay::Ptr display);
  virtual ~ImageDisplayerBase() = default;

 protected:
  /// @brief Send provided images to the attached displayers
  /// @param images
  void display(const ImageToDisplayVector& images);

 private:
  //! The classes that will actually do the displaying
  std::vector<ImageDisplay::Ptr> displayers_;
};

/**
 * @brief Immediately displays images via the attached displays
 *
 */
class ImageBatchDisplayer : public ImageDisplayerBase {
 public:
  void display(const ImageToDisplay& image);
  void display(const std::string& name, const cv::Mat& image);
};

using ImageDisplayQueue = ThreadsafeQueue<ImageToDisplay>;

/**
 * @brief  An non threaded image pipeline class that pulls a batch of images
 * from the provided ImageDisplayQueue and sends them to any added
 * ImageDisplay's.
 *
 * We pull all images that are available in the queue upon displayFromQueue
 * since many images might have been added since the last call.
 *
 */
class ImageQueueDisplayer : public ImageDisplayerBase {
 public:
  DYNO_POINTER_TYPEDEFS(ImageQueueDisplayer)
  ImageQueueDisplayer(ImageDisplayQueue* display_queue);
  virtual ~ImageQueueDisplayer() = default;

  void displayFromQueue(size_t wait_time_ms = 2);

 private:
  bool getFromQueue(ImageToDisplayVector& images, size_t wait_time_ms);

 private:
  ImageDisplayQueue* display_queue_;
};

/**
 * @brief Smart OpenCV shower that displays images with im::show IF the gui
 * backend is available. This allows us to run the same code on remote/headless
 * systems without getting errors.
 *
 * The class is 'smart' in the sense it tries to detect if the system is
 * headless or not.
 *
 */
class SmartOpenCvDisplayer : public ImageDisplay {
 public:
  SmartOpenCvDisplayer();
  void display(const ImageToDisplayVector& images) override;

  inline bool isDisplayAvailable() const { return can_display_; }

 private:
  static bool isGuiAvailable();

 private:
  const bool can_display_;
};

}  // namespace dyno
