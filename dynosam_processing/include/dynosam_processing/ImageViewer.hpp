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

using ImageDisplayQueue = ThreadsafeQueue<ImageToDisplay>;

/// @brief Base class to actually display a set of images
class ImageDisplay {
 public:
  DYNO_POINTER_TYPEDEFS(ImageDisplay)
  virtual ~ImageDisplay() = default;

  virtual void display(const std::vector<ImageToDisplay>& images) = 0;
};

class ImageBatchDisplayer {
 public:
  DYNO_POINTER_TYPEDEFS(ImageBatchDisplayer)
  ImageBatchDisplayer(ImageDisplayQueue* display_queue);
  virtual ~ImageBatchDisplayer() = default;

  void displayFromQueue(size_t wait_time_ms = 2);
  void addDisplay(ImageDisplay::Ptr display);

 protected:
  void display(const std::vector<ImageToDisplay>& images);

 private:
  bool getFromQueue(std::vector<ImageToDisplay>& images, size_t wait_time_ms);

 private:
  ImageDisplayQueue* display_queue_;
  //! The classes that will actually do the displaying
  std::vector<ImageDisplay::Ptr> displayers_;
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
  void display(const std::vector<ImageToDisplay>& images) override;

  inline bool isDisplayAvailable() const { return is_gui_available_; }

 private:
  static bool isGuiAvailable();

 private:
  const bool is_gui_available_;
};

}  // namespace dyno
