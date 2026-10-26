#pragma once

#include <opencv2/cudaoptflow.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <vector>

#include "dynosam_cv/CvUtils.hpp"
#include "dynosam_sensors/ImageTypes.hpp"

namespace dyno2 {

class OpticalFlowLKBase {
 public:
  static constexpr float kMaxErr = 20.0f;

  const cv::Size win_size;
  const int max_level;
  const cv::TermCriteria criteria;

  OpticalFlowLKBase(int iterations)
      : win_size(24, 24),
        max_level(4),
        criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, iterations,
                 0.01) {}

  OpticalFlowLKBase() : OpticalFlowLKBase(50) {}
  virtual ~OpticalFlowLKBase() = default;

  struct ImplResult {
    std::vector<uchar> status;
    std::vector<float> error;
    std::vector<cv::Point2f> prev;
    std::vector<cv::Point2f> next;

    size_t size() const { return status.size(); }
  };

  struct Result {
    const ImplResult forward_result;
    const ImplResult reverse_result;

    Result(const ImplResult& forward, const ImplResult& reverse)
        : forward_result(forward), reverse_result(reverse) {}

    const std::vector<cv::Point2f>& predictedPoints() const {
      return forward_result.next;
    }

    const std::vector<cv::Point2f>& fromPoints() const {
      return forward_result.prev;
    }

    const std::vector<float>& error() const { return forward_result.error; }

    size_t size() const { return forward_result.size(); }

    bool isGood(size_t i) const {
      const bool both_status_good =
          forward_result.status.at(i) && reverse_result.status.at(i);

      // compare distance betwen original pixel local and reverse predicted
      // pixel location (which should be the same)
      const bool within_distance =
          dyno::utils::distance(forward_result.prev.at(i),
                                reverse_result.next.at(i)) <= 0.5;

      const bool within_error = forward_result.error[i] < kMaxErr &&
                                reverse_result.error[i] < kMaxErr;

      return both_status_good && within_distance && within_error;
    }
  };

  inline Result operator()(const dyno::WrappedRGBMono& prev_img,
                           const dyno::WrappedRGBMono& next_img,
                           const std::vector<cv::Point2f>& prev_points) const {
    return this->compute(prev_img, next_img, prev_points);
  }

 protected:
  virtual Result compute(const dyno::WrappedRGBMono& prev_img,
                         const dyno::WrappedRGBMono& next_img,
                         const std::vector<cv::Point2f>& prev_points) const = 0;

  // Result operator()(const std::vector<cv::Mat>& prev_pyr,
  //                   const std::vector<cv::Mat>& next_pyr,
  //                   const std::vector<cv::Point2f>& prev_points) const;
};

class OpticalFlowLK_CPU : public OpticalFlowLKBase {
 public:
  typedef OpticalFlowLK_CPU::Result Result;
  OpticalFlowLK_CPU(int iterations) : OpticalFlowLKBase(iterations) {}

 private:
  Result compute(const dyno::WrappedRGBMono& prev_img,
                 const dyno::WrappedRGBMono& next_img,
                 const std::vector<cv::Point2f>& prev_points) const override;
};

class OpticalFlowLK_GPU : public OpticalFlowLKBase {
 public:
  typedef OpticalFlowLK_CPU::Result Result;
  OpticalFlowLK_GPU(int iterations);

 private:
  Result compute(const dyno::WrappedRGBMono& prev_img,
                 const dyno::WrappedRGBMono& next_img,
                 const std::vector<cv::Point2f>& prev_points) const override;

  //  // Ensure that a pinned host buffer has sufficient capacity.
  // static void ensureHostCapacity(
  //     cv::cuda::HostMem& buffer,
  //     int rows,
  //     int cols,
  //     int type);

  // // Ensure that a device buffer has the required dimensions and type.
  // static void ensureDeviceCapacity(
  //     cv::cuda::GpuMat& buffer,
  //     int rows,
  //     int cols,
  //     int type);

  static void ensureBuffer(cv::cuda::GpuMat& buffer, int rows, int cols,
                           int type);

 private:
  cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> impl_;

  // A single stream orders uploads, computation, and downloads.
  mutable cv::cuda::Stream stream_;

  // // Pinned host image buffers.
  // mutable cv::cuda::HostMem h_prev_img_;
  // mutable cv::cuda::HostMem h_next_img_;

  // // Pinned host point buffers.
  // mutable cv::cuda::HostMem h_prev_points_;
  // mutable cv::cuda::HostMem h_forward_points_;
  // mutable cv::cuda::HostMem h_reverse_points_;

  // // Pinned host status and error buffers.
  // mutable cv::cuda::HostMem h_forward_status_;
  // mutable cv::cuda::HostMem h_forward_error_;
  // mutable cv::cuda::HostMem h_reverse_status_;
  // mutable cv::cuda::HostMem h_reverse_error_;

  // // Reusable device image buffers.
  // mutable cv::cuda::GpuMat d_prev_img_;
  // mutable cv::cuda::GpuMat d_next_img_;

  // // Reusable device point buffers.
  // mutable cv::cuda::GpuMat d_prev_points_;
  // mutable cv::cuda::GpuMat d_forward_points_;
  // mutable cv::cuda::GpuMat d_reverse_points_;

  // // Reusable device status and error buffers.
  // mutable cv::cuda::GpuMat d_forward_status_;
  // mutable cv::cuda::GpuMat d_forward_error_;
  // mutable cv::cuda::GpuMat d_reverse_status_;
  // mutable cv::cuda::GpuMat d_reverse_error_;

  //  mutable cv::cuda::Stream stream_;

  // Reusable GPU image buffers.
  mutable cv::cuda::GpuMat d_prev_img_;
  mutable cv::cuda::GpuMat d_next_img_;

  // Reusable GPU point buffers.
  mutable cv::cuda::GpuMat d_prev_points_;
  mutable cv::cuda::GpuMat d_forward_points_;
  mutable cv::cuda::GpuMat d_reverse_points_;

  // Reusable GPU status and error buffers.
  mutable cv::cuda::GpuMat d_forward_status_;
  mutable cv::cuda::GpuMat d_forward_error_;
  mutable cv::cuda::GpuMat d_reverse_status_;
  mutable cv::cuda::GpuMat d_reverse_error_;
};

}  // namespace dyno2
