#include "dynosam2/frontend/SparseFlow.hpp"

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaoptflow.hpp>

namespace dyno2 {

OpticalFlowLK_CPU::Result OpticalFlowLK_CPU::compute(
    const dyno::WrappedRGBMono& prev_img, const dyno::WrappedRGBMono& next_img,
    const std::vector<cv::Point2f>& prev_points) const {
  const auto prev_pyr =
      prev_img.computeImagePyramid(win_size, max_level).levels;

  const auto next_pyr =
      next_img.computeImagePyramid(win_size, max_level).levels;

  CHECK_EQ(prev_pyr.size(), next_pyr.size());

  ImplResult forward;
  forward.next = prev_points;
  forward.prev = prev_points;

  cv::calcOpticalFlowPyrLK(prev_pyr, next_pyr, forward.prev, forward.next,
                           forward.status, forward.error, win_size, max_level,
                           criteria, 0);

  CHECK_EQ(forward.size(), prev_points.size());

  ImplResult reverse;
  // now do reverse flow
  reverse.status.resize(forward.size());
  reverse.error.resize(forward.size());
  reverse.next = forward.next;
  reverse.prev = forward.next;

  cv::calcOpticalFlowPyrLK(next_pyr, prev_pyr, forward.next, reverse.next,
                           reverse.status, reverse.error, win_size, max_level,
                           criteria, cv::OPTFLOW_USE_INITIAL_FLOW);

  return OpticalFlowLK_CPU::Result{forward, reverse};
}

OpticalFlowLK_GPU::OpticalFlowLK_GPU(int iterations)
    : OpticalFlowLKBase(iterations) {
  impl_ =
      cv::cuda::SparsePyrLKOpticalFlow::create(win_size, max_level, iterations);
}

// OpticalFlowLK_GPU::Result OpticalFlowLK_GPU::compute(
//     const dyno::WrappedRGBMono& prev_img, const dyno::WrappedRGBMono&
//     next_img, const std::vector<cv::Point2f>& prev_points) const
// {
//   const cv::Mat& prev = prev_img;
//   const cv::Mat& next = next_img;

//   //TODO: pinned memory
//   cv::cuda::GpuMat d_prev(prev);
//   cv::cuda::GpuMat d_next(next);

//   cv::cuda::GpuMat d_next_pts(prev.size(), CV_32FC2);
//   cv::cuda::GpuMat d_status;
//   cv::cuda::GpuMat d_err;

//   cv::cuda::GpuMat d_previous_pts(prev.size(), CV_32FC2);
//   d_previous_pts.upload(prev_points);

//   impl_->setUseInitialFlow(false);
//   impl_->calc(d_prev, d_next, d_previous_pts, d_next_pts, d_status, d_err);

//   cv::cuda::GpuMat d_reverse_next_pts = d_previous_pts;
//   cv::cuda::GpuMat d_reverse_status;
//   cv::cuda::GpuMat d_reverse_err;

//   impl_->setUseInitialFlow(true);
//   impl_->calc(d_next, d_prev, d_next_pts, d_reverse_next_pts,
//   d_reverse_status, d_reverse_err);

//   ImplResult forward;
//   forward.status.resize(d_status.cols);
//   d_status.download(forward.status);

//   forward.error.resize(d_err.cols);
//   d_err.download(forward.error);

//   forward.prev = prev_points;

//   forward.next.resize(d_next_pts.cols);
//   d_next_pts.download(forward.next);

//   ImplResult reverse;
//   reverse.status.resize(d_status.cols);
//   d_status.download(reverse.status);

//   reverse.error.resize(d_err.cols);
//   d_err.download(reverse.error);

//   reverse.prev = forward.next;

//   reverse.next.resize(d_reverse_next_pts.cols);
//   d_reverse_next_pts.download(reverse.next);

//   return OpticalFlowLK_GPU::Result{forward, reverse};

// }

void OpticalFlowLK_GPU::ensureBuffer(cv::cuda::GpuMat& buffer, int rows,
                                     int cols, int type) {
  if (rows <= 0 || cols <= 0) {
    return;
  }

  if (buffer.empty() || buffer.type() != type || buffer.rows < rows ||
      buffer.cols < cols) {
    buffer.create(rows, cols, type);
  }
}

OpticalFlowLK_GPU::Result OpticalFlowLK_GPU::compute(
    const dyno::WrappedRGBMono& prev_img, const dyno::WrappedRGBMono& next_img,
    const std::vector<cv::Point2f>& prev_points) const {
  const cv::Mat prev = dyno::ImageType::RGBMono::toMono(prev_img);
  const cv::Mat next = dyno::ImageType::RGBMono::toMono(next_img);
  ;

  if (prev.empty() || next.empty()) {
    throw std::invalid_argument("OpticalFlowLK_GPU received an empty image");
  }

  if (prev.size() != next.size() || prev.type() != next.type()) {
    throw std::invalid_argument(
        "OpticalFlowLK_GPU requires matching image sizes and types");
  }

  const int n = static_cast<int>(prev_points.size());

  ImplResult forward;
  ImplResult reverse;

  forward.prev = prev_points;

  if (n == 0) {
    return Result{std::move(forward), std::move(reverse)};
  }

  // Allocate only when the image dimensions or point count grow.
  ensureBuffer(d_prev_img_, prev.rows, prev.cols, prev.type());
  ensureBuffer(d_next_img_, next.rows, next.cols, next.type());

  // ensureBuffer(d_prev_points_, n, 1, CV_32FC2);
  // ensureBuffer(d_forward_points_, n, 1, CV_32FC2);
  // ensureBuffer(d_reverse_points_, n, 1, CV_32FC2);

  // ensureBuffer(d_forward_status_, n, 1, CV_8UC1);
  // ensureBuffer(d_forward_error_, n, 1, CV_32FC1);

  // ensureBuffer(d_reverse_status_, n, 1, CV_8UC1);
  // ensureBuffer(d_reverse_error_, n, 1, CV_32FC1);

  ensureBuffer(d_prev_points_, 1, n, CV_32FC2);
  ensureBuffer(d_forward_points_, 1, n, CV_32FC2);
  ensureBuffer(d_reverse_points_, 1, n, CV_32FC2);

  ensureBuffer(d_forward_status_, 1, n, CV_8UC1);
  ensureBuffer(d_forward_error_, 1, n, CV_32FC1);

  ensureBuffer(d_reverse_status_, 1, n, CV_8UC1);
  ensureBuffer(d_reverse_error_, 1, n, CV_32FC1);

  // Upload images and points on the dedicated CUDA stream.
  d_prev_img_.upload(prev, stream_);
  d_next_img_.upload(next, stream_);
  d_prev_points_.upload(prev_points, stream_);

  // Forward optical flow: previous frame -> next frame.
  impl_->setUseInitialFlow(false);

  impl_->calc(d_prev_img_, d_next_img_, d_prev_points_, d_forward_points_,
              d_forward_status_, d_forward_error_, stream_);

  // Reverse optical flow: next frame -> previous frame.
  // Initial estimates are the original previous-frame points.
  d_reverse_points_.upload(prev_points, stream_);

  impl_->setUseInitialFlow(true);

  impl_->calc(d_next_img_, d_prev_img_, d_forward_points_, d_reverse_points_,
              d_reverse_status_, d_reverse_error_, stream_);

  // Download only when the CPU needs the results.
  // The stream overloads queue the transfers; waitForCompletion()
  // ensures the CPU buffers are safe to read before returning.
  forward.next.resize(n);
  forward.status.resize(n);
  forward.error.resize(n);

  reverse.prev.resize(n);
  reverse.next.resize(n);
  reverse.status.resize(n);
  reverse.error.resize(n);

  d_forward_points_.download(forward.next, stream_);
  d_forward_status_.download(forward.status, stream_);
  d_forward_error_.download(forward.error, stream_);

  d_reverse_points_.download(reverse.next, stream_);
  d_reverse_status_.download(reverse.status, stream_);
  d_reverse_error_.download(reverse.error, stream_);

  stream_.waitForCompletion();

  // The reverse pass starts from the forward-tracked locations.
  reverse.prev = forward.next;

  return Result{std::move(forward), std::move(reverse)};
}

}  // namespace dyno2
