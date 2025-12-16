#include "dynosam_nn/CudaUtils.hpp"

#include <glog/logging.h>

#include <opencv4/opencv2/core/cuda_stream_accessor.hpp>

namespace dyno {

CudaTimeGenerator::CudaTimeGenerator(cudaStream_t stream) : stream_(stream) {
  CUDA_API_CALL(cudaEventCreate(&start_event_));
  CUDA_API_CALL(cudaEventCreate(&stop_event_));
}

CudaTimeGenerator::~CudaTimeGenerator() {
  CUDA_API_CALL(cudaEventCreate(&start_event_));
  CUDA_API_CALL(cudaEventCreate(&stop_event_));
}

void CudaTimeGenerator::onStart() {
  CUDA_API_CALL(cudaEventRecord(start_event_, stream_));
}

double CudaTimeGenerator::calcDelta() const {
  // NO stream?
  CUDA_API_CALL(cudaEventRecord(stop_event_, stream_));

  CUDA_API_CALL(cudaEventSynchronize(stop_event_));
  float time_ms;
  CUDA_API_CALL(cudaEventElapsedTime(&time_ms, start_event_, stop_event_));

  /*
   * Note to future self and others:
   * according to NVIDIA, the elapsed time is expressed in ms,
   * with 0.5usec resolution.
   */
  return static_cast<double>(time_ms * 1.0e6);  // go from ms to ns
}

GpuTimingStats::GpuTimingStats(const std::string& tag, cudaStream_t stream,
                               int glog_level, bool construct_stopped)
    : This(CudaTimeGenerator(stream), tag, glog_level, construct_stopped) {}

GpuTimingStats::GpuTimingStats(const std::string& tag,
                               cv::cuda::Stream cv_stream, int glog_level,
                               bool construct_stopped)
    : This(CudaTimeGenerator(cv::cuda::StreamAccessor::getStream(cv_stream)),
           tag, glog_level, construct_stopped) {}

CudaStreamPool::CudaStreamPool(std::size_t pool_size, unsigned int flags) {
  CHECK_GT(pool_size, 0) << "Stream pool size must be greater than zero";
  streams_.reserve(pool_size);
  std::generate_n(std::back_inserter(streams_), pool_size,
                  [flags]() { return cv::cuda::Stream(flags); });
}

cv::cuda::Stream CudaStreamPool::getCvStream() const noexcept {
  return streams_[(next_stream.fetch_add(1, std::memory_order_relaxed)) %
                  streams_.size()];
}

cv::cuda::Stream& CudaStreamPool::getCvStream() noexcept {
  return streams_[(next_stream.fetch_add(1, std::memory_order_relaxed)) %
                  streams_.size()];
}

cudaStream_t CudaStreamPool::getCudaStream() const {
  return cv::cuda::StreamAccessor::getStream(getCvStream());
}

cv::cuda::Stream CudaStreamPool::getCvStream(std::size_t stream_id) const {
  return streams_[stream_id % streams_.size()];
}

std::size_t CudaStreamPool::poolSize() const noexcept {
  return streams_.size();
}

}  // namespace dyno
