#include <glog/logging.h>

#include "dynosam_nn/cuda_utils.hpp"

namespace dyno {

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
  return static_cast<cudaStream_t>(getCvStream().cudaPtr());
}
cv::cuda::Stream CudaStreamPool::getCvStream(std::size_t stream_id) const {
  return streams_[stream_id % streams_.size()];
}

std::size_t CudaStreamPool::poolSize() const noexcept {
  return streams_.size();
}

}  // namespace dyno
