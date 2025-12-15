#pragma once

#include <cuda_runtime_api.h>

#include <atomic>
#include <mutex>
#include <opencv2/core/cuda.hpp>

#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)                                                  \
  {                                                                          \
    cudaError_t error_code = callstr;                                        \
    if (error_code != cudaSuccess) {                                         \
      std::cerr << "CUDA error " << cudaGetErrorString(error_code) << " at " \
                << __FILE__ << ":" << __LINE__;                              \
      assert(0);                                                             \
    }                                                                        \
  }
#endif  // CUDA_CHECK

namespace dyno {

// class ThreadSafeCvStream {

// private:
//   cv::cuda::Stream stream_;
//   mutable std::mutex mutex_;

// };

class CudaStreamPool {
 public:
  //! Default stream pool size
  static constexpr std::size_t default_size{16};
  static constexpr unsigned int default_flags = cudaStreamDefault;

  explicit CudaStreamPool(std::size_t pool_size = default_size,
                          unsigned int flags = cudaStreamDefault);
  ~CudaStreamPool() = default;

  CudaStreamPool(CudaStreamPool&&) = delete;
  CudaStreamPool(CudaStreamPool const&) = delete;
  CudaStreamPool& operator=(CudaStreamPool&&) = delete;
  CudaStreamPool& operator=(CudaStreamPool const&) = delete;

  cv::cuda::Stream getCvStream() const noexcept;
  cv::cuda::Stream& getCvStream() noexcept;
  cudaStream_t getCudaStream() const;
  cv::cuda::Stream getCvStream(std::size_t stream_id) const;

  std::size_t poolSize() const noexcept;

 private:
  std::vector<cv::cuda::Stream> streams_;
  mutable std::atomic_size_t next_stream{};
};

}  // namespace dyno
