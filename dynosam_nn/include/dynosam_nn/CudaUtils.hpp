#pragma once

#include <cuda_runtime_api.h>

#include <atomic>
#include <mutex>
#include <opencv2/core/cuda.hpp>

#include "dynosam_common/utils/TimingStats.hpp"

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

#define CUDA_API_CALL(__CALL__)                                          \
  do {                                                                   \
    const cudaError_t a = __CALL__;                                      \
    if (a != cudaSuccess) {                                              \
      std::cerr << "CUDA Error: " << cudaGetErrorString(a)               \
                << " (err_num=" << a << ")"                              \
                << "at: " << __FILE__ << " | " << __LINE__ << std::endl; \
      cudaDeviceReset();                                                 \
      assert(0);                                                         \
    }                                                                    \
  } while (0)

namespace dyno {

struct CudaTimeGenerator {
  cudaStream_t stream_;
  cudaEvent_t start_event_;
  cudaEvent_t stop_event_;

  CudaTimeGenerator(cudaStream_t stream);
  ~CudaTimeGenerator();

  void onStart();
  void onStop() {}

  double calcDelta() const;
};

class GpuTimingStats
    : public utils::BaseTimingStatsCollector<CudaTimeGenerator> {
 public:
  using This = utils::BaseTimingStatsCollector<CudaTimeGenerator>;
  GpuTimingStats(const std::string& tag, cudaStream_t stream = 0,
                 int glog_level = 0, bool construct_stopped = false);

  GpuTimingStats(const std::string& tag, cv::cuda::Stream cv_stream,
                 int glog_level = 0, bool construct_stopped = false);
};

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
