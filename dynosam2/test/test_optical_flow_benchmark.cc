#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <random>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using Duration = std::chrono::duration<double, std::milli>;

struct BenchmarkConfig {
  int width;
  int height;
  int feature_count;
};

struct TimingStats {
  double median_ms = 0.0;
  double p95_ms = 0.0;
  double mean_ms = 0.0;

  double fps() const { return median_ms > 0.0 ? 1000.0 / median_ms : 0.0; }
};

struct BenchmarkResult {
  BenchmarkConfig config;
  TimingStats cpu;
  TimingStats gpu_resident;
  TimingStats gpu_end_to_end;
};

struct SyntheticSequence {
  cv::Mat prev;
  cv::Mat next;
  std::vector<cv::Point2f> points;
};

TimingStats calculateStats(std::vector<double> samples) {
  TimingStats stats;

  if (samples.empty()) {
    return stats;
  }

  std::sort(samples.begin(), samples.end());

  stats.mean_ms = std::accumulate(samples.begin(), samples.end(), 0.0) /
                  static_cast<double>(samples.size());

  // Nearest-rank percentile.
  const auto percentile = [&samples](double p) {
    const std::size_t index = static_cast<std::size_t>(std::ceil(
                                  p * static_cast<double>(samples.size()))) -
                              1;

    return samples[std::min(index, samples.size() - 1)];
  };

  stats.median_ms = percentile(0.50);
  stats.p95_ms = percentile(0.95);

  return stats;
}

SyntheticSequence makeSyntheticSequence(const BenchmarkConfig& config) {
  SyntheticSequence sequence;

  // Reproducible texture, independent of the benchmark iteration.
  cv::Mat noise(config.height, config.width, CV_8UC1);
  cv::RNG rng(0x12345678);

  rng.fill(noise, cv::RNG::UNIFORM, 0, 256);
  cv::GaussianBlur(noise, sequence.prev, cv::Size(3, 3), 0.8);

  // Introduce known image motion.
  const cv::Mat affine =
      (cv::Mat_<double>(2, 3) << 1.0, 0.0, 2.0, 0.0, 1.0, 1.0);

  cv::warpAffine(sequence.prev, sequence.next, affine, sequence.prev.size(),
                 cv::INTER_LINEAR, cv::BORDER_REFLECT101);

  // Generate deterministic points away from image boundaries.
  std::mt19937 point_rng(0xABCDEF);
  std::uniform_real_distribution<float> x_dist(
      30.0f, static_cast<float>(config.width - 30));
  std::uniform_real_distribution<float> y_dist(
      30.0f, static_cast<float>(config.height - 30));

  sequence.points.reserve(static_cast<std::size_t>(config.feature_count));

  for (int i = 0; i < config.feature_count; ++i) {
    sequence.points.emplace_back(x_dist(point_rng), y_dist(point_rng));
  }

  return sequence;
}

cv::Mat pointsToMat(const std::vector<cv::Point2f>& points) {
  // OpenCV CUDA sparse LK requires a 1 x N CV_32FC2 matrix.
  cv::Mat mat(1, static_cast<int>(points.size()), CV_32FC2);

  if (!points.empty()) {
    std::memcpy(mat.ptr<cv::Point2f>(), points.data(),
                points.size() * sizeof(cv::Point2f));
  }

  return mat;
}

std::vector<cv::Point2f> matToPoints(const cv::Mat& mat) {
  EXPECT_EQ(mat.type(), CV_32FC2);
  EXPECT_EQ(mat.rows, 1);

  std::vector<cv::Point2f> points(static_cast<std::size_t>(mat.cols));

  if (!points.empty()) {
    std::memcpy(points.data(), mat.ptr<cv::Point2f>(),
                points.size() * sizeof(cv::Point2f));
  }

  return points;
}

TimingStats benchmarkCpu(const SyntheticSequence& sequence,
                         int warmup_iterations, int measured_iterations) {
  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(measured_iterations));

  cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30,
                            0.01);

  std::vector<cv::Point2f> next_points;
  std::vector<unsigned char> status;
  std::vector<float> errors;

  const auto run_once = [&]() {
    next_points.clear();
    status.clear();
    errors.clear();

    cv::calcOpticalFlowPyrLK(sequence.prev, sequence.next, sequence.points,
                             next_points, status, errors, cv::Size(21, 21), 3,
                             criteria);
  };

  for (int i = 0; i < warmup_iterations; ++i) {
    run_once();
  }

  for (int i = 0; i < measured_iterations; ++i) {
    const auto start = Clock::now();
    run_once();
    const auto end = Clock::now();

    samples.push_back(Duration(end - start).count());
  }

  return calculateStats(std::move(samples));
}

struct GpuTimingStats {
  TimingStats resident;
  TimingStats end_to_end;
};

GpuTimingStats benchmarkGpu(const SyntheticSequence& sequence,
                            int warmup_iterations, int measured_iterations) {
  GpuTimingStats result;

  auto lk =
      cv::cuda::SparsePyrLKOpticalFlow::create(cv::Size(21, 21), 3, 30, false);

  cv::cuda::Stream stream;

  // Allocate and upload the resident working set once.
  cv::cuda::GpuMat d_prev;
  cv::cuda::GpuMat d_next;
  cv::cuda::GpuMat d_prev_points;
  cv::cuda::GpuMat d_next_points;
  cv::cuda::GpuMat d_status;
  cv::cuda::GpuMat d_errors;

  const cv::Mat host_points = pointsToMat(sequence.points);

  d_prev.upload(sequence.prev, stream);
  d_next.upload(sequence.next, stream);
  d_prev_points.upload(host_points, stream);
  stream.waitForCompletion();

  // Warm up CUDA kernels and internal buffers.
  for (int i = 0; i < warmup_iterations; ++i) {
    lk->calc(d_prev, d_next, d_prev_points, d_next_points, d_status, d_errors,
             stream);

    stream.waitForCompletion();
  }

  std::vector<double> resident_samples;
  resident_samples.reserve(static_cast<std::size_t>(measured_iterations));

  // GPU-resident: time LK execution and completion, excluding transfers.
  for (int i = 0; i < measured_iterations; ++i) {
    const auto start = Clock::now();

    lk->calc(d_prev, d_next, d_prev_points, d_next_points, d_status, d_errors,
             stream);

    stream.waitForCompletion();

    const auto end = Clock::now();
    resident_samples.push_back(Duration(end - start).count());
  }

  result.resident = calculateStats(std::move(resident_samples));

  // Preallocate host buffers to avoid measuring repeated host allocations.
  cv::Mat host_next_points(1, static_cast<int>(sequence.points.size()),
                           CV_32FC2);
  cv::Mat host_status(1, static_cast<int>(sequence.points.size()), CV_8UC1);

  std::vector<double> end_to_end_samples;
  end_to_end_samples.reserve(static_cast<std::size_t>(measured_iterations));

  // Warm up the complete transfer + compute + retrieval path.
  for (int i = 0; i < warmup_iterations; ++i) {
    d_prev.upload(sequence.prev, stream);
    d_next.upload(sequence.next, stream);
    d_prev_points.upload(host_points, stream);

    lk->calc(d_prev, d_next, d_prev_points, d_next_points, d_status, d_errors,
             stream);

    d_next_points.download(host_next_points, stream);
    d_status.download(host_status, stream);
    stream.waitForCompletion();
  }

  // End-to-end: both images and points are uploaded on each iteration.
  // Download tracked points and status, which the CPU frontend needs.
  for (int i = 0; i < measured_iterations; ++i) {
    const auto start = Clock::now();

    d_prev.upload(sequence.prev, stream);
    d_next.upload(sequence.next, stream);
    d_prev_points.upload(host_points, stream);

    lk->calc(d_prev, d_next, d_prev_points, d_next_points, d_status, d_errors,
             stream);

    d_next_points.download(host_next_points, stream);
    d_status.download(host_status, stream);

    stream.waitForCompletion();

    const auto end = Clock::now();
    end_to_end_samples.push_back(Duration(end - start).count());
  }

  result.end_to_end = calculateStats(std::move(end_to_end_samples));

  return result;
}

void printHeader() {
  std::cout
      << "\n"
      << "Sparse Pyramidal LK: CPU vs CUDA\n"
      << "All latencies in milliseconds; throughput uses median latency.\n\n"
      << std::left << std::setw(13) << "Resolution" << std::right
      << std::setw(9) << "Points" << std::setw(12) << "CPU med" << std::setw(12)
      << "CPU p95" << std::setw(12) << "GPU med" << std::setw(12) << "GPU p95"
      << std::setw(12) << "GPU E2E" << std::setw(12) << "E2E p95"
      << std::setw(12) << "Speedup"
      << "\n"
      << std::string(104, '-') << "\n";
}

void printResult(const BenchmarkResult& result) {
  const auto& c = result.config;

  const double speedup = result.cpu.median_ms / result.gpu_end_to_end.median_ms;

  const std::string resolution =
      std::to_string(c.width) + "x" + std::to_string(c.height);

  std::cout << std::left << std::setw(13) << resolution << std::right
            << std::setw(9) << c.feature_count << std::fixed
            << std::setprecision(3) << std::setw(12) << result.cpu.median_ms
            << std::setw(12) << result.cpu.p95_ms << std::setw(12)
            << result.gpu_resident.median_ms << std::setw(12)
            << result.gpu_resident.p95_ms << std::setw(12)
            << result.gpu_end_to_end.median_ms << std::setw(12)
            << result.gpu_end_to_end.p95_ms << std::setw(11) << speedup << "x"
            << "\n";
}

TEST(OpticalFlowBenchmark, CompareCpuAndCudaSparsePyrLK) {
  const int warmup_iterations = 10;
  const int measured_iterations = 100;

  const std::vector<BenchmarkConfig> configs = {
      {640, 480, 100},  {640, 480, 500},  {640, 480, 1000},  {640, 480, 2000},
      {848, 480, 100},  {848, 480, 500},  {848, 480, 1000},  {848, 480, 2000},
      {1280, 720, 100}, {1280, 720, 500}, {1280, 720, 1000}, {1280, 720, 2000},
  };

  if (cv::cuda::getCudaEnabledDeviceCount() <= 0) {
    GTEST_SKIP()
        << "No CUDA-enabled OpenCV device available. "
        << "Build OpenCV with CUDA and cudaoptflow to run this benchmark.";
  }

  cv::cuda::setDevice(0);

  printHeader();

  std::vector<BenchmarkResult> results;
  results.reserve(configs.size());

  for (const auto& config : configs) {
    const SyntheticSequence sequence = makeSyntheticSequence(config);

    const TimingStats cpu =
        benchmarkCpu(sequence, warmup_iterations, measured_iterations);

    const GpuTimingStats gpu =
        benchmarkGpu(sequence, warmup_iterations, measured_iterations);

    BenchmarkResult result{config, cpu, gpu.resident, gpu.end_to_end};

    printResult(result);
    results.push_back(result);
  }

  std::cout << "\n"
            << "GPU throughput is 1000 / median latency.\n"
            << "Speedup = CPU median / GPU end-to-end median.\n"
            << "Speedup > 1 means GPU end-to-end is faster.\n"
            << "GPU resident excludes transfers; GPU E2E includes uploads, "
               "LK, synchronization and output downloads.\n\n";

  // The benchmark reports results rather than enforcing a speedup threshold.
  // Performance depends on the device, OpenCV build and workload.
  SUCCEED();
}

}  // namespace
