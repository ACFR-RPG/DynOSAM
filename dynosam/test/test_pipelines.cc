/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris
 (jesse.morris@sydney.edu.au)
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a
 copy
 *   of this software and associated documentation files (the "Software"), to
 deal
 *   in the Software without restriction, including without limitation the
 rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:

 *   The above copyright notice and this permission notice shall be included in
 all
 *   copies or substantial portions of the Software.

 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE
 *   SOFTWARE.
 */
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <type_traits>
#include <variant>

#include "dynosam/backend/RegularBackendModule.hpp"
#include "dynosam/pipeline/PipelineBase.hpp"
#include "dynosam_common/ModuleBase.hpp"
#include "dynosam_common/utils/SafeCast.hpp"
#include "internal/helpers.hpp"
#include "internal/simulator.hpp"

using namespace dyno;

namespace fs = std::filesystem;
class FrontendWithFiles : public ::testing::Test {
 public:
  FrontendWithFiles() {}

 protected:
  virtual void SetUp() { fs::create_directory(sandbox); }
  virtual void TearDown() { fs::remove_all(sandbox); }

  const fs::path sandbox{"/tmp/sandbox_json_backend"};
};

// TODO: why is this in the backend testing... oh well...
// TODO: bring back after VisionImuPacket is fixed with JSON!!
TEST_F(FrontendWithFiles, testLoadingFrontendWithJson) {
  // using namespace dyno;
  // auto scenario = dyno_testing::makeDefaultScenario();

  // std::map<FrameId, RGBDInstanceOutputPacket::Ptr> rgbd_output;

  // for (size_t i = 0; i < 10; i++) {
  //   auto output = scenario.getOutput(i);
  //   rgbd_output.insert({i, output.first});
  // }

  // fs::path tmp_bison_path = sandbox / "simple_bison.bson";
  // std::string tmp_bison_path_str = tmp_bison_path;

  // JsonConverter::WriteOutJson(rgbd_output, tmp_bison_path_str,
  //                             JsonConverter::Format::BSON);

  // FrontendOfflinePipeline<RegularBackendModule::ModuleTraits> offline_backed(
  //     "offline-rgbdfrontend", tmp_bison_path_str);

  // ThreadsafeQueue<FrontendOutputPacketBase::ConstPtr> queue;
  // offline_backed.registerOutputQueue(&queue);

  // int consumed = 0;
  // while (offline_backed.spinOnce()) {
  //   FrontendOutputPacketBase::ConstPtr base;
  //   bool result = queue.popBlocking(base);

  //   EXPECT_TRUE(result);
  //   EXPECT_TRUE(base != nullptr);
  //   RGBDInstanceOutputPacket::ConstPtr derived =
  //       safeCast<FrontendOutputPacketBase, RGBDInstanceOutputPacket>(base);
  //   EXPECT_TRUE(derived != nullptr);

  //   // this value indexed by consume will be wrong if pop blocking doesnt
  //   work EXPECT_EQ(*derived, *rgbd_output.at(consumed)); consumed++;
  // }

  // // should spin until the pipeline is shutdown which happens when we dont
  // have
  // // any more data
  // offline_backed.spin();
  // EXPECT_TRUE(offline_backed.isShutdown());
  // EXPECT_EQ(consumed, 10);  // we should habve processed data 10 tiuems
}

// https://www.cppstories.com/2018/09/visit-variants/
template <class... Ts>
struct overload : Ts... {
  using Ts::operator()...;
};
template <class... Ts>
overload(Ts...) -> overload<Ts...>;  // line not needed in C++20

// //must be T& not const T&
// template<typename T>
// struct Visitor {

//     void process(const T& t) {
//         LOG(INFO) << t;
//     }

//     void operator()(T& t) { process(t);}

// };

// template<typename T> struct is_variant : std::false_type {};

// template<typename ...Variants>
// struct is_variant<std::variant<Variants...>> : std::true_type {};

// template<typename T>
// inline constexpr bool is_variant_v=is_variant<T>::value;

// // main lookup logic of looking up a type in a list.
// //https://www.appsloveworld.com/cplus/100/22/how-do-i-check-if-an-stdvariant-can-hold-a-certain-type
// template<typename T, typename... Variants>
// struct isoneof : public std::false_type {};

// template<typename T, typename FrontVariant, typename... RestVariants>
// struct isoneof<T, FrontVariant, RestVariants...> : public
//   std::conditional<
//     std::is_same<T, FrontVariant>::value,
//     std::true_type,
//     isoneof<T, RestVariants...>
//   >::type {};

// // convenience wrapper for std::variant<>.
// template<typename T, typename Variants>
// struct isvariantmember : public std::false_type {};

// template<typename T, typename... Variants>
// struct isvariantmember<T, std::variant<Variants...>> : public isoneof<T,
// Variants...> {};

// template<typename T, typename... Variants>
// inline constexpr bool isvariantmember_v = isvariantmember<T,
// Variants...>::value;

// template<typename IPacket, typename OPacket, typename MInput = IPacket,
// typename MOutput = OPacket> class MBase { public:
//     using InputPacket = IPacket;
//     using OutputPacket = OPacket;
//     using ModuleInput = MInput;
//     using ModuleOutput = MOutput;

// public:
//     constexpr static bool IsInputPacketVariant = is_variant_v<IPacket>;
//     constexpr static bool IsOutputPacketVariant = is_variant_v<OutputPacket>;

//     constexpr static bool IsModuleInputVariant = is_variant_v<ModuleInput>;
//     constexpr static bool IsModuleOutputVariant = is_variant_v<ModuleOutput>;

//     //either InputPacket is a variant or the InputPacket and ModuleInput are
//     the same
//     //and no casting is required
//     static_assert(IsInputPacketVariant ||
//     std::is_same_v<ModuleInput,InputPacket>,
//         "InputPacket specified by IPacket is not a std::variant or
//         ModuleInput and InputPacket are not the same type");

//     static_assert(IsOutputPacketVariant ||
//     std::is_same_v<ModuleOutput,OutputPacket>,
//         "OutputPacket specified by OPacket is not a std::variant or
//         ModuleOutput and OutputPacket are not the same type");

//     // //if IsInputPacketVariant, then ModuleInput cannot also be a variant!!
//     static_assert(!(IsInputPacketVariant && IsModuleInputVariant),
//         "InputPacket is a variant, so ModuleInput cannot also be a variant,
//         but a type within the variant");

//     static_assert(!(IsOutputPacketVariant && IsModuleOutputVariant),
//         "OutputPacket is a variant, so ModuleOutput cannot also be a variant,
//         but a type within the variant");

//     static_assert(isvariantmember_v<ModuleInput, InputPacket> ||
//     std::is_same_v<ModuleInput,InputPacket>,
//         "If InputPacket is a variant, ModuleInput is not a type within the
//         variant, or ModuleInput and InputPacket types are not the same");

//     static_assert(isvariantmember_v<ModuleOutput, OutputPacket> ||
//     std::is_same_v<ModuleOutput,OutputPacket>,
//         "If InputPacket is a variant, ModuleOutput is not a type within the
//         variant, or ModuleOutput and InputPacket types are not the same");

// public:
//     ModuleInput cast(const InputPacket& packet) const {
//         if constexpr (IsInputPacketVariant) {
//             try {
//                 //packet should be a variant the ModuleOutput is a type
//                 within the std::variant
//                 //and the static asserts guarantee that ModuleInput is also
//                 not a variant return std::get<ModuleInput>(packet);
//             }
//             catch(const std::bad_variant_access& e) {
//                 throw std::runtime_error("Bad access");
//             }
//         }
//         else {
//            //if InputPacket is not a variant, then the static asserts
//            guarantee that InputPacket == ModuleInput; return packet;
//         }
//     }

//     virtual ModuleOutput process(const ModuleInput& input) {
//         LOG(INFO) << input;
//         return ModuleOutput{};
//     }

//     OutputPacket spinOnce(const InputPacket& packet) {
//         return process(cast(packet));
//     }

// };

// using VI = std::variant<double, std::string>;

// template<typename INPUT, typename OUTPUT>
// class Module : public MBase<VI, VI, INPUT, OUTPUT> {
// public:
//     using MBase<VI, VI, INPUT, OUTPUT>::process;
// };

// class DerivedModule : public Module<double, std::string> {

// public:
//     std::string process(const double& input) override {
//         LOG(INFO) << "Input" << input;
//         return "result";
//     }
// };

// TEST(PipelineBase, checkCompilation) {
//     using VariantInput = std::variant<double, int>;
//     //IPacket is variant so we need to specify ModuleInput
//     MBase<VariantInput, double, double>{};
//     MBase<int, double, int, double>{};
//     MBase<VariantInput, double, int, double>{};
//     // MBase<VariantInput, double, int, std::string>{};
//     // MBase<VariantInput, double, VariantInput, double>{};

//     // constexpr bool r = MBase<VariantInput, double, int,
//     double>::IsModuleInputVariantMember;
// }

// TEST(PipelineBase, moduleProcess) {
//     // using VariantInput = std::variant<double, std::string>;
//     // //IPacket is variant so we need to specify ModuleInput
//     // MBase<VariantInput, double, std::string> mb{};

//     // //this should cast the input to a double
//     // mb.spinOnce("hi");

//     DerivedModule dm;
//     dm.spinOnce("string");
// }

// TEST(PipelineBase, testWithVairant) {

//     using Var = std::variant<int, std::string>;
//     using VarPipeline = FunctionalSIMOPipelineModule<Var,
//     EmptyPayload>;

//     using VarModule = VariantModule<Var, EmptyPayload, std::string>;

//     VarPipeline::InputQueue input_queue;
//     VarPipeline::OutputQueue output_queue;
//     VarModule var_module;
//     // Visitor<std::string> string_visitor;
//     // Visitor<int> int_visitor;

//     VarPipeline p("var_module", &input_queue,
//         [&](const VarPipeline::InputConstSharedPtr& var_ptr) ->
//         VarPipeline::OutputConstSharedPtr {
//             // return var_module.spinOnce(var_ptr);
//             return var_module.spinOnce(var_ptr);

//         },
//         false);

//     // input_queue.push(std::make_shared<Var>((int)3));
//     input_queue.push(std::make_shared<Var>("string"));

//     p.spinOnce();
//     p.spinOnce();

// }

/*
 * Pipeline latency / overhead tests.
 *
 * These tests are intended to measure the overhead introduced by the
 * PipelineBase / PipelineModule / SIMOPipelineModule infrastructure.
 *
 * IMPORTANT:
 *   Run these tests in Release or RelWithDebInfo.
 *
 * Example:
 *   ./pipeline_test --gtest_filter=PipelineLatency*
 *
 * The tests print timing information but intentionally do not impose
 * hard performance thresholds. The purpose is to characterize the
 * overhead on the target machine.
 */

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <thread>
#include <vector>

#include "dynosam/pipeline/PipelineBase.hpp"

namespace dyno {
namespace {

using Clock = std::chrono::steady_clock;

struct TestInput {
  uint64_t sequence{0};
  Clock::time_point timestamp;
};

struct TestOutput {
  uint64_t sequence{0};
  Clock::time_point input_timestamp;
  Clock::time_point process_timestamp;
};

/**
 * Prevent the compiler from optimizing away the mock work.
 */
volatile double g_pipeline_benchmark_sink = 0.0;

/**
 * Deterministic CPU workload.
 *
 * This is deliberately not sleep_for(). A sleep-based workload makes it
 * difficult to resolve microsecond-scale pipeline overhead.
 */
inline void doMockWork(std::size_t iterations) {
  double value = 1.0;

  for (std::size_t i = 0; i < iterations; ++i) {
    value = value * 1.0000001 + 0.000001;
    value = std::sin(value);
  }

  g_pipeline_benchmark_sink += value;
}

/**
 * Convert a duration to nanoseconds.
 */
template <typename Duration>
double toNanoseconds(Duration duration) {
  return std::chrono::duration<double, std::nano>(duration).count();
}

/**
 * Simple statistics used for the printed benchmark results.
 */
struct TimingResult {
  double mean_ns{0.0};
  double min_ns{0.0};
  double max_ns{0.0};
  double p50_ns{0.0};
  double p95_ns{0.0};
  double p99_ns{0.0};
};

TimingResult calculateStats(std::vector<double> samples) {
  CHECK(!samples.empty());

  std::sort(samples.begin(), samples.end());

  TimingResult result;
  result.min_ns = samples.front();
  result.max_ns = samples.back();

  result.mean_ns = std::accumulate(samples.begin(), samples.end(), 0.0) /
                   static_cast<double>(samples.size());

  auto percentile = [&samples](double p) {
    const double index = p * static_cast<double>(samples.size() - 1);

    const std::size_t lower = static_cast<std::size_t>(std::floor(index));

    const std::size_t upper = static_cast<std::size_t>(std::ceil(index));

    if (lower == upper) {
      return samples[lower];
    }

    const double fraction = index - static_cast<double>(lower);

    return samples[lower] + fraction * (samples[upper] - samples[lower]);
  };

  result.p50_ns = percentile(0.50);
  result.p95_ns = percentile(0.95);
  result.p99_ns = percentile(0.99);

  return result;
}

void printTiming(const std::string& name, const TimingResult& result) {
  std::cout << std::left << std::setw(32) << name << " mean=" << std::setw(10)
            << result.mean_ns << " ns"
            << " p50=" << std::setw(10) << result.p50_ns << " ns"
            << " p95=" << std::setw(10) << result.p95_ns << " ns"
            << " p99=" << std::setw(10) << result.p99_ns << " ns"
            << " min=" << std::setw(10) << result.min_ns << " ns"
            << " max=" << result.max_ns << " ns" << '\n';
}

/**
 * A concrete pipeline used exclusively by the benchmark.
 */
class BenchmarkPipeline : public SIMOPipelineModule<TestInput, TestOutput> {
 public:
  using Base = SIMOPipelineModule<TestInput, TestOutput>;
  using InputConstSharedPtr = typename Base::InputConstSharedPtr;
  using OutputConstSharedPtr = typename Base::OutputConstSharedPtr;
  using InputQueue = typename Base::InputQueue;
  using OutputQueue = typename Base::OutputQueue;

  BenchmarkPipeline(const std::string& name, InputQueue* input_queue,
                    std::size_t work_iterations = 0, bool parallel_run = false)
      : Base(name, input_queue, parallel_run),
        work_iterations_(work_iterations) {}

  /**
   * Expose the actual process() implementation for the direct baseline.
   */
  OutputConstSharedPtr directProcess(const InputConstSharedPtr& input) {
    return process(input);
  }

  /**
   * Allow the benchmark to change the amount of mock work.
   */
  void setWorkIterations(std::size_t iterations) {
    work_iterations_ = iterations;
  }

 protected:
  OutputConstSharedPtr process(const InputConstSharedPtr& input) override {
    if (work_iterations_ > 0) {
      doMockWork(work_iterations_);
    }

    auto output = std::make_shared<TestOutput>();
    output->sequence = input->sequence;
    output->input_timestamp = input->timestamp;
    output->process_timestamp = Clock::now();

    return output;
  }

 private:
  std::size_t work_iterations_;
};

/**
 * Pipeline fixture.
 */
class PipelineLatencyTest : public ::testing::Test {
 protected:
  using InputQueue = PipelineTypeTraits<TestInput>::TypeQueue;

  using OutputQueue = PipelineTypeTraits<TestOutput>::TypeQueue;

  static constexpr std::size_t kWarmupPackets = 1000;
  static constexpr std::size_t kBenchmarkPackets = 10000;

  void SetUp() override {
    // The tests deliberately use a non-blocking pipeline unless explicitly
    // testing blocking behaviour.
    input_queue_ = std::make_unique<InputQueue>();
  }

  void TearDown() override {
    if (input_queue_) {
      input_queue_->shutdown();
    }

    output_queues_.clear();
  }

  std::shared_ptr<TestInput> makeInput(uint64_t sequence = 0) {
    auto input = std::make_shared<TestInput>();
    input->sequence = sequence;
    input->timestamp = Clock::now();
    return input;
  }

  void fillInputQueue(std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
      input_queue_->push(makeInput(i));
    }
  }

  std::unique_ptr<InputQueue> input_queue_;
  std::vector<std::unique_ptr<OutputQueue>> output_queues_;
};

/**
 * --------------------------------------------------------------------------
 * 1. DIRECT PROCESS BASELINE
 * --------------------------------------------------------------------------
 *
 * This measures:
 *
 *     make input
 *     virtual process()
 *     make output
 *
 * but does NOT include:
 *
 *     queue pop
 *     timing instrumentation
 *     PipelineModule::spinOnce()
 *     queue push
 *     callbacks
 *     atomics
 */
TEST_F(PipelineLatencyTest, DirectProcess) {
  constexpr std::size_t kWorkIterations = 0;

  BenchmarkPipeline pipeline("direct_process", input_queue_.get(),
                             kWorkIterations, false);

  auto input = makeInput();

  // Warm up.
  for (std::size_t i = 0; i < kWarmupPackets; ++i) {
    input->sequence = i;
    auto output = pipeline.directProcess(input);
    ASSERT_NE(output, nullptr);
  }

  std::vector<double> samples;
  samples.reserve(kBenchmarkPackets);

  for (std::size_t i = 0; i < kBenchmarkPackets; ++i) {
    input->sequence = i;

    const auto start = Clock::now();

    auto output = pipeline.directProcess(input);

    const auto end = Clock::now();

    ASSERT_NE(output, nullptr);

    samples.push_back(toNanoseconds(end - start));
  }

  const auto stats = calculateStats(std::move(samples));

  printTiming("Direct process", stats);
}

/**
 * --------------------------------------------------------------------------
 * 2. FULL PIPELINE WITH ZERO PROCESSING WORK
 * --------------------------------------------------------------------------
 *
 * This is the most important test.
 *
 * Compare this against DirectProcess.
 *
 * The difference approximates the pipeline framework overhead.
 */
TEST_F(PipelineLatencyTest, SpinOnceZeroWork) {
  BenchmarkPipeline pipeline("spin_zero_work", input_queue_.get(), 0, false);

  auto output_queue = std::make_unique<OutputQueue>();
  pipeline.registerOutputQueue(output_queue.get());

  // Warm up.
  fillInputQueue(kWarmupPackets);

  for (std::size_t i = 0; i < kWarmupPackets; ++i) {
    const auto result = pipeline.spinOnce();
    ASSERT_EQ(result, PipelineBase::ReturnCode::SUCCESS);
  }

  // Benchmark.
  fillInputQueue(kBenchmarkPackets);

  std::vector<double> samples;
  samples.reserve(kBenchmarkPackets);

  for (std::size_t i = 0; i < kBenchmarkPackets; ++i) {
    const auto start = Clock::now();

    const auto result = pipeline.spinOnce();

    const auto end = Clock::now();

    ASSERT_EQ(result, PipelineBase::ReturnCode::SUCCESS);

    samples.push_back(toNanoseconds(end - start));
  }

  const auto stats = calculateStats(std::move(samples));

  printTiming("spinOnce - zero work", stats);

  EXPECT_EQ(output_queue->size(), kBenchmarkPackets);
}

/**
 * --------------------------------------------------------------------------
 * 3. SMALL PROCESSING WORK
 * --------------------------------------------------------------------------
 *
 * Run the pipeline with increasing amounts of CPU work.
 *
 * This shows how the fixed pipeline overhead becomes less important as
 * actual processing becomes more expensive.
 */
TEST_F(PipelineLatencyTest, SpinOnceProcessingWork) {
  const std::vector<std::size_t> work_levels = {
      0, 1, 10, 100, 1000,
  };

  for (const std::size_t work : work_levels) {
    input_queue_->shutdown();

    // Create a fresh queue because shutdown() is permanent.
    input_queue_ = std::make_unique<InputQueue>();

    BenchmarkPipeline pipeline("spin_work", input_queue_.get(), work, false);

    auto output_queue = std::make_unique<OutputQueue>();
    pipeline.registerOutputQueue(output_queue.get());

    fillInputQueue(kWarmupPackets);

    for (std::size_t i = 0; i < kWarmupPackets; ++i) {
      ASSERT_EQ(pipeline.spinOnce(), PipelineBase::ReturnCode::SUCCESS);
    }

    fillInputQueue(kBenchmarkPackets);

    std::vector<double> samples;
    samples.reserve(kBenchmarkPackets);

    for (std::size_t i = 0; i < kBenchmarkPackets; ++i) {
      const auto start = Clock::now();

      ASSERT_EQ(pipeline.spinOnce(), PipelineBase::ReturnCode::SUCCESS);

      const auto end = Clock::now();

      samples.push_back(toNanoseconds(end - start));
    }

    const auto stats = calculateStats(std::move(samples));

    std::cout << "Mock work iterations = " << work << '\n';

    printTiming("  spinOnce", stats);
  }
}

/**
 * --------------------------------------------------------------------------
 * 4. OUTPUT FAN-OUT
 * --------------------------------------------------------------------------
 *
 * Measures the additional cost of pushing one packet to multiple queues.
 *
 * This is particularly relevant to your SIMO/MIMO use case.
 */
TEST_F(PipelineLatencyTest, OutputFanout) {
  const std::vector<std::size_t> output_counts = {
      1,
      2,
      4,
      8,
  };

  for (const std::size_t num_outputs : output_counts) {
    input_queue_->shutdown();
    input_queue_ = std::make_unique<InputQueue>();

    BenchmarkPipeline pipeline("fanout", input_queue_.get(), 0, false);

    output_queues_.clear();

    for (std::size_t i = 0; i < num_outputs; ++i) {
      auto queue = std::make_unique<OutputQueue>();

      pipeline.registerOutputQueue(queue.get());

      output_queues_.push_back(std::move(queue));
    }

    fillInputQueue(kWarmupPackets);

    for (std::size_t i = 0; i < kWarmupPackets; ++i) {
      ASSERT_EQ(pipeline.spinOnce(), PipelineBase::ReturnCode::SUCCESS);
    }

    fillInputQueue(kBenchmarkPackets);

    std::vector<double> samples;
    samples.reserve(kBenchmarkPackets);

    for (std::size_t i = 0; i < kBenchmarkPackets; ++i) {
      const auto start = Clock::now();

      ASSERT_EQ(pipeline.spinOnce(), PipelineBase::ReturnCode::SUCCESS);

      const auto end = Clock::now();

      samples.push_back(toNanoseconds(end - start));
    }

    const auto stats = calculateStats(std::move(samples));

    std::cout << "Number of output queues = " << num_outputs << '\n';

    printTiming("  spinOnce", stats);
  }
}

/**
 * --------------------------------------------------------------------------
 * 5. OUTPUT CALLBACK OVERHEAD
 * --------------------------------------------------------------------------
 *
 * Your implementation currently does:
 *
 *     for (OutputCallback cb : output_callbacks_)
 *
 * Notice that this copies the std::function on every invocation.
 *
 * This test quantifies the resulting overhead.
 */
TEST_F(PipelineLatencyTest, OutputCallbacks) {
  const std::vector<std::size_t> callback_counts = {
      0, 1, 2, 4, 8,
  };

  for (const std::size_t num_callbacks : callback_counts) {
    input_queue_->shutdown();
    input_queue_ = std::make_unique<InputQueue>();

    BenchmarkPipeline pipeline("callbacks", input_queue_.get(), 0, false);

    auto output_queue = std::make_unique<OutputQueue>();
    pipeline.registerOutputQueue(output_queue.get());

    std::atomic<uint64_t> callback_count{0};

    for (std::size_t i = 0; i < num_callbacks; ++i) {
      pipeline.registerOutputCallback(
          [&callback_count](
              const PipelineTypeTraits<TestOutput>::TypeConstSharedPtr&
                  output) {
            ASSERT_NE(output, nullptr);

            if (output) {
              callback_count.fetch_add(1, std::memory_order_relaxed);
            }
          });
    }

    fillInputQueue(kWarmupPackets);

    for (std::size_t i = 0; i < kWarmupPackets; ++i) {
      ASSERT_EQ(pipeline.spinOnce(), PipelineBase::ReturnCode::SUCCESS);
    }

    callback_count.store(0);

    fillInputQueue(kBenchmarkPackets);

    std::vector<double> samples;
    samples.reserve(kBenchmarkPackets);

    for (std::size_t i = 0; i < kBenchmarkPackets; ++i) {
      const auto start = Clock::now();

      ASSERT_EQ(pipeline.spinOnce(), PipelineBase::ReturnCode::SUCCESS);

      const auto end = Clock::now();

      samples.push_back(toNanoseconds(end - start));
    }

    const auto stats = calculateStats(std::move(samples));

    std::cout << "Number of callbacks = " << num_callbacks << '\n';

    printTiming("  spinOnce", stats);

    EXPECT_EQ(callback_count.load(), kBenchmarkPackets * num_callbacks);
  }
}

/**
 * --------------------------------------------------------------------------
 * 6. BLOCKING VS NON-BLOCKING INPUT
 * --------------------------------------------------------------------------
 *
 * Non-blocking:
 *
 *     queue.pop()
 *
 * Blocking:
 *
 *     queue.popBlocking()
 *
 * The blocking case must have another thread supplying the packet.
 */
TEST_F(PipelineLatencyTest, BlockingVsNonBlocking) {
  //
  // Non-blocking case.
  //
  {
    BenchmarkPipeline pipeline("nonblocking", input_queue_.get(), 0, false);

    auto output_queue = std::make_unique<OutputQueue>();
    pipeline.registerOutputQueue(output_queue.get());

    fillInputQueue(kWarmupPackets);

    for (std::size_t i = 0; i < kWarmupPackets; ++i) {
      ASSERT_EQ(pipeline.spinOnce(), PipelineBase::ReturnCode::SUCCESS);
    }

    fillInputQueue(kBenchmarkPackets);

    std::vector<double> samples;
    samples.reserve(kBenchmarkPackets);

    for (std::size_t i = 0; i < kBenchmarkPackets; ++i) {
      const auto start = Clock::now();

      ASSERT_EQ(pipeline.spinOnce(), PipelineBase::ReturnCode::SUCCESS);

      const auto end = Clock::now();

      samples.push_back(toNanoseconds(end - start));
    }

    const auto stats = calculateStats(std::move(samples));

    printTiming("Non-blocking pop", stats);
  }

  //
  // Blocking case.
  //
  input_queue_->shutdown();
  input_queue_ = std::make_unique<InputQueue>();

  BenchmarkPipeline pipeline("blocking", input_queue_.get(), 0, true);

  auto output_queue = std::make_unique<OutputQueue>();
  pipeline.registerOutputQueue(output_queue.get());

  std::atomic_bool producer_finished{false};

  std::thread producer([this, &producer_finished]() {
    for (std::size_t i = 0; i < kBenchmarkPackets; ++i) {
      input_queue_->push(makeInput(i));
    }

    producer_finished = true;
  });

  std::vector<double> samples;
  samples.reserve(kBenchmarkPackets);

  for (std::size_t i = 0; i < kBenchmarkPackets; ++i) {
    const auto start = Clock::now();

    ASSERT_EQ(pipeline.spinOnce(), PipelineBase::ReturnCode::SUCCESS);

    const auto end = Clock::now();

    samples.push_back(toNanoseconds(end - start));
  }

  producer.join();

  const auto stats = calculateStats(std::move(samples));

  printTiming("Blocking pop", stats);

  EXPECT_TRUE(producer_finished);
}

/**
 * --------------------------------------------------------------------------
 * 7. EMPTY QUEUE / NO WORK
 * --------------------------------------------------------------------------
 *
 * This measures the cost of repeatedly polling a pipeline which has no work.
 *
 * This is relevant if spin() is implemented as a tight loop around
 * spinOnce().
 */
TEST_F(PipelineLatencyTest, NoWork) {
  BenchmarkPipeline pipeline("no_work", input_queue_.get(), 0, false);

  constexpr std::size_t kIterations = 100000;

  std::vector<double> samples;
  samples.reserve(kIterations);

  for (std::size_t i = 0; i < kIterations; ++i) {
    const auto start = Clock::now();

    const auto result = pipeline.spinOnce();

    const auto end = Clock::now();

    EXPECT_EQ(result, PipelineBase::ReturnCode::GET_PACKET_FAILURE);

    samples.push_back(toNanoseconds(end - start));
  }

  const auto stats = calculateStats(std::move(samples));

  printTiming("Empty spinOnce", stats);
}

/**
 * --------------------------------------------------------------------------
 * 8. END-TO-END PACKET LATENCY
 * --------------------------------------------------------------------------
 *
 * This measures:
 *
 *     input queue push
 *          ↓
 *     queue pop
 *          ↓
 *     process
 *          ↓
 *     output queue push
 *          ↓
 *     output queue pop
 *
 * The timestamp is inserted before the input queue push.
 */
TEST_F(PipelineLatencyTest, EndToEndLatency) {
  BenchmarkPipeline pipeline("end_to_end", input_queue_.get(), 0, false);

  auto output_queue = std::make_unique<OutputQueue>();
  pipeline.registerOutputQueue(output_queue.get());

  constexpr std::size_t kPackets = 10000;

  // Warmup.
  for (std::size_t i = 0; i < kWarmupPackets; ++i) {
    input_queue_->push(makeInput(i));

    ASSERT_EQ(pipeline.spinOnce(), PipelineBase::ReturnCode::SUCCESS);

    typename PipelineTypeTraits<TestOutput>::TypeConstSharedPtr output;

    ASSERT_TRUE(output_queue->pop(output));
    ASSERT_NE(output, nullptr);
  }

  std::vector<double> samples;
  samples.reserve(kPackets);

  for (std::size_t i = 0; i < kPackets; ++i) {
    auto input = makeInput(i);

    const auto push_start = Clock::now();

    input_queue_->push(input);

    ASSERT_EQ(pipeline.spinOnce(), PipelineBase::ReturnCode::SUCCESS);

    typename PipelineTypeTraits<TestOutput>::TypeConstSharedPtr output;

    ASSERT_TRUE(output_queue->pop(output));

    const auto end = Clock::now();

    ASSERT_NE(output, nullptr);

    samples.push_back(toNanoseconds(end - push_start));
  }

  const auto stats = calculateStats(std::move(samples));

  printTiming("End-to-end push/pop", stats);
}

/**
 * --------------------------------------------------------------------------
 * 9. PIPELINE OVERHEAD SUMMARY
 * --------------------------------------------------------------------------
 *
 * Prints the most useful comparison in one place:
 *
 *     direct process
 *     spinOnce
 *     estimated framework overhead
 */
TEST_F(PipelineLatencyTest, OverheadSummary) {
  constexpr std::size_t kWorkIterations = 0;

  BenchmarkPipeline pipeline("summary", input_queue_.get(), kWorkIterations,
                             false);

  auto output_queue = std::make_unique<OutputQueue>();
  pipeline.registerOutputQueue(output_queue.get());

  //
  // Direct process.
  //
  auto input = makeInput();

  for (std::size_t i = 0; i < kWarmupPackets; ++i) {
    auto output = pipeline.directProcess(input);
    ASSERT_NE(output, nullptr);
  }

  std::vector<double> direct_samples;
  direct_samples.reserve(kBenchmarkPackets);

  for (std::size_t i = 0; i < kBenchmarkPackets; ++i) {
    const auto start = Clock::now();

    auto output = pipeline.directProcess(input);

    const auto end = Clock::now();

    ASSERT_NE(output, nullptr);

    direct_samples.push_back(toNanoseconds(end - start));
  }

  const auto direct_stats = calculateStats(std::move(direct_samples));

  //
  // Full pipeline.
  //
  fillInputQueue(kWarmupPackets);

  for (std::size_t i = 0; i < kWarmupPackets; ++i) {
    ASSERT_EQ(pipeline.spinOnce(), PipelineBase::ReturnCode::SUCCESS);
  }

  fillInputQueue(kBenchmarkPackets);

  std::vector<double> pipeline_samples;
  pipeline_samples.reserve(kBenchmarkPackets);

  for (std::size_t i = 0; i < kBenchmarkPackets; ++i) {
    const auto start = Clock::now();

    ASSERT_EQ(pipeline.spinOnce(), PipelineBase::ReturnCode::SUCCESS);

    const auto end = Clock::now();

    pipeline_samples.push_back(toNanoseconds(end - start));
  }

  const auto pipeline_stats = calculateStats(std::move(pipeline_samples));

  const double estimated_overhead =
      pipeline_stats.mean_ns - direct_stats.mean_ns;

  const double overhead_percentage =
      (estimated_overhead / pipeline_stats.mean_ns) * 100.0;

  std::cout << '\n';
  std::cout << "========================================\n";
  std::cout << "Pipeline overhead summary\n";
  std::cout << "========================================\n";

  printTiming("Direct process", direct_stats);

  printTiming("Full spinOnce", pipeline_stats);

  std::cout << std::left << std::setw(32) << "Estimated framework overhead"
            << " mean=" << estimated_overhead << " ns\n";

  std::cout << std::left << std::setw(32) << "Overhead percentage"
            << overhead_percentage << " %\n";

  std::cout << "========================================\n\n";

  EXPECT_GT(pipeline_stats.mean_ns, 0.0);
}

}  // namespace
}  // namespace dyno
