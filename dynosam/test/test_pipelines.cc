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
#include "dynosam/common/ModuleBase.hpp"
#include "dynosam/frontend/FrontendPipeline.hpp"
#include "dynosam/pipeline/PipelineBase.hpp"
#include "dynosam/pipeline/PipelinePayload.hpp"
#include "dynosam/utils/SafeCast.hpp"
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
//     NullPipelinePayload>;

//     using VarModule = VariantModule<Var, NullPipelinePayload, std::string>;

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
