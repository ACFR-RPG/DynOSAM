/*
 *   Copyright (c) 2023 ACFR-RPG, University of Sydney, Jesse Morris
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

#pragma once

#include "dynosam/frontend/FrontendInputPacket.hpp"
#include "dynosam/frontend/FrontendOutputPacket.hpp"
#include "dynosam/logger/Logger.hpp"
#include "dynosam/pipeline/PipelineModuleProcessor.hpp"

namespace dyno {

using FrontendPipeline =
    PipelineModuleProcessor<FrontendInputPacketBase, VisionImuPacket>;

/// @brief module traits of the associated backend...
/// @tparam MODULE_TRAITS
template <class MODULE_TRAITS>
class FrontendOfflinePipeline
    : public MIMOPipelineModule<NullPipelinePayload, VisionImuPacket> {
 public:
  using This = FrontendOfflinePipeline<MODULE_TRAITS>;
  using Base = MIMOPipelineModule<NullPipelinePayload, VisionImuPacket>;
  using ModuleTraits = MODULE_TRAITS;
  // A Dervied BackedInputPacket type (e.g. RGBDOutputPacketType)
  using DerivedPacketType = typename ModuleTraits::DerivedPacketType;
  using DerivedPacketTypePtr = std::shared_ptr<DerivedPacketType>;

  DYNO_POINTER_TYPEDEFS(This)

  //! Map of frames to (derived) frontend packets
  //! Same format as the seralized bson from the frontend
  using FrontendPacketMap = std::map<FrameId, DerivedPacketTypePtr>;

  using typename Base::InputConstSharedPtr;
  using typename Base::OutputConstSharedPtr;

  explicit FrontendOfflinePipeline(const std::string& module_name,
                                   const std::string& bsjon_file_path,
                                   int ending_frame = -1)
      : Base(module_name) {
    // TODO: after seralize fix with VisionImuPacket
    //  if(!JsonConverter::ReadInJson(frontend_output_packets_, bsjon_file_path,
    //  JsonConverter::BSON)) {
    //      LOG(FATAL) << "Failed to load frontend bson from: " <<
    //      bsjon_file_path;
    //  }
    //  current_packet_ = frontend_output_packets_.cbegin();
    //  ending_frame_ =  getFrontendOutputPackets().rbegin()->first;
    //  if(ending_frame != -1) {
    //      ending_frame_ = (FrameId)std::min(ending_frame, (int)ending_frame_);
    //  }
    //  LOG(INFO) << "Setting frontend offline ending frame = " <<
    //  ending_frame_;
  }

  InputConstSharedPtr getInputPacket() override {
    static auto output = std::make_shared<NullPipelinePayload>();
    return output;
  }

  inline OutputConstSharedPtr process(const InputConstSharedPtr&) override {
    if (!hasWork()) {
      this->shutdown();
      return nullptr;
    }
    auto output = current_packet_->second;
    current_packet_++;
    return output;
  }

  inline const FrontendPacketMap& getFrontendOutputPackets() const {
    return frontend_output_packets_;
  }
  inline FrameId endingFrame() const { return (FrameId)ending_frame_; }

 protected:
  void shutdownQueues() override {
    current_packet_ = frontend_output_packets_.end();
  }
  bool hasWork() const override {
    return current_packet_ != frontend_output_packets_.cend() &&
           current_packet_->first <= (FrameId)ending_frame_;
  }

 private:
  //! same format as the output_packet_record_ in the frontend
  //!  expected to be seralized in this form to a json file which this pipeline
  //!  will then load and send directly to the frontend
  FrontendPacketMap frontend_output_packets_;
  typename FrontendPacketMap::const_iterator current_packet_;

  int ending_frame_;
};

}  // namespace dyno
