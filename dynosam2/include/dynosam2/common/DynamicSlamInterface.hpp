#pragma once

#include "dynosam2/common/StateEstimate.hpp"
#include "dynosam_common/ModuleBase.hpp"
#include "dynosam_common/Trajectories.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_processing/ImageViewer.hpp"
#include "dynosam_processing/ThreadSafeQueue.hpp"
#include "dynosam_sensors/VIFrontendInput.hpp"

namespace dyno2 {

struct UpdatedStates {
  std::vector<dyno::FrameId> vo;
  std::vector<dyno::TemporalObjectIdentifier> objects;
};

/// @brief Realtime publishing data struct.
struct PublicationData {
  DYNO_POINTER_TYPEDEFS(PublicationData)

  dyno::FrameId frame_id;
  dyno::Timestamp timestamp;

  // TODO: add seen objects! becuase we might upate objects via FULL BA but we
  // dont want to draw their trajectoties!
  UpdatedStates updated_states;
  //! Seen objects with motion this frame
  std::vector<dyno::ObjectId> observed_objects;
};

class DynamicSlamInterface
    : public dyno::ModuleBase<dyno::VIFrontendInput, StateEstimate> {
 public:
  typedef dyno::ModuleBase<dyno::VIFrontendInput, StateEstimate> Base;

  DynamicSlamInterface();
  virtual ~DynamicSlamInterface() = default;

  void setPublicationQueue(ThreadsafeQueue<PublicationData::ConstPtr>* queue);
  void setImageDisplayQueue(dyno::ImageDisplayQueue* queue);

  bool pushImageToDisplayQueue(const std::string& wname, const cv::Mat& image);
  bool pushPublicationDataToQueue(PublicationData::ConstPtr publication_data);

 private:
  ThreadsafeQueue<PublicationData::ConstPtr>* publication_queue_{nullptr};
  dyno::ImageDisplayQueue* image_display_queue_{nullptr};

 protected:
  virtual StateEstimate::Ptr spinOnce(
      dyno::VIFrontendInput::ConstPtr input) = 0;

 private:
  inline Base::SpinReturn boostrapSpin(
      dyno::VIFrontendInput::ConstPtr input) override {
    return {Base::State::Nominal, spinOnce(input)};
  }
  inline Base::SpinReturn nominalSpin(
      dyno::VIFrontendInput::ConstPtr input) override {
    return {Base::State::Nominal, spinOnce(input)};
  }
  void validateInput(
      const dyno::VIFrontendInput::ConstPtr& input) const override;
};

}  // namespace dyno2
