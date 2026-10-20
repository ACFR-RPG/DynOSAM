#pragma once

#include "dynosam2/common/StateEstimate.hpp"
#include "dynosam_common/ModuleBase.hpp"
#include "dynosam_common/Trajectories.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_core/DynoState.hpp"
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

  PublicationData(dyno::FrameId fi, dyno::Timestamp t)
      : frame_id(fi), timestamp(t) {}
  PublicationData() {}

  // tracking state information
  dyno::ViTrackingState vi_tracking_state;
  //! This also indicates well-tracked/observed objects!
  std::vector<dyno::DoTrackingState> do_tracking_states;

  // TODO: draw links between affected objects: covisible!
  UpdatedStates updated_states;
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
