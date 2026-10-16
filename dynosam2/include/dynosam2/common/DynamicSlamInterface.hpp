#pragma once

#include "dynosam2/common/StateEstimate.hpp"
#include "dynosam_common/ModuleBase.hpp"
#include "dynosam_processing/ImageViewer.hpp"
#include "dynosam_processing/ThreadSafeQueue.hpp"
#include "dynosam_sensors/VIFrontendInput.hpp"

namespace dyno2 {

/// @brief Realtime publishing data struct.
struct PublicationData {
  DYNO_POINTER_TYPEDEFS(PublicationData)
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
