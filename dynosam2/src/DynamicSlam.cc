#include "dynosam2/DynamicSlam.hpp"

#include <glog/logging.h>

namespace dyno2 {

DynamicSlam::DynamicSlam(const PipelineParameters& parameters) {}

DynamicSlam::~DynamicSlam() {}

StateEstimate::Ptr DynamicSlam::spinOnce(
    dyno::VIFrontendInput::ConstPtr input) {
  LOG(INFO) << "DynamicSlam::spinOnce " << input->getFrameId();

  dyno::ImageContainer::Ptr image_container = input->image_container_;
  const dyno::WrappedRGBMono wrapped_rgb = image_container->rgb();
  const cv::Mat rgb = wrapped_rgb.image();

  pushImageToDisplayQueue("rgb", rgb);

  PublicationData::Ptr publication_data = std::make_shared<PublicationData>();
  pushPublicationDataToQueue(publication_data);

  return nullptr;
}

}  // namespace dyno2
