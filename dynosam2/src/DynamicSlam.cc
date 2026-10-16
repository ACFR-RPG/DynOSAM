#include "dynosam2/DynamicSlam.hpp"

#include <glog/logging.h>

namespace dyno2 {

DynamicSlam::DynamicSlam(const PipelineParameters& parameters,
                         CanonicalSensorRig::Ptr sensor_rig)
    : parameters_(parameters),
      sensor_rig_(sensor_rig),
      frontend_(parameters.frontend, sensor_rig->cannonicalCamera()) {}

DynamicSlam::~DynamicSlam() {}

StateEstimate::Ptr DynamicSlam::spinOnce(
    dyno::VIFrontendInput::ConstPtr input) {
  LOG(INFO) << "DynamicSlam::spinOnce " << input->getFrameId();

  dyno::ImageContainer& image_container = *input->image_container_;
  FeatureBlockContainer* features{nullptr};
  dyno::ObjectDetectionResult object_detection;

  frontend_.detectAndTrack(features, object_detection, image_container);

  CHECK_NOTNULL(features);

  // dyno::ImageContainer::Ptr image_container = input->image_container_;
  const dyno::WrappedRGBMono wrapped_rgb = image_container.rgb();
  const cv::Mat rgb = wrapped_rgb.image();

  pushImageToDisplayQueue("rgb", features->draw(rgb, true));

  // PublicationData::Ptr publication_data =
  // std::make_shared<PublicationData>();
  // pushPublicationDataToQueue(publication_data);

  return nullptr;
}

}  // namespace dyno2
