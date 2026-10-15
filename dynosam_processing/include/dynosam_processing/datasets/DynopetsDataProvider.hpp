#include "dynosam_common/Types.hpp"
#include "dynosam_common/utils/GtsamUtils.hpp"
#include "dynosam_cv/CvUtils.hpp"
#include "dynosam_processing/DatasetReader.hpp"
#include "dynosam_sensors/VIFrontendInput.hpp"

namespace dyno {

// depth, motion masks, gt
using DynoeptsProvider =
    DatasetReader<cv::Mat, cv::Mat, GroundTruthInputPacket>;

/**
 * @brief
 */
class DynopetsLoader : public DynoeptsProvider {
 public:
  DynopetsLoader(const fs::path& dataset_path);

  CanonicalSensorRig::Ptr sensorRig() const override { return sensor_rig_; }

 private:
  BasicDynoSensorRig::Ptr sensor_rig_;
};

}  // namespace dyno
