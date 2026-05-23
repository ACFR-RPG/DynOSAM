#include "dynosam/dataprovider/DatasetProvider.hpp"
#include "dynosam/frontend/VIFrontendInput.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_common/utils/GtsamUtils.hpp"
#include "dynosam_common/utils/OpenCVUtils.hpp"

namespace dyno {

// depth, motion masks, gt
using DynoeptsProvider =
    DynoDatasetProvider<cv::Mat, cv::Mat, GroundTruthInputPacket>;

/**
 * @brief
 */
class DynopetsLoader : public DynoeptsProvider {
 public:
  DynopetsLoader(const fs::path& dataset_path);

  SensorRigBase::Ptr sensorRig() const override { return sensor_rig_; }

 private:
  BasicDynoSensorRig::Ptr sensor_rig_;
};

}  // namespace dyno
