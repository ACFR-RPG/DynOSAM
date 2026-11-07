#include <gflags/gflags.h>
#include <glog/logging.h>

#include "dynosam2/common/Parameters.hpp"
#include "dynosam2/frontend/FeatureTracker.hpp"
#include "dynosam_processing/datasets/OMDDataProvider.hpp"

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;
  FLAGS_colorlogtostderr = 1;
  FLAGS_log_prefix = 1;
  FLAGS_v = 60;

  // KittiDataLoader::Params params;
  // KittiDataLoader loader("/root/data/vdo_slam/kitti/kitti/0020/", params);
  // ClusterSlamDataLoader loader("/root/data/cluster_slam/CARLA-S2");
  // loader.setStartingFrame(600);
  dyno::OMDDataLoader loader(
      "/root/data/vdo_slam/omd/omd/swinging_4_unconstrained_stereo/");

  LOG(INFO) << "Making camera";
  auto camera = std::make_shared<dyno::Camera>(loader.getCameraParams());
  LOG(INFO) << "Made camera";

  dyno2::FrontendParameters frontend;
  frontend.prefer_provided_object_detection = true;
  dyno2::FeatureTracker tracker(frontend, camera);

  loader.registerImageContainerCallback(
      [&](ImageContainer::Ptr container) -> void {
        LOG(INFO) << container->frameId() << " " << container->timestamp();

        auto tracking_result = tracker.track(container);

        const WrappedRGBMono wrapped_rgb = container->rgb();
        const cv::Mat rgb = wrapped_rgb.image();
        cv::imshow("Fast Track", tracking_result.featues->draw(rgb, true));

        cv::waitKey(1);
      });

  while (loader.spin()) {
    LOG(INFO) << dyno::utils::Statistics::Print();
  }
}
