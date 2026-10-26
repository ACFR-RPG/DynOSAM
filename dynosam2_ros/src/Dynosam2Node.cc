#include "dynosam2_ros/Dynosam2Node.hpp"

#include <config_utilities/parsing/yaml.h>
#include <gflags/gflags.h>

#include "dynosam2_ros/Dynosam2Node-impl.hpp"
#include "dynosam_common/utils/FileSystem.hpp"
#include "dynosam_common/utils/Statistics.hpp"
#include "dynosam_ros_common/CameraSystem.hpp"
#include "dynosam_ros_common/RosUtils.hpp"
#include "dynosam_ros_common/Subscriber.hpp"
#include "rosgraph_msgs/msg/clock.hpp"

DEFINE_int32(
    data_provider_type, 0,
    "Which data provider (loader) to use. Associated with specific datasets");

namespace dyno2 {

/* Check dyno params match the sensor configuration and update the params if
 * needed*/
static bool reconfigure(const SensorMode& sensor_mode,
                        PipelineParameters& parameters);

static std::string getFilePathFromParams(rclcpp::Node* node,
                                         const std::string& param_name,
                                         const std::string& description) {
  // use non-default version so that Builder throws exception if no
  // parameter is provided on the param server
  const std::string path = dyno::ros::Parameter::Builder(node, param_name)
                               .description(description)
                               .finish()
                               .get<std::string>();
  dyno::utils::throwExceptionIfPathInvalid(path);
  return path;
}

/// @brief
/// @tparam T Derived PipelineBase class
/// @param ptr
/// @return
template <typename T>
static dyno::Spinner::UniquePtr makePipelineSpinner(std::unique_ptr<T>& ptr) {
  return std::make_unique<dyno::Spinner>(std::bind(&T::spin, ptr.get()),
                                         ptr->moduleName() + "-spinner");
}

static const auto params_path_message =
    "Path to folder containing dynosam2.yaml file or a direct path to a .yaml "
    "file";

Dynosam2Node::Impl::Impl(rclcpp::Node& node)
    : node_(node),
      params_path_(
          getFilePathFromParams(&node, "params_path", params_path_message)) {
  rclcpp::Node* node_ptr = &node_;
  RCLCPP_INFO_STREAM(node_.get_logger(), "Starting Dynosam2Node");

  std::string cuda_enabled_message;
  dyno::utils::opencvCudaAvailable(&cuda_enabled_message);
  RCLCPP_INFO_STREAM(node_.get_logger(), cuda_enabled_message);

  PipelineParameters system_paramters;
  loadParameters(system_paramters);

  bool is_online;
  data_reader_ = createDataReader(system_paramters, &is_online);
  CHECK_NOTNULL(data_reader_);

  auto sensor_rig = data_reader_->sensorRig();
  CHECK_NOTNULL(sensor_rig);

  //   / setup slam
  slam_ = std::make_shared<DynamicSlam>(system_paramters, sensor_rig);

  constexpr static bool kRunParallel = true;

  VLOG(10) << "Sensor Rig: " << *sensor_rig;
  data_interface_ =
      std::make_unique<dyno::DataInterfacePipeline>(sensor_rig, kRunParallel);

  data_reader_->registerImageContainerCallback(
      std::bind(&dyno::DataInterfacePipeline::fillImageContainerQueue,
                data_interface_.get(), std::placeholders::_1));

  // ground truth
  data_reader_->registerGroundTruthPacketCallback(
      std::bind(&dyno::DataInterfacePipeline::addGroundTruthPacket,
                data_interface_.get(), std::placeholders::_1));

  // register single and multi IMU callbacks to the data loader
  data_reader_->registerImuSingleCallback(
      std::bind(static_cast<void (dyno::DataInterfacePipeline::*)(
                    const dyno::ImuMeasurement&)>(
                    &dyno::DataInterfacePipeline::fillImuQueue),
                data_interface_.get(), std::placeholders::_1));

  data_reader_->registerImuMultiCallback(
      std::bind(static_cast<void (dyno::DataInterfacePipeline::*)(
                    const dyno::ImuMeasurements&)>(
                    &dyno::DataInterfacePipeline::fillImuQueue),
                data_interface_.get(), std::placeholders::_1));

  // preprocessing
  data_interface_->registerImageContainerPreprocessor(
      std::bind(&dyno::DataReader::imageContainerPreprocessor,
                data_reader_.get(), std::placeholders::_1));

  // push data to SLAM pipeline
  data_interface_->registerOutputQueue(&input_queue_);

  if (!is_online) {
    rclcpp::Publisher<rosgraph_msgs::msg::Clock>::SharedPtr clock_pub =
        node_.create_publisher<rosgraph_msgs::msg::Clock>("/clock", 10);
    data_interface_->registerPreQueueContainerCallback(
        [clock_pub](const ImageContainer::Ptr image_container) -> void {
          auto msg = rosgraph_msgs::msg::Clock();
          msg.clock = dyno::ros::toRosTime(image_container->timestamp());
          CHECK_NOTNULL(clock_pub)->publish(msg);
        });
  }

  pipeline_ = std::make_unique<DynamicSlamPipeline>("dynosam2-pipeline",
                                                    &input_queue_, slam_);

  pipeline_->parallelRun(kRunParallel);
  // send to viz
  // pipeline_->registerOutputQueue(&output_queue_);

  const SlamBackend& estimator = slam_->getEstimator();
  auto viz_node = node_.create_sub_node("output");
  viz_ = std::make_unique<DynamicSlamVisualizerRos>(
      viz_node, sensor_rig, &estimator, &publication_queue_);

  dyno::ImageDisplayQueue* display_queue = &display_queue_;
  viz_->parallelRun(kRunParallel);
  // enable image displayer
  viz_->setDisplayQueue(display_queue);
  viz_->addImageDisplay(std::make_shared<SmartOpenCvDisplayer>());

  // ROS image displayer that will use images pulled from the display_queue
  auto ros_image_displayer = std::make_shared<RosImageDisplayer>(viz_node);
  // matches name currently in DynamicSLAM
  // eventually maybe the DynamicSlamVisualizerRos will draw some images too?
  ros_image_displayer->setupImageTopic("Tracks", "visual_tracks");
  viz_->addImageDisplay(ros_image_displayer);

  // connect pipeline to viz
  slam_->setPublicationQueue(&publication_queue_);
  slam_->setImageDisplayQueue(display_queue);

  // launch spinners
  viz_spinner_ = makePipelineSpinner(viz_);
  data_interface_spinner_ = makePipelineSpinner(data_interface_);
  pipeline_spinner_ = makePipelineSpinner(pipeline_);

  auto spinOnce = [&]() -> void {
    // spin while there is work to do!
    // TODO: and SLAM is still running!!
    if (!(data_reader_->spin() || pipeline_->isWorking())) {
      auto context = node_.get_node_options().context();
      context->shutdown("Dynosam2 pipeline has finished processing all data");
    }

    // do display
    // even if nothing is connected to the display queue, this will ensure
    // memory does not grow as we will try and pull images from the display
    // queue
    viz_->displayImages();
  };

  // so that all process is encapsulated within this class we use a timer
  // to trigger the pipelines spin function
  // this hides a bit of complex behaviour because depending on the setup
  // (ie. parallel_run, is_online etc) this may or may not limit the processing
  // rate of the pipeline
  // in most cases this should not matter (ie. when we are processing dataset
  // the processing speed is not an issue) in either case we expect the limit of
  // processing to be about 15-20Hz (which is realtime anyway) so we attempt to
  // spin at this rate regardless
  // NOTE: MUST be MutuallyExclusive to prevent execution in parallel to itself
  spin_timer_group_ = node_ptr->create_callback_group(
      rclcpp::CallbackGroupType::MutuallyExclusive);
  spin_timer_ = node_ptr->create_wall_timer(std::chrono::milliseconds(30),
                                            spinOnce, spin_timer_group_);

  // dyno::ros::Parameter publish_diagnostics_param =
  //   dyno::ros::Parameter::Builder(node_ptr, "publish_diagnostics", true)
  //         .description(
  //             "If the diagnostics publisher should run, reporting stats for
  //             all" " pipelines and modules.")
  //         .finish();

  // if (publish_diagnostics_param.get<bool>()) {
  //   diagnostics_updater_ =
  //       std::make_unique<dyno::DynoDiagnosticsTaskManager>(node_ptr);

  //   hooks->register_diagnostics_task = [&](const std::string& name,
  //                                          dyno::DiagnosticTaskRunner* task)
  //                                          {
  //     CHECK_NOTNULL(task);
  //     RCLCPP_INFO_STREAM(node_ptr->get_logger(),
  //                        "Registering diagnostics task: " << name);
  //     diagnostics_updater_->registerTask(
  //         std::make_shared<dyno::DynoDiagnosticsTask>(name, task));
  //   };
  // }
}

Dynosam2Node::Impl::~Impl() {
  if (data_interface_) data_interface_->shutdown();
  if (pipeline_) pipeline_->shutdown();
  if (viz_) viz_->shutdown();

  if (data_interface_spinner_) data_interface_spinner_->shutdown();
  if (pipeline_spinner_) pipeline_spinner_->shutdown();
  if (viz_spinner_) viz_spinner_->shutdown();
  //  stop diagnostic updater before pipeline as all the pipeline owns
  //  the memory for all diagnostics tasks which are only stored in the
  //  updater with references/raw pointers
  if (diagnostics_updater_) diagnostics_updater_.reset(nullptr);
  if (slam_) slam_.reset();
  LOG(INFO) << dyno::utils::Statistics::Print();
}

void resolveParamsFile(const std::string& provided_params_path,
                       std::string& full_params_path) {
  // params can either be provided directly as a .yaml file or as a folder with
  // a dynosam2.yaml file inside it
  namespace fs = std::filesystem;
  const fs::path params_path_fs(provided_params_path);

  fs::path yaml_path;
  if (fs::is_regular_file(params_path_fs)) {
    // A file was provided directly; validate its extension.
    const std::string extension = params_path_fs.extension().string();

    if (extension != ".yaml" && extension != ".yml") {
      throw std::runtime_error(
          "Dynosam2 parameters file must have a .yaml or .yml extension: " +
          provided_params_path);
    }

    yaml_path = params_path_fs;
  } else if (fs::is_directory(params_path_fs)) {
    // A directory was provided; look for dynosam2.yaml inside it.
    yaml_path = params_path_fs / "dynosam2.yaml";
  } else {
    throw std::runtime_error(
        "Dynosam2 parameters path does not exist or is not a regular "
        "file/directory: " +
        provided_params_path);
  }

  // Resolve to an absolute path and validate the YAML file.
  yaml_path = fs::absolute(yaml_path);

  if (!fs::is_regular_file(yaml_path)) {
    throw std::runtime_error("Dynosam2 parameters YAML file does not exist: " +
                             yaml_path.string());
  }

  full_params_path = yaml_path.string();
}

bool Dynosam2Node::Impl::loadParameters(PipelineParameters& parameters) {
  // params can either be provided directly as a .yaml file or as a folder with
  // a dynosam2.yaml file inside it
  std::string config_file_path;
  resolveParamsFile(params_path_, config_file_path);
  RCLCPP_INFO_STREAM(node_.get_logger(),
                     "Loading Dynosam2 params from file: " << config_file_path);

  parameters = config::fromYamlFile<PipelineParameters>(config_file_path);
  return true;
}

dyno::DataReader::Ptr Dynosam2Node::Impl::createDataReader(
    PipelineParameters& parameters, bool* is_online) {
  dyno::ros::Parameter is_online_param =
      dyno::ros::Parameter::Builder(&node_, "online", false)
          .description("If the online DataReader should be used")
          .finish();

  if (is_online) *is_online = is_online_param.get<bool>();

  if (is_online_param.get<bool>()) {
    auto mode_param = dyno::ros::Parameter::Builder(&node_, "input_image_mode",
                                                    "rgb+aligned_depth")
                          .description(
                              "Specify sensor and camera stream modes (ie. "
                              "rgb+aligned_depth, "
                              "+stereo, +imu etc.)")
                          .finish();

    dyno::SensorMode sensor_mode(mode_param.get<std::string>());
    reconfigure(sensor_mode, parameters);

    dyno::LoadingSourceParams loading_sources;
    loading_sources.params_folder = params_path_;
    loading_sources.cameras_from_ros = true;

    dyno::SensorSystem::Ptr sensor_system =
        std::make_shared<dyno::SensorSystem>(
            node_.create_sub_node("dataprovider"), sensor_mode.depthRigType(),
            loading_sources);

    for (const auto& configs : sensor_mode.configs()) {
      sensor_system->addCamera(configs);
    }

    sensor_system->enableImu(sensor_mode.useImu());
    sensor_system->finalise();

    // do better with the subnodes
    auto subscriber = std::make_shared<dyno::Subscriber>(
        sensor_system, node_.create_sub_node("images"));
    return subscriber;
  } else {
    std::string dataset_path(getFilePathFromParams(&node_, "dataset_path",
                                                   "Path to dataset folder"));

    RCLCPP_INFO_STREAM(node_.get_logger(),
                       "Loading dataset from: " << dataset_path);

    dyno::DataReader::Ptr data_loader = dyno::DataProviderFactory::Create(
        dataset_path, params_path_,
        static_cast<dyno::DatasetType>(FLAGS_data_provider_type));
    return data_loader;
  }
}

// reconfigure any parameters for online mode
bool reconfigure(const SensorMode& sensor_mode,
                 PipelineParameters& parameters) {
  bool any_changed = false;
  auto& frontend = parameters.frontend;
  // if (frontend.prefer_provided_optical_flow &&
  //     !sensor_mode.hasStreamType(StreamConfig::Types::OpticalFlow)) {
  //   LOG(WARNING) << "Sensor mode:" << sensor_mode.rawSensorMode()
  //                << " does not include optical flow "
  //                << " but prefer_provided_optical_flow=true! DynoParams will
  //                "
  //                   "be reconfigured";
  //   frontend.prefer_provided_optical_flow = false;
  //   any_changed = true;
  // }

  if (frontend.prefer_provided_object_detection &&
      !sensor_mode.hasStreamType(StreamConfig::Types::Mask)) {
    LOG(WARNING) << "Sensor mode:" << sensor_mode.rawSensorMode()
                 << " does not include object mask "
                 << " but prefer_provided_object_detection=true! DynoParams "
                    "will be reconfigured!\n"
                 << " NOTE: ground truth object id's may now no longer match "
                    "the tracked id's from Dynosam!!!";
    frontend.prefer_provided_object_detection = false;
    any_changed = true;
  }
  return any_changed;
}

Dynosam2Node::Dynosam2Node(const rclcpp::NodeOptions& options)
    : rclcpp::Node("dynosam2_node", "dynosam", options),
      impl_(std::make_unique<Impl>(*this)) {}
Dynosam2Node::~Dynosam2Node() { impl_.reset(); }

}  // namespace dyno2

// #include <rclcpp_components/register_node_macro.hpp>
// RCLCPP_COMPONENTS_REGISTER_NODE(dyno::DynosamComposableNode)
