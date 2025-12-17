/*
 *   Copyright (c) 2023 Jesse Morris
 *   All rights reserved.
 */
#pragma once

#include <functional>

#include "dynosam/dataprovider/DataInterfacePipeline.hpp"
#include "dynosam/frontend/imu/ImuParams.hpp"
#include "dynosam/frontend/vision/Frame.hpp"
#include "dynosam_common/Exceptions.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_cv/CameraParams.hpp"

namespace dyno {

/**
 * @brief A data provider is a module that actually gets the inididual image
 * and/or IMU data from some source and packets it into a Frame and IMU form and
 * sends it the the DataInterfacePipeline via callback functions where the data
 * is synchronized and sent to the frontend
 *
 *
 */
class DataProvider {
 public:
  DYNO_POINTER_TYPEDEFS(DataProvider)
  DYNO_DELETE_COPY_CONSTRUCTORS(DataProvider)

  using ImageContainerCallback = std::function<void(ImageContainer::Ptr)>;
  using ImuSingleInputCallback = std::function<void(const ImuMeasurement&)>;
  using ImuMultiInputCallback = std::function<void(const ImuMeasurements&)>;

  using GroundTruthPacketCallback =
      std::function<void(const GroundTruthInputPacket&)>;

  // this one will not guarnatee a binding of bind the data prover module
  DataProvider() = default;

  /**
   * @brief Calls the virtual DataProvider#shutdown() function
   *
   */
  virtual ~DataProvider();

  inline void registerImuSingleCallback(
      const ImuSingleInputCallback& callback) {
    imu_single_input_callback_ = callback;
  }

  inline void registerImuMultiCallback(const ImuMultiInputCallback& callback) {
    imu_multi_input_callback_ = callback;
  }

  inline void registerImageContainerCallback(
      const ImageContainerCallback& callback) {
    image_container_callback_ = callback;
  }

  inline void registerGroundTruthPacketCallback(
      const GroundTruthPacketCallback& callback) {
    ground_truth_packet_callback_ = callback;
  }

  /**
   * @brief Virtual Image Container Preprocessor function that is registered to
   * the data interface pipeline
   *
   * By default does nothing and just returns the argument.
   * This function is called (via callback) when a ImageContainer is sent to the
   * pipeline via DataProvider::image_container_callback_ and enables each
   * data-provider to implement their own data-preprocessing as necessary
   *
   * @param image_container
   * @return ImageContainer::Ptr
   */
  inline virtual ImageContainer::Ptr imageContainerPreprocessor(
      ImageContainer::Ptr image_container) {
    return image_container;
  }

  inline void registerOnFinishCallback(const std::function<void()>& callback) {
    on_finish_callbacks_.push_back(callback);
  }

  /**
   * @brief Get the size of the dataset - if this is not valid (i.e. the
   * provider is online) this function should return -1
   *
   * @return int
   */
  virtual int datasetSize() const = 0;

  /**
   * @brief Spins the dataset for one "step" of the dataset
   *
   * Returns true if the dataset still has data to process
   * @return true
   * @return false
   */
  virtual bool spin() = 0;

  virtual void shutdown();

  /**
   * @brief Provides functionality to get camera paramters for the main
   * pipeline, as these may be available from the dataset.
   *
   * By default returns std::nullopt but the derived data-provider can overwrite
   * this and return the loaded params
   *
   * @return CameraParams::Optional
   */
  virtual CameraParams::Optional getCameraParams() const { return {}; }

  virtual ImuParams::Optional getImuParams() const { return {}; }

 protected:
  // This class does not know when the data finishes finishes (indeed this is
  // only really valid for datasets) so we make a protected function that can be
  // called by a derived class that will trigger the callbacks we need the
  // callbacks in the base class becuase the pipelinemanager only has access to
  // the DataProvider::Ptr and NOT the derived classes
  void emitOnFinishCallbacks() {
    for (auto cb : on_finish_callbacks_) cb();
  }

 protected:
  ImageContainerCallback image_container_callback_;
  ImuSingleInputCallback imu_single_input_callback_;
  ImuMultiInputCallback imu_multi_input_callback_;

  GroundTruthPacketCallback ground_truth_packet_callback_;

  // Shutdown switch to stop data provider.
  std::atomic_bool shutdown_ = {false};

 private:
  std::vector<std::function<void()>> on_finish_callbacks_;
};

}  // namespace dyno
