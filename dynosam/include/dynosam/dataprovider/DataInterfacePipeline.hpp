/*
 *   Copyright (c) ACFR-RPG, University of Sydney, Jesse Morris
 * (jesse.morris@sydney.edu.au) All rights reserved.
 */
#pragma once

#include "dynosam/frontend/VIFrontendInput.hpp"
#include "dynosam/frontend/imu/ThreadSafeImuBuffer.hpp"
#include "dynosam/pipeline/PipelineBase.hpp"
#include "dynosam/pipeline/ThreadSafeQueue.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_sensors/SensorRig.hpp"

namespace dyno {

/**
 * @brief Takes data, synchronizes it and sends it to the output queue which
 * should be connected to the frontend User needs to implement
 * InputConstSharedPtr getInputPacket() = 0;
 * that takes data from the internal queues and processes them
 */
class DataInterfacePipeline
    : public MIMOPipelineModule<VIFrontendInput, VIFrontendInput> {
 public:
  DYNO_POINTER_TYPEDEFS(DataInterfacePipeline)

  using MIMO = MIMOPipelineModule<VIFrontendInput, VIFrontendInput>;
  using OutputQueue = typename MIMO::OutputQueue;

  using ImageContainerPreprocesser =
      std::function<ImageContainer::Ptr(ImageContainer::Ptr)>;
  using PreQueueContainerCallback =
      std::function<void(const ImageContainer::Ptr)>;

  DataInterfacePipeline(CanonicalSensorRig::ConstPtr sensor_rig,
                        bool parallel_run = false);
  virtual ~DataInterfacePipeline() = default;

  virtual VIFrontendInput::ConstPtr getInputPacket() override;

  void fillImuQueue(const ImuMeasurements& imu_measurements);

  void fillImuQueue(const ImuMeasurement& imu_measurement);

  void fillImageContainerQueue(ImageContainer::Ptr image_container);

  void addGroundTruthPacket(const GroundTruthInputPacket& gt_packet);

  void registerImageContainerPreprocessor(
      const ImageContainerPreprocesser& func);

  void registerPreQueueContainerCallback(const PreQueueContainerCallback& func);

  SharedGroundTruth getSharedGroundTruth() const;

 protected:
  virtual void onShutdown() {}

 private:
  inline MIMO::OutputConstSharedPtr process(
      const MIMO::InputConstSharedPtr& input) override {
    return input;
  }

  virtual bool hasWork() const override;

  //! Called when general shutdown of PipelineModule is triggered.
  void shutdownQueues() override;

  /**
   * @brief Preferred step to take with current frame based on IMU status
   */
  enum class FrameAction {
    Drop,     // None of the timestamp invariants check out
    Wait,     // We need more data to use the frame
    Use,      // The frame is good, use
    Shutdown  // The IMU queue was shutdown and this action should be emitted
  };

  /**
   * @brief getTimeSyncedImuMeasurements Time synchronizes the IMU buffer
   * with the given timestamp (this is typically the timestamp of a left img)
   *
   * False if synchronization failed, true otherwise.
   *
   * @param timestamp const Timestamp& for the IMU data to query
   * @param imu_meas ImuMeasurements* IMU measurements to be populated and
   * returned
   * @return true
   * @return false
   */
  FrameAction getTimeSyncedImuMeasurements(const Timestamp& timestamp,
                                           ImuMeasurements* imu_meas);

  void removeOldImuMeasurements(const ImuMeasurements& imu_meas);

 protected:
  CanonicalSensorRig::ConstPtr sensor_rig_;
  std::atomic_bool parallel_run_;
  ThreadsafeImuBuffer imu_buffer_;
  //! Time of last IMU synchronisation (and
  //! represents the time of the last frame)
  Timestamp timestamp_last_frame_;
  bool is_ready_;

  ThreadsafeQueue<ImageContainer::Ptr> packet_queue_;
  GroundTruthPublisher ground_truth_publisher_;
  SharedGroundTruth shared_ground_truth_;

  // callback to handle dataset specific pre-processing of the images before
  // they are sent to the frontend if one is registered, called immediately
  // before adding any ImageContainers to the packet_queue. Registered functions
  // should return quickly
  ImageContainerPreprocesser image_container_preprocessor_;
  //! Function that is called immediately before the ImageContainer is put into
  //! the packet_queue_
  PreQueueContainerCallback pre_queue_container_calback_;
};

}  // namespace dyno
