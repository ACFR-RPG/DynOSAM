/*
 *   Copyright (c) ACFR-RPG, University of Sydney, Jesse Morris
 * (jesse.morris@sydney.edu.au) All rights reserved.
 */
#pragma once

#include <mutex>

#include "dynosam/common/SensorModels.hpp"
#include "dynosam/common/Types.hpp"
#include "dynosam/frontend/FrontendInputPacket.hpp"
#include "dynosam/frontend/imu/ThreadSafeImuBuffer.hpp"
#include "dynosam/pipeline/PipelineBase.hpp"
#include "dynosam/pipeline/ThreadSafeQueue.hpp"
#include "dynosam/pipeline/ThreadSafeTemporalBuffer.hpp"

namespace dyno {

class ExternalMeasurementHandler {
 public:
  ExternalMeasurementHandler(Timestamp sync_delta = 0)
      : sync_delta_(sync_delta) {}
  virtual ~ExternalMeasurementHandler() {}

  inline void fillExternalMeasurementQueue(
      FunctionalMeasurement::Ptr external) {
    external_measurement_buffer_.addValue(external->timestamp, external);
  }

  FunctionalMeasurement::Ptr getTimeSyncedExternalMeasurements(
      const Timestamp& timestamp) const;

 protected:
  //! Allowable time-delta for synchronization in seconds. 0 means exact
  Timestamp sync_delta_{0};
  ThreadsafeTemporalBuffer<FunctionalMeasurement::Ptr>
      external_measurement_buffer_;
};

class MultiExternalMeasurementHandler {
 public:
  MultiExternalMeasurementHandler(Timestamp default_sync_delta = 0.05)
      : default_sync_delta_(default_sync_delta) {}
  virtual ~MultiExternalMeasurementHandler() {}

  /**
   * @brief Sets the allowable synchronisation delta for a
   * ExternalMeasurementHandler. If the topic does not exist, it will be
   * created.
   *
   * Does nothing if topic already exists!!? (So should be called before any
   * fillExternalMeasurementQueue)
   *
   * @param topic const std::string&
   * @param sync_delta Timestamp
   */
  void setSyncDelta(const std::string& topic, Timestamp sync_delta);

  void fillExternalMeasurementQueue(const std::string& topic,
                                    FunctionalMeasurement::Ptr external);

 protected:
  void getTimeSyncedExternalMeasurements(
      const Timestamp& timestamp,
      std::vector<FunctionalMeasurement::Ptr>& external_measurements) const;

 protected:
  Timestamp default_sync_delta_;
  gtsam::FastMap<std::string, ExternalMeasurementHandler>
      external_measurement_handlers_;
  mutable std::mutex mutex_;
};

class ImuInterfaceHandler {
 public:
  ImuInterfaceHandler();
  virtual ~ImuInterfaceHandler() {}

  virtual inline void fillImuQueue(const ImuMeasurements& imu_measurements) {
    imu_buffer_.addMeasurements(imu_measurements.timestamps_,
                                imu_measurements.acc_gyr_);
  }

  virtual inline void fillImuQueue(const ImuMeasurement& imu_measurement) {
    imu_buffer_.addMeasurement(imu_measurement.timestamp_,
                               imu_measurement.acc_gyr_);
  }

  /**
   * @brief Preferred step to take with current frame based on IMU status
   */
  enum class FrameAction {
    Drop,     // None of the timestamp invariants check out
    Wait,     // We need more data to use the frame
    Use,      // The frame is good, use
    Shutdown  // The IMU queue was shutdown and this action should be emitted
  };

 protected:
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

 protected:
  ThreadsafeImuBuffer imu_buffer_;

  Timestamp timestamp_last_frame_{
      0};  //! Time of last IMU synchronisation (and
           //! represents the time of the last frame)
};

/**
 * @brief Takes data, synchronizes it and sends it to the output queue which
 * should be connected to the frontend User needs to implement
 * InputConstSharedPtr getInputPacket() = 0;
 * that takes data from the internal queues and processes them
 */
class DataInterfacePipeline
    : public MIMOPipelineModule<FrontendInputPacketBase,
                                FrontendInputPacketBase>,
      public ImuInterfaceHandler,
      public MultiExternalMeasurementHandler {
 public:
  DYNO_POINTER_TYPEDEFS(DataInterfacePipeline)

  using MIMO =
      MIMOPipelineModule<FrontendInputPacketBase, FrontendInputPacketBase>;
  using OutputQueue = typename MIMO::OutputQueue;

  using ImageContainerPreprocesser =
      std::function<ImageContainer::Ptr(ImageContainer::Ptr)>;
  using PreQueueContainerCallback =
      std::function<void(const ImageContainer::Ptr)>;

  DataInterfacePipeline(bool parallel_run = false);
  virtual ~DataInterfacePipeline() = default;

  // TODO: later should be vision only module
  virtual FrontendInputPacketBase::ConstPtr getInputPacket() override;

  // expects input packet
  // TODO: I dont think should be vrtual?
  virtual inline void fillImageContainerQueue(
      ImageContainer::Ptr image_container) {
    ImageContainer::Ptr processed_container = image_container;
    if (image_container_preprocessor_) {
      processed_container = image_container_preprocessor_(processed_container);
    }
    CHECK_NOTNULL(processed_container);

    if (pre_queue_container_calback_)
      pre_queue_container_calback_(processed_container);
    packet_queue_.push(processed_container);
  }

  inline void addGroundTruthPacket(const GroundTruthInputPacket& gt_packet) {
    ground_truth_packets_[gt_packet.frame_id_] = gt_packet;
  }

  inline void registerImageContainerPreprocessor(
      const ImageContainerPreprocesser& func) {
    image_container_preprocessor_ = func;
  }
  inline void registerPreQueueContainerCallback(
      const PreQueueContainerCallback& func) {
    pre_queue_container_calback_ = func;
  }

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

 protected:
  ThreadsafeQueue<ImageContainer::Ptr> packet_queue_;
  std::atomic_bool parallel_run_;

  std::map<FrameId, GroundTruthInputPacket> ground_truth_packets_;

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
