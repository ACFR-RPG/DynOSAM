/*
 *   Copyright (c) 2023 Jesse Morris (jesse.morris@sydney.edu.au)
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a
 copy
 *   of this software and associated documentation files (the "Software"), to
 deal
 *   in the Software without restriction, including without limitation the
 rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:

 *   The above copyright notice and this permission notice shall be included in
 all
 *   copies or substantial portions of the Software.

 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE
 *   SOFTWARE.
 */

#include "dynosam/dataprovider/DataInterfacePipeline.hpp"

#include <glog/logging.h>

#include "dynosam_common/utils/Numerical.hpp"
#include "dynosam_common/utils/SafeCast.hpp"
#include "dynosam_common/utils/Statistics.hpp"

namespace dyno {

DataInterfacePipeline::DataInterfacePipeline(bool parallel_run)
    : MIMO("data-interface"), parallel_run_(parallel_run) {}

void DataInterfacePipeline::shutdownQueues() {
  packet_queue_.shutdown();
  // call the virtual shutdown method for the derived dataprovider module
  this->onShutdown();
}

FrontendInputPacketBase::ConstPtr DataInterfacePipeline::getInputPacket() {
  if (isShutdown()) {
    return nullptr;
  }

  utils::StatsCollector queue_size_stats("data-interface_queue_size #");
  queue_size_stats.AddSample(packet_queue_.size());

  bool queue_state;
  ImageContainer::Ptr packet = nullptr;

  if (parallel_run_) {
    queue_state = packet_queue_.pop(packet);
  } else {
    queue_state = packet_queue_.popBlocking(packet);
  }

  if (!queue_state) {
    //  LOG_EVERY_N(WARNING, 10)
    //     << "Module: " << MIMO::module_name_ << " - queue is down";
    return nullptr;
  }

  CHECK(packet);

  GroundTruthInputPacket::Optional ground_truth;
  if (ground_truth_packets_.find(packet->frameId()) !=
      ground_truth_packets_.end()) {
    VLOG(5) << "Gotten ground truth packet for frame id " << packet->frameId()
            << ", timestamp=" << packet->timestamp();
    ground_truth = ground_truth_packets_.at(packet->frameId());
  }

  const Timestamp& timestamp = packet->timestamp();
  ImuMeasurements::Optional imu_meas;
  imu_meas.emplace();
  FrameAction action = getTimeSyncedImuMeasurements(timestamp, &(*imu_meas));
  switch (action) {
    case FrameAction::Use:
      CHECK(imu_meas);
      break;
    case FrameAction::Wait:
    case FrameAction::Drop:
      imu_meas.reset();
      break;
  }

  auto frontend_input =
      std::make_shared<FrontendInputPacketBase>(packet, ground_truth);
  frontend_input->imu_measurements = imu_meas;

  // in some cases (ie datasets) we know ahead of time which imu packet is
  // associated with which frame
  //  if it is provided, check that our getTimeSyncedImuMeasurements did the
  //  right thing
  if (imu_meas && imu_meas->synchronised_frame_id) {
    CHECK_EQ(packet->frameId(), imu_meas->synchronised_frame_id.value());
  }

  return frontend_input;
}

bool DataInterfacePipeline::hasWork() const {
  return !packet_queue_.empty() && !packet_queue_.isShutdown();
}

ImuInterfaceHandler::ImuInterfaceHandler()
    : imu_buffer_(-1), timestamp_last_frame_(InvalidTimestamp) {}

ImuInterfaceHandler::FrameAction
ImuInterfaceHandler::getTimeSyncedImuMeasurements(const Timestamp& timestamp,
                                                  ImuMeasurements* imu_meas) {
  if (imu_buffer_.isShutdown() || imu_buffer_.size() == 0u) {
    return FrameAction::Drop;
  }

  CHECK_NOTNULL(imu_meas);
  CHECK_LT(timestamp_last_frame_, timestamp)
      << "Image timestamps out of order: " << timestamp_last_frame_
      << "[s] (last) >= " << timestamp << "[s] (curr)";

  if (imu_buffer_.size() == 0) {
    VLOG(1) << "No IMU measurements available yet, dropping this frame.";
    return FrameAction::Drop;
  }

  if (timestamp_last_frame_ == InvalidTimestamp) {
    // TODO(Toni): wouldn't it be better to get all IMU measurements up to
    // this
    // timestamp? We should add a method to the IMU buffer for that.
    VLOG(1) << "Skipping first frame, because we do not have a concept of "
               "a previous frame timestamp otherwise.";
    timestamp_last_frame_ = timestamp;
    return FrameAction::Drop;
  }

  // // Do a very coarse timestamp correction to make sure that the IMU data
  // // is aligned enough to send packets to the front-end. This is assumed
  // // to be very inaccurate and should not be enabled without some other
  // // actual time alignment in the frontend
  // if (do_coarse_imu_camera_temporal_sync_) {
  //   ImuMeasurement newest_imu;
  //   imu_data_.imu_buffer_.getNewestImuMeasurement(&newest_imu);
  //   // this is delta = imu.timestamp - frame.timestamp so that when querying,
  //   // we get query = new_frame.timestamp + delta = frame_delta +
  //   imu.timestamp imu_timestamp_correction_ = newest_imu.timestamp_ -
  //   timestamp; do_coarse_imu_camera_temporal_sync_ = false; LOG(WARNING) <<
  //   "Computed intial coarse time alignment of "
  //                << UtilsNumerical::NsecToSec(imu_timestamp_correction_)
  //                << "[s]";
  // }

  // imu_time_shift_ can be externally, asynchronously modified.
  // Caching here prevents a nasty race condition and avoids locking
  // const Timestamp curr_imu_time_shift = imu_time_shift_ns_;
  // const Timestamp imu_timestamp_last_frame =
  //     timestamp_last_frame_ + imu_timestamp_correction_ +
  //     curr_imu_time_shift;
  // const Timestamp imu_timestamp_curr_frame =
  //     timestamp + imu_timestamp_correction_ + curr_imu_time_shift;

  const Timestamp imu_timestamp_last_frame = timestamp_last_frame_;
  const Timestamp imu_timestamp_curr_frame = timestamp;

  // NOTE: using interpolation on both borders instead of just the upper
  // as before because without a measurement on the left-hand side we are
  // missing some of the motion or overestimating depending on the
  // last timestamp's relationship to the nearest imu timestamp.
  // For some datasets this caused an incorrect motion estimate
  ThreadsafeImuBuffer::QueryResult query_result =
      imu_buffer_.getImuDataInterpolatedBorders(
          imu_timestamp_last_frame, imu_timestamp_curr_frame,
          &imu_meas->timestamps_, &imu_meas->acc_gyr_);
  // logQueryResult(timestamp, query_result);

  switch (query_result) {
    case ThreadsafeImuBuffer::QueryResult::kDataAvailable:
      break;  // handle this below
    case ThreadsafeImuBuffer::QueryResult::kDataNotYetAvailable:
      return FrameAction::Wait;
    case ThreadsafeImuBuffer::QueryResult::kQueueShutdown:
      // MISO::shutdown();
      return FrameAction::Drop;
    case ThreadsafeImuBuffer::QueryResult::kDataNeverAvailable:
      timestamp_last_frame_ = timestamp;
      return FrameAction::Drop;
    case ThreadsafeImuBuffer::QueryResult::kTooFewMeasurementsAvailable:
    default:
      return FrameAction::Drop;
  }

  timestamp_last_frame_ = timestamp;

  // adjust the timestamps for the frontend
  // imu_meas->timestamps_.array() -=
  //     imu_timestamp_correction_ + curr_imu_time_shift;
  VLOG(200) << "////////////////////////////////////////// Creating packet!\n"
            << "STAMPS IMU rows : \n"
            << imu_meas->timestamps_.rows() << '\n'
            << "STAMPS IMU cols : \n"
            << imu_meas->timestamps_.cols() << '\n'
            << "STAMPS IMU: \n"
            << imu_meas->timestamps_ << '\n'
            << "ACCGYR IMU rows : \n"
            << imu_meas->acc_gyr_.rows() << '\n'
            << "ACCGYR IMU cols : \n"
            << imu_meas->acc_gyr_.cols() << '\n'
            << "ACCGYR IMU: \n"
            << imu_meas->acc_gyr_;
  return FrameAction::Use;
}

}  // namespace dyno
