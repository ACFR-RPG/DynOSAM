/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris
 (jesse.morris@sydney.edu.au)
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

#include "simulator.hpp"

#include <random>
#include <variant>

std::mt19937 gen(42);  // Fixed seed for reproducibility

namespace dyno_testing {

using namespace dyno;

RGBDScenario::Output RGBDScenario::getOutput(FrameId frame_id) const {
  static std::random_device rd;
  // static std::mt19937 gen(rd());

  VisionImuPacket::CameraTracks camera_tracks, noisy_camera_tracks;
  VisionImuPacket::ObjectTrackMap object_tracks, noisy_object_tracks;

  GroundTruthInputPacket gt_packet;
  gt_packet.frame_id_ = frame_id;

  MotionEstimateMap motions, noisy_motions;
  const gtsam::Pose3 X_world_k = cameraPose(frame_id);
  gt_packet.X_world_ = X_world_k;

  gtsam::Pose3 noisy_X_world_k;
  gtsam::Pose3 w_T_k_1_k;
  gtsam::Pose3 noisy_w_T_k_1_k;
  if (frame_id > 0) {
    // add noise on relative transformation of camera pose using gt poses to
    // calculate gt realtive pose
    const gtsam::Pose3 X_world_k_1 = cameraPose(frame_id - 1u);
    w_T_k_1_k = X_world_k_1.inverse() * X_world_k;

    gtsam::Vector6 pose_sigmas;
    pose_sigmas.head<3>().setConstant(noise_params_.X_R_sigma);
    pose_sigmas.tail<3>().setConstant(noise_params_.X_t_sigma);
    noisy_w_T_k_1_k = dyno::utils::perturbWithNoise(w_T_k_1_k, pose_sigmas);

    CHECK(noisy_camera_poses_.exists(frame_id - 1u));
    noisy_X_world_k = noisy_camera_poses_.at(frame_id - 1u) * noisy_w_T_k_1_k;
  } else {
    noisy_X_world_k = X_world_k;
  }

  camera_tracks.X_W_k = X_world_k;
  noisy_camera_tracks.X_W_k = noisy_X_world_k;

  camera_tracks.T_k_1_k = w_T_k_1_k;
  noisy_camera_tracks.T_k_1_k = noisy_w_T_k_1_k;

  // tracklets should be uniqyue but becuase we use the DynamicPointSymbol
  // they only need to be unique per frame
  for (const auto& [object_id, object] : object_bodies_) {
    if (objectInScenario(object_id, frame_id)) {
      const gtsam::Pose3 H_world_k = object->motionWorld(frame_id);
      const gtsam::Pose3 L_world_k = object->pose(frame_id);
      TrackedPoints points_world = object->getPointsWorld(frame_id);

      VisionImuPacket::ObjectTracks object_track, noisy_object_track;

      ObjectPoseGT object_pose_gt;
      object_pose_gt.frame_id_ = frame_id;
      object_pose_gt.object_id_ = object_id;
      object_pose_gt.L_world_ = L_world_k;
      object_pose_gt.prev_H_current_world_ = H_world_k;
      gt_packet.object_poses_.push_back(object_pose_gt);

      FrameId previous_frame;
      if (frame_id > 0) {
        previous_frame = frame_id - 1u;
      } else {
        previous_frame = 0u;  // hack? should actually skip this case
      }

      object_track.H_W_k_1_k = Motion3ReferenceFrame(
          H_world_k, Motion3ReferenceFrame::Style::F2F, ReferenceFrame::GLOBAL,
          previous_frame, frame_id);
      object_track.L_W_k = L_world_k;

      gtsam::Vector6 motion_sigmas;
      motion_sigmas.head<3>().setConstant(noise_params_.H_R_sigma);
      motion_sigmas.tail<3>().setConstant(noise_params_.H_t_sigma);
      const gtsam::Pose3 noisy_H_world_k =
          dyno::utils::perturbWithNoise(H_world_k, motion_sigmas);

      noisy_object_track.H_W_k_1_k = Motion3ReferenceFrame(
          noisy_H_world_k, Motion3ReferenceFrame::Style::F2F,
          ReferenceFrame::GLOBAL, previous_frame, frame_id);
      noisy_object_track.L_W_k = L_world_k;

      // convert to status vectors
      for (const TrackedPoint& tracked_p_world : points_world) {
        auto tracklet_id = tracked_p_world.first;
        auto p_world = tracked_p_world.second;

        // currently no keypoints!
        MeasurementWithCovariance<Keypoint> keypoint(dyno::Keypoint{});
        MeasurementWithCovariance<Keypoint> noisy_keypoint(dyno::Keypoint{});

        const Point3Measurement p_camera(X_world_k.inverse() * p_world);
        const Point3Measurement p_camera_noisy = addNoiseDynamicPoint(p_camera);

        CameraMeasurement measurement(keypoint);
        CameraMeasurement measurement_noisy(noisy_keypoint);

        measurement.landmark(p_camera);
        measurement_noisy.landmark(p_camera_noisy);

        object_track.measurements.push_back(
            CameraMeasurementStatus(measurement, frame_id, tracklet_id,
                                    object_id, ReferenceFrame::LOCAL));

        noisy_object_track.measurements.push_back(
            CameraMeasurementStatus(measurement_noisy, frame_id, tracklet_id,
                                    object_id, ReferenceFrame::LOCAL));
      }

      object_tracks.insert2(object_id, object_track);
      noisy_object_tracks.insert2(object_id, noisy_object_track);
    }
  }

  // add static points
  const TrackedPoints static_points_world =
      static_points_generator_->getPointsWorld(frame_id);

  // convert to status vectors
  for (const TrackedPoint& tracked_p_world : static_points_world) {
    auto tracklet_id = tracked_p_world.first;
    auto p_world = tracked_p_world.second;

    const Point3Measurement p_camera(X_world_k.inverse() * p_world);
    Point3Measurement noisy_p_camera = addNoiseStaticPoint(p_camera);

    MeasurementWithCovariance<Keypoint> keypoint(dyno::Keypoint{});
    MeasurementWithCovariance<Keypoint> noisy_keypoint(dyno::Keypoint{});

    CameraMeasurement measurement(keypoint);
    CameraMeasurement measurement_noisy(noisy_keypoint);

    measurement.landmark(p_camera);
    measurement_noisy.landmark(noisy_p_camera);

    camera_tracks.measurements.push_back(
        CameraMeasurementStatus(measurement, frame_id, tracklet_id,
                                background_label, ReferenceFrame::LOCAL));

    noisy_camera_tracks.measurements.push_back(
        CameraMeasurementStatus(measurement_noisy, frame_id, tracklet_id,
                                background_label, ReferenceFrame::LOCAL));
  }

  VisionImuPacket::Ptr ground_truth_input = std::make_shared<VisionImuPacket>();
  VisionImuPacket::Ptr noisy_input = std::make_shared<VisionImuPacket>();

  ground_truths_.insert2(frame_id, gt_packet);
  noisy_camera_poses_.insert2(frame_id, noisy_X_world_k);

  ground_truth_input->groundTruthPacket(gt_packet);
  ground_truth_input->cameraTracks(camera_tracks);
  ground_truth_input->objectTracks(object_tracks);
  ground_truth_input->frameId(frame_id);
  ground_truth_input->timestamp(frame_id);

  noisy_input->groundTruthPacket(gt_packet);
  noisy_input->cameraTracks(noisy_camera_tracks);
  noisy_input->objectTracks(noisy_object_tracks);
  noisy_input->frameId(frame_id);
  noisy_input->timestamp(frame_id);

  return {ground_truth_input, noisy_input};
}

Point3Measurement RGBDScenario::addNoiseStaticPoint(
    const Point3Measurement& p_local) const {
  return addNoisePoint(p_local, noise_params_.static_point_noise,
                       params_.static_outlier_ratio);
}
Point3Measurement RGBDScenario::addNoiseDynamicPoint(
    const Point3Measurement& p_local) const {
  return addNoisePoint(p_local, noise_params_.dynamic_point_noise,
                       params_.dynamic_outlier_ratio);
}

Point3Measurement RGBDScenario::addNoisePoint(const Point3Measurement& p_local,
                                              const PointNoise& options,
                                              double outlier_ratio) const {
  gtsam::Point3 noisy_p_local = p_local.measurement();
  gtsam::SharedGaussian model = p_local.model();

  if (std::holds_alternative<NaivePoint3dNoiseParams>(options)) {
    NaivePoint3dNoiseParams point_noise_params =
        std::get<NaivePoint3dNoiseParams>(options);
    noisy_p_local = dyno::utils::perturbWithNoise(p_local.measurement(),
                                                  point_noise_params.sigma);

    model = gtsam::noiseModel::Isotropic::Sigma(3, point_noise_params.sigma);

  } else if (std::holds_alternative<Point3NoiseParams>(options)) {
    Point3NoiseParams point_noise_params = std::get<Point3NoiseParams>(options);
    std::tie(noisy_p_local, model) = addAnisotropicNoiseToPoint(
        noisy_p_local, point_noise_params.sigma_xy, point_noise_params.sigma_z);
  } else if (std::holds_alternative<RGBDNoiseParams>(options)) {
    LOG(FATAL) << "Not implemented!";
  }

  if (outlier_ratio > 0 && outlier_dist(gen) < outlier_ratio) {
    // simulate out of distribution noise
    noisy_p_local =
        dyno::utils::perturbWithUniformNoise(noisy_p_local, -30, 30);
  }

  CHECK(model);

  Point3Measurement noisy_model(noisy_p_local, model);
  return noisy_model;
}

std::pair<gtsam::Point3, gtsam::SharedGaussian>
RGBDScenario::addAnisotropicNoiseToPoint(const gtsam::Point3& p,
                                         double sigma_xy,
                                         double sigma_z) const {
  double z = p.z();

  // Standard deviations for noise
  double s_x = sigma_xy * z;
  double s_y = sigma_xy * z;
  double s_z = sigma_z * z * z;

  std::normal_distribution<double> dist_x(0.0, s_x);
  std::normal_distribution<double> dist_y(0.0, s_y);
  std::normal_distribution<double> dist_z(0.0, s_z);

  Eigen::Vector3d noise(dist_x(gen), dist_y(gen), dist_z(gen));

  gtsam::Vector3 sigmas;
  sigmas << s_x, s_y, s_z;

  return {p + noise, gtsam::noiseModel::Isotropic::Sigmas(sigmas)};
}

}  // namespace dyno_testing
