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

#include "dynosam/dataprovider/OMDDataProvider.hpp"

#include <glog/logging.h>

#include <filesystem>

#include "dynosam/dataprovider/DataProviderUtils.hpp"
#include "dynosam/frontend/vision/VisionTools.hpp"  //for getObjectLabels
#include "dynosam/pipeline/ThreadSafeTemporalBuffer.hpp"
#include "dynosam_common/Algorithms.hpp"
#include "dynosam_common/utils/CsvParser.hpp"
#include "dynosam_common/utils/GtsamUtils.hpp"
#include "dynosam_common/utils/OpenCVUtils.hpp"
#include "dynosam_common/viz/Colour.hpp"

namespace dyno {

// class OMDAllLoader {

// public:
//     DYNO_POINTER_TYPEDEFS(OMDAllLoader)

//     OMDAllLoader(const std::string& file_path)
//     :   rgbd_folder_path_(file_path + "/rgbd"),
//         instance_masks_folder_(file_path + "/instance_masks"),
//         optical_flow_folder_path_(file_path + "/optical_flow"),
//         vicon_file_path_(file_path + "/vicon.csv"),
//         // kalibr_file_path_(file_path + "/manufacturer.yaml"),
//         kalibr_file_path_(file_path + "/kalibr.yaml"),
//         vicon_calibration_file_path_(file_path + "/vicon.yaml")
//     {
//         utils::throwExceptionIfPathInvalid(rgbd_folder_path_);
//         utils::throwExceptionIfPathInvalid(instance_masks_folder_);
//         utils::throwExceptionIfPathInvalid(optical_flow_folder_path_);
//         utils::throwExceptionIfPathInvalid(vicon_file_path_);
//         utils::throwExceptionIfPathInvalid(kalibr_file_path_);
//         utils::throwExceptionIfPathInvalid(vicon_calibration_file_path_);

//         //first load images and size
//         //the size will be used as a refernce for all other loaders
//         //size is number of images
//         //we use flow to set the size as for this dataset, there will be one
//         less
//         //optical flow as the flow ids index t to t+1 (which is gross, I
//         know!!)
//         //TODO: code and comments replicated in ClusterSlamDataProvider
//         utils::loadFlowImagesAndSize(optical_flow_image_paths_,
//         dataset_size_);

//         //load rgb and aligned depth image file paths into rgb_image_paths_
//         and aligned_depth_image_paths_
//         //does some sanity checks
//         //as well as timestamps
//         utils::loadRGBDImagesAndTime();

//         //load instance masks
//         loadPathsInDirectory(instance_masks_image_paths_,
//         instance_masks_folder_);
//         //remove the ones up to the dataset size
//         instance_masks_image_paths_.resize(dataset_size_);

//         //want one less rgb image than optical flow
//         CHECK_EQ(optical_flow_image_paths_.size() - 1,
//         rgb_image_paths_.size());
//         CHECK_EQ(instance_masks_image_paths_.size(),
//         optical_flow_image_paths_.size());

//         setGroundTruthPacketFromVicon(ground_truth_packets_);

//         setIntrisicsAndTransforms();
//     }

//     size_t size() const {
//         return dataset_size_;
//     }

//     cv::Mat getRGB(size_t idx) const {
//         CHECK_LT(idx, rgb_image_paths_.size());

//         cv::Mat rgb;
//         utils::loadRGB(rgb_image_paths_.at(idx), rgb);
//         CHECK(!rgb.empty());

//         //this set of images are loaded as 8UC4
//         // CHECK_EQ(rgb.type(), CV_8UC4) << "Somehow the image type has
//         changed...";
//         // rgb.convertTo(rgb, CV_8UC3);

//         //debug check -> draw keypoints on image!
//         // const LandmarksMap& kps_map = left_landmarks_map_.at(idx);

//         // for(const auto&[landmark_id, kp] : kps_map) {
//         //     const auto cluster_id = landmark_mapping_.at(landmark_id);

//         //     cv::Point2f pt(utils::gtsamPointToCv(kp));
//         //     utils::drawCircleInPlace(rgb, pt,
//         ColourMap::getObjectColour(cluster_id));
//         // }

//         return rgb;
//     }

//     cv::Mat getOpticalFlow(size_t idx) const {
//         CHECK_LT(idx,optical_flow_image_paths_.size());

//         cv::Mat flow;
//         utils::loadFlow(optical_flow_image_paths_.at(idx), flow);
//         CHECK(!flow.empty());
//         return flow;
//     }

//     cv::Mat getInstanceMask(size_t idx) const {
//         CHECK_LT(idx, rgb_image_paths_.size());
//         CHECK_LT(idx, dataset_size_);

//         cv::Mat mask, relabelled_mask;
//         utils::loadMask(instance_masks_image_paths_.at(idx), mask);
//         CHECK(!mask.empty());

//         associateGTWithObject(mask, getDepthImage(idx), idx,
//         relabelled_mask);

//         // cv::Mat mask_viz =
//         ImageType::SemanticMask::toRGB(relabelled_mask);

//         // cv::imshow("Relabelled mask", mask_viz);
//         // cv::waitKey(1);

//         return relabelled_mask;
//     }

//     cv::Mat getDepthImage(size_t idx) const {
//         CHECK_LT(idx, aligned_depth_image_paths_.size());

//         cv::Mat depth;
//         utils::loadDepth(aligned_depth_image_paths_.at(idx), depth);
//         // utils::loadRGB(aligned_depth_image_paths_.at(idx), depth);
//         CHECK(!depth.empty());
//         // CHECK(depth.type() == CV_16UC1);
//         // depth.convertTo(depth, CV_64F);

//         //The D435 publishes depth in "16-bit unsigned integers in millimeter
//         resolution."
//         //TODO: paramterise and check why this is different for OMD
//         //why is dyno-sam so different that it needs the baseline scale
//         factor as well as the scaling term
//         //is this just depth to dispartiy?
//         //imD.at<float>(i,j) = mbf/(imD.at<float>(i,j)/mDepthMapFactor);

//          const cv::Mat K = rgbd_camera_params_.getCameraMatrix();
//         //expect D to be zeros
//         const cv::Mat D = rgbd_camera_params_.getDistortionCoeffs();
//         cv::Mat new_K;
//         new_K = cv::getOptimalNewCameraMatrix(K, D,
//         rgbd_camera_params_.imageSize(), 1); cv::Mat undistorted_image =
//         depth;
//         // cv::undistort(depth, undistorted_image, K, D, new_K);

//         for (int i = 0; i < undistorted_image.rows; i++)
//         {
//             for (int j = 0; j < undistorted_image.cols; j++)
//             {
//                 // LOG(INFO) << undistorted_image.at<double>(i, j);
//                 if (undistorted_image.at<double>(i, j) <= 0) {
//                     undistorted_image.at<double>(i, j) = 0;
//                 }
//                 else {
//                     // baseline = 50mm (0.05)
//                     //bfm = 0.05 * fx (618.3587036132812)
//                     undistorted_image.at<double>(i, j) =
//                     (undistorted_image.at<double>(i, j) / 1000.0);
//                 }
//             }
//         }

//         // // CHECK_EQ()
//         // //check image sizes are the same!

//         // // LOG(INFO) << "Processing depth map";
//         // for (int i = 0; i < undistorted_image.rows; i++) {
//         //     for (int j = 0; j < undistorted_image.cols; j++) {
//         //         if (undistorted_image.at<double>(i,j)<0) {
//         //             undistorted_image.at<double>(i,j)=0;
//         //         }
//         //         else
//         //         {
//         //             undistorted_image.at<double>(i,j) =
//         (undistorted_image.at<double>(i,j) / 1000.0);
//         //             // undistorted_image.at<double>(i,j) = 387.5744 /
//         (undistorted_image.at<double>(i,j) / 1000.0);
//         //         }

//         //         if(std::isinf(undistorted_image.at<double>(i,j))) {
//         //             undistorted_image.at<double>(i,j) = 0;
//         //         }
//         //     }
//         // }
//         // undistorted_image.convertTo(undistorted_image, CV_64F);
//         // LOG(INFO ) << "Finished processing";

//         // depth /= 1000.0;

//         // cv::Mat depth;
//         // const_disparity.copyTo(depth);
//         return undistorted_image;
//     }

//     const GroundTruthInputPacket& getGtPacket(size_t idx) const {
//         CHECK(ground_truth_packets_.exists(idx)) << " Idx not contained in gt
//         packet: " << idx; return ground_truth_packets_.at(idx);
//     }

//     const CameraParams& getLeftCameraParams() const {
//         return rgbd_camera_params_;
//     }

//     double getTimestamp(size_t idx) {
//         return static_cast<double>(times_.at(idx));
//     }

// private:
//     void utils::loadFlowImagesAndSize(std::vector<std::string>& images_paths,
//     size_t& dataset_size) {
//         std::vector<std::filesystem::path> files_in_directory =
//         getAllFilesInDir(optical_flow_folder_path_); dataset_size =
//         files_in_directory.size(); CHECK_GT(dataset_size, 0);

//         for (const std::string file_path : files_in_directory) {
//             utils::throwExceptionIfPathInvalid(file_path);
//             images_paths.push_back(file_path);
//         }
//    }

//    inline Timestamp toTimestamp(int time_sec, int time_nsec) const {
//     return static_cast<Timestamp>(time_sec) +
//     static_cast<Timestamp>(time_nsec) / 1e+9;
//    }

//    void utils::loadRGBDImagesAndTime() {
//         // from dataset contains *_aligned_depth.png and *_color.png
//         // need to construct both
//         auto image_files = getAllFilesInDir(rgbd_folder_path_);

//         //maps the image label prefix number with the two image paths - rgb
//         (.first) and depth (.second)
//         //this enables us to ensure that each image number (which use the
//         same prefix) has both
//         //rgb and depth images available in the dataset
//         gtsam::FastMap<int, std::pair<std::string, std::string>>
//         image_label_to_path_pair;

//         const auto colour_substring = "color.png";
//         const auto depth_substring = "aligned_depth.png";

//         namespace fs = std::filesystem;

//         for(const std::string file_path : image_files) {
//             //extract image id
//             //this should be in the form (XXXX_aligned_depth or XXXX_color)
//             //get the filename from the file path (as the full file path
//             contains "_" in other places
//             //e.g
//             root/data/omm/swinging_4_unconstrained/rgbd/010707_color.png
//             //we just want the 010707_color.png componenent of the file path
//             auto split_lines =
//             trimAndSplit(std::string(fs::path(file_path).filename()), "_");
//             int image_number = -1;

//             try {
//                 image_number = std::stoi(split_lines.at(0));
//             }
//             catch(const std::invalid_argument& ex) {
//                 //this fails on the csv file
//             }

//             //if image number extracted successfully
//             if(image_number != -1) {
//                  if(!image_label_to_path_pair.exists(image_number)) {
//                     image_label_to_path_pair.insert2(image_number,
//                     std::make_pair("", ""));
//                 }

//                 auto& file_path_pair =
//                 image_label_to_path_pair.at(image_number);

//                 if(file_path.find(colour_substring) != std::string::npos) {
//                     //is rgb image
//                     file_path_pair.first = file_path;
//                 }
//                 else if (file_path.find(depth_substring) !=
//                 std::string::npos) {
//                     //is depth image
//                     file_path_pair.second = file_path;
//                 }
//             }
//             //special case - rgbd.csv file is inside this folder
//             //and gives timestamp per frame number
//             else if(file_path.find("rgbd.csv") != std::string::npos) {
//                 loadTimestampsFromRGBDCsvFile(file_path);
//             }
//             else {
//                 LOG(FATAL) << "Could not load rgbd image at path " <<
//                 file_path;
//             }
//         }

//         //go through the map and add all image paths that have both image
//         paths
//         //assume everything is ordered by FastMap
//         for(const auto& [image_num, file_path_pair] :
//         image_label_to_path_pair) {
//             (void)image_num;
//             if(file_path_pair.first.empty() || file_path_pair.second.empty())
//             {
//                 continue;
//             }

//             rgb_image_paths_.push_back(file_path_pair.first);
//             aligned_depth_image_paths_.push_back(file_path_pair.second);

//             //want one less than the number of flow images
//             if(rgb_image_paths_.size() + 1 >= dataset_size_) {
//                 break;
//             }
//         }

//         CHECK_EQ(rgb_image_paths_.size(), aligned_depth_image_paths_.size());
//         CHECK_EQ(times_.size(), rgb_image_paths_.size());

//         LOG(INFO) << "Loaded " << rgb_image_paths_.size() << " rgbd images";
//    }

//    void loadTimestampsFromRGBDCsvFile(const std::string& file_path) {
//         //exepect a csv file with the header [frame_num, time_sec, time_nsec]
//         //inside the rgbd folder
//         //TODO: for now dont use a csv reader!!
//         std::ifstream file(file_path);
//         CsvReader csv_reader(file);
//         auto it = csv_reader.begin();
//         //skip header
//         ++it;

//         for(; it != csv_reader.end(); it++) {
//             const auto row = *it;
//             int frame_num = row.at<int>(0);
//             int time_sec = row.at<int>(1);
//             int time_nsec = row.at<int>(2);

//             Timestamp time = toTimestamp(time_sec, time_nsec);

//             times_.push_back(time);
//             timestamp_frame_map_.insert({frame_num, time});

//             //want one less than the number of flow images
//             if(times_.size() + 1 >= dataset_size_) {
//                 break;
//             }
//         }
//         LOG(INFO) << "Loaded " << times_.size() << " timestamps from RGBD csv
//         file: " << file_path;

//    }

//    struct ViconPoseData {
//         Timestamp timestamp;
//         gtsam::Pose3 T_world_object; //or T_OA using the omd dataset
//         notation, the inverse of what is given by the dataset (T_AO) ObjectId
//         object_id;
//    };

//     void setGroundTruthPacketFromVicon(GroundTruthPacketMap&
//     ground_truth_packets) {
//         LOG(INFO) << "Loading object/camera pose gt from vicon: " <<
//         vicon_file_path_; std::ifstream file(vicon_file_path_); CsvReader
//         csv_reader(file); auto it = csv_reader.begin();
//         //skip header
//         ++it;

//         std::vector<ViconPoseData> vicon_pose_data;
//         gtsam::FastMap<FrameId, std::vector<ObjectPoseGT>> temp_object_poses;
//         //vicon timestamps that will replace the internal variable times_
//         (which are initally camera timestamps)
//         //should have the same length
//         std::vector<Timestamp> vicon_timestamps;

//         //  np.array([[1, 0, 0, 0],
//         // [0, 0, -1, 0],
//         // [0, 1, 0, 0],
//         // [0, 0, 0, 1]], dtype=np.float32)
//         // const gtsam::Pose3 T_cv_robotic =
//         gtsam::Pose3(gtsam::Rot3::RzRyRx(M_PI/2.0, 0, M_PI/2.0),
//         gtsam::traits<gtsam::Point3>::Identity()); const gtsam::Pose3
//         T_cv_robotic = gtsam::Pose3(
//             gtsam::Rot3(
//                 1, 0, 0,
//                 0, 0, -1,
//                 0, 1, 0),
//             gtsam::traits<gtsam::Point3>::Identity());
//         const gtsam::Pose3 T_omd_cv =
//         gtsam::Pose3(gtsam::Rot3::RzRyRx(-M_PI/2.0, 0, 0),
//         gtsam::traits<gtsam::Point3>::Identity()) ;

//         for(; it != csv_reader.end(); it++) {
//             const auto row = *it;
//             //Though the stereo camera, RGB-D camera, and IMU were recorded
//             on the same machine,
//             //they are not hardware synchronized. The Vicon was recorded on a
//             separate system with an unknown temporal offset
//             //and clock drift.
//             int time_sec = row.at<int>(0);
//             int time_nsec = row.at<int>(1);
//             Timestamp vicon_time = toTimestamp(time_sec, time_nsec);
//             //object is in the form boxX or sensor payload (the camera)
//             std::string object = row.at<std::string>(2);
//             ObjectId object_id;
//             if(object == "sensor_payload") {
//                 //camera
//                 object_id = 0;
//             }
//             else if(object.find("box") != std::string::npos) {
//                 //should be in boxX, so extract the last character as a
//                 number ObjectId box_id = object.back() - '0';
//                 CHECK_NE(box_id, 0); ///canot be 0, this is reserved for the
//                 camera object_id = box_id;
//             }
//             else {
//                 LOG(FATAL) << "Unknown object type: " << object;
//             }

//             double qx = row.at<double>(3);
//             double qy = row.at<double>(4);
//             double qz = row.at<double>(5);
//             double qw = row.at<double>(6);

//             double tx = row.at<double>(7);
//             double ty = row.at<double>(8);
//             double tz = row.at<double>(9);

//             //transformation from object to world frame where world is the
//             vicon frame
//             //given as T_AO in dataset paper
//             gtsam::Pose3 T_object_world_robotic(gtsam::Rot3(qw, qx, qy, qz),
//             gtsam::Point3(tx, ty, tz));
//             //world is the vicon frame but is defined in the ROBOTIC
//             convention (but with y forward, becuase reasons...?)
//             //see figure.2. in paper where x is to the right and z is up
//             gtsam::Pose3 T_world_object_robotic = T_object_world_robotic;
//             //rotate the reference 90\deg around z to put into real robotic
//             convention; z up , x forward, y left
//             //so apply a -90 transformation
//             // T_world_object_robotic = gtsam::Pose3(gtsam::Rot3::RzRyRx(0,
//             0, M_PI_2), gtsam::traits<gtsam::Point3>::Identity()) *
//             T_world_object_robotic;

//             // //rotation that takes a transform from the robotic convention
//             //this is from actually the omd robotic convention
//             // const gtsam::Pose3 T_cv_robotic = gtsam::Pose3(
//             //     gtsam::Rot3(
//             //         1, 0, 0,
//             //         0, 0, -1,
//             //         0, 1, 0
//             //     ),
//             //     gtsam::traits<gtsam::Point3>::Identity());
//             // gtsam::Pose3 T_world_object_camera = T_cv_robotic *
//             T_world_object_robotic * T_cv_robotic.inverse();

//             ViconPoseData vicon_data;
//             vicon_data.timestamp = vicon_time;
//             vicon_data.object_id = object_id;
//             vicon_data.T_world_object = T_object_world_robotic;
//             // vicon_data.T_world_object = T_world_object_robotic;
//             vicon_pose_data.push_back(vicon_data);

//         }

//         // //convert camera times to map for lookup
//         ThreadsafeTemporalBuffer<FrameId> camera_timestamp_buffer;
//         for(const auto& [frame_id, timestamp] : timestamp_frame_map_) {
//             camera_timestamp_buffer.addValue(timestamp, frame_id);
//         }
//         //dataset size is number of optical flow images - we want one less
//         camera image!! CHECK_EQ(camera_timestamp_buffer.size(), dataset_size_
//         - 1); const auto earliest_camera_timestamp =
//         camera_timestamp_buffer.getOldestTimestamp(); const auto
//         latest_camera_timestamp =
//         camera_timestamp_buffer.getNewestTimestamp();

//         //this is very slow!!!
//         for(size_t i = 0; i  < vicon_pose_data.size(); i++) {
//             const auto vicon_data = vicon_pose_data.at(i);
//         // for(const auto& vicon_data : vicon_pose_data) {
//             const ObjectId object_id = vicon_data.object_id;
//             const gtsam::Pose3& T_world_object = vicon_data.T_world_object;
//             const Timestamp& vicon_timestamp = vicon_data.timestamp;

//             //dont include if less than first or greater than last camera
//             time if (vicon_timestamp < earliest_camera_timestamp ||
//             vicon_timestamp > latest_camera_timestamp) {
//                 continue;
//             }

//             //get associated
//             //get closest camera timestamp to the vicon timestamp - this is
//             the value we want to interpolate
//             //to as as this will be the timestamp of the frame we actually
//             use
//             //the min delta should really be the delta between camera frames.
//             According to the associated RA-L
//             //paper the rate of the RGBD sensor is 30Hz = 0.033
//             //the rate of the vicon system is 200Hz
//             constexpr static auto approx_rgbd_frame_rate = 0.033;
//             FrameId frame_id;
//             Timestamp camera_timestamp;
//             if(!camera_timestamp_buffer.getNearestValueToTime(vicon_timestamp,
//             approx_rgbd_frame_rate, &frame_id, &camera_timestamp)) {
//                 continue;
//                 //TODO: throw warning?
//             }

//             // LOG(INFO) << std::setprecision(15) <<  camera_timestamp;

//             //get associated camera timestamp

//             if (fpEqual(camera_timestamp, vicon_timestamp)) {

//                 if(!ground_truth_packets.exists(frame_id)) {
//                     GroundTruthInputPacket gt_packet;
//                     gt_packet.frame_id_ = frame_id;
//                     gt_packet.timestamp_ = camera_timestamp;
//                     ground_truth_packets.insert2(frame_id, gt_packet);
//                 }

//                 if(!temp_object_poses.exists(frame_id)) {
//                     temp_object_poses.insert2(frame_id,
//                     std::vector<ObjectPoseGT>{});
//                 }

//                 GroundTruthInputPacket& gt_packet =
//                 ground_truth_packets.at(frame_id);

//                 std::vector<ObjectPoseGT>& tmp_object_vector =
//                 temp_object_poses.at(frame_id);

//                 if(object_id == 0) {
//                     //if object is sensor payload we need to apply extra
//                     transforms to get T_world_depthcam from T_world_object
//                     //in this case T_world_object = T_world_apparatus (T_OA)
//                     as the object is the apparatus frame measured from the
//                     vicon
//                     //T_world_depthcam =  T_world_apparatus *
//                     T_apparatus_leftstereo * T_leftstereo_depthcam
//                     //where T_apparatus_leftstereo is given from the
//                     vicon.yaml config
//                     //and T_leftstereo_depthcam is constructed from the
//                     kalibr config
//                     //this is actually the omd robotic convention
//                     // LOG(INFO) << T_cv_robotic;
//                     //apply frame convention change to cancel out the frame
//                     convention change that is inbuilt into T_apparatus_rgbd_
//                     //specifically this happens in the T_apparatus_left
//                     // gt_packet.X_world_ = T_world_object * T_cv_robotic *
//                     T_apparatus_rgbd_;
//                     // gt_packet.X_world_ = T_world_object *
//                     T_apparatus_rgbd_;
//                     // gt_packet.X_world_ = T_leftstereo_depthcam_.inverse()
//                     * T_apparatus_left_.inverse() * T_world_object;
//                     // gt_packet.X_world_ = T_leftstereo_depthcam_.inverse()
//                     * T_apparatus_left_.inverse() * T_world_object;

//                     //T_DO (relates rgbD camera to the origin frame O)
//                     //T_DO = T_AD^{-1} * T_AO
//                     // gt_packet.X_world_ = T_apparatus_rgbd_.inverse() *
//                     T_world_object;
//                     // gt_packet.X_world_ =  T_apparatus_rgbd_.inverse() *
//                     T_world_object; gt_packet.X_world_ =  T_world_object *
//                     T_apparatus_rgbd_; gt_packet.X_world_ = T_cv_robotic *
//                     gt_packet.X_world_ * T_cv_robotic.inverse();

//                     // gt_packet.X_world_ = T_world_object *
//                     T_apparatus_rgbd_;
//                     //  gt_packet.X_world_ = T_world_object;
//                      //now in camera convention!!!
//                     // gt_packet.X_world_ = camera_to_world.inverse() *
//                     gt_packet.X_world_;
//                 }
//                 else {
//                     ObjectPoseGT object_pose_gt;
//                     object_pose_gt.frame_id_ = frame_id;
//                     object_pose_gt.object_id_ = object_id;
//                     //if this is object, the transform is the object as seen
//                     in the vicon frame
//                     //we want it in the depth sensor frame
//                     //T_world_object -> T_vicon_object
//                     object_pose_gt.L_world_ =  T_cv_robotic * T_world_object
//                     * T_cv_robotic.inverse();
//                     // object_pose_gt.L_world_ = camera_to_world.inverse() *
//                     object_pose_gt.L_world_;
//                     //cannot set L_camera yet as we might not have the camera
//                     pose

//                     //if object is already in the vector dont add
//                     //this is 'wrong' but currently we dont interpolate the
//                     gt data auto it_this =
//                     std::find_if(tmp_object_vector.begin(),
//                     tmp_object_vector.end(),
//                         [=](const ObjectPoseGT& gt_object) { return
//                         gt_object.object_id_ == object_id; });
//                     if(it_this == tmp_object_vector.end()) {
//                         tmp_object_vector.push_back(object_pose_gt);
//                     }
//                 }
//             }
//             else {
//                 LOG(FATAL) << "Currently expecting vicon and camera tiemstamp
//                 data to have at least one synched timestamp per frame!!";
//             }

//             //it seems that everything is very fast so (maybe a hack?) we'll
//             just take the cloestst timestamp

//         }

//         CHECK_EQ(ground_truth_packets.size(), times_.size());

//         //assumes we get one for every frame!!!!?
//         gtsam::Pose3 initial_pose = gtsam::Pose3::Identity();
//         bool set_initial_pose = false;
//         FrameId previous_frame = 0;
//         for(auto& [frame_id, gt_packet] : ground_truth_packets) {
//             vicon_timestamps.push_back(gt_packet.timestamp_);

//             if(!set_initial_pose) {
//                 initial_pose = gt_packet.X_world_;
//                 set_initial_pose = true;
//             }

//             // offset initial pose so we start at "0, 0, 0"
//             gtsam::Pose3 X_world_aligned = initial_pose.inverse() *
//             gt_packet.X_world_;
//             //put into cv convention with Z-axis forward
//             // X_world_aligned = T_omd_cv * X_world_aligned *
//             T_omd_cv.inverse();

//             //get object vector for the tmp list
//             auto& unprocessed_objects =
//             temp_object_poses.at(gt_packet.frame_id_); for(auto&
//             object_pose_gt : unprocessed_objects) {

//                 // // pose of the object in the camera frame usign the vicon
//                 world frame and not our aligned one

//                 // using the original pose of the camera put into camera
//                 frame gtsam::Pose3 object_pose_camera =
//                 gt_packet.X_world_.inverse() * object_pose_gt.L_world_;
//                 gtsam::Pose3 object_pose_world_aligned = X_world_aligned *
//                 object_pose_camera; gtsam::Pose3 object_pose_camera_aligned =
//                 X_world_aligned.inverse() * object_pose_world_aligned;

//                 // gtsam::Pose3 object_pose_world_aligned =
//                 object_pose_gt.L_world_ * X_world_aligned * T_omd_cv;
//                 // gtsam::Pose3 object_pose_camera_aligned =
//                 X_world_aligned.inverse() * object_pose_world_aligned;

//                 object_pose_gt.L_camera_ = object_pose_camera_aligned;
//                 object_pose_gt.L_world_ = object_pose_world_aligned;
//                 CHECK_EQ(frame_id, object_pose_gt.frame_id_);
//             }

//             //update camera pose gt with the aligned one
//             gt_packet.X_world_ = X_world_aligned;
//             gt_packet.object_poses_ = unprocessed_objects;

//             if(frame_id > 0) {
//                 const GroundTruthInputPacket& previous_gt_packet =
//                 ground_truth_packets.at(frame_id - 1);
//                 gt_packet.calculateAndSetMotions(previous_gt_packet);
//                 CHECK_EQ(frame_id - 1, previous_frame) << "Frames are not in
//                 ascending order!!!";
//             }

//             previous_frame = frame_id;
//         }

//         CHECK_EQ(ground_truth_packets.size(), times_.size());
//         CHECK_EQ(vicon_timestamps.size(), times_.size());

//         //update times with vicon timestamps as this is what is used
//         externally by the dataprovider times_ = vicon_timestamps;
//     }

//     void setIntrisicsAndTransforms() {
//         //using manufacturer
//         //cam0 -> rgbd
//         //cam1 -> stereo left
//         //cam2 -> stereo right
//         //kalirb file gives the rigid body transforms from the sensors to the
//         apparatus frame
//         //cam0 -> stereo left
//         //cam1 -> stereo right
//         //cam2 -> rgbd
//         //using the camera (opencv) convention, ie. with z forward
//         YamlParser yaml_parser(kalibr_file_path_);

//         // std::vector<double> v_cam2_cam1;
//         // yaml_parser.getNestedYamlParam("cam2", "T_cn_cnm1", &v_cam2_cam1);
//         // gtsam::Pose3 T_cam1_cam2 =
//         utils::poseVectorToGtsamPose3(v_cam2_cam1).inverse();
//         // gtsam::Pose3 T_left_right = T_cam1_cam2;

//         std::vector<double> v_cam1_cam0;
//         yaml_parser.getNestedYamlParam("cam1", "T_cn_cnm1", &v_cam1_cam0);
//         gtsam::Pose3 T_cam0_cam1 =
//         utils::poseVectorToGtsamPose3(v_cam1_cam0).inverse(); const
//         gtsam::Pose3 T_left_right = T_cam0_cam1;

//         std::vector<double> v_cam2_cam1;
//         yaml_parser.getNestedYamlParam("cam2_undistort", "T_cn_cnm1",
//         &v_cam2_cam1);
//         //transform from cam2 into camera 1 frame
//         gtsam::Pose3 T_cam1_cam2 =
//         utils::poseVectorToGtsamPose3(v_cam2_cam1).inverse(); const
//         gtsam::Pose3 T_right_rgbd = T_cam1_cam2;

//         // // //transformation from cam2 INTO cam0
//         // // //in this case camera2 is the RGBD camera and cam0 is the
//         stereo left camera
//         // // gtsam::Pose3 T_cam0_cam2 = T_cam0_cam1 * T_cam1_cam2;
//         // // T_leftstereo_depthcam_ = T_cam0_cam2;

//         // //now load apparaturs to left camera (cam0) transform to put cam0
//         into the sensor frame (or A) YamlParser
//         vicon_yaml_parser(vicon_calibration_file_path_); std::vector<double>
//         v_apparatus_left; vicon_yaml_parser.getYamlParam("T_apparatus_left",
//         &v_apparatus_left);
//         //transformation from cam0 (left) into the apparatus (A) frame
//         gtsam::Pose3 T_apparatus_cam0 =
//         utils::poseVectorToGtsamPose3(v_apparatus_left);
//         // //this contains a (partial?) frame change convention from opencv
//         to omd's robotic convention T_apparatus_left_ = T_apparatus_cam0;
//         // T_leftstereo_depthcam_ = T_left_right * T_right_rgbd;
//         // T_apparatus_rgbd_ = T_apparatus_left_;
//         T_apparatus_rgbd_ = T_apparatus_left_ * T_left_right * T_right_rgbd;

//         //NOTE: expect the rgbd camera to always be cam2
//         CameraParams::IntrinsicsCoeffs intrinsics;
//         std::vector<double> intrinsics_v;
//         yaml_parser.getNestedYamlParam("cam2_undistort", "intrinsics",
//         &intrinsics_v);
//         // yaml_parser.getNestedYamlParam("cam0", "intrinsics",
//         &intrinsics_v); CHECK_EQ(intrinsics_v.size(), 4u);
//         intrinsics.resize(4u);
//         // Move elements from one to the other.
//         std::copy_n(std::make_move_iterator(intrinsics_v.begin()),
//                     intrinsics.size(),
//                     intrinsics.begin());

//         CameraParams::DistortionCoeffs distortion({0, 0, 0, 0});

//         std::vector<int> resolution;
//         yaml_parser.getNestedYamlParam("cam2_undistort", "resolution",
//         &resolution); CHECK_EQ(resolution.size(), 2); cv::Size
//         image_size(resolution[0], resolution[1]);

//         std::string distortion_model, camera_model;
//         yaml_parser.getNestedYamlParam("cam2_undistort", "camera_model",
//         &camera_model); auto model =
//         CameraParams::stringToDistortion("radtan", camera_model);

//         rgbd_camera_params_ = CameraParams(
//             intrinsics,
//             distortion,
//             image_size,
//             model
//         );

//         LOG(INFO) << rgbd_camera_params_.toString();

//     }

//     void associateGTWithObject(const cv::Mat& instance_mask, const cv::Mat&
//     depth, FrameId frame_id, cv::Mat& relabelled_mask) const {
//         const GroundTruthInputPacket& gt_packet = getGtPacket(frame_id);
//         const ObjectIds gt_object_ids = gt_packet.getObjectIds();

//         ObjectIds object_ids = vision_tools::getObjectLabels(instance_mask);
//         const size_t n = object_ids.size();

//         instance_mask.copyTo(relabelled_mask);

//         if(n == 0) {
//             return;
//         }

//         //instance id (track id -> all keypoints for that cluster)
//         std::set<ObjectId> instance_id_set;
//         gtsam::FastMap<ObjectId, Keypoints> instance_kps;
//         gtsam::FastMap<ObjectId, Landmarks> instance_lmks;

//         Camera camera(rgbd_camera_params_);

//         //sample sparsely across image and fill instance_kps
//         int step = 3;
//         for (int i = 0; i < instance_mask.rows - step; i = i + step)
//         {
//             for (int j = 0; j < instance_mask.cols - step; j = j + step)
//             {
//                 ObjectId instance_label = instance_mask.at<ObjectId>(i, j);
//                 if(instance_label == background_label) {
//                     continue;
//                 }
//                 CHECK_NE(instance_label, background_label);

//                 instance_id_set.insert(instance_label);

//                 Keypoint keypoint(j, i);

//                 if(!instance_kps.exists(instance_label)) {
//                     instance_kps.insert2(instance_label, Keypoints{});
//                 }
//                 instance_kps.at(instance_label).push_back(keypoint);

//                 //lmk in camera frame
//                 const Depth d = functional_keypoint::at<Depth>(keypoint,
//                 depth); Landmark lmk; camera.backProject(keypoint, d, &lmk);

//                 if(!instance_lmks.exists(instance_label)) {
//                     instance_lmks.insert2(instance_label, Landmarks{});
//                 }
//                 instance_lmks.at(instance_label).push_back(lmk);
//             }
//         }

//         std::vector<int> instance_ids(instance_id_set.begin(),
//         instance_id_set.end()); CHECK_EQ(instance_ids.size(), n);

//         //want to assign instance_ids to gt labels
//         const size_t m = gt_object_ids.size();

//         Eigen::MatrixXd cost;
//         cost.resize(n, m);

//         for(size_t rows = 0; rows < n; rows++) {
//             int instance_id = instance_ids.at(rows);
//             // all lmks in the camera frame for the detected mask
//             auto sampled_lmks = instance_lmks.at(instance_id);
//             Landmark avg_lmk_camera = gtsam::mean(sampled_lmks);
//             Landmark avg_lmk_world = gt_packet.X_world_ * avg_lmk_camera;

//             for(size_t cols = 0; cols < m; cols++) {
//                 // LOG(INFO) << cols;
//                 ObjectId gt_object_id = gt_object_ids.at(cols);
//                 // LOG(INFO) << gt_object_id;

//                 ObjectPoseGT gt_object_pose;
//                 CHECK(gt_packet.getObject(gt_object_id, gt_object_pose));

//                 gtsam::Point3 gt_translation =
//                 gt_object_pose.L_world_.translation();

//                 //distance between the gt object pose (translation) and the
//                 cluster of detected points cost(rows, cols) =
//                 gtsam::distance3(gt_translation, avg_lmk_world);

//             }
//         }

//         Eigen::VectorXi assignment;
//         internal::HungarianAlgorithm().solve(cost, assignment);

//         // LOG(INFO) << " With " << n << " original ids and " << m << " gt
//         ids";

//         ObjectIds old_labels = instance_ids;
//         ObjectIds new_labels;
//         for(size_t i = 0; i < assignment.size(); i++) {
//             int j = assignment[i];

//             // //the i-th object is assignment to the j-th cluster
//             // ObjectId instance_id = instance_ids.at(i);
//             int assigned_object_id = gt_object_ids.at(j);
//             new_labels.push_back(assigned_object_id);

//             // LOG(INFO) << "Relabelled " << instance_id << " to " <<
//             assigned_object_id;

//             // const cv::Mat object_mask = relabelled_mask == instance_id;
//             // relabelled_mask.setTo(cv::Scalar(assigned_object_id),
//             object_mask);
//         }

//         vision_tools::relabelMasks(instance_mask, relabelled_mask,
//         old_labels, new_labels);
//         // LOG(INFO) << "Done";

//     }

// private:
//     const std::string rgbd_folder_path_;
//     const std::string instance_masks_folder_;
//     const std::string optical_flow_folder_path_;
//     const std::string vicon_file_path_; //vicon measurement (.csv) file
//     const std::string kalibr_file_path_;
//     const std::string vicon_calibration_file_path_; //vicon calibration
//     (.yaml) file giving transform between left stereo and apparatus

//     std::vector<std::string> rgb_image_paths_; //index from 0 to match the
//     naming convention of the dataset std::vector<std::string>
//     aligned_depth_image_paths_; std::vector<Timestamp> times_; //loaded from
//     rgbd.csv file and associated with rgbd images. Should be the same length
//     as rgb/aligned depth files
//     //timestamps between vicon (gt) and camera (from rgbd) are not
//     synchronized!!
//     //camera frame ids to vicon frame ids
//     gtsam::FastMap<FrameId, Timestamp> timestamp_frame_map_;

//     std::vector<std::string> optical_flow_image_paths_;
//     std::vector<std::string> instance_masks_image_paths_;

//     GroundTruthPacketMap ground_truth_packets_;
//     size_t dataset_size_; //set in setGroundTruthPacket. Reflects the number
//     of files in the /optical_flow folder which is one per frame

//     //below are set in the setIntrisicsAndTransforms
//     CameraParams rgbd_camera_params_;
//     gtsam::Pose3 T_apparatus_left_; //T_AL or the transform from the left
//     stereo to the apparaturs frame, as given in vicon.yaml gtsam::Pose3
//     T_leftstereo_depthcam_; //T_LD or the transfrom from the depthcam to the
//     left stereo. Given in the kalibr.yaml gtsam::Pose3 T_apparatus_rgbd_;
//     //T_AL * T_LD = T_AD

// };

class OMDOldAllLoader {
 public:
  DYNO_POINTER_TYPEDEFS(OMDOldAllLoader)

  OMDOldAllLoader(const std::string& file_path) {
    loadAll(file_path);
    setCameraParams(file_path);
  }

  size_t size() const { return rgb_file_names_.size(); }

  double getTimestamp(size_t idx) {
    return static_cast<double>(timestamps_.at(idx));
  }

  cv::Mat getRGB(size_t idx) const {
    CHECK_LT(idx, rgb_file_names_.size());

    cv::Mat rgb;
    utils::loadRGB(rgb_file_names_.at(idx), rgb);
    CHECK(!rgb.empty()) << "Empty at  " << rgb_file_names_.at(idx);

    img_size_ = rgb.size();
    return rgb;
  }

  cv::Mat getDepthImage(size_t idx) const {
    CHECK_LT(idx, depth_file_names_.size());

    cv::Mat disp;
    utils::loadDepth(depth_file_names_.at(idx), disp);
    CHECK(!disp.empty());

    constexpr auto depth_type = ImageType::Depth::OpenCVType;
    cv::Mat depth_image = cv::Mat::zeros(disp.size(), depth_type);

    const auto baseline = base_line_;
    const auto fx = rgbd_camera_params_.fx();

    // Get depth from disparity
    for (int i = 0u; i < disp.rows; i++) {
      // Loop over rows
      const double* disp_ptr = disp.ptr<double>(i);
      double* depth_ptr = depth_image.ptr<double>(i);

      for (int j = 0u; j < disp.cols; j++) {
        // Loop over cols
        const double depth =
            (baseline * fx) / (static_cast<double>(disp_ptr[j]) / 256.0);
        *(depth_ptr + j) = depth;
      }
    }

    return depth_image;
  }

  cv::Mat getOpticalFlow(size_t idx) const {
    CHECK_LT(idx, flow_file_names_.size());

    cv::Mat rgb;
    utils::loadFlow(flow_file_names_.at(idx), rgb);
    CHECK(!rgb.empty());

    return rgb;
  }

  cv::Mat getInstanceMask(size_t idx) const {
    CHECK_LT(idx, semantic_file_names_.size());

    cv::Mat rgb;
    utils::loadSemanticMask(semantic_file_names_.at(idx), img_size_, rgb);
    CHECK(!rgb.empty());

    return rgb;
  }

  GroundTruthInputPacket getGtPacket(size_t idx) const {
    return ground_truths_.at(idx);
  }

  void loadAll(const std::string& path_to_sequence) {
    std::ifstream times_stream;
    std::string strPathTimeFile = path_to_sequence + "/times.txt";
    utils::throwExceptionIfPathInvalid(strPathTimeFile);

    const gtsam::Pose3 T_cv_robotic =
        gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0, 0, -1, 0, 1, 0),
                     gtsam::traits<gtsam::Point3>::Identity());

    times_stream.open(strPathTimeFile.c_str());
    while (!times_stream.eof()) {
      std::string s;
      getline(times_stream, s);
      if (!s.empty()) {
        std::stringstream ss;
        ss << s;
        double t;
        ss >> t;
        timestamps_.push_back(t);
      }
    }
    times_stream.close();
    LOG(INFO) << "Loaded " << timestamps_.size() << " timestamps";

    // +++ image, depth, semantic and moving object tracking mask +++
    std::string strPrefixImage =
        path_to_sequence + "/image_0/";  // image  image_0
    std::string strPrefixDepth =
        path_to_sequence + "/depth/";  // depth_gt  depth  depth_mono_stereo
    std::string strPrefixSemantic =
        path_to_sequence + "/semantic/";  // semantic_gt  semantic
    std::string strPrefixFlow = path_to_sequence + "/flow/";  // flow_gt  flow

    // const int nTimes = timestamps_.size();
    // rgb_file_names_.resize(nTimes);
    // depth_file_names_.resize(nTimes);
    // semantic_file_names_.resize(nTimes);
    // flow_file_names_.resize(nTimes);
    utils::loadPathsInDirectory(
        rgb_file_names_, strPrefixImage, [](const std::string& file) -> bool {
          return boost::algorithm::ends_with(&file[0], ".png");  // true
        });
    utils::loadPathsInDirectory(
        depth_file_names_, strPrefixDepth, [](const std::string& file) -> bool {
          return boost::algorithm::ends_with(&file[0], ".png");  // true
        });
    utils::loadPathsInDirectory(
        semantic_file_names_, strPrefixSemantic,
        [](const std::string& file) -> bool {
          return boost::algorithm::ends_with(&file[0],
                                             ".txt");  // true
        });
    utils::loadPathsInDirectory(
        flow_file_names_, strPrefixFlow, [](const std::string& file) -> bool {
          return boost::algorithm::ends_with(&file[0], ".flo");  // true
        });

    const int nTimes = rgb_file_names_.size();
    LOG(INFO) << rgb_file_names_.back();

    LOG(INFO) << "Loaded " << nTimes;

    // for (int i = 0; i < nTimes; i++)
    // {
    //     std::stringstream ss;
    //     ss << std::setfill('0') << std::setw(6) << i;
    //     rgb_file_names_.push_back(strPrefixImage + ss.str() + ".png");
    //     depth_file_names_.push_back(strPrefixDepth + ss.str() + ".png");
    //     semantic_file_names_.push_back(strPrefixSemantic + ss.str() +
    //     ".txt"); flow_file_names_.push_back(strPrefixFlow + ss.str() +
    //     ".flo");
    // }

    // +++ ground truth pose +++
    std::string strFilenamePose =
        path_to_sequence + "/pose_gt.txt";  //  pose_gt.txt kevin_extrinsics.txt
    utils::throwExceptionIfPathInvalid(strFilenamePose);
    // vPoseGT.resize(nTimes);
    std::ifstream fPose;
    fPose.open(strFilenamePose.c_str());
    LOG(INFO) << "OPened pose file";

    gtsam::Pose3 initial_pose;

    bool has_initial_pose = false;

    while (!fPose.eof()) {
      std::string s;
      getline(fPose, s);
      if (!s.empty()) {
        std::stringstream ss;
        ss << s;
        int t;
        ss >> t;
        cv::Mat Pose_tmp = cv::Mat::eye(4, 4, CV_64F);

        ss >> Pose_tmp.at<double>(0, 0) >> Pose_tmp.at<double>(0, 1) >>
            Pose_tmp.at<double>(0, 2) >> Pose_tmp.at<double>(0, 3) >>
            Pose_tmp.at<double>(1, 0) >> Pose_tmp.at<double>(1, 1) >>
            Pose_tmp.at<double>(1, 2) >> Pose_tmp.at<double>(1, 3) >>
            Pose_tmp.at<double>(2, 0) >> Pose_tmp.at<double>(2, 1) >>
            Pose_tmp.at<double>(2, 2) >> Pose_tmp.at<double>(2, 3) >>
            Pose_tmp.at<double>(3, 0) >> Pose_tmp.at<double>(3, 1) >>
            Pose_tmp.at<double>(3, 2) >> Pose_tmp.at<double>(3, 3);

        // std::vector<double>
        // vec(Pose_tmp.begin<double>(),Pose_tmp.end<double>());
        // vPoseGT_.push_back(parsePose(vec));
        gtsam::Pose3 pose = utils::cvMatToGtsamPose3(Pose_tmp);

        // pose = T_cv_robotic * pose * T_cv_robotic.inverse();

        if (!has_initial_pose) {
          initial_pose = pose;
          has_initial_pose = true;
        }

        // pose = initial_pose.inverse() * pose;

        vPoseGT_.push_back(pose);
      }
    }
    fPose.close();

    // +++ ground truth object pose +++
    std::string strFilenameObjPose = path_to_sequence + "/object_pose.txt";
    utils::throwExceptionIfPathInvalid(strFilenameObjPose);
    std::ifstream fObjPose;
    fObjPose.open(strFilenameObjPose.c_str());
    LOG(INFO) << "Opened object pose file";

    while (!fObjPose.eof()) {
      std::string s;
      getline(fObjPose, s);
      if (!s.empty()) {
        std::stringstream ss;
        ss << s;

        std::vector<double> ObjPose_tmp(8, 0);
        ss >> ObjPose_tmp[0] >> ObjPose_tmp[1] >> ObjPose_tmp[2] >>
            ObjPose_tmp[3] >> ObjPose_tmp[4] >> ObjPose_tmp[5] >>
            ObjPose_tmp[6] >> ObjPose_tmp[7];

        ObjectPoseGT object_pose;
        object_pose.frame_id_ = static_cast<size_t>(ObjPose_tmp[0]);
        object_pose.object_id_ = static_cast<size_t>(ObjPose_tmp[1]);

        const auto frame_id = object_pose.frame_id_;

        // assign t vector
        cv::Mat t(3, 1, CV_64F);
        t.at<double>(0) = ObjPose_tmp[2];
        t.at<double>(1) = ObjPose_tmp[3];
        t.at<double>(2) = ObjPose_tmp[4];

        // from axis-angle to Rotation Matrix
        cv::Mat R(3, 3, CV_64F);
        cv::Mat Rvec(3, 1, CV_64F);

        // assign r vector
        // Rvec.at<double>(0, 0) = ObjPose_tmp[5];
        // Rvec.at<double>(0, 1) = ObjPose_tmp[6];
        // Rvec.at<double>(0, 2) = ObjPose_tmp[7];
        Rvec.at<double>(0) = ObjPose_tmp[5];
        Rvec.at<double>(1) = ObjPose_tmp[6];
        Rvec.at<double>(2) = ObjPose_tmp[7];

        // *******************************************************************

        const double angle = std::sqrt(ObjPose_tmp[5] * ObjPose_tmp[5] +
                                       ObjPose_tmp[6] * ObjPose_tmp[6] +
                                       ObjPose_tmp[7] * ObjPose_tmp[7]);

        if (angle > 0) {
          // Rvec.at<double>(0, 0) = Rvec.at<double>(0, 0) / angle;
          // Rvec.at<double>(0, 1) = Rvec.at<double>(0, 1) / angle;
          // Rvec.at<double>(0, 2) = Rvec.at<double>(0, 2) / angle;
          Rvec.at<double>(0) = Rvec.at<double>(0) / angle;
          Rvec.at<double>(1) = Rvec.at<double>(1) / angle;
          Rvec.at<double>(2) = Rvec.at<double>(2) / angle;
        }

        const double s = std::sin(angle);
        const double c = std::cos(angle);

        const double v = 1 - c;
        // const double x = Rvec.at<double>(0, 0);
        // const double y = Rvec.at<double>(0, 1);
        // const double z = Rvec.at<double>(0, 2);
        const double x = Rvec.at<double>(0);
        const double y = Rvec.at<double>(1);
        const double z = Rvec.at<double>(2);
        const double xyv = x * y * v;
        const double yzv = y * z * v;
        const double xzv = x * z * v;

        R.at<double>(0, 0) = x * x * v + c;
        R.at<double>(0, 1) = xyv - z * s;
        R.at<double>(0, 2) = xzv + y * s;
        R.at<double>(1, 0) = xyv + z * s;
        R.at<double>(1, 1) = y * y * v + c;
        R.at<double>(1, 2) = yzv - x * s;
        R.at<double>(2, 0) = xzv - y * s;
        R.at<double>(2, 1) = yzv + x * s;
        R.at<double>(2, 2) = z * z * v + c;

        // construct 4x4 transformation matrix
        cv::Mat Pose = cv::Mat::eye(4, 4, CV_64F);
        Pose.at<double>(0, 0) = R.at<double>(0, 0);
        Pose.at<double>(0, 1) = R.at<double>(0, 1);
        Pose.at<double>(0, 2) = R.at<double>(0, 2);
        Pose.at<double>(0, 3) = t.at<double>(0);
        Pose.at<double>(1, 0) = R.at<double>(1, 0);
        Pose.at<double>(1, 1) = R.at<double>(1, 1);
        Pose.at<double>(1, 2) = R.at<double>(1, 2);
        Pose.at<double>(1, 3) = t.at<double>(1);
        Pose.at<double>(2, 0) = R.at<double>(2, 0);
        Pose.at<double>(2, 1) = R.at<double>(2, 1);
        Pose.at<double>(2, 2) = R.at<double>(2, 2);
        Pose.at<double>(2, 3) = t.at<double>(2);

        object_pose.L_world_ = utils::cvMatToGtsamPose3(Pose);
        // object_pose.L_world_ = T_cv_robotic * object_pose.L_world_ *
        // T_cv_robotic.inverse();

        vObjPoseGT_.push_back(object_pose);
      }
    }
    fObjPose.close();
    LOG(INFO) << "Loaded object poses";

    // organise gt poses into vector of arrays
    std::vector<std::vector<size_t>> vObjPoseID(rgb_file_names_.size());
    for (size_t i = 0; i < vObjPoseGT_.size(); ++i) {
      size_t f_id = vObjPoseGT_[i].frame_id_;
      if (f_id >= rgb_file_names_.size()) {
        break;
      }
      vObjPoseID[f_id].push_back(i);
    }
    LOG(INFO) << "Organised object poses";

    // now read image image and add grount truths
    for (size_t frame_id = 0; frame_id < nTimes - 1; frame_id++) {
      Timestamp timestamp = timestamps_[frame_id];
      GroundTruthInputPacket gt_packet;
      gt_packet.timestamp_ = timestamp;
      gt_packet.frame_id_ = frame_id;

      auto original_camera_pose = vPoseGT_[frame_id];
      auto aligned_camera_pose = initial_pose.inverse() * original_camera_pose;
      // auto aligned_camera_pose = initial_pose.inverse() *
      // original_camera_pose;

      gt_packet.X_world_ = aligned_camera_pose;

      // add ground truths for this fid
      for (int i = 0; i < vObjPoseID[frame_id].size(); i++) {
        auto gt_object = vObjPoseGT_[vObjPoseID[frame_id][i]];
        auto relative_object_pose =
            original_camera_pose.inverse() * gt_object.L_world_;
        auto aligned_object_pose = aligned_camera_pose * relative_object_pose;

        gt_object.L_camera_ = relative_object_pose;
        gt_object.L_world_ = aligned_object_pose;

        gt_packet.object_poses_.push_back(gt_object);
        // sanity check
        CHECK_EQ(gt_packet.object_poses_[i].frame_id_, frame_id);
      }

      if (frame_id > 0) {
        auto& previous_gt = ground_truths_.at(frame_id - 1);
        gt_packet.calculateAndSetMotions(previous_gt);
      }

      ground_truths_.push_back(gt_packet);
    }
    LOG(INFO) << "Set GT";
  }

  const CameraParams& getLeftCameraParams() const {
    return rgbd_camera_params_;
  }

  void setCameraParams(const std::string& file_path) {
    std::string params_file = file_path + "oxford.yaml";

    YamlParser yaml_parser(params_file);

    std::vector<double> intrinsics_v(4);
    yaml_parser.getYamlParam<double>("Camera.fx", intrinsics_v.data());
    yaml_parser.getYamlParam<double>("Camera.fy", intrinsics_v.data() + 1);
    yaml_parser.getYamlParam<double>("Camera.cx", intrinsics_v.data() + 2);
    yaml_parser.getYamlParam<double>("Camera.cy", intrinsics_v.data() + 3);

    CameraParams::IntrinsicsCoeffs intrinsics;
    intrinsics.resize(4u);
    // Move elements from one to the other.
    std::copy_n(std::make_move_iterator(intrinsics_v.begin()),
                intrinsics.size(), intrinsics.begin());

    CameraParams::DistortionCoeffs distortion({0, 0, 0, 0});

    double width, height;
    yaml_parser.getYamlParam<double>("Camera.width", &width);
    yaml_parser.getYamlParam<double>("Camera.height", &height);
    cv::Size image_size(width, height);

    yaml_parser.getYamlParam<double>("Camera.baseline", &base_line_);

    auto model = CameraParams::stringToDistortion("radtan", "pinhole");

    rgbd_camera_params_ =
        CameraParams(intrinsics, distortion, image_size, model);

    LOG(INFO) << "Camera params " << rgbd_camera_params_.toString();
  }

 private:
  std::vector<std::string> rgb_file_names_;
  std::vector<std::string> depth_file_names_;
  std::vector<std::string> flow_file_names_;
  std::vector<std::string> semantic_file_names_;
  std::vector<Timestamp> timestamps_;

  std::vector<gtsam::Pose3> vPoseGT_;
  std::vector<ObjectPoseGT> vObjPoseGT_;
  // per frame ground truth
  std::vector<GroundTruthInputPacket> ground_truths_;

  mutable cv::Size img_size_;
  CameraParams rgbd_camera_params_;
  double base_line_;
};

struct OMMTimestampLoader : public TimestampBaseLoader {
  OMDOldAllLoader::Ptr loader_;

  OMMTimestampLoader(OMDOldAllLoader::Ptr loader)
      : loader_(CHECK_NOTNULL(loader)) {}
  std::string getFolderName() const override { return ""; }

  size_t size() const override { return loader_->size(); }

  double getItem(size_t idx) override { return loader_->getTimestamp(idx); }
};

OMDDataLoader::OMDDataLoader(const fs::path& dataset_path)
    : OMDDatasetProvider(dataset_path) {
  LOG(INFO) << "Starting OMDDataLoader with path" << dataset_path;

  // this would go out of scope but we capture it in the functional loaders
  auto loader = std::make_shared<OMDOldAllLoader>(dataset_path);
  auto timestamp_loader = std::make_shared<OMMTimestampLoader>(loader);

  left_camera_params_ = loader->getLeftCameraParams();

  CHECK(getCameraParams());

  auto rgb_loader = std::make_shared<FunctionalDataFolder<cv::Mat>>(
      [loader](size_t idx) { return loader->getRGB(idx); });

  auto optical_flow_loader = std::make_shared<FunctionalDataFolder<cv::Mat>>(
      [loader](size_t idx) { return loader->getOpticalFlow(idx); });

  auto depth_loader = std::make_shared<FunctionalDataFolder<cv::Mat>>(
      [loader](size_t idx) { return loader->getDepthImage(idx); });

  auto instance_mask_loader = std::make_shared<FunctionalDataFolder<cv::Mat>>(
      [loader](size_t idx) { return loader->getInstanceMask(idx); });

  auto gt_loader =
      std::make_shared<FunctionalDataFolder<GroundTruthInputPacket>>(
          [loader](size_t idx) { return loader->getGtPacket(idx); });

  this->setLoaders(timestamp_loader, rgb_loader, optical_flow_loader,
                   depth_loader, instance_mask_loader, gt_loader);

  auto callback = [&](size_t frame_id, Timestamp timestamp, cv::Mat rgb,
                      cv::Mat optical_flow, cv::Mat depth,
                      cv::Mat instance_mask,
                      GroundTruthInputPacket gt_object_pose_gt) -> bool {
    CHECK_EQ(timestamp, gt_object_pose_gt.timestamp_);

    CHECK(ground_truth_packet_callback_);
    if (ground_truth_packet_callback_)
      ground_truth_packet_callback_(gt_object_pose_gt);

    // ImageContainer::Ptr image_container = nullptr;
    // image_container = ImageContainer::Create(
    //         timestamp,
    //         frame_id,
    //         ImageWrapper<ImageType::RGBMono>(rgb),
    //         ImageWrapper<ImageType::Depth>(depth),
    //         ImageWrapper<ImageType::OpticalFlow>(optical_flow),
    //         ImageWrapper<ImageType::MotionMask>(instance_mask));
    // CHECK(image_container);

    ImageContainer image_container(frame_id, timestamp);
    image_container.rgb(rgb)
        .depth(depth)
        .opticalFlow(optical_flow)
        .objectMotionMask(instance_mask);

    CHECK(image_container_callback_);
    if (image_container_callback_)
      image_container_callback_(
          std::make_shared<ImageContainer>(image_container));
    return true;
  };

  this->setCallback(callback);

  // first valid frame is 1
  //  setStartingFrame(1u);
}

}  // namespace dyno
