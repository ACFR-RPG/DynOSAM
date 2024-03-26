/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris (jesse.morris@sydney.edu.au)
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a copy
 *   of this software and associated documentation files (the "Software"), to deal
 *   in the Software without restriction, including without limitation the rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:

 *   The above copyright notice and this permission notice shall be included in all
 *   copies or substantial portions of the Software.

 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *   SOFTWARE.
 */

#include "dynosam/dataprovider/ClusterSlamDataProvider.hpp"
#include "dynosam/dataprovider/DataProviderUtils.hpp"
#include "dynosam/utils/OpenCVUtils.hpp"
#include "dynosam/visualizer/ColourMap.hpp"
#include "dynosam/utils/GtsamUtils.hpp"

#include <boost/filesystem.hpp>

#include "dynosam/frontend/vision/VisionTools.hpp" //for getObjectLabels

#include <glog/logging.h>
#include <fstream>

#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string.hpp>

#include <png++/png.hpp> //libpng-dev


namespace dyno {



/**
 * @brief As there is so much interdependancy between the folders in the dataset,
 * we just have the one loader so all the data can be accessed in the same class.
 * This class then provides interfaces for each dyno::DataFolder<....> so that the Dynodataset
 * can use it!
 *
 */
class ClusterSlamAllLoader {

public:
    DYNO_POINTER_TYPEDEFS(ClusterSlamAllLoader)

    /**
     * @brief Construct a new Cluster Slam Gt Loader object.
     *
     * File path is the full file path to a clusterslam folder, containing
     * /images
     * /pose....
     *
     * @param file_path const std::string&
     */
    ClusterSlamAllLoader(const std::string& file_path)
    :   left_images_folder_path_(file_path + "/images/left"),
        optical_flow_folder_path_(file_path + "/optical_flow"),
        pose_folder_path_(file_path + "/pose"),
        instance_masks_folder_(file_path + "/instance_masks"),
        left_landmarks_folder_(file_path + "/landmarks/left"),
        landmark_mapping_file_path_(file_path + "/landmark_mapping.txt"),
        intrinsics_file_path_(file_path + "/intrinsic.txt")
    {

        throwExceptionIfPathInvalid(left_images_folder_path_);
        throwExceptionIfPathInvalid(optical_flow_folder_path_);
        throwExceptionIfPathInvalid(pose_folder_path_);
        throwExceptionIfPathInvalid(instance_masks_folder_);
        throwExceptionIfPathInvalid(left_landmarks_folder_);
        throwExceptionIfPathInvalid(landmark_mapping_file_path_);
        throwExceptionIfPathInvalid(intrinsics_file_path_);

        //first load images and size
        //the size will be used as a refernce for all other loaders
        //size is number of images
        //we use flow to set the size as for this dataset, there will be one less
        //optical flow as the flow ids index t to t+1 (which is gross, I know!!)
        loadFlowImagesAndSize(optical_flow_image_paths_, dataset_size_);
        loadLeftImages(left_rgb_image_paths_);
        loadInstanceMasksImages(instance_masks_image_paths_);
        loadLeftLandmarks(left_landmarks_map_);

        loadLandMarkMapping(landmark_mapping_);

        // //sanity check -> should have the same size of left_landmarks_map_ as landmark_mapping_
        // //as they both represent landmark ids!!
        // CHECK_EQ(left_landmarks_map_.size(), landmark_mapping_.size());

        CHECK_EQ(optical_flow_image_paths_.size(), left_rgb_image_paths_.size() - 1);
        CHECK_EQ(instance_masks_image_paths_.size(), optical_flow_image_paths_.size());
        CHECK_EQ(size(), optical_flow_image_paths_.size());
        setGroundTruthPacket(ground_truth_packets_);

    }

    size_t size() const {
        return dataset_size_;
    }

    cv::Mat getOpticalFlow(size_t idx) const {
        CHECK_LT(idx,optical_flow_image_paths_.size());

        cv::Mat flow;
        loadFlow(optical_flow_image_paths_.at(idx), flow);
        CHECK(!flow.empty());
        return flow;
    }

    cv::Mat getRGB(size_t idx) const {
        CHECK_LT(idx, left_rgb_image_paths_.size());

        cv::Mat rgb;
        loadRGB(left_rgb_image_paths_.at(idx), rgb);
        CHECK(!rgb.empty());

        //debug check -> draw keypoints on image!
        const LandmarksMap& kps_map = left_landmarks_map_.at(idx);

        for(const auto&[landmark_id, kp] : kps_map) {
            const auto cluster_id = landmark_mapping_.at(landmark_id);

            cv::Point2f pt(utils::gtsamPointToCv(kp));
            utils::drawCircleInPlace(rgb, pt, ColourMap::getObjectColour(cluster_id));
        }

        return rgb;
    }

    cv::Mat getInstanceMask(size_t idx) const {
        CHECK_LT(idx, left_rgb_image_paths_.size());
        CHECK_LT(idx, dataset_size_);

        cv::Mat mask, relabelled_mask;
        loadMask(instance_masks_image_paths_.at(idx), mask);
        CHECK(!mask.empty());

        associateDetectedBBWithObject(mask, idx, relabelled_mask);

        return relabelled_mask;
    }

    const GroundTruthInputPacket& getGtPacket(size_t idx) const {
        return ground_truth_packets_.at(idx);
    }

private:

   void loadFlowImagesAndSize(std::vector<std::string>& images_paths, size_t& dataset_size) {
    std::vector<std::filesystem::path> files_in_directory = getAllFilesInDir(optical_flow_folder_path_);
    dataset_size = files_in_directory.size();
    CHECK_GT(dataset_size, 0);

    for (const std::string file_path : files_in_directory) {
        throwExceptionIfPathInvalid(file_path);
        images_paths.push_back(file_path);
    }
   }

    void loadLeftImages(std::vector<std::string>& images_paths) {
        std::vector<std::filesystem::path> files_in_directory = getAllFilesInDir(left_images_folder_path_);

        for (const std::string file_path : files_in_directory) {
            throwExceptionIfPathInvalid(file_path);
            images_paths.push_back(file_path);
        }
    }

    void loadInstanceMasksImages(std::vector<std::string>& images_paths) {
        std::vector<std::filesystem::path> files_in_directory = getAllFilesInDir(instance_masks_folder_);

        for (const std::string file_path : files_in_directory) {
            throwExceptionIfPathInvalid(file_path);
            images_paths.push_back(file_path);
        }
    }



    //mapping of landmark id -> u, v coordiante
    using LandmarksMap = gtsam::FastMap<int, Keypoint>;
    using LandmarksMapPerFrame = gtsam::FastMap<FrameId, LandmarksMap>;

    void loadLeftLandmarks(LandmarksMapPerFrame& detected_landmarks) {
        std::vector<std::filesystem::path> files_in_directory = getAllFilesInDir(left_landmarks_folder_);

        for (const std::filesystem::path& file_path : files_in_directory) {

            //this should be the file name which we will then decompose into the frame id
            //e.g. .../pose/0000.txt -> filename = 0000 -> frame id = int(0)
            const std::string file_name = file_path.filename();
            const int frame =  std::stoi(file_name);

            std::ifstream fstream;
            fstream.open((std::string)file_path, std::ios::in);
            if(!fstream) {
                throw std::runtime_error("Could not open file " + (std::string)file_path + " when trying to load Cluster Slam landmarks information!");
            }

            //check we've never processed this frame before
            CHECK(!detected_landmarks.exists(frame));

            LandmarksMap detected_lmks_this_frame;

            //expected size of each line (landamrk_id u v)
            constexpr static size_t kExpectedSize = 3u;
            while (!fstream.eof()) {
                std::vector<std::string> split_line;
                if(!getLine(fstream, split_line)) continue;

                CHECK_EQ(split_line.size(), kExpectedSize) << "Line present in landmarks file does not have the right length - " << container_to_string(split_line);

                const int landmark_id = std::stoi(split_line.at(0));
                const double u = std::stod(split_line.at(1));
                const double v = std::stod(split_line.at(2));

                detected_lmks_this_frame.insert2(landmark_id, Keypoint(u, v));
            }

            detected_landmarks.insert2(frame, detected_lmks_this_frame);
        }

        VLOG(20) << "Loaded " << detected_landmarks.size() << " detected landmarks...";

    }

    //Landmark to cluster id mapping.
    //the landmark id is the key in LandmarksMap
    void loadLandMarkMapping(gtsam::FastMap<int, int>& landmark_mapping) {
        std::ifstream fstream;
        fstream.open(landmark_mapping_file_path_, std::ios::in);
        if(!fstream) {
            throw std::runtime_error("Could not open file " + landmark_mapping_file_path_ + " when trying to load Cluster Slam landmarks mapping!");
        }

        //expected size of each line (landamrk_id cluster_id)
        constexpr static size_t kExpectedSize = 2u;
        while (!fstream.eof()) {
            std::vector<std::string> split_line;
            if(!getLine(fstream, split_line)) continue;

            CHECK_EQ(split_line.size(), kExpectedSize) << "Line present in landmark mapping file does not have the right length - " << container_to_string(split_line);

            const int landmark_id = std::stoi(split_line.at(0));
            //clusters start at 0 here, but we need them to start at 1
            // such that the tracking is consistent with out labelling system
            //which has background = 0
            const int cluster_id = std::stoi(split_line.at(1)) + 1;

            landmark_mapping.insert2(landmark_id, cluster_id);
        }
    }

    //we're going to have to associate the detected bounding boxes
    //with the actual landmark ids so that the gt labels match with the
    //tracking labels....
    //we will have to do this by taking all the landmarks of detected features
    //and check which objects lies within the bounding box....
    //this is almost the definitions of heinous but like, whatever....
    //this function updates
    //returns the new instance mask
    //from the original mask, relabel the pixel values using the gt tracks
    void associateDetectedBBWithObject(const cv::Mat& instance_mask, FrameId frame_id, cv::Mat& relabelled_mask) const {
        const GroundTruthInputPacket& gt_packet = getGtPacket(frame_id);
        ObjectIds object_ids = vision_tools::getObjectLabels(instance_mask);



        instance_mask.copyTo(relabelled_mask);

        //sort kps map into a map of clusters -> [kps....]
        const LandmarksMap& kps_map = left_landmarks_map_.at(frame_id);
        //cluster id (track id -> all keypoints for that cluster)
        gtsam::FastMap<int, Keypoints> clustered_kps;
        for(const auto&[landmark_id, kp] : kps_map) {
            const auto cluster_id = landmark_mapping_.at(landmark_id);

            if(!clustered_kps.exists(cluster_id)) {
                clustered_kps.insert2(cluster_id, Keypoints());
            }
            clustered_kps.at(cluster_id).push_back(kp);
        }

        //alright ive done something wrong.....
        for(const auto&[cluster_id, kps] : clustered_kps) {
            //vector for each of the object ids
            //containts a pair identifying which object in the instance mask
            //and HOW many keypoints for this (cluster id) object fell inside the detected bounding box
            std::vector<std::pair<ObjectId, int>> voting;

            //for each object in the instance mask, try and associate it with a cluster
            for(ObjectId object_id : object_ids) {
                //get the binary mask just for this object, we will use this too find a bounding box
                //and count how many kp's fall into this mask
                const cv::Mat object_mask = relabelled_mask == object_id;

                cv::Rect bb;
                if(!vision_tools::findObjectBoundingBox(instance_mask, object_id, bb)) {
                    VLOG(10) << "No bb could be found for detected object " << object_id << " at frame " << frame_id << ". Skipping association and removing object from mask";
                    relabelled_mask.setTo(cv::Scalar(0), object_mask);
                    continue;
                }

                std::pair<ObjectId, int> object_count{object_id, 0};
                //iteate over all kps in this cluster from gt and see if they lie in the detected bb
                for(const Keypoint& kp : kps) {
                    cv::Point2f pt(utils::gtsamPointToCv(kp));

                    if(bb.contains(pt)) {
                        object_count.second++;
                    }
                }

                voting.push_back(object_count);

            }

            if(voting.empty()) {
                continue;
            }

            //sort in descening order via the object coutn
            auto sort_pair_int = [](const std::pair<int, int>& a, const std::pair<int, int>& b) -> bool {
                return (a.second > b.second);
            };
            std::sort(voting.begin(), voting.end(), sort_pair_int);

            //the detected object label with the most kps in it from cluster_id
            //we therefore need to relabel all pixels with the value associated_object_label
            //to cluster label
            auto associated_object_label = voting[0].first;
            const cv::Mat assciated_object_mask = relabelled_mask == associated_object_label;
            relabelled_mask.setTo(cv::Scalar(cluster_id), assciated_object_mask);

            LOG(INFO) << "Relabelled object id " << associated_object_label << " to " << cluster_id;

        }
    }


    void setGroundTruthPacket(GroundTruthPacketMap& ground_truth_packets) {
        //load the poses
        //first line contains camera pose
        //consequative lines contain object ids which are indexed from 0, but we will index from 1
        //expect the file name to give us the frame we are working with!!
        std::vector<std::filesystem::path> files_in_directory = getAllFilesInDir(pose_folder_path_);

        for (const std::filesystem::path& pose_file_path : files_in_directory) {

            //this should be the file name which we will then decompose into the frame id
            //e.g. .../pose/0000.txt -> filename = 0000 -> frame id = int(0)
            const std::string file_name = pose_file_path.filename();
            const int frame =  std::stoi(file_name);
            const gtsam::Pose3Vector poses = processPoseFile(pose_file_path, ground_truth_packets);

            //check we've never processed this frame before
            CHECK(!ground_truth_packets.exists(frame));

            if(frame > 0) {
                //check we have the previous frame!! (ie.e we process in order!)
                CHECK(ground_truth_packets.exists(frame-1));
            }

            GroundTruthInputPacket gt_packet;
            gt_packet.frame_id_ = frame;
            // Quoting the dataset home page:
            // For each trajectory file under pose/ directory, each line in each file represents
            // the pose of each cluster except for the first line - that line represents the camera pose.
            for(size_t i = 0; i < poses.size(); i++) {
                if(i == 0) {
                    gt_packet.X_world_ = poses.at(i);
                }
                else {
                    ObjectPoseGT object_pose_gt;
                    object_pose_gt.frame_id_ = frame;
                    object_pose_gt.object_id_ = i;
                    object_pose_gt.L_world_ = poses.at(i);
                    object_pose_gt.L_camera_ = gt_packet.X_world_.inverse() * object_pose_gt.L_world_;
                    gt_packet.object_poses_.push_back(object_pose_gt);
                }
            }

            if(frame > 0) {
                const GroundTruthInputPacket& previous_gt_packet = ground_truth_packets.at(frame - 1);
                gt_packet.calculateAndSetMotions(previous_gt_packet);
            }

            ground_truth_packets.insert2(frame, gt_packet);

        }

    }
    gtsam::Pose3Vector processPoseFile(const std::string& pose_file, GroundTruthPacketMap& ground_truth_packets) {
        std::ifstream fstream;
        fstream.open(pose_file, std::ios::in);
        if(!fstream) {
            throw std::runtime_error("Could not open file " + pose_file + " when trying to load Cluster Slam pose information!");
        }

        //expected size of each line (3 translation components + 4 for quaternion)
        constexpr static size_t kExpectedSize = 7u;

        //pose string expected to be in form x,y,z, qx, qy, qz, qw
        auto construct_pose =[](const std::vector<std::string>& pose_string) -> gtsam::Pose3 {
            const double x = std::stod(pose_string.at(0));
            const double y = std::stod(pose_string.at(1));
            const double z = std::stod(pose_string.at(2));

            const double qx = std::stod(pose_string.at(3));
            const double qy = std::stod(pose_string.at(4));
            const double qz = std::stod(pose_string.at(5));
            const double qw = std::stod(pose_string.at(6));

            return gtsam::Pose3(
                gtsam::Rot3(qw, qx, qy, qz),
                gtsam::Point3(x, y, z)
            );
        };

        gtsam::Pose3Vector poses;
        while (!fstream.eof()) {
            std::vector<std::string> split_line;
            if(!getLine(fstream, split_line)) continue;

            CHECK_EQ(split_line.size(), kExpectedSize) << "Line present in pose file does not have the right length - " << container_to_string(split_line);
            poses.push_back(construct_pose(split_line));
        }
        return poses;
    }

     /**
     * @brief Returns a ORDERED vector of all files in the given directory.
     *
     * @param folder_path
     * @return std::vector<std::filesystem::path>
     */
    std::vector<std::filesystem::path> getAllFilesInDir(const std::string& folder_path) {
        std::vector<std::filesystem::path> files_in_directory;
        std::copy(std::filesystem::directory_iterator(folder_path), std::filesystem::directory_iterator(), std::back_inserter(files_in_directory));
        std::sort(files_in_directory.begin(), files_in_directory.end());
        return files_in_directory;

    }

private:
    const std::string left_images_folder_path_;
    const std::string optical_flow_folder_path_;
    const std::string pose_folder_path_;
    const std::string left_landmarks_folder_;
    const std::string instance_masks_folder_;
    const std::string landmark_mapping_file_path_;
    const std::string intrinsics_file_path_;

    std::vector<std::string> left_rgb_image_paths_; //index from 0 to match the naming convention of the dataset
    std::vector<std::string> optical_flow_image_paths_;
    std::vector<std::string> instance_masks_image_paths_;
    LandmarksMapPerFrame left_landmarks_map_;

    //instance masks per frame - we need a chace of them as we're going to change
    //the index of each object to match the gt labels!!
    gtsam::FastMap<FrameId, cv::Mat> instance_masks_;

    //Landmark to cluster id mapping
    //cluster should be the actual track
    //the landmark id is the key in LandmarksMap
    gtsam::FastMap<int, int> landmark_mapping_;

    GroundTruthPacketMap ground_truth_packets_;
    size_t dataset_size_; //set in setGroundTruthPacket. Reflects the number of files in the /pose folder which is one per frame

};

struct ClusterSlamTimestampLoader : public TimestampBaseLoader {

    ClusterSlamAllLoader::Ptr loader_;

    ClusterSlamTimestampLoader(ClusterSlamAllLoader::Ptr loader) : loader_(CHECK_NOTNULL(loader)) {}
    std::string getFolderName() const override { return ""; }

    size_t size() const override {
        return loader_->size();
    }

    double getItem(size_t idx) override {
        //we dont have a time? for now just make idx
        return static_cast<double>(idx);
    }
};



ClusterSlamDataLoader::ClusterSlamDataLoader(const fs::path& dataset_path) : ClusterSlamDatasetProvider(dataset_path)
{
    LOG(INFO) << "Starting ClusterSlamDataLoader with path" << dataset_path;

    //this would go out of scope but we capture it in the functional loaders
    auto loader = std::make_shared<ClusterSlamAllLoader>(dataset_path);
    auto timestamp_loader = std::make_shared<ClusterSlamTimestampLoader>(loader);

    auto rgb_loader = std::make_shared<FunctionalDataFolder<cv::Mat>>(
        [loader](size_t idx) {
            return loader->getRGB(idx);
        }
    );

    auto optical_flow_loader = std::make_shared<FunctionalDataFolder<cv::Mat>>(
        [loader](size_t idx) {
            return loader->getOpticalFlow(idx);
        }
    );

    auto depth_loader = std::make_shared<FunctionalDataFolder<cv::Mat>>(
        [loader](size_t idx) {
            return cv::Mat();
        }
    );

    auto instance_mask_loader = std::make_shared<FunctionalDataFolder<cv::Mat>>(
        [loader](size_t idx) {
            return loader->getInstanceMask(idx);
        }
    );


    auto gt_loader = std::make_shared<FunctionalDataFolder<GroundTruthInputPacket>>(
        [loader](size_t idx) {
            return loader->getGtPacket(idx);
        }
    );

    this->setLoaders(
        timestamp_loader,
        rgb_loader,
        optical_flow_loader,
        depth_loader,
        instance_mask_loader,
        gt_loader
    );
}


}
