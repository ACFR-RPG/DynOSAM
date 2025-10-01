/*
 *   Copyright (c) 2023 ACFR-RPG, University of Sydney, Jesse Morris
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

#include "dynosam/dataprovider/DataProviderFactory.hpp"

#include <glog/logging.h>

#include "dynosam/dataprovider/ClusterSlamDataProvider.hpp"
#include "dynosam/dataprovider/DataProvider.hpp"
#include "dynosam/dataprovider/KittiDataProvider.hpp"
#include "dynosam/dataprovider/OMDDataProvider.hpp"
#include "dynosam/dataprovider/ProjectAriaDataProvider.hpp"
#include "dynosam/dataprovider/TartanAirShibuya.hpp"
#include "dynosam/dataprovider/ViodeDataProvider.hpp"
#include "dynosam/dataprovider/VirtualKittiDataProvider.hpp"
#include "dynosam_common/utils/YamlParser.hpp"

DEFINE_int32(starting_frame, -1,
             "Starting frame of the dataset. If -1 use the default which is "
             "the starting frame=0");
DEFINE_int32(ending_frame, -1,
             "Ending frame of the dataset. If -1 use the default which is the "
             "ending_frame=dataset_size");

namespace dyno {

DataProvider::Ptr DataProviderFactory::Create(
    const std::string& dataset_folder_path,
    const std::string& params_folder_path, DatasetType dataset_type) {
  if (dataset_type == DatasetType::KITTI) {
    LOG(INFO) << "Using KITTI dataset at path: " << dataset_folder_path;
    KittiDataLoader::Params params =
        KittiDataLoader::Params::fromYaml(params_folder_path);
    auto loader =
        std::make_shared<KittiDataLoader>(dataset_folder_path, params);

    loader->setStartingFrame(FLAGS_starting_frame);
    loader->setEndingFrame(FLAGS_ending_frame);
    return loader;
  } else if (dataset_type == DatasetType::VIRTUAL_KITTI) {
    LOG(INFO) << "Using Virtual KITTI dataset at path: " << dataset_folder_path;
    VirtualKittiDataLoader::Params params =
        VirtualKittiDataLoader::Params::fromYaml(params_folder_path);
    auto loader =
        std::make_shared<VirtualKittiDataLoader>(dataset_folder_path, params);
    loader->setStartingFrame(FLAGS_starting_frame);
    loader->setEndingFrame(FLAGS_ending_frame);
    return loader;
  } else if (dataset_type == DatasetType::CLUSTER) {
    LOG(INFO) << "Using Cluster (SLAM) dataset at path: "
              << dataset_folder_path;
    auto loader = std::make_shared<ClusterSlamDataLoader>(dataset_folder_path);
    loader->setStartingFrame(FLAGS_starting_frame);
    loader->setEndingFrame(FLAGS_ending_frame);
    return loader;
  } else if (dataset_type == DatasetType::OMD) {
    LOG(INFO) << "Using Oxford Multi-motion Dataset dataset at path: "
              << dataset_folder_path;
    auto loader = std::make_shared<OMDDataLoader>(dataset_folder_path);
    loader->setStartingFrame(FLAGS_starting_frame);
    loader->setEndingFrame(FLAGS_ending_frame);
    return loader;
  } else if (dataset_type == DatasetType::ARIA) {
    LOG(INFO) << "Using ARIA dataset at path: " << dataset_folder_path;
    auto loader = std::make_shared<ProjectARIADataLoader>(dataset_folder_path);
    loader->setStartingFrame(FLAGS_starting_frame);
    loader->setEndingFrame(FLAGS_ending_frame);
    return loader;
  } else if (dataset_type == DatasetType::TARTAN_AIR_SHIBUYA) {
    LOG(INFO) << "Using TARTAN_AIR_SHIBUYA dataset at path: "
              << dataset_folder_path;
    auto loader = std::make_shared<TartanAirShibuyaLoader>(dataset_folder_path);
    loader->setStartingFrame(FLAGS_starting_frame);
    loader->setEndingFrame(FLAGS_ending_frame);
    return loader;
  } else if (dataset_type == DatasetType::VIODE) {
    LOG(INFO) << "Using VIODE dataset at path: " << dataset_folder_path;
    auto loader = std::make_shared<ViodeLoader>(dataset_folder_path);
    loader->setStartingFrame(FLAGS_starting_frame);
    loader->setEndingFrame(FLAGS_ending_frame);
    return loader;
  } else {
    throw std::runtime_error(
        "Unable to construct Dataprovider - unknown dataset type: " +
        static_cast<int>(dataset_type));
  }
}

}  // namespace dyno
