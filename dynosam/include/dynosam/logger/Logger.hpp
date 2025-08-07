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
#pragma once

#include <glog/logging.h>
#include <stdlib.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

#include "dynosam/common/GroundTruthPacket.hpp"
#include "dynosam/common/PointCloudProcess.hpp"
#include "dynosam/common/Types.hpp"
#include "dynosam/utils/CsvParser.hpp"
#include "dynosam/utils/JsonUtils.hpp"

namespace dyno {

namespace fs = std::filesystem;

/**
 * @brief Constructs a file path (using the file name) where the root of the
 * path is defined by the FLAGS_output_path param.
 *
 * This flag defines the output folder of all output log/debug files.
 * The resulting file path with then be FLAGS_output_path/file_name
 *
 * @param file_name
 * @return std::string
 */
std::string getOutputFilePath(const std::string& file_name);

/**
 * @brief Writes all the statistics to a csv file on the path defined by
 * getOutputFilePath(file_name).
 *
 * The output csv file has a header "label, samples"
 * Where ALL recorded samples per label are saved to file.
 *
 * Internally uses Statistics::WriteAllSamplesToCsvFile.
 *
 * Default file name is statistics_samples.csv.
 *
 * @param file_name
 * @return true
 * @return false
 */
void writeStatisticsSamplesToFile(
    const std::string& file_name = "statistics_samples.csv");

/**
 * @brief Writes a summary of the statistics to a csv file on the path defined
 * by getOutputFilePath(file_name).
 *
 * Internally uses Statistics::WriteSummaryToCsvFile.
 *
 * Default file name is statistics.csv.
 *
 * @param file_name
 * @return true
 * @return false
 */
void writeStatisticsSummaryToFile(
    const std::string& file_name = "statistics.csv");

/**
 * @brief Writes statistics per module to csv files on path defined by
 * getOutputFilePath(). Interrnally uses WritePerModuleSummariesToCsvFile
 *
 */
void writeStatisticsModuleSummariesToFile();

/**
 * @brief Create a directory at the given path.
 *
 * If the directory does not exist it will be created, otherwise nothing
 * happens.
 *
 * @param path
 * @return true
 * @return false
 */
bool createDirectory(const std::string& path);

// Open files with name output_filename, and checks that it is valid
static inline void OpenFile(const std::string& output_filename,
                            std::ofstream* output_file,
                            bool append_mode = false) {
  CHECK_NOTNULL(output_file);
  output_file->open(output_filename.c_str(),
                    append_mode ? std::ios_base::app : std::ios_base::out);
  output_file->precision(20);
  CHECK(output_file->is_open()) << "Cannot open file: " << output_filename;
  CHECK(output_file->good()) << "File in bad state: " << output_filename;
}

// Wrapper for std::ofstream to open/close it when created/destructed.
class OfstreamWrapper {
 public:
  OfstreamWrapper(const std::string& filename,
                  const bool& open_file_in_append_mode = false);

  OfstreamWrapper(const std::string& filename, const std::string& output_path,
                  const bool& open_file_in_append_mode = false);

  virtual ~OfstreamWrapper();
  void closeAndOpenLogFile();

  static bool WriteOutCsvWriter(const CsvWriter& csv,
                                const std::string& filename);

  fs::path getFilePath() const;

 public:
  const std::string filename_;
  const std::string output_path_;
  const bool open_file_in_append_mode_ = false;

  std::ofstream ofstream_;

 protected:
  void openLogFile(bool open_file_in_append_mode = false);
};

// wrapper to write a type t that is json seralizable to an open ofstream
// with a set width
template <typename T>
std::ofstream& writeJson(std::ofstream& os, const T& t) {
  // T must be json seralizable
  const json j = t;
  os << std::setw(4) << j;
  return os;
};

class JsonConverter {
 public:
  using json = nlohmann::json;

  enum Format { NORMAL, BSON };

  // T must be json seralizable
  template <typename T>
  static void WriteOutJson(const T& value, const std::string& filepath,
                           const Format& fmt = Format::BSON,
                           bool append = false) {
    if (fmt == Format::BSON) {
      WriteBson<T>(value, filepath, append);
    } else {
      CHECK(false) << "normal json not implemented";
    }
  }

  template <typename T>
  static bool ReadInJson(T& value, const std::string& filepath,
                         const Format& fmt = Format::BSON) {
    if (fmt == Format::BSON) {
      return ReadBson(value, filepath);
    } else {
      CHECK(false) << "normal json not implemented";
    }
  }

 private:
  template <typename T>
  static void WriteBson(const T& value, const std::string& filepath,
                        bool append) {
    VLOG(5) << "Writing bson data to filepath: " << filepath;

    json j;
    j["data"] = value;
    auto v_bson = json::to_bson(j);

    std::ofstream bsonfile;
    if (!append) {
      bsonfile.open(filepath.c_str(), std::ios_base::out | std::ios::binary);
    } else {
      bsonfile.open(filepath.c_str(),
                    std::ios_base::app | std::ios_base::out | std::ios::binary);
    }

    bsonfile.write((char*)&v_bson[0], v_bson.size() * sizeof(v_bson[0]));
    bsonfile.close();
  }

  template <typename T>
  static bool ReadBson(T& value, const std::string& filepath) {
    VLOG(5) << "Reading bson data from filepath: " << filepath;
    std::ifstream file(filepath, std::ios::binary);

    if (!file.is_open()) {
      return false;
    }

    // Get the size of the file
    file.seekg(0, std::ios::end);
    const std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Create a vector to store the file content
    std::vector<std::uint8_t> content;
    content.reserve(file_size);  // Reserve space to avoid reallocation

    // Read the file content into the vector
    content.insert(content.begin(), std::istreambuf_iterator<char>(file),
                   std::istreambuf_iterator<char>());
    const json json_instance_read = json::from_bson(content);

    value = json_instance_read["data"].template get<T>();
    return true;
  }
};

class EstimationModuleLogger {
 public:
  DYNO_POINTER_TYPEDEFS(EstimationModuleLogger)

  /**
   * @brief Construct a new Estimation Module Logger object.
   *
   * Module name is prefixed to the output file names which are hardcoded per
   * log function. Unless otherwise specified, all the (base) functions operate
   * per frame.
   *
   * Will write to the file path sepcified by FLAGS_output_path
   *
   * @param module_name
   */
  EstimationModuleLogger(const std::string& module_name);
  // write to file on destructor
  virtual ~EstimationModuleLogger();

  // logs to motion errors
  virtual std::optional<size_t> logObjectMotion(
      FrameId frame_id, const MotionEstimateMap& motion_estimates,
      const std::optional<GroundTruthPacketMap>& gt_packets = {});

  virtual std::optional<size_t> logObjectMotion(
      const ObjectMotionMap& object_motions,
      const std::optional<GroundTruthPacketMap>& gt_packets = {});

  // logs object pose (to a differnet file)
  virtual std::optional<size_t> logObjectPose(
      FrameId frame_id, const PoseEstimateMap& object_poses,
      const std::optional<GroundTruthPacketMap>& gt_packets = {});

  virtual std::optional<size_t> logObjectPose(
      const ObjectPoseMap& object_poses,
      const std::optional<GroundTruthPacketMap>& gt_packets = {});

  // logs camera pose (to a differnet file)
  virtual std::optional<size_t> logCameraPose(
      FrameId frame_id, const gtsam::Pose3& T_world_camera,
      const std::optional<GroundTruthPacketMap>& gt_packets = {});

  virtual void logPoints(FrameId frame_id, const gtsam::Pose3& T_world_local_k,
                         const StatusLandmarkVector& landmarks);

  virtual void logMapPoints(const StatusLandmarkVector& landmarks);

  // logs to object bounding boxes
  virtual void logObjectBbxes(FrameId frame_id,
                              const BbxPerObject& object_bbxes);

  // interum solution since we only log FrameIds to file and not Tiemstamps
  // TODO: better is to log both (or actually just timestamp since our eval is
  // designed to use Timestamps and we just have a hack ;)
  void logFrameIdToTimestamp(FrameId frame_id, Timestamp timestamp);

  inline const std::string& moduleName() const { return module_name_; }

 protected:
  bool logObjectMotion(
      CsvWriter& writer, const Motion3ReferenceFrame& motion, FrameId frame_id,
      ObjectId object_id,
      const std::optional<GroundTruthPacketMap>& gt_packets = {});
  bool logObjectPose(
      CsvWriter& writer, const gtsam::Pose3& pose, FrameId frame_id,
      ObjectId object_id,
      const std::optional<GroundTruthPacketMap>& gt_packets = {});

 protected:
  const std::string module_name_;
  const std::string object_pose_file_name_;
  const std::string object_motion_file_name_;
  const std::string object_bbx_file_name_;
  const std::string camera_pose_file_name_;
  const std::string map_points_file_name_;
  const std::string frame_id_to_timestamp_file_name_;

  CsvWriter::UniquePtr object_pose_csv_;
  CsvWriter::UniquePtr object_motion_csv_;
  CsvWriter::UniquePtr object_bbx_csv_;
  CsvWriter::UniquePtr camera_pose_csv_;
  CsvWriter::UniquePtr map_points_csv_;
  CsvWriter::UniquePtr frame_id_timestamp_csv_;
};

}  // namespace dyno
