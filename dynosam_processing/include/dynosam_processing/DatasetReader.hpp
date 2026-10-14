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

#pragma once

#include <glog/logging.h>

#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <opencv2/viz.hpp>
#include <opencv4/opencv2/opencv.hpp>

#include "dynosam_common/Exceptions.hpp"
#include "dynosam_common/utils/Macros.hpp"
#include "dynosam_common/utils/TimingStats.hpp"
#include "dynosam_common/utils/Tuple.hpp"
#include "dynosam_processing/DataReader.hpp"

namespace dyno {

class TimestampDataFolderDoesNotExist : public DynosamException {
 public:
  TimestampDataFolderDoesNotExist()
      : DynosamException(
            "TimestampBaseLoader does not exist. Have you called "
            "setLoaders()?") {}
};

class DynoDatasetException : public DynosamException {
 public:
  DynoDatasetException(const std::string& what) : DynosamException(what) {}
};

template <typename... DynoDataTypes>
struct _DynoDatasetConstructor {
  using DynoDataTypesTuple =
      std::tuple<DynoDataTypes...>;  //! All the dyno datypes in one definition
                                     //! (default at the start and then the
                                     //! optional ones)

  using DataInputCallback = std::function<bool(size_t, const DynoDataTypes...)>;

  virtual ~_DynoDatasetConstructor() {}

  void setCallback(const DataInputCallback& data_input_callback) {
    data_input_callback_ = data_input_callback;
  }

  DataInputCallback data_input_callback_;
};

namespace fs = std::filesystem;

/**
 * @brief Defines a set of datafiles within a folder to be laoded by a dataset
 * (partially inspired by pytorch's dataset class). Each folder exists within a
 * parent folder (defined by the dataset) and contains a number of files that
 * need to be loaded by this class. As a number of different input image types
 * (motion/instance segmenation, with and without depth etc.) exist the class is
 * expected to handle the different types itself and return the loaded type
 *
 * @tparam T
 */
template <typename T = cv::Mat>
class DataFolder {
 public:
  DYNO_POINTER_TYPEDEFS(DataFolder)
  using This = DataFolder<T>;
  using Type = T;
  DataFolder() = default;

  virtual ~DataFolder() {}

  /**
   * @brief Set the Absolute Folder Path object to be used when an item is
   * loaded This needs to be a setter function since we do not know the absolute
   * path at the time of construction.
   *
   * Once set, the virtual onPathInit function is called, which can act like a
   * constructor to the derived classes
   *
   * In the context of a dataset laoder, this function is guaranteed to be
   * called be
   *
   * @param absolute_folder_path
   */
  void initAbsolutePath(const fs::path& absolute_folder_path) {
    absolute_folder_path_ = absolute_folder_path;
    try {
      onPathInit();
    } catch (const std::exception& e) {
      LOG(FATAL) << "Exception caught when running virtual onPathInit for "
                    "DataFolder with absolute path: "
                 << absolute_folder_path << ". Error: " << e.what();
    }
  }

  /**
   * @brief Indicates that the absolute folder path has been set correctly with
   * initAbsolutePath Same behaviour as is isAbsolutePathSet but can operate as
   * a boolean operator
   *
   * @return true
   * @return false
   */
  explicit operator bool() const { return isAbsolutePathSet(); }

  /**
   * @brief Checks if the absolute path has been set and the data folder is
   * ready to get items
   *
   * @return true
   * @return false
   */
  inline bool isAbsolutePathSet() const { return (bool)absolute_folder_path_; }

  fs::path getAbsolutePath() const {
    if (!absolute_folder_path_) {
      throw std::runtime_error(
          "Error accessing absolute folder path for DataFolder (" +
          getFolderName() +
          "). Has the path been set with initAbsolutePath()?");
    }
    return absolute_folder_path_.value();
  }

  virtual std::string getFolderName() const = 0;
  // virtual size_t size() const = 0;

  /**
   * @brief Get the Item at an index in the dataset.
   *
   * The idx should essentually correspond to the frame_id (starting at 0) for
   * this datastream in the dataset.
   *
   * @param idx
   * @return T
   */
  virtual T getItem(size_t idx) = 0;

 protected:
  /**
   * @brief Virtual function that is called once the absolute path to this
   * folder is known This happens after construction as the dataset folder (the
   * parent folder) is known by the DataFolderStructure which will set the path
   * on its construction.
   *
   * The absolute folder path can be accessed This::getAbsolutePath
   *
   */
  virtual void onPathInit() {}

 private:
  //! Set when init is called since we need a default constructor. This will not
  //! have the ending "/" on it Initially set to nullopt so we can check that
  //! the folder has been properly initalised via initAbsoluteFolderPath
  std::optional<fs::path> absolute_folder_path_{std::nullopt};
};

template <typename T>
class FunctionalDataFolder : public dyno::DataFolder<T> {
 public:
  using Type = T;
  using This = FunctionalDataFolder<T>;
  using Base = dyno::DataFolder<T>;

  using GetItemFunction = std::function<Type(size_t)>;

  FunctionalDataFolder(const GetItemFunction& func) : func_(func) {}

  std::string getFolderName() const override { return ""; }
  Type getItem(size_t idx) override { return func_(idx); }

 private:
  GetItemFunction func_;
};

class RGBDataFolder : public dyno::DataFolder<cv::Mat> {
 public:
  DYNO_POINTER_TYPEDEFS(RGBDataFolder)
  RGBDataFolder() {}

  /**
   * @brief "image_0" as folder name
   *
   * @return std::string
   */
  std::string getFolderName() const override;
  cv::Mat getItem(size_t idx) override;
};

class OpticalFlowDataFolder : public dyno::DataFolder<cv::Mat> {
 public:
  OpticalFlowDataFolder() {}

  /**
   * @brief "flow" as folder name
   *
   * @return std::string
   */
  std::string getFolderName() const override;
  cv::Mat getItem(size_t idx) override;
};

class DepthDataFolder : public dyno::DataFolder<cv::Mat> {
 public:
  DepthDataFolder() {}

  /**
   * @brief "depth" as folder name
   *
   * @return std::string
   */
  inline std::string getFolderName() const override { return "depth"; }
  cv::Mat getItem(size_t idx) override;
};

class SegMaskFolder : public dyno::DataFolder<cv::Mat> {
 public:
  DYNO_POINTER_TYPEDEFS(SegMaskFolder)
  SegMaskFolder(RGBDataFolder::Ptr rgb_data_folder)
      : rgb_data_folder_(rgb_data_folder) {
    CHECK(rgb_data_folder_);
  }
  virtual ~SegMaskFolder() {}

  cv::Mat getItem(size_t idx) override;

 protected:
  RGBDataFolder::Ptr rgb_data_folder_;
};

class InstantanceSegMaskFolder : public SegMaskFolder {
 public:
  InstantanceSegMaskFolder(RGBDataFolder::Ptr rgb_data_folder)
      : SegMaskFolder(rgb_data_folder) {}

  inline std::string getFolderName() const override { return "semantic"; }
};

class MotionSegMaskFolder : public SegMaskFolder {
 public:
  MotionSegMaskFolder(RGBDataFolder::Ptr rgb_data_folder)
      : SegMaskFolder(rgb_data_folder) {}

  inline std::string getFolderName() const override { return "motion"; }
};

class TimestampBaseLoader : public dyno::DataFolder<double> {
 public:
  DYNO_POINTER_TYPEDEFS(TimestampBaseLoader)
  TimestampBaseLoader() {}

  virtual ~TimestampBaseLoader() = default;
  virtual size_t size() const = 0;
};

// TODO: (jesse) add comments - this is the really important one as we base the
// size of the dataset on this data!!!
class TimestampFile : public TimestampBaseLoader {
 public:
  DYNO_POINTER_TYPEDEFS(TimestampFile)
  TimestampFile() {}

  virtual ~TimestampFile() = default;

  /**
   * @brief "times.txt" as file name
   *
   * @return std::string
   */
  std::string getFolderName() const override;
  double getItem(size_t idx) override;

  // TODO: coment!!
  size_t size() const override {
    // only valid after loading
    // we go up to -1 becuase the optical flow has ONE LESS image
    return times.size() - 1u;
  }

 protected:
  /**
   * @brief Setup ifstream and read everything in the data vector
   *
   */
  virtual void onPathInit() override;

 private:
  std::vector<double> times;
};

/// @brief enum to indicate what type of mask to load (if both are available)
enum MaskType { MOTION, SEMANTIC_INSTANCE };

/**
 * @brief Converts an input string into a enun MaskType
 *
 * @param mask_type
 * @return MaskType
 */
inline MaskType maskTypeFromString(const std::string& mask_type) {
  if (mask_type == "motion") {
    return MaskType::MOTION;
  } else if (mask_type == "semantic") {
    return MaskType::SEMANTIC_INSTANCE;
  }

  throw std::runtime_error(
      "Cannot load mask type from string - unknown input " + mask_type);
}

// Helper type trait to check if a type is std::optional
template <typename T>
struct is_optional : std::false_type {};

template <typename T>
struct is_optional<std::optional<T>> : std::true_type {};

// Helper type trait to extract the underlying type from std::optional
template <typename T>
struct unwrap_optional {
  using type = T;
};

template <typename T>
struct unwrap_optional<std::optional<T>> {
  using type = T;
};

// Define a variadic template alias to convert DataTypes to a std::tuple of
// regular types template<typename... DataTypes> constexpr auto toTuple =
// std::tuple<typename unwrap_optional<DataTypes>::type...>;

/**
 * @brief Defines the structure, image and type of different datasets (partially
 * inspired by pytorch's dataset class). Each dataset contains a number of
 * datafolders which defines the data for a particular input. Each datafolder is
 * responsible for loading the file and all must contain the same number of
 * input files (or some equivalent "length") so the data can be packaged
 * together.
 *
 * This class intends to just enable access to all the datafolders by defining
 * all the templated tuples
 */

template <typename... DataTypes>
class DataFolderStructure {
 public:
  using This = DataFolderStructure<DataTypes...>;
  using DataTypeTuple = std::tuple<DataTypes...>;
  using TypedDataFolderTuple =
      std::tuple<typename DataFolder<DataTypes>::Ptr...>;

  DYNO_POINTER_TYPEDEFS(This)
  static constexpr size_t N = sizeof...(DataTypes);

  template <size_t I>
  using DataType = std::tuple_element_t<I, DataTypeTuple>;

  template <size_t I>
  using DataFolderType = std::tuple_element_t<I, TypedDataFolderTuple>;

  DataFolderStructure(const fs::path& dataset_path,
                      typename DataFolder<DataTypes>::Ptr... data_folders)
      : dataset_path_(dataset_path), data_folders_(data_folders...) {}

  // will be of type DataFolder<T>::Ptr where T is the type returned by the data
  // folder as index at DataTypeTuple<I>
  template <size_t I>
  auto getDataFolder() const {
    return std::get<I>(data_folders_);
  }

  template <size_t I>
  fs::path getAbsoluteFolderPath() const {
    const auto data_folder = this->template getDataFolder<I>();
    fs::path folder_path = dataset_path_;
    // technically this can also be a file path
    folder_path /= fs::path(data_folder->getFolderName());
    return fs::absolute(folder_path);
  }

  /**
   * @brief Get top level (parent) dataset path.
   *
   * All internal folders should be then found in dataset_path/folder
   *
   * @return const fs::path&
   */
  const fs::path& getDatasetPath() const { return dataset_path_; }

 protected:
  const fs::path
      dataset_path_;  //! Path to the dataset directory (ie. the parent folder)
  TypedDataFolderTuple data_folders_;
};

/**
 * @brief GenericDataset extends the DataFolderStructure to include
 * functionality such as validation of each folder and implemets the runtime
 * loading (per index) of each datafolder, updating the internal data structures
 * containing the actual data
 *
 * @tparam DataTypes
 */
template <typename... DataTypes>
class GenericDataset : public DataFolderStructure<DataTypes...> {
 public:
  using This = GenericDataset<DataTypes...>;
  using Base = DataFolderStructure<DataTypes...>;

  //! Tuple of vectors containing each datatype
  //! If DataTypes = [double, int], then this would be an alias
  //! for std::tuple<std::vector<double>, std::vector<int>>;
  using DataStorageTuple = std::tuple<std::vector<DataTypes>...>;

  template <size_t I>
  using DataStorageVector = std::vector<typename Base::DataType<I>>;

  DYNO_POINTER_TYPEDEFS(This)

  GenericDataset(const fs::path& dataset_path,
                 typename DataFolder<DataTypes>::Ptr... data_folders)
      : Base(dataset_path, data_folders...) {
    // this will also set the absolute folder/file path on each DataFolder
    validateFoldersExist();
    LOG(INFO) << "Done folder validate";
  }

  template <size_t I>
  inline DataStorageVector<I>& getDataVector() {
    return std::get<I>(data_);
  }

  virtual ~GenericDataset() {}

  /**
   * @brief Load the data at an index (which we index from 1 so that the same of
   * the vector matches the max index querired) This means that access the
   * DataStorageVector for a loaded index will be index-1
   *
   * @param idx
   * @return true
   * @return false
   */
  bool load(size_t idx) {
    if (idx < 1u) {
      throw std::runtime_error(
          "Failure when loading data in GenericDataset - loading index's start "
          "at 1!");
    }
    for (size_t i = 0; i < Base::N; i++) {
      internal::select_apply<Base::N>(i, [&](auto stream_index) {
        using Type = typename Base::template DataType<stream_index>;
        auto data_folder = this->template getDataFolder<stream_index>();
        // loade at idx-1 since we expect all the indexing and loading to be
        // appropiately zero idnexed
        Type loaded_data = data_folder->getItem(idx - 1);

        auto& data_vector = this->template getDataVector<stream_index>();

        // add loaded data to data vector. resize if necessary but we assume in
        // most cases data will be loaded in order
        if (idx > data_vector.size()) {
          data_vector.resize(idx);
        }

        // index the data storage vector at I-1 to account for the zero indexing
        data_vector.at(idx - 1) = loaded_data;
      });
    }

    return true;  //?
  }

 private:
  /**
   * @brief For each datafolder, iterate through and check that the folders
   * actually exist for each std::runtime_exception is thrown if a folder is not
   * valid.
   *
   * This function also sets absolute file path for each data folder
   *
   */
  void validateFoldersExist() {
    for (size_t i = 0; i < Base::N; i++) {
      internal::select_apply<Base::N>(i, [&](auto stream_index) {
        const fs::path absolute_file_path =
            this->template getAbsoluteFolderPath<stream_index>();

        if (!fs::exists(absolute_file_path)) {
          throw std::runtime_error(
              "Error loading datafolder - absolute file path " +
              std::string(absolute_file_path) + " does not exist");
        }

        LOG(INFO) << "Validated dataset folder: " << absolute_file_path;
        auto data_folder = this->template getDataFolder<stream_index>();
        data_folder->initAbsolutePath(absolute_file_path);
      });
    }
  }

 private:
  DataStorageTuple data_;
};

/**
 * @brief Specalisation of the generic dataset for the expected dynosam dataset
 * structure.
 *
 * Includes some default Datafolders which reflect the minimum requirements for
 * the system (timestamp, rgb, optical flow) which are protected within the
 * dataset class. Additional datafolders must be added on top of this and the
 * class is templated on the extra data folders which also inidicates the total
 * number of datastreams to load for this dataset
 *
 * @tparam DataTypes Types that should be used to construct a frame
 */
template <typename... DataTypes>
class Dataset : public _DynoDatasetConstructor<Timestamp, cv::Mat, cv::Mat,
                                               DataTypes...> {
  // TODO: timestamp really shouldn't be default as not all datasets will have
  // this! We can get time some other way!! maybe just dont have default
  // TODO: (jesse) print folder structure functions and checks!!
  // TODO: (jesse) mix use of the term folder and loader (should revert
  // everything to loader!!)
 public:
  using TypedGenericDataset = GenericDataset<DataTypes...>;
  using DefaultDataset = GenericDataset<Timestamp, cv::Mat, cv::Mat>;

  using Base =
      _DynoDatasetConstructor<Timestamp, cv::Mat, cv::Mat, DataTypes...>;
  using DynoDataTypesTuple =
      typename Base::DynoDataTypesTuple;  //! All the dataypes declared in one
                                          //! tuple (Timestamp, cv::Mat,
                                          //! cv::Mat, ....)

  // TODO: check that the type created by the _DynoDatasetConstructor is the
  // same as the concat of the
  //  DefaultDataset and TypedGenericDataset

  static constexpr size_t TimestampFileIdx = 0u;  //!
  static constexpr size_t RGBFolderIdx =
      TimestampFileIdx + 1u;  //! Index in Default Dataset
  static constexpr size_t OpticalFlowFolderIdx =
      RGBFolderIdx + 1u;  //! Index in Default Dataset

  //! The type stored at the tuple idnex of the GenericDataset
  template <size_t I>
  using GenericDatasetType = typename TypedGenericDataset::DataType<I>;

  //! The type stored at the tuple idnex of the DefaultDataset
  template <size_t I>
  using DefaultDatasetType = typename DefaultDataset::DataType<I>;

  using This = Dataset<DataTypes...>;

  //! Fully defined tuple type containing all the dataset types in order
  //! starting with the default types
  // using ConcatDataTypes =
  // decltype(std::tuple_cat(DefaultDataset::DataTypeTuple{}, typename
  // TypedGenericDataset::DataTypeTuple{}));

  // uses timestamp file to set the size
  Dataset(const fs::path& dataset_path) : dataset_path_(dataset_path) {}

  virtual ~Dataset() {}

  // timestamp used to determine the size of the dataset
  virtual void setLoaders(TimestampBaseLoader::Ptr timestamp_loader,
                          dyno::DataFolder<cv::Mat>::Ptr rgb_loader,
                          dyno::DataFolder<cv::Mat>::Ptr optical_flow_loader,
                          typename DataFolder<DataTypes>::Ptr... data_folders) {
    timestamp_file_ = CHECK_NOTNULL(timestamp_loader);
    default_dataset_ = std::make_unique<DefaultDataset>(
        dataset_path_, timestamp_file_, rgb_loader, optical_flow_loader);

    LOG(INFO) << "Made default dataset";

    dataset_ =
        std::make_unique<TypedGenericDataset>(dataset_path_, data_folders...);
    LOG(INFO) << "Made generic dataset";

    LOG(ERROR) << "Dataset prepared with an expected size of "
               << getDatasetSize() << " - loaded from timestamps file found at "
               << default_dataset_->getAbsoluteFolderPath<TimestampFileIdx>();
    are_loaders_set_ = true;
  }

  /**
   * @brief Returns true when setLoaders(...) has been called and the internal
   * datasets have been constructed correctly After this point spin() and
   * getDatasetSize() can be called as they use the laoded datasets
   *
   * @return true
   * @return false
   */
  bool isProviderValid() const { return are_loaders_set_; }

  // only valid after setLoaders
  virtual size_t getDatasetSize() const {
    if (!timestamp_file_) {
      throw TimestampDataFolderDoesNotExist();
    }
    return timestamp_file_->size();
  }
  const std::string getDatasetPath() const { return dataset_path_; }

  bool processSingle(size_t frame_id) {
    // TODO: change to check with isProviderValid!!
    if (!dataset_) {
      LOG(ERROR) << "Dataset not loaded with setLoaders()! Skipping processing";
      return false;
    }
    if (frame_id >= getDatasetSize()) {
      throw std::runtime_error("Failure when processing a single frame_id!");
    }

    const size_t loading_idx = frame_id + 1;  // odd indexing... oops
    CHECK(default_dataset_->load(loading_idx));
    CHECK(dataset_->load(loading_idx));

    DefaultDataset::DataTypeTuple default_data;
    typename TypedGenericDataset::DataTypeTuple extra_data;
    constexpr size_t default_data_size =
        std::tuple_size<DefaultDataset::DataTypeTuple>::value;

    // iterate over default dataset portion of the ConcatDataTypes
    for (size_t data_idx = 0; data_idx < default_data_size; data_idx++) {
      internal::select_apply<default_data_size>(data_idx, [&](auto I) {
        // I should be data_idx -> access the data vector we just loaded
        auto per_folder_data_vector = default_dataset_->getDataVector<I>();

        // the data idx should also correspond with the index in the
        // DynoDataTypesTuple tuple update data
        std::get<I>(default_data) = per_folder_data_vector.at(frame_id);
      });
    }

    constexpr size_t extra_data_size =
        std::tuple_size<typename TypedGenericDataset::DataTypeTuple>::value;
    // //iterate over extra dataset portion of the ConcatDataTypes
    for (size_t data_idx = 0; data_idx < extra_data_size; data_idx++) {
      internal::select_apply<extra_data_size>(data_idx, [&](auto I) {
        // I should be data_idx -> access the data vector we just loaded
        auto per_folder_data_vector = dataset_->template getDataVector<I>();

        // the data idx should also correspond with the index in the
        // DynoDataTypesTuple tuple update data
        std::get<I>(extra_data) = per_folder_data_vector.at(frame_id);
      });
    }

    // concat the default and extra data into the full tuple
    // this is a bit annoting as we will have to copy everything over
    // get default data
    DynoDataTypesTuple loaded_data = std::tuple_cat(default_data, extra_data);

    if (this->data_input_callback_) {
      auto construct_input_func = [&](auto&&... args) {
        return this->data_input_callback_(frame_id, args...);
      };
      return std::apply(construct_input_func, loaded_data);
    } else {
      LOG(WARNING) << "Data input callback not set with Dataset::setCallback!";
      return false;
    }
  }

  // //TODO: set start and end idx?
  // void processRange() {
  //     for (size_t frame_id = 0; frame_id < getDatasetSize(); frame_id++) {
  //         processSingle(frame_id);
  //     }
  // }

  // TODO: atm only default - could enable if on size so the same function could
  // be used to get from default and extra dataset as we have to zero index them
  // both (or use consexpr? to select and re index)
  template <size_t I>
  auto getLoader() {
    // TODO: test
    return default_dataset_->template getDataFolder<I>();
  }

 private:
  const fs::path dataset_path_;
  typename TypedGenericDataset::UniquePtr dataset_;
  typename DefaultDataset::UniquePtr default_dataset_;

  TimestampBaseLoader::Ptr timestamp_file_;
  bool are_loaders_set_{false};
};

/**
 * @brief A very simple sensor rig used for all basic datasets.
 * Only contains the cannonical camera params, default reference frame
 * values and the extrinsics are set to cv -> robotic convention.
 * By default IMU is disabled.
 *
 */
class BasicDynoSensorRig : public CanonicalSensorRig {
 public:
  DYNO_POINTER_TYPEDEFS(BasicDynoSensorRig)

  //! For ease of use for dataprovider classes, make public
  bool imu_enabled{false};

  BasicDynoSensorRig(const CameraParams& canonical_params,
                     const DepthRigType depth_rig_type,
                     const ImuCalibration& imu_calibration = {},
                     const ReferenceFrames& reference_frames = {});

  inline CameraParams getCanonicalParams() const override {
    return canonical_params_;
  }
  inline DepthRigType depthRigType() const override { return depth_rig_type_; }
  inline ReferenceFrames getReferenceFrames() const override {
    return reference_frames_;
  }

  inline ImuCalibration getImuParams() const override {
    return imu_calibration_;
  }
  inline bool imuEnabled() const { return imu_enabled; }

 private:
  CameraParams canonical_params_;
  DepthRigType depth_rig_type_;
  ImuCalibration imu_calibration_;
  ReferenceFrames reference_frames_;
};

template <typename... DataTypes>
class DatasetReader : public Dataset<DataTypes...>, public DataReader {
 public:
  using BaseDynoDataset = Dataset<DataTypes...>;
  using This = DatasetReader<DataTypes...>;
  DYNO_POINTER_TYPEDEFS(This)

  using BaseDynoDataset::isProviderValid;

  // does not accept DataProviderModule* for dataprovider
  DatasetReader(const fs::path& dataset_path)
      : BaseDynoDataset(dataset_path), DataReader() {}

  virtual ~DatasetReader() = default;

  // ignore if -1. How to reset to original starting frame? we only have -1 so
  // that in main we can call setStartingFrame without doing any checks
  void setStartingFrame(int starting_frame) {
    if (starting_frame != -1) requested_starting_frame_id_ = starting_frame;
    setEndingFrameFromRequested();
  }
  // ignore if -1
  void setEndingFrame(int ending_frame) {
    if (ending_frame != -1) requested_ending_frame_id_ = ending_frame;

    LOG(WARNING) << "Updating ending frame" << requested_ending_frame_id_;
    setEndingFrameFromRequested();
  }

  /**
   * @brief Resets the dataset to the "start", such that the next call to spin
   * will begin the dataset again from the starting frame id. The
   * starting/ending id will return to whatever the last call to
   * setStartingFrame/setEndingFrame set the values to, or default if they have
   * not been called
   *
   * Whatever changes to the dataset, ie. updating the starting and ending
   * frames will take affect after reset is called
   *
   */
  void reset() {
    is_first_spin_ = true;
    active_frame_id_ = 0;
  }

  // returns the effetive size of the dataset (after the ending frame has been
  // set) only valid after setLoaders is called, but this should happen at the
  // constructor level anyway!
  virtual int datasetSize() const override { return (int)ending_frame_id_; }

  virtual bool spin() override {
    if (!isProviderValid()) {
      LOG(ERROR) << "Cannot spin as provider is invalid";
      return false;
    }

    if (active_frame_id_ >= ending_frame_id_) {
      LOG_FIRST_N(INFO, 1) << "Finished dataset";
      emitOnFinishCallbacks();
      return false;
    }

    utils::ChronoTimingStats dataset_spin_timer("dataset_spin");
    if (!BaseDynoDataset::processSingle(active_frame_id_)) {
      LOG(INFO) << "Processing single frame failed at frame id "
                << active_frame_id_;
      return false;
    }

    active_frame_id_++;
    return true;
  }

  size_t getActiveFrameId() const { return active_frame_id_; }
  size_t getEndingFrameId() const { return ending_frame_id_; }

 protected:
  // overrride loaders so we can add our own post set laoders behaviour
  // in this case, dataset size is now going to be valid so we can set the
  // starting/end frame!
  void setLoaders(
      TimestampBaseLoader::Ptr timestamp_loader,
      dyno::DataFolder<cv::Mat>::Ptr rgb_loader,
      dyno::DataFolder<cv::Mat>::Ptr optical_flow_loader,
      typename DataFolder<DataTypes>::Ptr... data_folders) override {
    BaseDynoDataset::setLoaders(timestamp_loader, rgb_loader,
                                optical_flow_loader, data_folders...);
    // now we have a valid timestamp loader set the active/ending frame ids
    // using the requested starting/ending frames and the actual dataset size,
    // which we get from the base
    setEndingFrameFromRequested();
  }

 private:
  void setEndingFrameFromRequested() {
    // call base version - this will also check that the timestamp laoder is
    // valid!
    const size_t dataset_size = BaseDynoDataset::getDatasetSize();

    // by default start dataset at 0
    size_t starting_frame = 0u;
    // by default end dataset at the last frame
    size_t ending_frame = dataset_size;

    if (requested_starting_frame_id_ >= 0) {
      starting_frame = requested_starting_frame_id_;
    }

    if (requested_ending_frame_id_ >= 0) {
      ending_frame = requested_ending_frame_id_;
    }

    // check ending frame first
    if (ending_frame > dataset_size) {
      throw DynoDatasetException(
          "Requested ending frame is greater than the size of the dataset (" +
          std::to_string(ending_frame) + " > " + std::to_string(dataset_size));
    }

    if (starting_frame > ending_frame) {
      throw DynoDatasetException(
          "Requested starting frame is invalid - either less than 0 or greater "
          "than the ending frame (starting frame= " +
          std::to_string(starting_frame) +
          ", ending frame= " + std::to_string(ending_frame));
    }

    // update varibles actually used in the spin
    active_frame_id_ = starting_frame;
    ending_frame_id_ = ending_frame;
  }

 private:
  size_t active_frame_id_ = 0u;
  size_t ending_frame_id_ =
      0u;  // Start at 0, but this will be set upon the first call to spin()

  int requested_starting_frame_id_{-1};
  int requested_ending_frame_id_{-1};

  bool is_first_spin_{true};

  using BaseDynoDataset::getDatasetSize;
};

}  // namespace dyno
