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
#include <opencv4/opencv2/opencv.hpp>
#include <optional>

#include "dynosam_common/utils/Macros.hpp"
#include "dynosam_common/utils/Tuple.hpp"

namespace dyno {

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
  DataStorageVector<I>& getDataVector() {
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

}  // namespace dyno
