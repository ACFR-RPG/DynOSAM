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

#include "dynosam/dataprovider/DataProvider.hpp"
#include "dynosam/dataprovider/DatasetLoader.hpp"
#include "dynosam_common/Exceptions.hpp"
#include "dynosam_common/utils/Macros.hpp"
#include "dynosam_common/utils/Tuple.hpp"

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
class DynoDataset : public _DynoDatasetConstructor<Timestamp, cv::Mat, cv::Mat,
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

  using This = DynoDataset<DataTypes...>;

  //! Fully defined tuple type containing all the dataset types in order
  //! starting with the default types
  // using ConcatDataTypes =
  // decltype(std::tuple_cat(DefaultDataset::DataTypeTuple{}, typename
  // TypedGenericDataset::DataTypeTuple{}));

  // uses timestamp file to set the size
  DynoDataset(const fs::path& dataset_path) : dataset_path_(dataset_path) {}

  virtual ~DynoDataset() {}

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

    LOG(ERROR) << "DynoDataset prepared with an expected size of "
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
      LOG(WARNING)
          << "Data input callback not set with DynoDataset::setCallback!";
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

template <typename... DataTypes>
class DynoDatasetProvider : public DynoDataset<DataTypes...>,
                            public DataProvider {
 public:
  using BaseDynoDataset = DynoDataset<DataTypes...>;
  using This = DynoDatasetProvider<DataTypes...>;
  DYNO_POINTER_TYPEDEFS(This)

  using BaseDynoDataset::isProviderValid;

  // does not accept DataProviderModule* for dataprovider
  DynoDatasetProvider(const fs::path& dataset_path)
      : BaseDynoDataset(dataset_path), DataProvider() {}

  virtual ~DynoDatasetProvider() = default;

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

    utils::TimingStatsCollector dataset_spin_timer("dataset_spin");
    if (!BaseDynoDataset::processSingle(active_frame_id_)) {
      LOG(ERROR) << "Processing single frame failed at frame id "
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
