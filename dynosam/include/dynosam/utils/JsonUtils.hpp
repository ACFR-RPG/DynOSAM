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

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/ISAM2Result.h>

#include <boost/optional.hpp>  //only becuase current gtsam version needs this!!!
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <optional>
#include <ostream>
#include <sstream>

#include "dynosam/common/Types.hpp"
#include "dynosam/frontend/Frontend-Definitions.hpp"

using json = nlohmann::json;

namespace dyno {

// map KeyPointType values to JSON as strings
NLOHMANN_JSON_SERIALIZE_ENUM(KeyPointType, {
                                               {STATIC, "static"},
                                               {DYNAMIC, "dynamic"},
                                           })

// map ReferenceFrame values to JSON as strings
NLOHMANN_JSON_SERIALIZE_ENUM(ReferenceFrame, {
                                                 {GLOBAL, "global"},
                                                 {LOCAL, "local"},
                                                 {OBJECT, "object"},
                                             })

NLOHMANN_JSON_SERIALIZE_ENUM(FrontendType, {
                                               {kRGBD, "RGB"},
                                               {kMono, "Mono"},
                                           })

NLOHMANN_JSON_SERIALIZE_ENUM(MotionRepresentationStyle, {
                                                            {F2F, "F2F"},
                                                            {KF, "KF"},
                                                        })

// forward declare - IO definition is in .cc file
class FrontendOutputPacketBase;
class RGBDInstanceOutputPacket;
class VisionImuPacket;

}  // namespace dyno

/**
 * @brief Provide direct specalisations of nlohmann::adl_serializer
 * for gtsam/opencv types that we want to seralize. These are not declared in
 * the header file as they are template specalisations. They will directly be
 * found by the json library. We write them as specalisations and not free
 * functions as per the guidelines of the library (essentially to avoid
 * namespace hijaking...)
 *
 */

namespace nlohmann {
/**
 * @brief ADL (argument-dependant lookup) seralizer for std::optional types.
 * We want an std::optional seralizer as we frequently use this type
 * within our data structs so this saves having to write additional boiler-plate
 * code.
 *
 * A free to_json/from_json within the dyno (or type specific namespace) must be
 * available for T (see basic example or the one that already exists for the
 * GroundTruthPackets) must exist.
 *
 * Taken directly from the nlohmann::json readme.
 *
 * @tparam T
 */
template <typename T>
struct adl_serializer<std::optional<T>> {
  static void to_json(json& j, const std::optional<T>& opt) {
    if (opt == std::nullopt) {
      j = nullptr;
    } else {
      j = *opt;  // this will call adl_serializer<T>::to_json which will
                 // find the free function to_json in T's namespace!
    }
  }

  static void from_json(const json& j, std::optional<T>& opt) {
    if (j.is_null()) {
      opt = std::nullopt;
    } else {
      opt = j.template get<T>();  // same as above, but with
                                  // adl_serializer<T>::from_json
    }
  }
};

template <typename T>
struct adl_serializer<boost::optional<T>> {
  static void to_json(json& j, const boost::optional<T>& opt) {
    if (opt == boost::none) {
      j = nullptr;
    } else {
      j = *opt;  // this will call adl_serializer<T>::to_json which will
                 // find the free function to_json in T's namespace!
    }
  }

  static void from_json(const json& j, boost::optional<T>& opt) {
    if (j.is_null()) {
      opt = boost::none;
    } else {
      opt = j.template get<T>();  // same as above, but with
                                  // adl_serializer<T>::from_json
    }
  }
};

template <typename T>
struct adl_serializer<std::shared_ptr<T>> {
  static void to_json(json& j, const std::shared_ptr<T>& opt) {
    if (opt == nullptr) {
      j = nullptr;
    } else {
      j = *opt;  // this will call adl_serializer<T>::to_json which will
                 // find the free function to_json in T's namespace!
    }
  }

  static void from_json(const json& j, std::shared_ptr<T>& opt) {
    if (j.is_null()) {
      opt = opt;
    } else {
      opt = std::make_shared<T>(j.template get<T>());
    }
  }
};

/**
 * @brief General template on Eigen. Somewhat experimental...
 *
 * We would LOVE to be able to template on Derived and specalise on
 * Eigen::MatrixBase<Derived> but for reaons (outlined in this extensiive PR
 * from ma boi: https://github.com/nlohmann/json/issues/3267) we cannot, at
 * least with our version of nlohmann's library. For later jesse or users: I
 * installed the json lib with apt install in April 2024 so maybe this apt has
 * not been updated for a while and the souce code has a bug fix to this issue.
 *
 * NOTE: i imagine this will also fail for dynamic sized arrays....
 *
 * @tparam Scalar
 * @tparam Rows
 * @tparam Cols
 */
template <typename Scalar, int Rows, int Cols>
struct adl_serializer<Eigen::Matrix<Scalar, Rows, Cols>> {
  // static_assert(Rows != Eigen::Dynamic, "Not implemented for dynamic rows");
  // static_assert(Cols != Eigen::Dynamic, "Not implemented for dynamic cols");

  static void to_json(json& j,
                      const Eigen::Matrix<Scalar, Rows, Cols>& matrix) {
    for (int row = 0; row < matrix.rows(); ++row) {
      json column = json::array();
      for (int col = 0; col < matrix.cols(); ++col) {
        column.push_back(matrix(row, col));
      }
      j.push_back(column);
    }
  }

  static void from_json(const json& j,
                        Eigen::Matrix<Scalar, Rows, Cols>& matrix) {
    for (std::size_t row = 0; row < j.size(); ++row) {
      const auto& jrow = j.at(row);
      for (std::size_t col = 0; col < jrow.size(); ++col) {
        const auto& value = jrow.at(col);
        value.get_to(matrix(row, col));
      }
    }
  }
};

// begin POSE3
template <>
struct adl_serializer<gtsam::Pose3> {
  static void to_json(json& j, const gtsam::Pose3& pose) {
    j["tx"] = pose.x();
    j["ty"] = pose.y();
    j["tz"] = pose.z();

    const auto& rotation = pose.rotation();
    const auto& quaternion = rotation.toQuaternion();

    j["qx"] = quaternion.x();
    j["qy"] = quaternion.y();
    j["qz"] = quaternion.z();
    j["qw"] = quaternion.w();
  }

  static void from_json(const json& j, gtsam::Pose3& pose) {
    gtsam::Point3 translation(j["tx"].template get<double>(),
                              j["ty"].template get<double>(),
                              j["tz"].template get<double>());

    // Note: w, x, y, z order
    gtsam::Rot3 rotation(
        j["qw"].template get<double>(), j["qx"].template get<double>(),
        j["qy"].template get<double>(), j["qz"].template get<double>());

    pose = gtsam::Pose3(rotation, translation);
  }
};
// end POSE3

namespace internal {

template <typename KEY, typename VALUE>
struct StringMapConverterBase {
  //! Alias to base type of gtsam::FastMap<KEY, VALUE> which is a std::map with
  //! custom allocator
  using Stdmap_with_allocator_KEY = typename gtsam::FastMap<KEY, VALUE>::Base;
  //! Alias to base type of gtsam::FastMap<std::string, VALUE> which is a
  //! std::map with custom allocator
  using Stdmap_with_allocator_STRING =
      typename gtsam::FastMap<std::string, VALUE>::Base;
};

/**
 * @brief Special handling of the std::Map/gtsam::FastMap case - nlohmann will
 * not properly convert the map to json format if the KEY is not a std::string.
 * We want to therere apply conversion as best we can to do this. We also try to
 * use the inbuit std::map conversion functionality where possible. since we
 * pretty much always use gtsam::FastMap (which is built on std::map with custom
 * allocators) I write custom functions for gtsam::FastMap::Base, in the case
 * that the KEY is not a std::string
 *
 * @tparam KEY
 * @tparam VALUE
 */
template <typename KEY, typename VALUE>
struct StringMapConverter : public StringMapConverterBase<KEY, VALUE> {
  using Base = StringMapConverterBase<KEY, VALUE>;
  using Stdmap_with_allocator_KEY = typename Base::Stdmap_with_allocator_KEY;
  using Stdmap_with_allocator_STRING =
      typename Base::Stdmap_with_allocator_STRING;

  static Stdmap_with_allocator_STRING to_string_map(
      const Stdmap_with_allocator_KEY& typed_map) {
    Stdmap_with_allocator_STRING string_map;
    for (const auto& [key, value] : typed_map) {
      string_map[std::to_string(key)] = value;
    }
    return string_map;
  }

  static Stdmap_with_allocator_KEY from_string_map(
      const Stdmap_with_allocator_STRING& string_map) {
    Stdmap_with_allocator_KEY typed_map;
    for (const auto& [string_key, value] : string_map) {
      KEY key;

      if constexpr (std::is_integral_v<KEY>) {
        key = static_cast<KEY>(std::stoll(string_key));
      } else if constexpr (std::is_floating_point_v<KEY>) {
        key = static_cast<KEY>(std::stod(string_key));
      } else {
        throw std::invalid_argument(
            "Unsupported key type for deserialization. Type is " +
            dyno::type_name<KEY>());
      }

      typed_map[key] = value;
    }
    return typed_map;
  }
};

/**
 * @brief Specalise StringMapConverter for the case when the map already as
 * std::string as its KEY type.
 *
 * @tparam VALUE
 */
template <typename VALUE>
struct StringMapConverter<std::string, VALUE>
    : public StringMapConverterBase<std::string, VALUE> {
  using Base = StringMapConverterBase<std::string, VALUE>;
  using Stdmap_with_allocator_KEY = typename Base::Stdmap_with_allocator_KEY;
  using Stdmap_with_allocator_STRING =
      typename Base::Stdmap_with_allocator_STRING;

  // still copying so could be more efficient!
  static Stdmap_with_allocator_STRING to_string_map(
      const Stdmap_with_allocator_KEY& typed_map) {
    return typed_map;
  }

  // still copying so could be more efficient!
  static Stdmap_with_allocator_KEY from_string_map(
      const Stdmap_with_allocator_STRING& string_map) {
    return string_map;
  }
};

}  // namespace internal

// begin gtsam::FastMap
// in the case where key is NOT an std::string we have to do some versions
// as nlohmann will not properly convert the map to json format if the KEY is
// not a std::string
template <typename KEY, typename VALUE>
struct adl_serializer<gtsam::FastMap<KEY, VALUE>> {
  using Map = gtsam::FastMap<KEY, VALUE>;

  static void to_json(json& j, const gtsam::FastMap<KEY, VALUE>& map) {
    j = internal::StringMapConverter<KEY, VALUE>::to_string_map(
        static_cast<typename Map::Base>(map));
  }

  static void from_json(const json& j, gtsam::FastMap<KEY, VALUE>& map) {
    // this is a truly truly awful line of code...
    // now all maps will be string -> value so all stored maps will be a string
    // map therefore need to tell JSON to to expect the type as a string map
    // before converting it into the expected std::map type (e.g. with correct
    // KEY, VALUE) and then loading it into to fast map. lots of copying etc but
    // this should not happen often...
    map = Map(internal::StringMapConverter<KEY, VALUE>::from_string_map(
        j.template get<typename internal::StringMapConverter<
            KEY, VALUE>::Stdmap_with_allocator_STRING>()));
  }
};

// specalise for when the key is std::string as no special behaviour is needed
template <typename VALUE>
struct adl_serializer<gtsam::FastMap<std::string, VALUE>> {
  using Map = gtsam::FastMap<std::string, VALUE>;

  static void to_json(json& j, const Map& map) {
    j = static_cast<typename Map::Base>(map);
  }

  static void from_json(const json& j, Map& map) {
    map = Map(j.template get<typename Map::Base>());
  }
};
// end gtsam::FastMap

// begin gtsam::FastSet
template <typename VALUE>
struct adl_serializer<gtsam::FastSet<VALUE>> {
  using Set = gtsam::FastSet<VALUE>;

  static void to_json(json& j, const Set& set) {
    j = static_cast<typename Set::Base>(set);
  }

  static void from_json(const json& j, Set& set) {
    set = Set(j.template get<typename Set::Base>());
  }
};
// end gtsam::FastSet

// begin dyno::GenericObjectCentricMap
template <typename VALUE>
struct adl_serializer<dyno::GenericObjectCentricMap<VALUE>> {
  using Map = dyno::GenericObjectCentricMap<VALUE>;

  static void to_json(json& j,
                      const dyno::GenericObjectCentricMap<VALUE>& map) {
    j = static_cast<typename Map::Base>(map);
  }

  static void from_json(const json& j,
                        dyno::GenericObjectCentricMap<VALUE>& map) {
    map = Map(j.template get<typename Map::Base>());
  }
};
// end dyno::GenericObjectCentricMap

template <typename T>
struct adl_serializer<cv::Rect_<T>> {
  static void to_json(json& j, const cv::Rect_<T>& rect) {
    j["x"] = rect.x;
    j["y"] = rect.y;
    j["width"] = rect.width;
    j["height"] = rect.height;
  }

  static void from_json(const json& j, cv::Rect_<T>& rect) {
    T x = j["x"].template get<T>();
    T y = j["y"].template get<T>();
    T width = j["width"].template get<T>();
    T height = j["height"].template get<T>();

    rect = cv::Rect_<T>(x, y, width, height);
  }
};

// begin ISAM2Result::DetailedResults::VariableStatus
template <>
struct adl_serializer<gtsam::ISAM2Result::DetailedResults::VariableStatus> {
  static void to_json(json& j,
                      const gtsam::ISAM2Result::DetailedResults::VariableStatus&
                          variable_status) {
    j["is_reeliminated"] = variable_status.isReeliminated;
    j["is_above_relin_threshold"] = variable_status.isAboveRelinThreshold;
    j["is_relinearized_involved"] = variable_status.isRelinearizeInvolved;
    j["is_relinearized"] = variable_status.isRelinearized;
    j["is_observed"] = variable_status.isObserved;
    j["is_new"] = variable_status.isNew;
    j["is_root_clique"] = variable_status.inRootClique;
  }

  static void from_json(const json&,
                        gtsam::ISAM2Result::DetailedResults::VariableStatus&) {
    // TODO:
  }
};
// end ISAM2Result::DetailedResults::VariableStatus

// begin ISAM2Result::DetailedResults
template <>
struct adl_serializer<gtsam::ISAM2Result::DetailedResults> {
  static void to_json(
      json& j, const gtsam::ISAM2Result::DetailedResults& detailed_result) {
    j["variable_status"] = detailed_result.variableStatus;
  }

  static void from_json(const json&, gtsam::ISAM2Result::DetailedResults&) {
    // TODO:
  }
};
// end ISAM2Result::DetailedResults

// begin ISAM2Result
template <>
struct adl_serializer<gtsam::ISAM2Result> {
  static void to_json(json& j, const gtsam::ISAM2Result& result) {
    j["error_before"] = result.errorBefore;
    j["error_after"] = result.errorAfter;
    j["variables_relinearized"] = result.variablesRelinearized;
    j["variables_reeliminated"] = result.variablesReeliminated;
    j["factors_recalculated"] = result.factorsRecalculated;
    j["cliques"] = result.cliques;
    j["error_after"] = result.errorAfter;
    j["unused_keys"] = result.unusedKeys;
    j["observed_keys"] = result.observedKeys;
    j["observed_keys"] = result.observedKeys;
    j["keys_with_factors_removed"] = result.keysWithRemovedFactors;
    j["marked_keys"] = result.markedKeys;
    j["detailed_results"] = result.detail;
  }

  static void from_json(const json&, gtsam::ISAM2Result&) {
    // TODO
  }
};
// end ISAM2Result

// definitions of seralization functions.
// implementation in .cc
template <>
struct adl_serializer<dyno::RGBDInstanceOutputPacket> {
  static void to_json(json& j, const dyno::RGBDInstanceOutputPacket& input);
  static dyno::RGBDInstanceOutputPacket from_json(const json& j);
};

template <>
struct adl_serializer<dyno::VisionImuPacket> {
  static void to_json(json& j, const dyno::VisionImuPacket& input);
  static dyno::VisionImuPacket from_json(const json& j);
};

template <typename T>
struct adl_serializer<dyno::ReferenceFrameValue<T>> {
  static void to_json(json& j, const dyno::ReferenceFrameValue<T>& value) {
    j["estimate"] = static_cast<T>(value);
    j["reference_frame"] = static_cast<dyno::ReferenceFrame>(value);
  }
  static void from_json(const json& j, dyno::ReferenceFrameValue<T>& value) {
    T estimate = j["estimate"].template get<T>();
    dyno::ReferenceFrame rf =
        j["reference_frame"].template get<dyno::ReferenceFrame>();
    value = dyno::ReferenceFrameValue<T>(estimate, rf);
  }
};

template <typename T>
struct adl_serializer<dyno::MotionReferenceFrame<T>> {
  static void to_json(json& j, const dyno::MotionReferenceFrame<T>& value) {
    j["estimate"] = static_cast<T>(value);
    j["reference_frame"] = static_cast<dyno::ReferenceFrame>(value);
    j["style"] = value.style();
    j["from"] = value.from();
    j["to"] = value.to();
  }
  static void from_json(const json& j, dyno::MotionReferenceFrame<T>& value) {
    T estimate = j["estimate"].template get<T>();
    dyno::ReferenceFrame rf =
        j["reference_frame"].template get<dyno::ReferenceFrame>();
    dyno::FrameId from = j["from"].template get<dyno::FrameId>();
    dyno::FrameId to = j["to"].template get<dyno::FrameId>();
    dyno::MotionRepresentationStyle style =
        j["style"].template get<dyno::MotionRepresentationStyle>();

    value = dyno::MotionReferenceFrame<T>(estimate, style, rf, from, to);
  }
};

template <typename T>
struct adl_serializer<dyno::MeasurementWithCovariance<T>> {
  static void to_json(json& j,
                      const dyno::MeasurementWithCovariance<T>& value) {
    j["measurement"] = value.measurement();
    if (value.hasModel()) {
      j["covariance"] = value.covariance();
    }
  }
  static void from_json(const json& j,
                        dyno::MeasurementWithCovariance<T>& value) {
    T measurement = j["measurement"].template get<T>();

    // check if the json has a covariance value, if it does not it means the
    // original object did not have a sensor model. We should then use the
    // measurement only constructor to ennsure the new model does not have a
    // model eitehr and hasModel() returns false!
    if (j.contains("covariance")) {
      typename dyno::MeasurementWithCovariance<T>::Covariance cov =
          j["covariance"]
              .template get<
                  typename dyno::MeasurementWithCovariance<T>::Covariance>();
      value = dyno::MeasurementWithCovariance<T>(measurement, cov);
      return;
    } else {
      // measurement only constructor
      value = dyno::MeasurementWithCovariance<T>(measurement);
    }
  }
};

template <typename T>
struct adl_serializer<dyno::TrackedValueStatus<T>> {
  static void to_json(json& j, const dyno::TrackedValueStatus<T>& status) {
    using nlohmann::to_json;
    // expect value to be seralizable
    j["value"] = (json)status.value();
    j["frame_id"] = status.frameId();
    j["tracklet_id"] = status.trackletId();
    j["object_id"] = status.objectId();
    j["reference_frame"] = status.referenceFrame();
  }
  static void from_json(const json& j, dyno::TrackedValueStatus<T>& status) {
    typename dyno::TrackedValueStatus<T>::Value value =
        j["value"].template get<typename dyno::TrackedValueStatus<T>::Value>();

    using namespace dyno;
    FrameId frame_id = j["frame_id"].template get<FrameId>();
    TrackletId tracklet_id = j["tracklet_id"].template get<TrackletId>();
    ObjectId object_id = j["object_id"].template get<ObjectId>();
    ReferenceFrame rf = j["reference_frame"].template get<ReferenceFrame>();
    status = dyno::TrackedValueStatus<T>(value, frame_id, tracklet_id,
                                         object_id, rf);
  }
};

// sinnce VisualMeasurementWithCovStatus is aliased it becomes its own type
// so we need to specify the adl for this
// we just specify the exact type and then cast to the known type
// (TrackedValueStatus) which should allow the adl to find the right type I dont
// know a better way of doing this expect for writing an adl_serializer for
// every class which I dont want to do since the MeasurementStatis classes all
// designed to be defined by typdef
template <typename T>
struct adl_serializer<dyno::VisualMeasurementWithCovStatus<T>> {
  static void to_json(json& j,
                      const dyno::VisualMeasurementWithCovStatus<T>& status) {
    using Type = typename dyno::VisualMeasurementWithCovStatus<T>::Value;
    using TrackedStatusValue = dyno::TrackedValueStatus<Type>;

    const auto& cast_status = static_cast<const TrackedStatusValue&>(status);
    j = cast_status;
  }
  static void from_json(const json& j,
                        dyno::VisualMeasurementWithCovStatus<T>& status) {
    using Type = typename dyno::VisualMeasurementWithCovStatus<T>::Value;
    using TrackedStatusValue = dyno::TrackedValueStatus<Type>;

    status = j.template get<TrackedStatusValue>();
  }
};

// begin dyno::GenericTrackedStatusVector
// The default json seraliser only works with primitive types and the stl
// containers could write own seraliser but also just iterate over things
template <typename DERIVEDSTATUS>
struct adl_serializer<dyno::GenericTrackedStatusVector<DERIVEDSTATUS>> {
  using Vector = dyno::GenericTrackedStatusVector<DERIVEDSTATUS>;

  static void to_json(
      json& j, const dyno::GenericTrackedStatusVector<DERIVEDSTATUS>& map) {
    for (const auto& value : map) {
      json j_value = value;
      j.push_back(j_value);
    }
  }

  static void from_json(const json& j,
                        dyno::GenericTrackedStatusVector<DERIVEDSTATUS>& map) {
    for (const json& j_value : j) {
      DERIVEDSTATUS status = j_value.template get<DERIVEDSTATUS>();
      map.push_back(status);
    }
  }
};
// end dyno::GenericTrackedStatusVector

}  // namespace nlohmann
