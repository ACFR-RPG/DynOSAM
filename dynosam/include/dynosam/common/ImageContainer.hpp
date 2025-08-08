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

#include <exception>
#include <opencv4/opencv2/opencv.hpp>
#include <type_traits>

#include "dynosam/common/Exceptions.hpp"
#include "dynosam/common/ImageTypes.hpp"
#include "dynosam/common/Types.hpp"
#include "dynosam/utils/OpenCVUtils.hpp"
#include "dynosam/utils/Tuple.hpp"

namespace dyno {

class ImageContainerConstructionException : public DynosamException {
 public:
  ImageContainerConstructionException(const std::string& what)
      : DynosamException(what) {}
};

class MismatchedImageWrapperTypes : public DynosamException {
 public:
  MismatchedImageWrapperTypes(const std::string& requested_type,
                              const std::string& stored_type)
      : DynosamException("Attempting to retrieve value with requested type \"" +
                         requested_type + "\", but stored type is " +
                         stored_type),
        requested_type_name(requested_type),
        stored_type_name(stored_type) {}

  const std::string requested_type_name;
  const std::string stored_type_name;
};

class ImageKeyAlreadyExists : public DynosamException {
 public:
  ImageKeyAlreadyExists(const std::string& image_key)
      : DynosamException("Image key \"" + image_key +
                         "\" already exists in the container"),
        key(image_key) {}

  const std::string key;
};

class ImageKeyDoesNotExist : public DynosamException {
 public:
  ImageKeyDoesNotExist(const std::string& image_key)
      : DynosamException("Image key \"" + image_key +
                         "\" does not exist in the container"),
        key(image_key) {}

  const std::string key;
};

/**
 * @brief Basic class to contain Images which are the input for the DynoSAM
 * system. Instead of just containing a map of cv::Mat's we wrap each image
 * using a ImageType which provides a way of abstracting the 'meaning' of each
 * image given its purpose (ie. an rgb image is different in both data-type and
 * value meaning than the motion-mask).
 *
 * This allows for checking that each image has the correct properties and
 * allows for easier handling of common functionn that differ in implementation
 * for each image type.
 *
 * Each image is associated with a key, allowing dynamic allocation of images
 * and input depending on what is abvailable ie. this class can represent a
 * stereo-pair/RGB-D pair and monocular camera, plus the pre-processed optical
 * flow and semantic/motion mask for a single timestamp.
 *
 * This class also atttempt to mimic the copy/assignment behaviour of the
 * cv::Mat class everything is a shallow copy until specified with clone
 *
 */
class ImageContainer {
 private:
  /**
   * @brief Internal data-structure to manage a key-image mapping pair
   *
   * The image is stored as a type erased ImageWrapper which is defined upon
   * construction and recoverable by dynamic pointer casting.
   *
   * Note: Move constructors and assignments are made explicit.
   * According to chat-gpt (seems to check out with the documentation):
   * "you declare any constructor, especially a templated constructor,
   *  the compiler does not generate copy/move constructors or assignments
   * unless you explicitly say".
   *
   * We can make them default to let the compiler figure out how to define them
   * since all types are well defined.
   *
   */
  struct KeyImagePair {
    const std::string key;
    const std::string stored_type;
    std::unique_ptr<ImageBase> ptr;

    // only for when the type is externally known as the ImageBase contains no
    // information about the (IMAGETYPE) type
    KeyImagePair(const std::string& name, const std::string& type,
                 std::unique_ptr<ImageBase> image_wrapper)
        : key(name), stored_type(type), ptr(std::move(image_wrapper)) {}

    template <typename IMAGETYPE>
    KeyImagePair(const std::string& name,
                 std::unique_ptr<ImageWrapper<IMAGETYPE>> image_wrapper)
        : key(name),
          stored_type(type_name<IMAGETYPE>()),
          ptr(std::move(image_wrapper)) {}

    // Copy constructor
    KeyImagePair(const KeyImagePair& other)
        : key(other.key),
          stored_type(other.stored_type),
          ptr(other.ptr ? other.ptr->shallowCopy() : nullptr) {}

    // Copy assignment operator
    KeyImagePair& operator=(const KeyImagePair& other) {
      if (this != &other) {
        // key and stored_type are const, so normally non-assignable,
        // so this only works if you can guarantee they are the same or make
        // them non-const. Otherwise, you may want to remove const qualifier on
        // key and stored_type.
        assert(this->stored_type == other.stored_type);
        assert(this->key == other.key);

        // For simplicity, assume non-const or only allow assignment when key is
        // the same
        ptr = other.ptr ? other.ptr->shallowCopy() : nullptr;
      }
      return *this;
    }

    // Move constructor
    KeyImagePair(KeyImagePair&&) noexcept = default;
    // Move assignment
    KeyImagePair& operator=(KeyImagePair&&) noexcept = default;

    template <typename IMAGETYPE>
    static KeyImagePair Create(const std::string& key, const cv::Mat& image) {
      return KeyImagePair(key,
                          std::make_unique<ImageWrapper<IMAGETYPE>>(image));
    }

    KeyImagePair clone() const {
      CHECK_NOTNULL(ptr);
      return KeyImagePair(key, stored_type, ptr ? ptr->deepCopy() : nullptr);
    }

    template <typename IMAGETYPE>
    const ImageWrapper<IMAGETYPE>& cast() const {
      return castImpl<IMAGETYPE, true>();
    }

    template <typename IMAGETYPE>
    ImageWrapper<IMAGETYPE>& cast() {
      return castImpl<IMAGETYPE, false>();
    }

    template <typename IMAGETYPE, bool IsConst>
    decltype(auto) castImpl() const {
      using WrapperType = ImageWrapper<IMAGETYPE>;
      using PointerType =
          std::conditional_t<IsConst, const WrapperType*, WrapperType*>;

      PointerType casted = dynamic_cast<PointerType>(ptr.get());

      if (!casted) {
        const auto requested_type = type_name<IMAGETYPE>();
        throw MismatchedImageWrapperTypes(requested_type, stored_type);
      }

      return *casted;
    }
  };

  template <typename Container, typename IMAGETYPE>
  static decltype(auto) atImpl(Container* image_container,
                               const std::string& key) {
    if (!image_container->exists(key)) {
      throw ImageKeyDoesNotExist(key);
    }

    decltype(auto) key_image = image_container->images_.at(key);
    return key_image.template cast<IMAGETYPE>();
  }

  FrameId frame_id_;
  Timestamp timestamp_;
  gtsam::FastMap<std::string, KeyImagePair> images_;

 public:
  DYNO_POINTER_TYPEDEFS(ImageContainer)

  static constexpr char kRGB[] = "rgb";
  static constexpr char kOPticalFlow[] = "opticalflow";
  static constexpr char kDepth[] = "depth";
  static constexpr char kObjectMask[] = "objectmask";
  static constexpr char kRightRgb[] = "rightrgb";

 public:
  ImageContainer(FrameId frame_id, Timestamp timestamp)
      : frame_id_(frame_id), timestamp_(timestamp), images_() {}
  ImageContainer() : frame_id_(0), timestamp_(InvalidTimestamp), images_() {}

  ImageContainer(const ImageContainer& other)
      : frame_id_(other.frame_id_), timestamp_(other.timestamp_) {
    for (const auto& [k, v] : other.images_) {
      images_.emplace(
          k, v);  // Uses KeyImagePair copy ctor above (ie. shallow image copy)
    }
  }

  ImageContainer& operator=(const ImageContainer& other) {
    if (this != &other) {
      frame_id_ = other.frame_id_;
      timestamp_ = other.timestamp_;
      images_.clear();
      for (const auto& [k, v] : other.images_) {
        images_.emplace(k, v);
      }
    }
    return *this;
  }

  ImageContainer(ImageContainer&&) noexcept = default;
  ImageContainer& operator=(ImageContainer&&) noexcept = default;

  template <typename IMAGETYPE>
  ImageContainer& add(const std::string& key, const cv::Mat& image) {
    if (exists(key)) {
      throw ImageKeyAlreadyExists(key);
    }

    KeyImagePair key_image = KeyImagePair::Create<IMAGETYPE>(key, image);
    images_.insert({key, std::move(key_image)});

    return *this;
  }

  template <typename IMAGETYPE>
  const ImageWrapper<IMAGETYPE>& at(const std::string& key) const {
    return atImpl<const ImageContainer, IMAGETYPE>(this, key);
  }

  template <typename IMAGETYPE>
  ImageWrapper<IMAGETYPE>& at(const std::string& key) {
    return atImpl<ImageContainer, IMAGETYPE>(const_cast<ImageContainer*>(this),
                                             key);
  }

  ImageContainer clone() const;

  inline bool exists(const std::string& key) const {
    return images_.exists(key);
  }
  inline size_t size() const { return images_.size(); }

  // Specific getters for known/expected image types
  inline bool hasRgb() const { return exists(kRGB); }
  inline bool hasDepth() const { return exists(kDepth); }
  inline bool hasOpticalFlow() const { return exists(kOPticalFlow); }
  inline bool hasObjectMask() const { return exists(kObjectMask); }
  inline bool hasRightRgb() const { return exists(kRightRgb); }

  const ImageWrapper<ImageType::RGBMono>& rgb() const;
  const ImageWrapper<ImageType::Depth>& depth() const;
  const ImageWrapper<ImageType::OpticalFlow>& opticalFlow() const;
  const ImageWrapper<ImageType::MotionMask>& objectMotionMask() const;
  const ImageWrapper<ImageType::RGBMono>& rightRgb() const;

  ImageWrapper<ImageType::RGBMono>& rgb();
  ImageWrapper<ImageType::Depth>& depth();
  ImageWrapper<ImageType::OpticalFlow>& opticalFlow();
  ImageWrapper<ImageType::MotionMask>& objectMotionMask();
  ImageWrapper<ImageType::RGBMono>& rightRgb();

  ImageContainer& rgb(const cv::Mat& image);
  ImageContainer& depth(const cv::Mat& image);
  ImageContainer& opticalFlow(const cv::Mat& image);
  ImageContainer& objectMotionMask(const cv::Mat& image);
  ImageContainer& rightRgb(const cv::Mat& image);

  Timestamp timestamp() const { return timestamp_; }
  FrameId frameId() const { return frame_id_; }

  std::string toString() const;

  friend std::ostream& operator<<(std::ostream& os,
                                  const ImageContainer& image_container);
};

}  // namespace dyno
