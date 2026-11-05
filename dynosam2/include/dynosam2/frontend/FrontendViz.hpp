#pragma once

#include "dynosam2/estimator/SlamBackend.hpp"
#include "dynosam2/frontend/Frame.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_common/viz/Colour.hpp"
#include "dynosam_core/DynamicObjects.hpp"
#include "dynosam_sensors/ImageContainer.hpp"

namespace dyno2 {

/**
 * @brief Responsible for visualizing matching and tracking results
 *
 */
class FrontendViz {
 public:
  struct Data {
    // TODO: plus tracking state!
    template <typename FRAME>
    struct FrameTracks {
      std::shared_ptr<FRAME> frame;
      std::vector<TrackletId> correspondences;
      std::vector<KeypointIdentifier> measurements;
    };
    typedef FrameTracks<ViFrame> ViFrameTracks;
    typedef FrameTracks<DoFrame> DoFrameTracks;

    dyno::ImageContainer::Ptr images;
    dyno::ObjectDetectionResult object_detection;

    ViFrameTracks vi_tracks;
    std::unordered_map<dyno::ObjectId, DoFrameTracks> do_tracks;

    typedef std::shared_ptr<Data> Ptr;
  };

  FrontendViz(const SlamBackend& estimator);

  cv::Mat draw(Data::Ptr data);

 private:
  const SlamBackend& estimator_;

  struct TemporalObjectState {
    dyno::ObjectId object_id;

    dyno::Timestamp first_seen_time = -1.0;
    dyno::Timestamp last_seen_time = -1.0;

    float appear_progress = 0.0f;

    mutable std::array<cv::Point2f, 4> filtered_corners;
    mutable bool corners_initialized = false;
  };
  //! How long (in seconds) for 'lock on'
  static constexpr float appear_duration_sec_ = 0.6f;
  //! higher = snappier
  static constexpr float corner_smoothing_alpha_ = 0.6f;
  std::unordered_map<ObjectId, TemporalObjectState> states_;

  inline static const cv::Scalar red{Color::red().bgra()};
  inline static const cv::Scalar green{Color::green().bgra()};
  inline static const cv::Scalar blue{Color::blue().bgra()};
};

}  // namespace dyno2
