#include "dynosam2/frontend/FrontendViz.hpp"

#include "dynosam_common/utils/GtsamUtils.hpp"
#include "dynosam_common/utils/TimingStats.hpp"

namespace dyno2 {

FrontendViz::FrontendViz(const SlamBackend& estimator)
    : estimator_(estimator) {}

cv::Mat FrontendViz::draw(FrontendViz::Data::Ptr data) {
  dyno::utils::ChronoTimingStats timer("frontend.viz-tracking");
  const dyno::FrameId fid = data->images->frameId();
  const dyno::FrameId timestamp = data->images->timestamp();

  const ImageWrapper<ImageType::RGBMono>& img_wrapper = data->images->rgb();
  cv::Mat img_rgb = img_wrapper.toRGB().clone();

  // draw static correspondences
  using DoFrameTracks = FrontendViz::Data::DoFrameTracks;
  using ViFrameTracks = FrontendViz::Data::ViFrameTracks;

  const VioGraphEstimator& vi_estimator = estimator_.vioEstimator();
  const ViFrameTracks& vi_frame_tracks = data->vi_tracks;
  const ViFrame::Ptr vi_frame = vi_frame_tracks.frame;
  auto relative_motion = vi_frame->getRelativePose();
  CHECK(relative_motion);
  // sanity check
  CHECK_EQ(relative_motion->to(), fid);

  // frame against which we tracked
  const dyno::FrameId ref_frame = relative_motion->from();
  const ViFrame::Ptr ref_vi_frame = vi_estimator.getFramePtr(ref_frame);
  CHECK_NOTNULL(ref_vi_frame);

  for (dyno::TrackletId tid : vi_frame_tracks.correspondences) {
    const dyno::Keypoint pt = vi_frame_tracks.frame->keypointById(tid);
    const cv::Point2f pt_cv = dyno::utils::gtsamPointToCv(pt);

    // this point is in the map
    // TODO: draw projection
    if (vi_estimator.landmarkExists(tid)) {
      cv::circle(img_rgb, pt_cv, 2, blue, cv::FILLED, cv::LINE_AA);
    } else {
      // this should always be in the successful tracking case
      // TODO: ie tracking quality is good! (becuase if trackign quality is good
      // we should have tracks EACTLY from this frame!)
      // TODO: otherwise we should have no/few tracks (ie motion propogation)
      auto it = ref_vi_frame->local_indices.find(tid);
      if (it != ref_vi_frame->local_indices.end()) {
        cv::Point2f ref_pt_cv;
        ref_vi_frame->getCvKeypointByIndex(it->second, ref_pt_cv);

        cv::circle(img_rgb, pt_cv, 2, red, 1);

        cv::arrowedLine(img_rgb, ref_pt_cv, pt_cv, red, 1);
      }
    }
  }

  // we should only have tracks for valid objects (ie those with a motion
  // estimate) therefore each should have a relative motion as well
  for (const auto& [oid, do_track] : data->do_tracks) {
    const DoGraphEstimator& do_estimator = estimator_.doEstimator(oid);
    const DoFrame::Ptr do_frame = do_track.frame;

    const auto& G = do_frame->getRelativeMotion();
    // sanity check
    CHECK_EQ(G.to(), fid);
    // frame against which we tracked
    const dyno::FrameId ref_frame = G.from();
    const DoFrame::Ptr ref_do_frame = do_estimator.getFramePtr(ref_frame);
    CHECK_NOTNULL(ref_do_frame);

    const dyno::Color object_colour = dyno::Color::uniqueId(oid);
    const cv::Scalar lmk_colour = object_colour.brightness(0.9f).bgra();
    const cv::Scalar tracked_colour = object_colour.brightness(0.5f).bgra();

    for (dyno::TrackletId tid : do_track.correspondences) {
      const dyno::Keypoint pt = do_frame->keypointById(tid);
      const cv::Point2f pt_cv = dyno::utils::gtsamPointToCv(pt);

      // this point is in the map
      // TODO: draw projection
      if (do_estimator.landmarkExists(tid)) {
        cv::circle(img_rgb, pt_cv, 3, lmk_colour, cv::FILLED, cv::LINE_AA);
      } else {
        // this should always be in the successful tracking case
        // TODO: ie tracking quality is good! (becuase if trackign quality is
        // good we should have tracks EACTLY from this frame!)
        // TODO: otherwise we should have no/few tracks (ie motion propogation)
        auto it = ref_do_frame->local_indices.find(tid);
        if (it != ref_do_frame->local_indices.end()) {
          cv::Point2f ref_pt_cv;
          ref_do_frame->getCvKeypointByIndex(it->second, ref_pt_cv);

          cv::circle(img_rgb, pt_cv, 2, tracked_colour, 1);
          cv::arrowedLine(img_rgb, ref_pt_cv, pt_cv, tracked_colour, 1);
        }
      }
    }
  }

  return img_rgb;
}

}  // namespace dyno2
