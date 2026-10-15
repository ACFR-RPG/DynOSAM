#include "dynosam2/frontend/SparseFlow.hpp"

namespace dyno2 {

OpticalFlowLK::Result OpticalFlowLK::operator()(
    const std::vector<cv::Mat>& prev_pyr, const std::vector<cv::Mat>& next_pyr,
    const std::vector<cv::Point2f>& prev_points) const {
  CHECK_EQ(prev_pyr.size(), next_pyr.size());

  ImplResult forward;
  forward.next = prev_points;
  forward.prev = prev_points;

  cv::calcOpticalFlowPyrLK(prev_pyr, next_pyr, forward.prev, forward.next,
                           forward.status, forward.error, win_size, max_level,
                           criteria, 0);

  CHECK_EQ(forward.size(), prev_points.size());

  ImplResult reverse;
  // now do reverse flow
  reverse.status.resize(forward.size());
  reverse.error.resize(forward.size());
  reverse.next = forward.next;
  reverse.prev = forward.next;

  cv::calcOpticalFlowPyrLK(next_pyr, prev_pyr, forward.next, reverse.next,
                           reverse.status, reverse.error, win_size, max_level,
                           criteria, cv::OPTFLOW_USE_INITIAL_FLOW);

  return Result{forward, reverse};
}

}  // namespace dyno2
