#pragma once

#include <opencv4/opencv2/opencv.hpp>
#include <vector>

#include "dynosam_cv/CvUtils.hpp"

namespace dyno2 {

struct OpticalFlowLK {
  static constexpr float kMaxErr = 20.0f;

  const cv::Size win_size;
  const int max_level;
  const cv::TermCriteria criteria;

  OpticalFlowLK(int iterations)
      : win_size(24, 24),
        max_level(4),
        criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, iterations,
                 0.01) {}

  OpticalFlowLK() : OpticalFlowLK(50) {}

  struct ImplResult {
    std::vector<uchar> status;
    std::vector<float> error;
    std::vector<cv::Point2f> prev;
    std::vector<cv::Point2f> next;

    size_t size() const { return status.size(); }
  };

  struct Result {
    const ImplResult forward_result;
    const ImplResult reverse_result;

    Result(const ImplResult& forward, const ImplResult& reverse)
        : forward_result(forward), reverse_result(reverse) {}

    const std::vector<cv::Point2f>& predictedPoints() const {
      return forward_result.next;
    }

    const std::vector<cv::Point2f>& fromPoints() const {
      return forward_result.prev;
    }

    const std::vector<float>& error() const { return forward_result.error; }

    size_t size() const { return forward_result.size(); }

    bool isGood(size_t i) const {
      const bool both_status_good =
          forward_result.status.at(i) && reverse_result.status.at(i);

      // compare distance betwen original pixel local and reverse predicted
      // pixel location (which should be the same)
      const bool within_distance =
          dyno::utils::distance(forward_result.prev.at(i),
                                reverse_result.next.at(i)) <= 0.5;

      const bool within_error = forward_result.error[i] < kMaxErr &&
                                reverse_result.error[i] < kMaxErr;

      return both_status_good && within_distance && within_error;
    }
  };

  Result operator()(const std::vector<cv::Mat>& prev_pyr,
                    const std::vector<cv::Mat>& next_pyr,
                    const std::vector<cv::Point2f>& prev_points) const;
};

}  // namespace dyno2
