#include "dynosam_nn/trackers/byte_tracker/Rect.hpp"

#include <algorithm>

template <typename T>
byte_track::Rect<T>::Rect(const T& x, const T& y, const T& width,
                          const T& height)
    : tlwh({x, y, width, height}) {}

template <typename T>
byte_track::Rect<T>::~Rect() {}

template <typename T>
const T& byte_track::Rect<T>::x() const {
  return tlwh[0];
}

template <typename T>
const T& byte_track::Rect<T>::y() const {
  return tlwh[1];
}

template <typename T>
const T& byte_track::Rect<T>::width() const {
  return tlwh[2];
}

template <typename T>
const T& byte_track::Rect<T>::height() const {
  return tlwh[3];
}

template <typename T>
T& byte_track::Rect<T>::x() {
  return tlwh[0];
}

template <typename T>
T& byte_track::Rect<T>::y() {
  return tlwh[1];
}

template <typename T>
T& byte_track::Rect<T>::width() {
  return tlwh[2];
}

template <typename T>
T& byte_track::Rect<T>::height() {
  return tlwh[3];
}

template <typename T>
const T& byte_track::Rect<T>::tl_x() const {
  return tlwh[0];
}

template <typename T>
const T& byte_track::Rect<T>::tl_y() const {
  return tlwh[1];
}

template <typename T>
T byte_track::Rect<T>::br_x() const {
  return tlwh[0] + tlwh[2];
}

template <typename T>
T byte_track::Rect<T>::br_y() const {
  return tlwh[1] + tlwh[3];
}

template <typename T>
byte_track::Tlbr<T> byte_track::Rect<T>::getTlbr() const {
  return {
      tlwh[0],
      tlwh[1],
      tlwh[0] + tlwh[2],
      tlwh[1] + tlwh[3],
  };
}

template <typename T>
byte_track::Xyah<T> byte_track::Rect<T>::getXyah() const {
  return {
      tlwh[0] + tlwh[2] / 2,
      tlwh[1] + tlwh[3] / 2,
      tlwh[2] / tlwh[3],
      tlwh[3],
  };
}

// template <typename T>
// float byte_track::Rect<T>::calcIoU(const Rect<T>& other) const {
//   const float box_area = (other.tlwh[2] + 1) * (other.tlwh[3] + 1);
//   const float iw = std::min(tlwh[0] + tlwh[2], other.tlwh[0] + other.tlwh[2])
//   -
//                    std::max(tlwh[0], other.tlwh[0]) + 1;
//   float iou = 0;
//   if (iw > 0) {
//     const float ih =
//         std::min(tlwh[1] + tlwh[3], other.tlwh[1] + other.tlwh[3]) -
//         std::max(tlwh[1], other.tlwh[1]) + 1;
//     if (ih > 0) {
//       const float ua = (tlwh[0] + tlwh[2] - tlwh[0] + 1) *
//                            (tlwh[1] + tlwh[3] - tlwh[1] + 1) +
//                        box_area - iw * ih;
//       iou = iw * ih / ua;
//     }
//   }
//   return iou;
// }
template <typename T>
float byte_track::Rect<T>::calcIoU(const Rect<T>& other) const {
  // Intersection.
  const float x1 = std::max(tlwh[0], other.tlwh[0]);
  const float y1 = std::max(tlwh[1], other.tlwh[1]);
  const float x2 = std::min(tlwh[0] + tlwh[2], other.tlwh[0] + other.tlwh[2]);
  const float y2 = std::min(tlwh[1] + tlwh[3], other.tlwh[1] + other.tlwh[3]);

  const float iw = std::max(0.0f, x2 - x1 + 1.0f);
  const float ih = std::max(0.0f, y2 - y1 + 1.0f);
  const float intersection = iw * ih;

  // Union.
  const float area1 = (tlwh[2] + 1.0f) * (tlwh[3] + 1.0f);
  const float area2 = (other.tlwh[2] + 1.0f) * (other.tlwh[3] + 1.0f);
  const float union_area = area1 + area2 - intersection;

  const float iou = union_area > 0.0f ? intersection / union_area : 0.0f;

  // Squared distance between box centres.
  const float cx1 = tlwh[0] + tlwh[2] * 0.5f;
  const float cy1 = tlwh[1] + tlwh[3] * 0.5f;
  const float cx2 = other.tlwh[0] + other.tlwh[2] * 0.5f;
  const float cy2 = other.tlwh[1] + other.tlwh[3] * 0.5f;

  const float centre_dist_sq =
      (cx1 - cx2) * (cx1 - cx2) + (cy1 - cy2) * (cy1 - cy2);

  // Squared diagonal of the smallest enclosing rectangle.
  const float enclosing_w =
      std::max(tlwh[0] + tlwh[2], other.tlwh[0] + other.tlwh[2]) -
      std::min(tlwh[0], other.tlwh[0]);

  const float enclosing_h =
      std::max(tlwh[1] + tlwh[3], other.tlwh[1] + other.tlwh[3]) -
      std::min(tlwh[1], other.tlwh[1]);

  const float enclosing_diag_sq =
      enclosing_w * enclosing_w + enclosing_h * enclosing_h;

  const float distance_penalty =
      enclosing_diag_sq > 0.0f ? centre_dist_sq / enclosing_diag_sq : 0.0f;

  return iou - distance_penalty;
}

template <typename T>
byte_track::Rect<T> byte_track::generate_rect_by_tlbr(
    const byte_track::Tlbr<T>& tlbr) {
  return byte_track::Rect<T>(tlbr[0], tlbr[1], tlbr[2] - tlbr[0],
                             tlbr[3] - tlbr[1]);
}

template <typename T>
byte_track::Rect<T> byte_track::generate_rect_by_xyah(
    const byte_track::Xyah<T>& xyah) {
  const auto width = xyah[2] * xyah[3];
  return byte_track::Rect<T>(xyah[0] - width / 2, xyah[1] - xyah[3] / 2, width,
                             xyah[3]);
}

// explicit instantiation
template class byte_track::Rect<int>;
template class byte_track::Rect<float>;

template byte_track::Rect<int> byte_track::generate_rect_by_tlbr<int>(
    const byte_track::Tlbr<int>&);
template byte_track::Rect<float> byte_track::generate_rect_by_tlbr<float>(
    const byte_track::Tlbr<float>&);

template byte_track::Rect<int> byte_track::generate_rect_by_xyah<int>(
    const byte_track::Xyah<int>&);
template byte_track::Rect<float> byte_track::generate_rect_by_xyah<float>(
    const byte_track::Xyah<float>&);
