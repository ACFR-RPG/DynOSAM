#include "dynosam2/frontend/Frame.hpp"

namespace dyno2 {

void ViFrame::setRelativePose(const Motion3ReferenceFrame& T_ij) {
  CHECK_EQ(T_ij.to(), this->getFrameId());
  T_ij_ = T_ij;
}

}  // namespace dyno2
