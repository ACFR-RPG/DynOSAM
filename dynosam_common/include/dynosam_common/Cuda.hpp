#pragma once

#include <opencv2/core/cvdef.h>
#include <opencv2/cvconfig.h>  // for the HAVE_CUDA define

///! Re-expose OpenCV's CUDA define under our own project macro
#ifdef HAVE_CUDA
#define DYNO_CUDA_OPENCV_ENABLED
#endif

constexpr bool isOpencvCudaEnabled() {
#ifdef DYNO_CUDA_OPENCV_ENABLED
  return true;
#else
  return false;
#endif
}
