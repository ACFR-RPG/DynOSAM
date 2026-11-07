#pragma once

namespace dyno {

/**
   * @brief Constants from model architecture used to access output data.
   * NOTE: We pulled these out of the YoloModelInfo struct so that we could include
   * only the necessary headers in the .cu files (otherwise nvcc on ARM woulld file to compile!)
   *
   */
struct YoloV8Constants {
    constexpr static int BoxOffset = 0;
    constexpr static int ClassConfOffset = 4;
    constexpr static int NumMasks = 32;
    static constexpr int MaxDetections = 8400;

    static int MaskCoeffOffset(int num_classes) {
        return num_classes + ClassConfOffset;
    }
};

}