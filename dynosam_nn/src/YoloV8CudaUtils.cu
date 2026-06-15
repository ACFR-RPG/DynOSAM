#include "dynosam_nn/YoloV8CudaUtils.hpp"
#include "dynosam_nn/YoloV8ObjectDetector.hpp"
#include "dynosam_nn/CudaUtils.hpp"
#include "dynosam_common/DynamicObjects.hpp"


#include <cuda_runtime.h>
#include <cstdio>
#include <stdio.h>

#include <opencv4/opencv2/core/cuda.hpp>
#include <opencv4/opencv2/cudawarping.hpp>
#include <opencv4/opencv2/cudaimgproc.hpp>
#include <opencv4/opencv2/cudaarithm.hpp>

#include <opencv4/opencv2/opencv.hpp>

#include <glog/logging.h>

__global__ void letterBoxToBlobKernel(const uchar* __restrict__ srcData,
                                      int srcH, int srcW, int srcStep, int srcChannels,
                                      float* __restrict__ dstBlob,
                                      int dstH, int dstW,
                                      int newH, int newW,
                                      int padTop, int padLeft,
                                      float scaleX, float scaleY) {
    // 2D grid mapping to the destination blob coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dstW || y >= dstH) return;

    int planeSize = dstH * dstW;
    constexpr float padNorm = 114.0f / 255.0f;
    constexpr float scale255 = 1.0f / 255.0f;

    // Check if current thread falls inside the padded image area
    if (y >= padTop && y < (padTop + newH) && x >= padLeft && x < (padLeft + newW)) {
        // Map destination coordinates back to source image coordinates (Bilinear interpolation)
        int srcY_resized = y - padTop;
        int srcX_resized = x - padLeft;

        // Bilinear interpolation coordinates
        float src_fX = srcX_resized * scaleX;
        float src_fY = srcY_resized * scaleY;

        int x1 = __float2int_rd(src_fX);
        int y1 = __float2int_rd(src_fY);
        int x2 = min(x1 + 1, srcW - 1);
        int y2 = min(y1 + 1, srcH - 1);

        float fx2 = src_fX - x1;
        float fx1 = 1.0f - fx2;
        float fy2 = src_fY - y1;
        float fy1 = 1.0f - fy2;

        int dstIdx = y * dstW + x;

        if (srcChannels == 3) {
            // Pointers to the source rows
            const uchar* row1 = srcData + y1 * srcStep;
            const uchar* row2 = srcData + y2 * srcStep;

            // Fetch the 4 neighboring pixels for B, G, R
            float b11 = row1[x1 * 3 + 0], b12 = row1[x2 * 3 + 0];
            float b21 = row2[x1 * 3 + 0], b22 = row2[x2 * 3 + 0];

            float g11 = row1[x1 * 3 + 1], g12 = row1[x2 * 3 + 1];
            float g21 = row2[x1 * 3 + 1], g22 = row2[x2 * 3 + 1];

            float r11 = row1[x1 * 3 + 2], r12 = row1[x2 * 3 + 2];
            float r21 = row2[x1 * 3 + 2], r22 = row2[x2 * 3 + 2];

            // Interpolate
            float b = (b11 * fx1 + b12 * fx2) * fy1 + (b21 * fx1 + b22 * fx2) * fy2;
            float g = (g11 * fx1 + g12 * fx2) * fy1 + (g21 * fx1 + g22 * fx2) * fy2;
            float r = (r11 * fx1 + r12 * fx2) * fy1 + (r21 * fx1 + r22 * fx2) * fy2;

            // Write out to Planar CHW format (BGR -> RGB conversion)
            dstBlob[dstIdx]               = r * scale255; // Red channel plane
            dstBlob[dstIdx + planeSize]     = g * scale255; // Green channel plane
            dstBlob[dstIdx + 2 * planeSize] = b * scale255; // Blue channel plane
        } else {
            // Single channel grayscale path
            const uchar* row1 = srcData + y1 * srcStep;
            const uchar* row2 = srcData + y2 * srcStep;

            float val11 = row1[x1], val12 = row1[x2];
            float val21 = row2[x1], val22 = row2[x2];

            float val = (val11 * fx1 + val12 * fx2) * fy1 + (val21 * fx1 + val22 * fx2) * fy2;
            dstBlob[dstIdx] = val * scale255;
        }
    } else {
        // We are in the padding area
        int dstIdx = y * dstW + x;
        dstBlob[dstIdx] = padNorm;
        if (srcChannels == 3) {
            dstBlob[dstIdx + planeSize]     = padNorm;
            dstBlob[dstIdx + 2 * planeSize] = padNorm;
        }
    }
}


// --- Device Kernel ---
//TODO: I think N, C, ClassConfOffset and MaskCoeffOffset should also be constexpr
// can use C to unroll!?
__global__ void YOLO_PostProcess_Kernel(
    const float* __restrict__ d_input,
    float* __restrict__ d_detections, // Treated as flat float array
    int* __restrict__ d_count,
    int C, float CONF_THRESHOLD,
    int MaskCoeffOffset)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    static constexpr int N =  dyno::YoloV8ModelInfo::Constants::MaxDetections;
    static constexpr int BoxOffset =  dyno::YoloV8ModelInfo::Constants::BoxOffset;
    static constexpr int ClassConfOffset =  dyno::YoloV8ModelInfo::Constants::ClassConfOffset;

    if (i >= N) return;

    // 1. Find Max Confidence
    float maxConf = 0.0f;
    int classId = -1;

    // Optional: #pragma unroll if C is small constant
    for (int c = 0; c < C; ++c) {
        float conf = d_input[(ClassConfOffset + c) * N + i];
        if (conf > maxConf) {
            maxConf = conf;
            classId = c;
        }
    }

    if (maxConf < CONF_THRESHOLD) return;

    // 2. Atomic Allocation
    int write_idx = atomicAdd(d_count, 1);
    if (write_idx >= N) return;

    // 3. Write Data (38 floats per detection)
    // const int STRIDE = 38; // sizeof(YoloDetection) / sizeof(float)
    static constexpr int STRIDE = sizeof(dyno::internal::AlignedYoloDetection) / sizeof(float);
    float* out_ptr = d_detections + (write_idx * STRIDE);

    out_ptr[0] = d_input[BoxOffset * N + i];       // x
    out_ptr[1] = d_input[(BoxOffset + 1) * N + i]; // y
    out_ptr[2] = d_input[(BoxOffset + 2) * N + i]; // w
    out_ptr[3] = d_input[(BoxOffset + 3) * N + i]; // h
    out_ptr[4] = maxConf;
    out_ptr[5] = (float)classId;

    // Copy mask coeffs
    #pragma unroll
    for (int m = 0; m < 32; ++m) {
        out_ptr[6 + m] = d_input[(MaskCoeffOffset + m) * N + i];
    }
}

// Custom kernal to perform binary thresholding with a float mat as input and unsigned char
// mat as output. This is different from the opencv implementation which requires converting the
// float mat before thresholding which takes up most of the computation
__global__ void binaryThreshold(
    cv::cuda::PtrStepSz<const float> in,
    cv::cuda::PtrStepSz<unsigned char> out,
    float thresh,
    unsigned char maxval
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < in.cols && y < in.rows) {
        out(y, x) = (in(y, x) > thresh) ? maxval : 0;
    }
}


__device__ __forceinline__ float sigmoidf(float x)
{
    return 1.0f / (1.0f + __expf(-x));
}

// Kernel to perform linear combination and sigmoid for ALL detections
__global__ void YOLO_Mask_Combination_Kernel(
    cv::cuda::PtrStepSz<const float> d_mask_coeffs, // N detections x 32 coeffs
    cv::cuda::PtrStepSz<const float> d_prototypes,  // 32*H*W linear array
    cv::cuda::PtrStepSz<float> d_output_masks,      // N detections x H*W output
    int N_Detections,
    int mask_h,
    int mask_w)
{
    // Map thread indices: x/y for the pixel
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // int det_idx = blockIdx.z;

    // if (det_idx >= N_Detections || x >= Mask_W || y >= Mask_H) return;

    float pixel_sum = 0.0f;
    if (y >= mask_h || x >= mask_w) return;

    float acc = 0.f;
    const int stride = mask_h * mask_w;   // elements per mask
    const int idx    = y * mask_w + x;    // pixel index inside one mask

    #pragma unroll
    for (int m = 0; m < 32; ++m) {
        float v = d_prototypes.ptr(m * mask_h)[y * mask_w + x];  // read-only
        acc += d_mask_coeffs[m] * v;
    }

    d_output_masks(y, x) = sigmoidf(acc);
}

template <typename T>
    T clamp(const T& val, const T& low, const T& high) {
    return std::max(low, std::min(val, high));
}

inline cv::Rect scaleCoords(const cv::Size& required_shape,
                              const cv::Rect& coords,
                              const cv::Size& originalShape,
                              bool p_Clip = true) {
    float gain =
        std::min((float)required_shape.height / (float)originalShape.height,
                 (float)required_shape.width / (float)originalShape.width);

    int pad_w = static_cast<int>(std::round(
        ((float)required_shape.width - (float)originalShape.width * gain) /
        2.f));
    int pad_h = static_cast<int>(std::round(
        ((float)required_shape.height - (float)originalShape.height * gain) /
        2.f));

    cv::Rect ret;
    ret.x =
        static_cast<int>(std::round(((float)coords.x - (float)pad_w) / gain));
    ret.y =
        static_cast<int>(std::round(((float)coords.y - (float)pad_h) / gain));
    ret.width = static_cast<int>(std::round((float)coords.width / gain));
    ret.height = static_cast<int>(std::round((float)coords.height / gain));

    if (p_Clip) {
      ret.x = clamp(ret.x, 0, originalShape.width);
      ret.y = clamp(ret.y, 0, originalShape.height);
      ret.width = clamp(ret.width, 0, originalShape.width - ret.x);
      ret.height = clamp(ret.height, 0, originalShape.height - ret.y);
    }

    return ret;
  }

namespace dyno {
namespace internal {

// Wrapper implementation to setup grid/block sizes
void launchLetterBoxKernel(const uchar* d_src, int srcH, int srcW, int srcStep, int srcChannels,
                           float* d_blob, int dstH, int dstW, int newH, int newW,
                           int padTop, int padLeft, float scaleX, float scaleY, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((dstW + block.x - 1) / block.x, (dstH + block.y - 1) / block.y);

    letterBoxToBlobKernel<<<grid, block, 0, stream>>>(
        d_src, srcH, srcW, srcStep, srcChannels,
        d_blob, dstH, dstW, newH, newW,
        padTop, padLeft, scaleX, scaleY
    );
}

// void letterBoxToBlobGPU(const cv::Mat& image, DeviceMemory<uchar>& d_image_cache, float* d_processed,
//                                  int targetChannels, const cv::Size& targetSize,
//                                  cv::Size& actualSize, bool dynamicShape,
//                                  cudaStream_t stream) {
//     const int srcH = image.rows;
//     const int srcW = image.cols;
//     int dstH = targetSize.height;
//     int dstW = targetSize.width;

//     // 1. Calculate scale (match Ultralytics exactly)
//     const float scale = std::min(static_cast<float>(dstH) / srcH,
//                                  static_cast<float>(dstW) / srcW);

//     // Ultralytics uses round() for new dimensions
//     int newH = static_cast<int>(std::round(srcH * scale));
//     int newW = static_cast<int>(std::round(srcW * scale));

//     // For dynamic shape, adjust to stride-aligned minimum size
//     if (dynamicShape) {
//         constexpr int stride = 32;
//         dstH = ((newH + stride - 1) / stride) * stride;
//         dstW = ((newW + stride - 1) / stride) * stride;
//     }

//     actualSize = cv::Size(dstW, dstH);

//     // 2. Ultralytics asymmetric padding with -0.1/+0.1 adjustment
//     const float dh = (dstH - newH) / 2.0f;
//     const float dw = (dstW - newW) / 2.0f;
//     const int padTop = static_cast<int>(std::round(dh - 0.1f));
//     const int padLeft = static_cast<int>(std::round(dw - 0.1f));

//     // Inverse scales for mapping target back to source pixels inside the kernel
//     float scaleX = static_cast<float>(srcW) / newW;
//     float scaleY = static_cast<float>(srcH) / newH;

//     // // 3. Allocate and copy the raw source image data to GPU
//     size_t srcBytes = image.step * srcH;
//     d_image_cache.
//     // uchar* d_srcData = nullptr;
//     // cudaMallocAsync(&d_srcData, srcBytes, stream);
//     // cudaMemcpyAsync(d_srcData, image.data, srcBytes, cudaMemcpyHostToDevice, stream);

//     // 4. Launch the integrated Resize + Pad + Format conversion Kernel
//     launchLetterBoxKernel(d_srcData, srcH, srcW, image.step, targetChannels,
//                           d_processed, dstH, dstW, newH, newW,
//                           padTop, padLeft, scaleX, scaleY, stream);

//     // Free the temporary source image memory on the device
//     cudaFreeAsync(d_srcData, stream);
// }

// --- Host Wrapper Function ---
int YoloOutputToDetections(
    const float* d_model_output,
    const YoloKernelConfig& config,
    AlignedYoloDetection* h_output_buffer,
    AlignedYoloDetection* d_output_buffer,
    int* d_count_buffer,
    int* h_count_buffer,
    void* stream_ptr)
{
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

    // // 1. Reset the atomic counter on Device
    // cudaMemsetAsync(d_count_buffer, 0, sizeof(int), stream);
    *h_count_buffer = 0;

    // 2. Launch Kernel
    static constexpr int threads = 256;
    static constexpr auto num_boxes = dyno::YoloV8ModelInfo::Constants::MaxDetections;
    const int blocks = (num_boxes + threads - 1) / threads;

    YOLO_PostProcess_Kernel<<<blocks, threads, 0, stream>>>(
        d_model_output,
        reinterpret_cast<float*>(d_output_buffer),
        d_count_buffer,
        config.num_classes,
        config.conf_threshold,
        config.mask_coeff_offset
    );

    cudaError_t err = cudaGetLastError();
    CUDA_CHECK(err);

    // cudaDeviceSynchronize();
    cudaStreamSynchronize(stream);

    return *h_count_buffer;
}

void YoloDetectionsToObjects(
    const YoloDetectionGpuMatDevice& wrappers,
    const cv::cuda::GpuMat& d_prototype_masks,
    const cv::Rect& prototype_crop_rect, // Pre-calculated crop area
    const AlignedYoloDetection* h_detection,
    const cv::Size& required_size,
    const cv::Size& original_size,
    const std::string& label,
    const int mask_h,
    const int mask_w,
    cv::cuda::Stream stream,
    dyno::ObjectDetection& detection)
{
    // --- 1. Launch Mask Combination Kernel (Run once for all detections) ---
    // Kernel launch configuration for 160x160 mask for N_Detections
    // dim3 block(16, 16);
    //32 and 8 is the default transform policy for opencv
    dim3 block(32, 8);
    dim3 grid(
        (mask_w + block.x - 1) / block.x,
        (mask_h + block.y - 1) / block.y
    );

    cudaStream_t c_stream = (cudaStream_t)stream.cudaPtr();

    cv::cuda::GpuMat d_combined_masks;
    d_combined_masks.create(
        mask_h,                         // rows: Treat as a flattened 1D array
        mask_w,    // cols: Total number of float pixels
        CV_32FC1                   // type: Output of sigmoid is a float (0.0 to 1.0)
    );
    // Assuming d_prototypes is a linear array, not pitched.
    dyno::GpuTimingStats mask_combination_kernel(
        "yolov8_detection.post_process.combination_kernel", c_stream, 5);
    YOLO_Mask_Combination_Kernel<<<grid, block, 0, c_stream>>>(
        wrappers.mask_coeffs,
        d_prototype_masks, // Assuming d_prototype_masks contains 32*H*W linear floats
        d_combined_masks,  // The N x mask_h x mask_w combined float mask buffer
        1, mask_h, mask_w
    );
    mask_combination_kernel.stop();


     dyno::GpuTimingStats timing_detections(
        "yolov8_detection.post_process.cv_processing", stream, 5);
    // --- 2. Sequential Post-Processing (Loop over detections, using cv::cuda) ---
    static cv::cuda::GpuMat d_final_masks_full_res;
    d_final_masks_full_res.create(original_size, CV_8UC1);

    // --- A. Define ROI wrappers for the combined mask ---
    // Create a wrapper for the i-th 160x160 mask plane
    cv::cuda::GpuMat d_combined_mask_plane = d_combined_masks;
    // --- B. Crop to Letterbox Area (GPU Sub-Region) ---
    cv::cuda::GpuMat d_cropped_mask =
        d_combined_mask_plane(prototype_crop_rect);

    // --- C. Resize to Original Dimensions (GPU) ---
    static cv::cuda::GpuMat d_resized_mask;
    d_resized_mask.create(original_size, CV_32FC1);
    cv::cuda::resize(d_cropped_mask, d_resized_mask, original_size,
                        0, 0, cv::INTER_LINEAR, stream);

    // // --- D. Threshold and Convert to Binary (GPU) ---
    // // stream?
    static cv::cuda::GpuMat d_binary_mask;
    d_binary_mask.create(original_size, CV_8UC1);
    // cv::cuda::threshold(d_resized_mask, d_binary_mask, 0.5, 255.0,
    //                     cv::THRESH_BINARY, stream);
    dim3 full_size_grid(
        (original_size.width + block.x - 1) / block.x,
        (original_size.height + block.y - 1) / block.y
    );

    // custom binary thresholding function that is faster than opencv
    binaryThreshold<<<full_size_grid, block, 0, c_stream>>>(d_resized_mask, d_binary_mask, 0.5, 255);

    // // Convert to 8-bit unsigned integer (CV_8UC1)
    // d_binary_mask.convertTo(d_binary_mask, CV_8UC1);

    // --- E. Final Crop/Copy to Bounding Box (GPU) ---
    // The final mask is only valid inside the detection box.

    // 1. Download bounding box data (Smallest possible transfer)
    // cudaDeviceSynchronize();

    cv::Mat box_data; // x, y, w, h
    cv::cuda::GpuMat box_row = wrappers.boxes.row(0);
    box_row.download(box_data, stream);

    // stream.waitForCompletion();


    const float bb_x = box_data.at<float>(0);
    const float bb_y = box_data.at<float>(1);
    const float bb_w = box_data.at<float>(2);
    const float bb_h = box_data.at<float>(3);
    const cv::Rect box(
        static_cast<int>(std::round(bb_x - bb_w / 2.0f)),
        static_cast<int>(std::round(bb_y - bb_h / 2.0f)),
        static_cast<int>(std::round(bb_w)),
        static_cast<int>(std::round(bb_h))
    );

    // LOG(INFO) << bb_x << " " << bb_y << " " << bb_w << " " << bb_h;

    // 2. Calculate final ROI on CPU
    cv::Rect bounding_box = scaleCoords(required_size, box,
            original_size, true); // Assuming this CPU helper exists

    cv::Rect roi = bounding_box;
    roi &= cv::Rect(0, 0, original_size.width, original_size.height);

    // // 3. Clear the output mask buffer and copy the binary mask into the ROI
    d_final_masks_full_res.setTo(cv::Scalar(0)); // Clear previous mask

    if (roi.area() > 0) {
        // Copy the relevant ROI data from the binary mask into the final
        // mask's ROI. The full binary mask is HxW, so we sub-region both.
        d_binary_mask(roi).copyTo(d_final_masks_full_res(roi));
    }

    // // --- F. Final Download (Minimal Transfer) ---
    // // Download only the final result and metadata for ObjectDetection struct
    // //TODO: pinned memory!!
    //TODO: this seems slower than doing CPU (maybe due to copy or constant realloc?)
    // cv::cuda::HostMem h_mem(original_size, CV_8U, cv::cuda::HostMem::PAGE_LOCKED);
    // static cv::cuda::HostMem h_mem(cv::cuda::HostMem::PAGE_LOCKED);
    // h_mem.create(original_size, CV_8U);
    // cv::cuda::HostMem h_mem;
    cv::Mat final_cpu_mask = cv::Mat::zeros(original_size, CV_8U);
    d_final_masks_full_res.download(final_cpu_mask, stream);
    // d_final_masks_full_res.download(h_mem, stream);
    stream.waitForCompletion();

    // // should synchronize!?
    // cv::Mat final_cpu_mask = h_mem.createMatHeader();

    const float confidence = h_detection->confidence;
    const int class_id = static_cast<int>(h_detection->class_id);

    detection = dyno::ObjectDetection{final_cpu_mask, bounding_box, label ,confidence};

}

} //internal
} //dyno
