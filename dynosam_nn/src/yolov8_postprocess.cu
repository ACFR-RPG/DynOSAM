#include "dynosam_nn/YoloV8Utils.hpp"
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

static const int STRIDE = sizeof(YoloDetection) / sizeof(float);

// // rect should be of length 4 in the form x,y, width, height
// __device__ void jaccardDistance(const float* __restrict__ rect_a,
//                                 const float* __restrict__ rect_b,
//                                 float* __restrict__ distance) {
//     const float Aa = rect_a[3] * rect_a[4];
//     const float Ab = rect_a[3] * rect_a[4];

//     if ((Aa + Ab) <= std::numeric_limits<float>::epsilon()) {
//         // jaccard_index = 1 -> distance = 0
//         *distance = 0.0;
//         return;
//     }

//     // compute intersection

//     const float x1 = fmaxf(rect_a[0], rect_b[0]);
//     const float y1 = fmaxf(rect_a[1], rect_b[1]);
//     const float x2 = fminf(rect_a[0] + rect_a[2], rect_b[0] + rect_b[2]);
//     const float y2 = fminf(rect_a[1] + rect_a[3], rect_b[1] + rect_b[3]);

//     // Compute width/height of intersection
//     float w = fmaxf(0.0f, x2 - x1);
//     float h = fmaxf(0.0f, y2 - y1);

//     // Intersection area (cast to double if you want double)
//     float Aab = w * h;
//     // distance = 1 - jaccard_index
//     *distance = 1.0 - Aab / (Aa + Ab - Aab);
// }


// --- Device Kernel ---
__global__ void YOLO_PostProcess_Kernel(
    const float* __restrict__ d_input,
    float* __restrict__ d_detections, // Treated as flat float array
    int* __restrict__ d_count,
    int N, int C, float CONF_THRESHOLD,
    int BoxOffset, int ClassConfOffset, int MaskCoeffOffset)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    // printf("In kernel\n");
    // printf("i %d, N %d\n", i, N);

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

    // printf("Class id %d\n", classId);

    if (maxConf < CONF_THRESHOLD) return;

    // 2. Atomic Allocation
    int write_idx = atomicAdd(d_count, 1);
    if (write_idx >= N) return;

    // 3. Write Data (38 floats per detection)
    // const int STRIDE = 38; // sizeof(YoloDetection) / sizeof(float)
    float* out_ptr = d_detections + (write_idx * STRIDE);

    out_ptr[0] = d_input[BoxOffset * N + i];       // x
    out_ptr[1] = d_input[(BoxOffset + 1) * N + i]; // y
    out_ptr[2] = d_input[(BoxOffset + 2) * N + i]; // w
    out_ptr[3] = d_input[(BoxOffset + 3) * N + i]; // h
    out_ptr[4] = maxConf;
    out_ptr[5] = (float)classId;

    // Copy mask coeffs
    for (int m = 0; m < 32; ++m) {
        out_ptr[6 + m] = d_input[(MaskCoeffOffset + m) * N + i];
    }
}

// --- Host Wrapper Function ---
int LaunchYoloPostProcess(
    const float* d_model_output,
    const YoloKernelConfig& config,
    YoloDetection* h_output_buffer,
    YoloDetection* d_output_buffer,
    int* d_count_buffer,
    void* stream_ptr)
{
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

    // 1. Reset the atomic counter on Device
    cudaMemsetAsync(d_count_buffer, 0, sizeof(int), stream);

    // 2. Launch Kernel
    const int threads = 256;
    const int blocks = (config.num_boxes + threads - 1) / threads;

    // LOG(INFO) << "Here " << blocks;
    // printf("Here");

    // Cast YoloDetection* to float* for the kernel
    YOLO_PostProcess_Kernel<<<blocks, threads, 0, stream>>>(
        d_model_output,
        reinterpret_cast<float*>(d_output_buffer),
        d_count_buffer,
        config.num_boxes,
        config.num_classes,
        config.conf_threshold,
        config.box_offset,
        config.class_conf_offset,
        config.mask_coeff_offset
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();
    fflush(stdout);

    // 3. Copy Count Back to Host
    // We need a small temporary host variable for the count.
    // Ideally, this is a pinned memory member of your class, but local var is okay for sync.
    int valid_count = 0;
    cudaMemcpyAsync(&valid_count, d_count_buffer, sizeof(int), cudaMemcpyDeviceToHost, stream);

    // SYNC POINT: We must know the count to know how much data to copy next.
    // If you want to be fully async, you'd need to copy the *max* buffer or use unified memory.
    // For simplicity/safety here, we sync the stream.
    cudaStreamSynchronize(stream);
    //for printf!

    // 4. Copy Valid Data Back to Host
    if (valid_count > 0) {
        size_t copy_size = valid_count * sizeof(YoloDetection);
        cudaMemcpyAsync(h_output_buffer, d_output_buffer, copy_size, cudaMemcpyDeviceToHost, stream);

        // Final sync ensures data is ready on CPU before function returns
        cudaStreamSynchronize(stream);
    }

    return valid_count;
}

// ITU-R BT.601 coefficients for Luminance (Y) calculation
#define R_WEIGHT 0.299f
#define G_WEIGHT 0.587f
#define B_WEIGHT 0.114f

__global__ void PlanarRGBToGrayscaleKernel(
    cv::cuda::PtrStepSz<const float> R,
    cv::cuda::PtrStepSz<const float> G,
    cv::cuda::PtrStepSz<const float> B,
    cv::cuda::PtrStepSz<float> output) // Output is a single-channel float GpuMat
{
    // Map thread indices to pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check using the output dimensions
    if (x < output.cols && y < output.rows) {

        // 1. Read the normalized channel values (0.0 to 1.0)
        // Since input GpuMats are wrappers on *contiguous* CHW memory,
        // using the accessors (y, x) ensures correct pitch handling (if any, though expected 0).
        float val_r = R(y, x);
        float val_g = G(y, x);
        float val_b = B(y, x);

        // 2. Calculate Luminance (Greyscale intensity)
        float grayscale_intensity = (R_WEIGHT * val_r) +
                                   (G_WEIGHT * val_g) +
                                   (B_WEIGHT * val_b);

        // 3. Write the result to the single-channel output GpuMat
        output(y, x) = grayscale_intensity;
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
    // Map thread indices: x/y for the pixel, z for the detection index
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
    // int pixel_idx = y * Mask_W + x; // Pixel index within the 160x160 plane

    // // 1. Linear Combination Loop (Sequential inside the thread)
    // for (int m = 0; m < 32; ++m) {
    //     // Access coefficient m for this detection row
    //     // d_mask_coeffs(det_idx, m) accesses the coefficient
    //     float coeff = d_mask_coeffs(det_idx, m);

    //     // Prototype memory index: (m * H + y) * W + x
    //     int proto_idx = (m * Mask_H * Mask_W) + pixel_idx;
    //     float proto_val = d_prototypes.data[proto_idx]; // Assuming d_prototypes is linear (often true for TensorRT)

    //     pixel_sum += coeff * proto_val;
    // }

    // // 2. Sigmoid Activation
    // float final_float_mask_val = 1.0f / (1.0f + __expf(-pixel_sum));

    // // 3. Write to the output buffer
    // d_output_masks.data[det_idx * Mask_H * Mask_W + pixel_idx] = final_float_mask_val;
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

void RunFullMaskPipelineGPU(
    const DetectionGpuMats& wrappers,
    const cv::cuda::GpuMat& d_prototype_masks,
    const cv::Rect& prototype_crop_rect, // Pre-calculated crop area
    const YoloDetection* h_detection,
    const cv::Size& required_size,
    const cv::Size& original_size,
    const int mask_h,
    const int mask_w,
    dyno::ObjectDetection& detection)
{
    //This should be 1!
    int N_Detections = wrappers.boxes.rows;
    CHECK_EQ(N_Detections, 1);
    if (N_Detections == 0) return;

    // --- 1. Launch Mask Combination Kernel (Run once for all detections) ---
    // Kernel launch configuration for 160x160 mask for N_Detections
    dim3 block(16, 16);
    dim3 grid(
        (mask_w + block.x - 1) / block.x,
        (mask_h + block.y - 1) / block.y
    );

    cv::cuda::GpuMat d_combined_masks;
    d_combined_masks.create(
        mask_h,                         // rows: Treat as a flattened 1D array
        mask_w,    // cols: Total number of float pixels
        CV_32FC1                   // type: Output of sigmoid is a float (0.0 to 1.0)
    );
    // Assuming d_prototypes is a linear array, not pitched.
    YOLO_Mask_Combination_Kernel<<<grid, block>>>(
        wrappers.mask_coeffs,
        d_prototype_masks, // Assuming d_prototype_masks contains 32*H*W linear floats
        d_combined_masks,  // The N x mask_h x mask_w combined float mask buffer
        N_Detections, mask_h, mask_w
    );

    LOG(INFO) << "Done YOLO comb mask";
    cudaDeviceSynchronize();


    // --- 2. Sequential Post-Processing (Loop over detections, using cv::cuda) ---
    cv::cuda::GpuMat d_final_masks_full_res(
        original_size, CV_8UC1); // Reusable buffer for final binary mask

    for(int i = 0; i < N_Detections; ++i) {
        // --- A. Define ROI wrappers for the combined mask ---
        // Create a wrapper for the i-th 160x160 mask plane
        cv::cuda::GpuMat d_combined_mask_plane = d_combined_masks;
        // --- B. Crop to Letterbox Area (GPU Sub-Region) ---
        cv::cuda::GpuMat d_cropped_mask =
            d_combined_mask_plane(prototype_crop_rect);

        // --- C. Resize to Original Dimensions (GPU) ---
        cv::cuda::GpuMat d_resized_mask(original_size, CV_32FC1);
        cv::cuda::resize(d_cropped_mask, d_resized_mask, original_size,
                         0, 0, cv::INTER_LINEAR);

        // // --- D. Threshold and Convert to Binary (GPU) ---
        // // stream?
        cv::cuda::GpuMat d_binary_mask(original_size, CV_8UC1);
        cv::cuda::threshold(d_resized_mask, d_binary_mask, 0.5, 255.0,
                            cv::THRESH_BINARY);

        // // Convert to 8-bit unsigned integer (CV_8UC1)
        d_binary_mask.convertTo(d_binary_mask, CV_8UC1);

        // --- E. Final Crop/Copy to Bounding Box (GPU) ---
        // The final mask is only valid inside the detection box.

        // 1. Download bounding box data (Smallest possible transfer)
        cv::Mat box_data; // x, y, w, h
        cv::cuda::GpuMat box_row = wrappers.boxes.row(i);
        box_row.download(box_data);

        const float bb_x = box_data.at<float>(0);
        const float bb_y = box_data.at<float>(1);
        const float bb_w = box_data.at<float>(2);
        const float bb_h = box_data.at<float>(3);
        float half_w = bb_w * 0.5f;
        float half_h = bb_h * 0.5f;
        const cv::Rect box(
            static_cast<int>(bb_x - half_w + 0.5f), // round
            static_cast<int>(bb_y - half_h + 0.5f), // round
            static_cast<int>(bb_w + 0.5f),
            static_cast<int>(bb_h + 0.5f)
        );

        LOG(INFO) << bb_x << " " << bb_y << " " << bb_w << " " << bb_h;

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
        cv::Mat final_cpu_mask = cv::Mat::zeros(original_size, CV_8U);
        d_final_masks_full_res.download(final_cpu_mask);


        const float confidence = h_detection->confidence;
        const int class_id = static_cast<int>(h_detection->class_id);
        // float confidence = wrappers.scores_and_classes.row(i).ptr<float>()[0]; // Re-download or pass via arg
        // int class_id = static_cast<int>(wrappers.scores_and_classes.row(i).ptr<float>()[1]);

        detection = dyno::ObjectDetection{final_cpu_mask, bounding_box, "",confidence};

        // LOG(INFO) << "class id " << class_id;

        // ... retrieve class_label and push to detections ...
        // (This final step requires the small download, but everything before it was GPU-accelerated).

    }
}

void LaunchGrayscaleConversion(
    const std::vector<cv::cuda::GpuMat>& channel_wrappers,
    cv::Mat& h_grayscale_out,
    cudaStream_t stream)
{

    cv::cuda::GpuMat d_grayscale_out;

    // Sanity check
    if (channel_wrappers.size() != 3 || channel_wrappers[0].type() != CV_32FC1) {
        throw std::runtime_error("Input channel wrappers must be 3x CV_32FC1.");
    }

    int H = channel_wrappers[0].rows;
    int W = channel_wrappers[0].cols;

    // 1. Allocate the output GpuMat (single channel, float)
    // This GpuMat will own its memory and use OpenCV's pitch.
    d_grayscale_out.create(H, W, CV_32FC1);

    // 2. Configure grid and block dimensions
    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x,
              (H + block.y - 1) / block.y);

    // 3. Launch the kernel
    PlanarRGBToGrayscaleKernel<<<grid, block, 0, stream>>>(
        channel_wrappers[0], // R
        channel_wrappers[1], // G
        channel_wrappers[2], // B
        d_grayscale_out        // Output (CV_32FC1)
    );

    d_grayscale_out.download(h_grayscale_out);
}
