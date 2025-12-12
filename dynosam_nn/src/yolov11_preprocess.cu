#include "dynosam_nn/YOLOV11.hpp"
#include "dynosam_nn/cuda_utils.hpp"
#include "device_launch_parameters.h"

// Host and device pointers for image buffers
static uint8_t* img_buffer_host = nullptr;    // Pinned memory on the host for faster transfers
static uint8_t* img_buffer_device = nullptr;  // Memory on the device (GPU)

// Structure to represent a 2x3 affine transformation matrix
struct AffineMatrix {
    float value[6]; // [m00, m01, m02, m10, m11, m12]
};

// CUDA kernel to perform affine warp on the image
__global__ void warpaffine_kernel(
    uint8_t* src,           // Source image on device
    int src_line_size,      // Number of bytes per source image row
    int src_width,          // Source image width
    int src_height,         // Source image height
    float* dst,             // Destination image on device (output)
    int dst_width,          // Destination image width
    int dst_height,         // Destination image height
    uint8_t const_value_st, // Constant value for out-of-bound pixels
    AffineMatrix d2s,       // Affine transformation matrix (destination to source)
    int edge                // Total number of pixels to process
) {
    // Calculate the global position of the thread
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= edge) return; // Exit if position exceeds total pixels

    // Extract affine matrix elements
    float m_x1 = d2s.value[0];
    float m_y1 = d2s.value[1];
    float m_z1 = d2s.value[2];
    float m_x2 = d2s.value[3];
    float m_y2 = d2s.value[4];
    float m_z2 = d2s.value[5];

    // Calculate destination pixel coordinates
    int dx = position % dst_width;
    int dy = position / dst_width;

    // Apply affine transformation to get source coordinates
    float src_x = m_x1 * dx + m_y1 * dy + m_z1 + 0.5f;
    float src_y = m_x2 * dx + m_y2 * dy + m_z2 + 0.5f;

    float c0, c1, c2; // Color channels (B, G, R)

    // Check if the source coordinates are out of bounds
    if (src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height) {
        // Assign constant value if out of range
        c0 = const_value_st;
        c1 = const_value_st;
        c2 = const_value_st;
    }
    else {
        // Perform bilinear interpolation

        // Get the integer parts of the source coordinates
        int y_low = floorf(src_y);
        int x_low = floorf(src_x);
        int y_high = y_low + 1;
        int x_high = x_low + 1;

        // Initialize constant values for out-of-bound pixels
        uint8_t const_value[] = { const_value_st, const_value_st, const_value_st };

        // Calculate the fractional parts
        float ly = src_y - y_low;
        float lx = src_x - x_low;
        float hy = 1 - ly;
        float hx = 1 - lx;

        // Compute the weights for the four surrounding pixels
        float w1 = hy * hx; // Top-left
        float w2 = hy * lx; // Top-right
        float w3 = ly * hx; // Bottom-left
        float w4 = ly * lx; // Bottom-right

        // Initialize pointers to the four surrounding pixels
        uint8_t* v1 = const_value;
        uint8_t* v2 = const_value;
        uint8_t* v3 = const_value;
        uint8_t* v4 = const_value;

        // Top-left pixel
        if (y_low >= 0) {
            if (x_low >= 0)
                v1 = src + y_low * src_line_size + x_low * 3;
            // Top-right pixel
            if (x_high < src_width)
                v2 = src + y_low * src_line_size + x_high * 3;
        }

        // Bottom-left and Bottom-right pixels
        if (y_high < src_height) {
            if (x_low >= 0)
                v3 = src + y_high * src_line_size + x_low * 3;
            if (x_high < src_width)
                v4 = src + y_high * src_line_size + x_high * 3;
        }

        // Perform bilinear interpolation for each color channel
        c0 = w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0]; // Blue
        c1 = w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1]; // Green
        c2 = w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2]; // Red
    }

    // Convert from BGR to RGB by swapping channels
    float t = c2;
    c2 = c0;
    c0 = t;

    // Normalize the color values to [0, 1]
    c0 = c0 / 255.0f;
    c1 = c1 / 255.0f;
    c2 = c2 / 255.0f;

    // Rearrange the output format from interleaved RGB to separate channels
    int area = dst_width * dst_height;
    float* pdst_c0 = dst + dy * dst_width + dx;        // Red channel
    float* pdst_c1 = pdst_c0 + area;                   // Green channel
    float* pdst_c2 = pdst_c1 + area;                   // Blue channel

    // Assign the normalized color values to the destination buffers
    *pdst_c0 = c0;
    *pdst_c1 = c1;
    *pdst_c2 = c2;
}

// Host function to perform CUDA-based preprocessing
void cuda_preprocess(
    uint8_t* src,        // Source image data on host
    int src_width,       // Source image width
    int src_height,      // Source image height
    float* dst,          // Destination buffer on device
    int dst_width,       // Destination image width
    int dst_height,      // Destination image height
    cudaStream_t stream  // CUDA stream for asynchronous execution
) {
    // Calculate the size of the image in bytes (3 channels: BGR)
    int img_size = src_width * src_height * 3;

    // Copy source image data to pinned host memory for faster transfer
    memcpy(img_buffer_host, src, img_size);

    // Asynchronously copy image data from host to device memory
    CUDA_CHECK(cudaMemcpyAsync(
        img_buffer_device,
        img_buffer_host,
        img_size,
        cudaMemcpyHostToDevice,
        stream
    ));

    // Define affine transformation matrices
    AffineMatrix s2d, d2s; // Source to destination and vice versa

    // Calculate the scaling factor to maintain aspect ratio
    float scale = std::min(
        dst_height / (float)src_height,
        dst_width / (float)src_width
    );

    // Initialize source-to-destination affine matrix (s2d)
    s2d.value[0] = scale;                  // m00
    s2d.value[1] = 0;                      // m01
    s2d.value[2] = -scale * src_width * 0.5f + dst_width * 0.5f; // m02
    s2d.value[3] = 0;                      // m10
    s2d.value[4] = scale;                  // m11
    s2d.value[5] = -scale * src_height * 0.5f + dst_height * 0.5f; // m12

    // Create OpenCV matrices for affine transformation
    cv::Mat m2x3_s2d(2, 3, CV_32F, s2d.value);
    cv::Mat m2x3_d2s(2, 3, CV_32F, d2s.value);

    // Invert the source-to-destination matrix to get destination-to-source
    cv::invertAffineTransform(m2x3_s2d, m2x3_d2s);

    // Copy the inverted matrix back to d2s
    memcpy(d2s.value, m2x3_d2s.ptr<float>(0), sizeof(d2s.value));

    // Calculate the total number of pixels to process
    int jobs = dst_height * dst_width;

    // Define the number of threads per block
    int threads = 256;

    // Calculate the number of blocks needed
    int blocks = ceil(jobs / (float)threads);

    // Launch the warp affine kernel
    warpaffine_kernel << <blocks, threads, 0, stream >> > (
        img_buffer_device,           // Source image on device
        src_width * 3,               // Source line size (bytes per row)
        src_width,                   // Source width
        src_height,                  // Source height
        dst,                         // Destination buffer on device
        dst_width,                   // Destination width
        dst_height,                  // Destination height
        128,                         // Constant value for out-of-bounds (gray)
        d2s,                         // Destination to source affine matrix
        jobs                         // Total number of pixels
        );

    // Optionally, you might want to check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
}

// Initialize CUDA preprocessing by allocating memory
void cuda_preprocess_init(int max_image_size) {
    // Allocate pinned (page-locked) memory on the host for faster transfers
    CUDA_CHECK(cudaMallocHost((void**)&img_buffer_host, max_image_size * 3));

    // Allocate memory on the device (GPU) for the image
    CUDA_CHECK(cudaMalloc((void**)&img_buffer_device, max_image_size * 3));
}

// Clean up and free allocated memory
void cuda_preprocess_destroy() {
    // Free device memory
    CUDA_CHECK(cudaFree(img_buffer_device));

    // Free pinned host memory
    CUDA_CHECK(cudaFreeHost(img_buffer_host));
}
