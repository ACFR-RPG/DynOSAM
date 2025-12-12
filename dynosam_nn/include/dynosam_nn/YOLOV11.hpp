/**
 * @file YOLOv11.h
 * @brief Header file for the YOLOv11 object detection model using TensorRT and
 * OpenCV.
 *
 * This class encapsulates the preprocessing, inference, and postprocessing
 * steps required to perform object detection using a YOLOv11 model with
 * TensorRT.
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <opencv2/opencv.hpp>

#include "NvInfer.h"

#ifdef API_EXPORTS
#if defined(_MSC_VER)
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif
#else

#if defined(_MSC_VER)
#define API __declspec(dllimport)
#else
#define API
#endif
#endif  // API_EXPORTS

#if NV_TENSORRT_MAJOR >= 8
#define TRT_NOEXCEPT noexcept
#define TRT_CONST_ENQUEUE const
#else
#define TRT_NOEXCEPT
#define TRT_CONST_ENQUEUE
#endif

/**
 * @brief Initialize CUDA resources for image preprocessing.
 *
 * Allocates resources and sets up the necessary environment for performing
 * image preprocessing on the GPU. This function should be called once before
 * using any preprocessing functions.
 *
 * @param max_image_size The maximum image size (in pixels) that will be
 * processed.
 */
void cuda_preprocess_init(int max_image_size);

/**
 * @brief Clean up and release CUDA resources.
 *
 * Frees any memory and resources allocated during initialization. This function
 * should be called when the preprocessing operations are no longer needed.
 */
void cuda_preprocess_destroy();

/**
 * @brief Preprocess an image using CUDA.
 *
 * This function resizes and converts the input image data (from uint8 to float)
 * using CUDA for faster processing. The result is stored in a destination
 * buffer, ready for inference.
 *
 * @param src Pointer to the source image data in uint8 format.
 * @param src_width The width of the source image.
 * @param src_height The height of the source image.
 * @param dst Pointer to the destination buffer to store the preprocessed image
 * in float format.
 * @param dst_width The desired width of the output image.
 * @param dst_height The desired height of the output image.
 * @param stream The CUDA stream to execute the preprocessing operation
 * asynchronously.
 */
void cuda_preprocess(uint8_t* src, int src_width, int src_height, float* dst,
                     int dst_width, int dst_height, cudaStream_t stream);

using namespace nvinfer1;
using namespace std;
using namespace cv;

const std::vector<std::string> CLASS_NAMES = {
    "person",        "bicycle",      "car",
    "motorcycle",    "airplane",     "bus",
    "train",         "truck",        "boat",
    "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench",        "bird",
    "cat",           "dog",          "horse",
    "sheep",         "cow",          "elephant",
    "bear",          "zebra",        "giraffe",
    "backpack",      "umbrella",     "handbag",
    "tie",           "suitcase",     "frisbee",
    "skis",          "snowboard",    "sports ball",
    "kite",          "baseball bat", "baseball glove",
    "skateboard",    "surfboard",    "tennis racket",
    "bottle",        "wine glass",   "cup",
    "fork",          "knife",        "spoon",
    "bowl",          "banana",       "apple",
    "sandwich",      "orange",       "broccoli",
    "carrot",        "hot dog",      "pizza",
    "donut",         "cake",         "chair",
    "couch",         "potted plant", "bed",
    "dining table",  "toilet",       "tv",
    "laptop",        "mouse",        "remote",
    "keyboard",      "cell phone",   "microwave",
    "oven",          "toaster",      "sink",
    "refrigerator",  "book",         "clock",
    "vase",          "scissors",     "teddy bear",
    "hair drier",    "toothbrush"};

const std::vector<std::vector<unsigned int>> COLORS = {
    {0, 114, 189},   {217, 83, 25},   {237, 177, 32},  {126, 47, 142},
    {119, 172, 48},  {77, 190, 238},  {162, 20, 47},   {76, 76, 76},
    {153, 153, 153}, {255, 0, 0},     {255, 128, 0},   {191, 191, 0},
    {0, 255, 0},     {0, 0, 255},     {170, 0, 255},   {85, 85, 0},
    {85, 170, 0},    {85, 255, 0},    {170, 85, 0},    {170, 170, 0},
    {170, 255, 0},   {255, 85, 0},    {255, 170, 0},   {255, 255, 0},
    {0, 85, 128},    {0, 170, 128},   {0, 255, 128},   {85, 0, 128},
    {85, 85, 128},   {85, 170, 128},  {85, 255, 128},  {170, 0, 128},
    {170, 85, 128},  {170, 170, 128}, {170, 255, 128}, {255, 0, 128},
    {255, 85, 128},  {255, 170, 128}, {255, 255, 128}, {0, 85, 255},
    {0, 170, 255},   {0, 255, 255},   {85, 0, 255},    {85, 85, 255},
    {85, 170, 255},  {85, 255, 255},  {170, 0, 255},   {170, 85, 255},
    {170, 170, 255}, {170, 255, 255}, {255, 0, 255},   {255, 85, 255},
    {255, 170, 255}, {85, 0, 0},      {128, 0, 0},     {170, 0, 0},
    {212, 0, 0},     {255, 0, 0},     {0, 43, 0},      {0, 85, 0},
    {0, 128, 0},     {0, 170, 0},     {0, 212, 0},     {0, 255, 0},
    {0, 0, 43},      {0, 0, 85},      {0, 0, 128},     {0, 0, 170},
    {0, 0, 212},     {0, 0, 255},     {0, 0, 0},       {36, 36, 36},
    {73, 73, 73},    {109, 109, 109}, {146, 146, 146}, {182, 182, 182},
    {219, 219, 219}, {0, 114, 189},   {80, 183, 189},  {128, 128, 0}};

/**
 * @struct Detection
 * @brief A structure representing a detected object.
 *
 * Contains the confidence score, class ID, and bounding box for a detected
 * object.
 */
struct Detection {
  float conf;    //!< Confidence score of the detection.
  int class_id;  //!< Class ID of the detected object.
  Rect bbox;     //!< Bounding box of the detected object.
};

/**
 * @class YOLOv11
 * @brief A class for running YOLOv11 object detection using TensorRT and
 * OpenCV.
 *
 * This class handles model initialization, inference, and postprocessing to
 * detect objects in images.
 */
class YOLOv11 {
 public:
  /**
   * @brief Constructor to initialize the YOLOv11 object.
   *
   * Loads the model and initializes TensorRT objects.
   *
   * @param model_path Path to the model engine or ONNX file.
   * @param logger Reference to a TensorRT logger for error reporting.
   */
  YOLOv11(string model_path, nvinfer1::ILogger& logger);

  /**
   * @brief Destructor to clean up resources.
   *
   * Frees the allocated memory and TensorRT resources.
   */
  ~YOLOv11();

  /**
   * @brief Preprocess the input image.
   *
   * Prepares the image for inference by resizing and normalizing it.
   *
   * @param image The input image to be preprocessed.
   */
  void preprocess(Mat& image);

  /**
   * @brief Run inference on the preprocessed image.
   *
   * Executes the TensorRT engine for object detection.
   */
  void infer();

  /**
   * @brief Postprocess the output from the model.
   *
   * Filters and decodes the raw output from the TensorRT engine into detection
   * results.
   *
   * @param output A vector to store the detected objects.
   */
  void postprocess(vector<Detection>& output);

  /**
   * @brief Draw the detected objects on the image.
   *
   * Overlays bounding boxes and class labels on the image for visualization.
   *
   * @param image The input image where the detections will be drawn.
   * @param output A vector of detections to be visualized.
   */
  void draw(Mat& image, const vector<Detection>& output);

 private:
  /**
   * @brief Initialize TensorRT components from the given engine file.
   *
   * @param engine_path Path to the serialized TensorRT engine file.
   * @param logger Reference to a TensorRT logger for error reporting.
   */
  void init(std::string engine_path, nvinfer1::ILogger& logger);

  float* gpu_buffers[2];     //!< The vector of device buffers needed for engine
                             //!< execution.
  float* cpu_output_buffer;  //!< Pointer to the output buffer on the host.

  cudaStream_t stream;  //!< CUDA stream for asynchronous execution.
  IRuntime* runtime;  //!< The TensorRT runtime used to deserialize the engine.
  ICudaEngine* engine;  //!< The TensorRT engine used to run the network.
  IExecutionContext*
      context;  //!< The context for executing inference using an ICudaEngine.

  // Model parameters
  int input_w;                   //!< Width of the input image.
  int input_h;                   //!< Height of the input image.
  int num_detections;            //!< Number of detections output by the model.
  int detection_attribute_size;  //!< Size of each detection attribute.
  int num_classes = 80;  //!< Number of object classes that can be detected.
  const int MAX_IMAGE_SIZE =
      4096 * 4096;  //!< Maximum allowed input image size.
  float conf_threshold =
      0.3f;  //!< Confidence threshold for filtering detections.
  float nms_threshold = 0.4f;  //!< Non-Maximum Suppression (NMS) threshold for
                               //!< filtering overlapping boxes.

  vector<Scalar> colors;  //!< A vector of colors for drawing bounding boxes.

  /**
   * @brief Build the TensorRT engine from the ONNX model.
   *
   * @param onnxPath Path to the ONNX file.
   * @param logger Reference to a TensorRT logger for error reporting.
   */
  void build(std::string onnxPath, nvinfer1::ILogger& logger);

  /**
   * @brief Save the TensorRT engine to a file.
   *
   * @param filename Path to save the serialized engine.
   * @return True if the engine was saved successfully, false otherwise.
   */
  bool saveEngine(const std::string& filename);
};
