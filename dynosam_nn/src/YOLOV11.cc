#include "dynosam_nn/YOLOV11.hpp"  // Header file for YOLOv11 class
// #include "logging.h"             // Logging utilities
#include "dynosam_nn/cuda_utils.hpp"  // CUDA utility functions
// #include "macros.h"              // Common macros
// #include "preprocess.h"          // Preprocessing functions
#include <NvOnnxParser.h>  // NVIDIA ONNX parser for TensorRT
// #include "common.h"              // Common definitions and utilities
#include <glog/logging.h>

#include <fstream>   // File stream operations
#include <iostream>  // Input/output stream operations

// Initialize a static logger instance
// static Logger logger;

// Define whether to use FP16 precision
#define isFP16 true

// Define whether to perform model warmup
#define warmup true

// Constructor for the YOLOv11 class
YOLOv11::YOLOv11(string model_path, nvinfer1::ILogger& logger) {
  LOG(INFO) << "Model path " << model_path;
  // Check if the model path does not contain ".onnx"
  if (model_path.find(".onnx") == std::string::npos) {
    // Initialize the engine from a serialized engine file
    LOG(INFO) << "INIT";
    init(model_path, logger);
  } else {
    LOG(INFO) << "BUILD";
    // Build the engine from an ONNX model
    build(model_path, logger);
    // Save the built engine to a file
    saveEngine(model_path);
  }

  // Handle input dimensions based on TensorRT version
#if NV_TENSORRT_MAJOR < 10
  // For TensorRT versions less than 10, get binding dimensions directly
  auto input_dims = engine->getBindingDimensions(0);
  input_h = input_dims.d[2];
  input_w = input_dims.d[3];
#else
  // For TensorRT versions 10 and above, use getTensorShape
  auto input_dims = engine->getTensorShape(engine->getIOTensorName(0));
  input_h = input_dims.d[2];
  input_w = input_dims.d[3];
#endif
}

// Initialize the engine from a serialized engine file
void YOLOv11::init(std::string engine_path, nvinfer1::ILogger& logger) {
  // Open the engine file in binary mode
  ifstream engineStream(engine_path, ios::binary);
  // Move to the end to determine file size
  engineStream.seekg(0, ios::end);
  const size_t modelSize = engineStream.tellg();
  // Move back to the beginning of the file
  engineStream.seekg(0, ios::beg);
  // Allocate memory to read the engine data
  unique_ptr<char[]> engineData(new char[modelSize]);
  // Read the engine data into memory
  engineStream.read(engineData.get(), modelSize);
  engineStream.close();

  // Create a TensorRT runtime instance
  runtime = createInferRuntime(logger);
  // Deserialize the CUDA engine from the engine data
  engine = runtime->deserializeCudaEngine(engineData.get(), modelSize);
  // Create an execution context for the engine
  context = engine->createExecutionContext();
  // Calculate the number of classes based on detection attributes
  num_classes = detection_attribute_size - 4;

  // Allocate CPU memory for output buffer
  cpu_output_buffer = new float[detection_attribute_size * num_detections];
  // Allocate GPU memory for input buffer (assuming 3 channels: RGB)
  CUDA_CHECK(
      cudaMalloc(&gpu_buffers[0], 3 * input_w * input_h * sizeof(float)));
  // Allocate GPU memory for output buffer
  CUDA_CHECK(cudaMalloc(&gpu_buffers[1], detection_attribute_size *
                                             num_detections * sizeof(float)));

  auto num_tensors = engine->getNbIOTensors();
  for (int i = 0; i < num_tensors; ++i) {
    std::string tname(engine->getIOTensorName(i));
    LOG(INFO) << "Tensor Name " << tname;
  }

  // Initialize CUDA preprocessing with maximum image size
  cuda_preprocess_init(MAX_IMAGE_SIZE);

  // Retrieve input dimensions from the engine
#if NV_TENSORRT_MAJOR < 10
  // For TensorRT versions less than 10, use getBindingDimensions
  input_h = engine->getBindingDimensions(0).d[2];
  input_w = engine->getBindingDimensions(0).d[3];
  detection_attribute_size = engine->getBindingDimensions(1).d[1];
  num_detections = engine->getBindingDimensions(1).d[2];
#else
  // For TensorRT versions 10 and above, use getTensorShape with tensor names
  auto input_dims = engine->getTensorShape(engine->getIOTensorName(0));
  input_h = input_dims.d[2];
  input_w = input_dims.d[3];

  auto output_dims = engine->getTensorShape(engine->getIOTensorName(1));
  detection_attribute_size = output_dims.d[1];
  num_detections = output_dims.d[2];

  // Set tensor addresses for TensorRT 10+
  context->setInputTensorAddress(engine->getIOTensorName(0), gpu_buffers[0]);
  context->setTensorAddress(engine->getIOTensorName(1), gpu_buffers[1]);
#endif

  // context>setInputTensorAddress(input_info.name.c_str(),
  //                               input_device_ptr_.device_pointer.get());

  // Create a CUDA stream for asynchronous operations
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Perform model warmup if enabled
  if (warmup) {
    for (int i = 0; i < 10; i++) {
      this->infer();  // Run inference to warm up the model
    }
    printf("model warmup 10 times\n");
  }
}

// Destructor for the YOLOv11 class
YOLOv11::~YOLOv11() {
  // Synchronize and destroy the CUDA stream
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaStreamDestroy(stream));
  // Free allocated GPU buffers
  for (int i = 0; i < 2; i++) CUDA_CHECK(cudaFree(gpu_buffers[i]));
  // Free CPU output buffer
  delete[] cpu_output_buffer;

  // Destroy CUDA preprocessing resources
  cuda_preprocess_destroy();
  // Delete TensorRT context, engine, and runtime
  delete context;
  delete engine;
  delete runtime;
}

// Preprocess the input image and transfer it to the GPU buffer
void YOLOv11::preprocess(Mat& image) {
  // Perform CUDA-based preprocessing
  cuda_preprocess(image.ptr(), image.cols, image.rows, gpu_buffers[0], input_w,
                  input_h, stream);
  // Synchronize the CUDA stream to ensure preprocessing is complete
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

// Perform inference using the TensorRT execution context
void YOLOv11::infer() {
#if NV_TENSORRT_MAJOR < 10
  // For TensorRT versions less than 10, use enqueueV2 with GPU buffers
  context->enqueueV2((void**)gpu_buffers, stream, nullptr);
#else
  // For TensorRT versions 10 and above, use enqueueV3 with the CUDA stream
  this->context->enqueueV3(this->stream);
#endif
}

// Postprocess the inference output to extract detections
void YOLOv11::postprocess(vector<Detection>& output) {
  // Asynchronously copy output from GPU to CPU
  CUDA_CHECK(
      cudaMemcpyAsync(cpu_output_buffer, gpu_buffers[1],
                      num_detections * detection_attribute_size * sizeof(float),
                      cudaMemcpyDeviceToHost, stream));
  // Synchronize the CUDA stream to ensure copy is complete
  CUDA_CHECK(cudaStreamSynchronize(stream));

  vector<Rect> boxes;         // Bounding boxes
  vector<int> class_ids;      // Class IDs
  vector<float> confidences;  // Confidence scores

  // Create a matrix view of the detection output
  const Mat det_output(detection_attribute_size, num_detections, CV_32F,
                       cpu_output_buffer);

  // Iterate over each detection
  for (int i = 0; i < det_output.cols; ++i) {
    // Extract class scores for the current detection
    const Mat classes_scores = det_output.col(i).rowRange(4, 4 + num_classes);
    Point class_id_point;
    double score;
    // Find the class with the maximum score
    minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);

    // Check if the confidence score exceeds the threshold
    if (score > conf_threshold) {
      // Extract bounding box coordinates
      const float cx = det_output.at<float>(0, i);
      const float cy = det_output.at<float>(1, i);
      const float ow = det_output.at<float>(2, i);
      const float oh = det_output.at<float>(3, i);
      Rect box;
      // Calculate top-left corner of the bounding box
      box.x = static_cast<int>((cx - 0.5 * ow));
      box.y = static_cast<int>((cy - 0.5 * oh));
      // Set width and height of the bounding box
      box.width = static_cast<int>(ow);
      box.height = static_cast<int>(oh);

      // Store the bounding box, class ID, and confidence
      boxes.push_back(box);
      class_ids.push_back(class_id_point.y);
      confidences.push_back(score);
    }
  }

  vector<int> nms_result;  // Indices after Non-Maximum Suppression (NMS)
  // Apply NMS to remove overlapping boxes
  dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, nms_result);

  // Iterate over NMS results and populate the output detections
  for (int i = 0; i < nms_result.size(); i++) {
    Detection result;
    int idx = nms_result[i];
    result.class_id = class_ids[idx];
    result.conf = confidences[idx];
    result.bbox = boxes[idx];
    output.push_back(result);
  }
}

// Build the TensorRT engine from an ONNX model
void YOLOv11::build(std::string onnxPath, nvinfer1::ILogger& logger) {
  // Create a TensorRT builder
  auto builder = createInferBuilder(logger);
#if NV_TENSORRT_MAJOR < 10
  // For TensorRT versions less than 10, use explicit batch flag
  const auto explicitBatch =
      1U << static_cast<uint32_t>(
          NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
#else
  // For TensorRT 10+, explicit batch is default, use 0 or kSTRONGLY_TYPED
  INetworkDefinition* network = builder->createNetworkV2(0);
#endif
  // Create builder configuration
  IBuilderConfig* config = builder->createBuilderConfig();
  // Enable FP16 precision if specified
  if (isFP16) {
    config->setFlag(BuilderFlag::kFP16);
  }
  // Create an ONNX parser
  nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
  // Parse the ONNX model file
  bool parsed = parser->parseFromFile(
      onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO));
  // Build the serialized network plan
  IHostMemory* plan{builder->buildSerializedNetwork(*network, *config)};

  // Create a TensorRT runtime
  runtime = createInferRuntime(logger);

  // Deserialize the CUDA engine from the serialized plan
  engine = runtime->deserializeCudaEngine(plan->data(), plan->size());

  // Create an execution context for the engine
  context = engine->createExecutionContext();

  // Clean up allocated resources
  delete network;
  delete config;
  delete parser;
  delete plan;
}

// Save the serialized TensorRT engine to a file
bool YOLOv11::saveEngine(const std::string& onnxpath) {
  // Generate the engine file path by replacing the extension with ".engine"
  std::string engine_path;
  size_t dotIndex = onnxpath.find_last_of(".");
  if (dotIndex != std::string::npos) {
    engine_path = onnxpath.substr(0, dotIndex) + ".engine";
  } else {
    return false;  // Return false if no extension is found
  }

  // Check if the engine is valid
  if (engine) {
    // Serialize the engine
    nvinfer1::IHostMemory* data = engine->serialize();
    std::ofstream file;
    // Open the engine file in binary write mode
    file.open(engine_path, std::ios::binary | std::ios::out);
    if (!file.is_open()) {
      std::cout << "Create engine file " << engine_path << " failed"
                << std::endl;
      return false;
    }
    // Write the serialized engine data to the file
    file.write((const char*)data->data(), data->size());
    file.close();

    // Free the serialized data memory
    delete data;
  }
  return true;
}

// Draw bounding boxes and labels on the image based on detections
void YOLOv11::draw(Mat& image, const vector<Detection>& output) {
  // Calculate the scaling ratios between input and original image dimensions
  const float ratio_h = input_h / (float)image.rows;
  const float ratio_w = input_w / (float)image.cols;

  // Iterate over each detection
  for (int i = 0; i < output.size(); i++) {
    auto detection = output[i];
    auto box = detection.bbox;
    auto class_id = detection.class_id;
    auto conf = detection.conf;
    // Assign a color based on the class ID
    cv::Scalar color = cv::Scalar(COLORS[class_id][0], COLORS[class_id][1],
                                  COLORS[class_id][2]);

    // Adjust bounding box coordinates based on aspect ratio
    if (ratio_h > ratio_w) {
      box.x = box.x / ratio_w;
      box.y = (box.y - (input_h - ratio_w * image.rows) / 2) / ratio_w;
      box.width = box.width / ratio_w;
      box.height = box.height / ratio_w;
    } else {
      box.x = (box.x - (input_w - ratio_h * image.cols) / 2) / ratio_h;
      box.y = box.y / ratio_h;
      box.width = box.width / ratio_h;
      box.height = box.height / ratio_h;
    }

    // Draw the bounding box on the image
    rectangle(image, Point(box.x, box.y),
              Point(box.x + box.width, box.y + box.height), color, 3);

    // Prepare the label text with class name and confidence
    string class_string =
        CLASS_NAMES[class_id] + ' ' + to_string(conf).substr(0, 4);
    // Calculate the size of the text for background rectangle
    Size text_size = getTextSize(class_string, FONT_HERSHEY_DUPLEX, 1, 2, 0);
    // Define the background rectangle for the text
    Rect text_rect(box.x, box.y - 40, text_size.width + 10,
                   text_size.height + 20);
    // Draw the background rectangle
    rectangle(image, text_rect, color, FILLED);
    // Put the text label on the image
    putText(image, class_string, Point(box.x + 5, box.y - 10),
            FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0), 2, 0);
  }
}
