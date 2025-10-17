#include <glog/logging.h>

#include <boost/python.hpp>
#include <opencv4/opencv2/opencv.hpp>

#include "dynosam_nn/PyObjectDetector.hpp"

int main(int argc, char* argv[]) {
  // Initialize the Python interpreter manually
  Py_Initialize();
  {
    // dyno::PyObjectDetectorWrapper engine;
    // FLAGS_logtostderr = 1;
    // FLAGS_colorlogtostderr = 1;
    // FLAGS_log_prefix = 1;

    // cv::Mat mat = cv::Mat::zeros(640, 480, CV_8UC1);
    // auto r = engine.process(mat);
    // LOG(INFO) << r;
    // engine.process(r);
    // Python-dependent objects go out of scope here and thus are destroyed.
  }

  // Finalize the Python interpreter.
  Py_FinalizeEx();
}
