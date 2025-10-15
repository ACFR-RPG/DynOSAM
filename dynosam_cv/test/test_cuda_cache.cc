#include <glog/logging.h>
#include <gtest/gtest.h>

// using namespace dyno;

#include <opencv2/core/cuda.hpp>

void printRefcount(const cv::cuda::GpuMat& gpu_mat,
                   const std::string& prefix = "") {
  if (gpu_mat.refcount == nullptr)
    LOG(INFO) << prefix << " ref count is nullptr";
  else
    LOG(INFO) << prefix << " ref count is " << *gpu_mat.refcount;
}

struct Buffer;

struct GpuContext {
  Buffer acquire(int rows, int cols, int type);

  std::unordered_map<int, Buffer> all_entries;
  std::vector<int> active_entries;
};

struct Buffer {
  struct Internal {
    Internal(Buffer* b) : buffer(b) {}
    ~Internal() {
      LOG(INFO) << "Destricting internal with ref count " << buffer->refCount();
    }

    Buffer* buffer;
  };
  // must be before gpu_mat so its destructor is called after!
  GpuContext* context_;
  Internal internal_;
  cv::cuda::GpuMat gpu_mat;

  Buffer(GpuContext* context) : context_(context), internal_(this), gpu_mat() {}
  Buffer(GpuContext* context, int rows, int cols, int type)
      : context_(context), internal_(this), gpu_mat(rows, cols, type) {}
  ~Buffer() { LOG(INFO) << "Destricting buffer with ref coutn " << refCount(); }

  int refCount() const {
    if (gpu_mat.refcount == nullptr)
      return 0;
    else
      return *gpu_mat.refcount;
  }
};

Buffer GpuContext::acquire(int rows, int cols, int type) {}

// TEST(GpuCache, testBasicBuffer) {
//     Buffer buf1;
//     LOG(INFO) << "buf1 " << buf1.refCount();

//     buf1.gpu_mat.create(480, 640, CV_8UC1);
//     LOG(INFO) << "buf1 after create " << buf1.refCount();

//     Buffer buf2 = buf1;
//     LOG(INFO) << "buf1 after assign " << buf1.refCount();
//     LOG(INFO) << "buf2 after assign " << buf2.refCount();

// }

TEST(GpuCache, testBasic) {
  cv::cuda::GpuMat gpu_mat;
  // ref count is nullptr
  printRefcount(gpu_mat);

  gpu_mat.create(480, 640, CV_8UC1);
  // ref count is 1
  printRefcount(gpu_mat);

  cv::cuda::GpuMat gpu_mat_2(480, 640, CV_8UC1);
  printRefcount(gpu_mat_2, "gpu mat 2");

  {
    // ref count is 2
    cv::cuda::GpuMat gpu_mat_1 = gpu_mat;
    printRefcount(gpu_mat, "gpu mat");
    printRefcount(gpu_mat_1, "gpu mat 1");
    //  gpu_mat_1.release();
    gpu_mat_1 = gpu_mat_2;
    printRefcount(gpu_mat_2, "gpu mat 2 after assignment");
  }

  // gpu_mat.release();
  // ref count is 1
  printRefcount(gpu_mat, "gpu mat post release");
  printRefcount(gpu_mat_2, "gpu mat  2 post release");
  // ref count nullptr for this mat
  // printRefcount(gpu_mat_1, "gpu mat 1 post release");
}
