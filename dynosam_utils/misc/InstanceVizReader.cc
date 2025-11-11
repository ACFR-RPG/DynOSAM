/*
 *   Copyright (c) 2023 ACFR-RPG, University of Sydney, Jesse Morris
 (jesse.morris@sydney.edu.au)
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a
 copy
 *   of this software and associated documentation files (the "Software"), to
 deal
 *   in the Software without restriction, including without limitation the
 rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:

 *   The above copyright notice and this permission notice shall be included in
 all
 *   copies or substantial portions of the Software.

 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE
 *   SOFTWARE.
 */

#include <glog/logging.h>

#include <dynosam/common/Types.hpp>
#include <fstream>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv4/opencv2/core.hpp>
#include <ostream>
#include <png++/png.hpp>
#include <set>

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;
  FLAGS_colorlogtostderr = 1;
  FLAGS_log_prefix = 1;

  const std::string inst_path =
      "/root/data/virtual_kitti/vkitti_2.0.3_instanceSegmentation/Scene01/"
      "clone/frames/instanceSegmentation/Camera_0/instancegt_00001.png";

  png::image<png::index_pixel> image(inst_path);
  // png::image< png::rgb_pixel > image(inst_path);

  png::palette palette = image.get_palette();
  // int num_colours = static_cast<int>((double)palette.size()/3.0);
  LOG(INFO) << palette.size();

  cv::Size size(image.get_width(), image.get_height());
  cv::Mat mat(size, CV_8UC3);

  std::set<int> object_ids;

  for (size_t y = 0; y < image.get_height(); ++y) {
    for (size_t x = 0; x < image.get_width(); ++x) {
      const auto& pixel = image.get_pixel(x, y);
      png::byte byte = pixel;

      // LOG(INFO) << (int)byte;
      object_ids.insert(static_cast<int>(byte));

      const auto& colour_p = palette.at((int)byte);
      LOG(INFO) << (int)colour_p.red << " " << (int)colour_p.green << " "
                << (int)colour_p.blue;
      cv::Vec3i color(colour_p.red, colour_p.green, colour_p.blue);
      mat.at<cv::Vec3i>(y, x) = color;
      // image[y][x] = png::rgb_pixel(x, y, x + y);
      // // non-checking equivalent of image.set_pixel(x, y, ...);
    }
  }

  LOG(INFO) << dyno::container_to_string(object_ids);
  cv::imshow("Colour", mat);
  cv::waitKey(0);
}
