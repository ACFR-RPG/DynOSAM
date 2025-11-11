/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris
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

#include "dynosam/dataprovider/DatasetProvider.hpp"

#include <glog/logging.h>

namespace dyno {

// PlaybackGui::PlaybackGui()
//     :   pause_button_rect_(20, 20, 100, 50),
//         resume_button_rect_(140, 20, 100, 50),
//         frame_(200, 300, CV_8UC3),
//         window_("Screen")
//     {
//         frame_.setTo(cv::Scalar(255, 255, 255)); // White background
//         window_.registerMouseCallback(onMouseCallback,
//                                                 (void*)this );
//         window_.registerKeyboardCallback([](const cv::viz::KeyboardEvent&
//         event,
//                                             void* t) { LOG(INFO) <<
//                                             event.action; },
//                                                (void*)this );

//         // cv::setMouseCallback( "image", onMouseCallback, (void*)this );
//     }

// void PlaybackGui::draw() {
//     // drawButtons();
//     // cv::Rect rect = cv::Rect(0, 0, frame_.size().width,
//     frame_.size().height);
//     // window_.showWidget("Screen", cv::viz::WImageOverlay(frame_, rect));
//     LOG(INFO) << "Spinning";
//     window_.spinOnce(1, true);
//     // cv::imshow("Dialog Box", frame_);
//     // cv::waitKey(0);
// }

// void PlaybackGui::onMouseCallback(const cv::viz::MouseEvent & event, void*
// v_gui) {
//     PlaybackGui* gui = (PlaybackGui*)v_gui;
//     CHECK_NOTNULL(gui);

//     LOG(INFO) << "Click";

//     // if (event == cv::EVENT_LBUTTONDOWN) {
//     //     if (gui->pause_button_rect_.contains(cv::Point(x, y))) {
//     //         LOG(INFO) << "PAUSED!";
//     //         // paused = true;
//     //     } else if (gui->resume_button_rect_.contains(cv::Point(x, y))) {
//     //         // paused = false;
//     //         LOG(INFO) << "RESUME!";
//     //     }
//     // }

// }

// void PlaybackGui::drawButtons() {
//      // Draw pause button
//     cv::rectangle(frame_, pause_button_rect_, cv::Scalar(0, 255, 0), -1);
//     cv::putText(frame_, "Pause", cv::Point(pause_button_rect_.x + 10,
//     pause_button_rect_.y + 30),
//                 cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 2);

//     // Draw resume button
//     cv::rectangle(frame_, resume_button_rect_, cv::Scalar(0, 0, 255), -1);
//     cv::putText(frame_, "Resume", cv::Point(resume_button_rect_.x + 10,
//     resume_button_rect_.y + 30),
//                 cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 2);
// }

}  // namespace dyno
