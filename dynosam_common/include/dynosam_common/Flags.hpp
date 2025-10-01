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

#pragma once

#include <gflags/gflags.h>

/**
 * @brief Declaration of common gflags that are DEFINED in Types.cc
 *
 */

// common glags used in multiple modules
DECLARE_bool(init_object_pose_from_gt);
DECLARE_bool(save_frontend_json);
DECLARE_bool(frontend_from_file);
DECLARE_bool(use_byte_tracker);
DECLARE_bool(refine_with_optical_flow);

// for now?
DECLARE_bool(use_vo_factor);
DECLARE_bool(use_identity_rot_L_for_init);
DECLARE_bool(corrupt_L_for_init);
DECLARE_double(corrupt_L_for_init_sigma);
DECLARE_bool(init_LL_with_identity);
DECLARE_bool(init_H_with_identity);
