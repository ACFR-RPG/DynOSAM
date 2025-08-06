/*
 *   Copyright (c) 2025 ACFR-RPG, University of Sydney, Jesse Morris
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

#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/backend/BackendInputPacket.hpp"
#include "dynosam/backend/BackendModule.hpp"

namespace dyno {

using RGBDBackendModuleTraits =
    BackendModuleTraits<VisionImuPacket, CameraMeasurement>;

enum RGBDFormulationType : int {

  WCME = 0,             // world-centric motion estimator
  WCPE = 1,             // world-centric pose estimator
  HYBRID = 2,           // full-hybrid
  PARALLEL_HYBRID = 3,  // associated to its own special class
  // the following are test formulations that were not specifcially part of a
  // paper but were used for (internal) development/research. they may not work
  // as intended and are included for posterity
  TESTING_HYBRID_SD = 4,  // (SD) structureless-decoupled
  TESTING_HYBRID_D = 5,   // (D) decoupled
  TESTING_HYBRID_S = 6,   // (S) structureless
  TESTING_HYBRID_SMF = 7  // (SFM) smart motion factor
};

}  // namespace dyno
