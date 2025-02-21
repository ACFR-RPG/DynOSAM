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

#include "dynosam/backend/Formulation.hpp"

namespace dyno {

// TODO: debug info
UpdateObservationResult& UpdateObservationResult::operator+=(
    const UpdateObservationResult& oth) {
  for (const auto& [key, value] : oth.objects_affected_per_frame) {
    objects_affected_per_frame[key].insert(value.begin(), value.end());
  }
  return *this;
}

void UpdateObservationResult::updateAffectedObject(FrameId frame_id,
                                                   ObjectId object_id) {
  if (!objects_affected_per_frame.exists(object_id)) {
    objects_affected_per_frame.insert2(object_id, std::set<FrameId>{});
  }
  objects_affected_per_frame[object_id].insert(frame_id);
}

const gtsam::Values& GraphUpdateResult::values() const {
  return all_new_values_;
}
const gtsam::NonlinearFactorGraph& GraphUpdateResult::factors() const {
  return all_new_factors_;
}

const gtsam::Values& GraphUpdateResult::dynamicValues() const {
  return new_dynamic_values_;
}
const gtsam::NonlinearFactorGraph& GraphUpdateResult::dynamicFactors() const {
  return new_dynamic_factors_;
}

const gtsam::Values& GraphUpdateResult::staticValues() const {
  return new_static_values_;
}
const gtsam::NonlinearFactorGraph& GraphUpdateResult::staticFactors() const {
  return new_static_factors_;
}

std::optional<std::reference_wrapper<const gtsam::Values>>
GraphUpdateResult::values(ObjectId object_id) const {
  if (collections_.exists(object_id)) {
    return collections_.at(object_id).values;
  }
  return {};
}
std::optional<std::reference_wrapper<const gtsam::NonlinearFactorGraph>>
GraphUpdateResult::factors(ObjectId object_id) const {
  if (collections_.exists(object_id)) {
    return collections_.at(object_id).factors;
  }
  return {};
}

GraphUpdateResult& GraphUpdateResult::add(const gtsam::Values& new_values,
                                          ObjectId object_id) {
  gtsam::Values* values_to_update;
  if (object_id == static_id_) {
    values_to_update = &new_static_values_;
  } else {
    values_to_update = &new_dynamic_values_;
    if (!collections_.exists(object_id)) {
      collections_.insert2(object_id, Collection{});
    }
    Collection& collection = collections_.at(object_id);
    collection.values.insert(new_values);
  }
  values_to_update->insert(new_values);
  all_new_values_.insert(new_values);
  return *this;
}
GraphUpdateResult& GraphUpdateResult::add(
    const gtsam::NonlinearFactorGraph& new_factors, ObjectId object_id) {
  gtsam::NonlinearFactorGraph* graph_to_update;
  if (object_id == static_id_) {
    graph_to_update = &new_static_factors_;
  } else {
    graph_to_update = &new_dynamic_factors_;
    if (!collections_.exists(object_id)) {
      collections_.insert2(object_id, Collection{});
    }
    Collection& collection = collections_.at(object_id);
    collection.factors += new_factors;
  }
  (*graph_to_update) += new_factors;
  all_new_factors_ += new_factors;
  return *this;
}

ObjectIds GraphUpdateResult::dynamicObjectIds() const {
  ObjectIds object_ids;
  for (const auto& [object_id, _] : collections_) {
    object_ids.push_back(object_id);
  }
  return object_ids;
}

}  // namespace dyno
