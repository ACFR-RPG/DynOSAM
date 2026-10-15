#include "dynosam2/frontend/FeatureBlockContainer.hpp"

#include "dynosam_common/viz/Colour.hpp"

using namespace dyno;

namespace dyno2 {

void FeatureBlockContainer::FeatureData::checkSizes() const {
  const size_t n = points.size();

  if (previous_points.size() != n || ids.size() != n || inlier.size() != n ||
      age.size() != n) {
    throw std::invalid_argument(
        "FeatureData: all feature arrays must have the same size");
  }
}

void FeatureBlockContainer::FeatureData::reserve(size_t n) {
  points.reserve(n);
  previous_points.reserve(n);
  ids.reserve(n);
  age.reserve(n);
  inlier.reserve(n);
  errors.reserve(n);
}

void FeatureBlockContainer::FeatureData::resize(size_t n) {
  points.resize(n);
  previous_points.resize(n);
  ids.resize(n);
  age.resize(n);
  inlier.resize(n);
  errors.resize(n);
}

// ---------------------------------------------------------------------
// Convenient bulk assignment
// ---------------------------------------------------------------------

void FeatureBlockContainer::BlockView::copyFrom(const FeatureData& data) {
  data.checkSizes();

  if (data.size() != size()) {
    throw std::invalid_argument(
        "FeatureSet::BlockView::copyFrom: "
        "FeatureData size does not match object size");
  }

  copyBlock(points(), data.points.data(), size());
  copyBlock(previousPoints(), data.previous_points.data(), size());
  copyBlock(ids(), data.ids.data(), size());
  copyBlock(age(), data.age.data(), size());
  copyBlock(inlier(), data.inlier.data(), size());
  copyBlock(errors(), data.errors.data(), size());
}

FeatureBlockContainer::FeatureBlockContainer(
    std::initializer_list<BlockDim> specs) {
  initialize(specs.begin(), specs.end());
}

FeatureBlockContainer::FeatureBlockContainer(
    const std::vector<BlockDim>& specs) {
  initialize(specs.begin(), specs.end());
}

size_t FeatureBlockContainer::size() const { return points.size(); }
size_t FeatureBlockContainer::objectCount() const {
  return block_layout_.size();
}

std::vector<FeatureBlockContainer::BlockView>
FeatureBlockContainer::objectViews() const {
  std::vector<BlockView> views;
  views.reserve(objectCount());

  for (const BlockLayout& block : block_layout_) {
    views.push_back(objectView(block.object_id));
  }

  return views;
}

bool FeatureBlockContainer::containsObject(ObjectId object_id) const {
  return object_lookup_.find(object_id) != object_lookup_.end();
}

FeatureBlockContainer FeatureBlockContainer::merge(
    const FeatureBlockContainer& other) const {
  // check for tracklet ids dupliactes
  // TODO: comment out for now - this makes creating empty or initalised but
  // inassigned FeatureSets invalid becuase all objectids/trackletids will
  // have the same id!
  //  std::unordered_set<int> feature_ids;
  //  feature_ids.reserve(ids.size() + other.ids.size());

  // for (const int id : ids)
  // {
  //     feature_ids.insert(id);
  // }

  // for (const int id : other.ids)
  // {
  //     if (!feature_ids.insert(id).second)
  //     {
  //         throw std::invalid_argument(
  //             "FeatureSet::merge: duplicate feature ID " +
  //             std::to_string(id));
  //     }
  // }

  // -------------------------------------------------------------------------
  // Build the resulting object layout.
  //
  // Existing objects retain their order.
  // New objects from `other` are appended in `other`'s order.
  // -------------------------------------------------------------------------

  std::vector<BlockDim> specs;
  specs.reserve(block_layout_.size() + other.block_layout_.size());

  // Existing objects.
  for (const BlockLayout& object : block_layout_) {
    const auto other_it = other.object_lookup_.find(object.object_id);

    const size_t other_size = other_it != other.object_lookup_.end()
                                  ? other.block_layout_[other_it->second].size()
                                  : 0;

    specs.push_back({object.object_id, object.size() + other_size});
  }

  // Objects which only exist in `other`.
  for (const BlockLayout& object : other.block_layout_) {
    if (object_lookup_.find(object.object_id) == object_lookup_.end()) {
      specs.push_back({object.object_id, object.size()});
    }
  }

  // -------------------------------------------------------------------------
  // Allocate the final FeatureSet exactly once.
  // -------------------------------------------------------------------------

  FeatureBlockContainer result(specs);

  // -------------------------------------------------------------------------
  // Copy the existing features into their final locations.
  // -------------------------------------------------------------------------

  for (const BlockLayout& object : block_layout_) {
    const BlockView source = objectView(object.object_id);

    BlockView destination = result.objectView(object.object_id);

    copyFeatures(destination, source);
  }

  // -------------------------------------------------------------------------
  // Append features from `other`.
  //
  // Existing objects are appended after their existing features.
  // New-only objects are copied starting at offset zero.
  // -------------------------------------------------------------------------

  for (const BlockLayout& other_object : other.block_layout_) {
    const BlockView source = other.objectView(other_object.object_id);

    BlockView destination = result.objectView(other_object.object_id);

    const auto existing_it = object_lookup_.find(other_object.object_id);

    const size_t destination_offset =
        existing_it != object_lookup_.end()
            ? block_layout_[existing_it->second].size()
            : 0;

    if (source.size() == 0) continue;

    copyFeatures(destination, source, destination_offset);
  }

  // object_ids are established by the FeatureSet constructor and therefore
  // don't need to be copied during the merge.

  result.checkInvariants();

  return result;
}

FeatureBlockContainer::BlockView FeatureBlockContainer::objectView(
    ObjectId object_id) {
  const BlockLayout& metadata = objectMetadata(object_id);
  return BlockView(this, metadata);
}

const FeatureBlockContainer::BlockView FeatureBlockContainer::objectView(
    ObjectId object_id) const {
  const BlockLayout& metadata = objectMetadata(object_id);
  return BlockView(const_cast<FeatureBlockContainer*>(this), metadata);
}

std::string FeatureBlockContainer::debugInfoString() const {
  std::stringstream ss;
  ss << "FeatureSet"
     << " | total_features=" << size() << " | objects=" << objectCount()
     << '\n';

  for (const BlockLayout& object : block_layout_) {
    ss << "  object_id=" << object.object_id << " | size=" << object.size()
       << " | range=[" << object.begin << ", " << object.end << ")" << '\n';
  }
  return ss.str();
}

void FeatureBlockContainer::checkInvariants() const {
#ifndef NDEBUG
  const size_t n = points.size();

  assert(previous_points.size() == n);
  assert(ids.size() == n);
  assert(object_ids.size() == n);
  assert(inlier.size() == n);
  assert(errors.size() == n);
  assert(age.size() == n);

  size_t expected_begin = 0;

  for (const BlockLayout& object : block_layout_) {
    assert(object.begin == expected_begin);
    assert(object.begin <= object.end);
    assert(object.end <= n);

    for (size_t i = object.begin; i < object.end; ++i) {
      assert(object_ids[i] == object.object_id);
    }

    expected_begin = object.end;
  }

  assert(expected_begin == n);
#endif
}

FeatureBlockContainer FeatureBlockContainer::reduceToInliers(
    IndexMapping* index_mapping) const {
  std::vector<BlockDim> specs;
  specs.reserve(block_layout_.size());

  std::vector<size_t> inlier_counts(block_layout_.size(), 0);

  size_t total_inliers = 0;

  for (size_t block_index = 0; block_index < block_layout_.size();
       ++block_index) {
    const BlockLayout& block = block_layout_[block_index];

    size_t count = 0;

    for (size_t i = block.begin; i < block.end; ++i) {
      count += static_cast<size_t>(inlier[i] != 0);
    }

    inlier_counts[block_index] = count;

    if (count > 0) {
      specs.push_back({block.object_id, count});
      total_inliers += count;
    }
  }

  // if no inliers no need to copy the data across
  // IndexMapping is not updated as no change to index's
  // TODO: (is this the desired behaviour?)
  if (total_inliers == this->size()) {
    return *this;
  }

  FeatureBlockContainer result(specs);

  if (index_mapping != nullptr) {
    index_mapping->clear();
  }

  size_t destination_block_index = 0;

  for (size_t source_block_index = 0; source_block_index < block_layout_.size();
       ++source_block_index) {
    const size_t count = inlier_counts[source_block_index];

    if (count == 0) {
      continue;
    }

    const BlockLayout& source_block = block_layout_[source_block_index];

    const BlockLayout& destination_block =
        result.block_layout_[destination_block_index];

    size_t destination_index = destination_block.begin;

    for (size_t source_index = source_block.begin;
         source_index < source_block.end; ++source_index) {
      if (inlier[source_index] == 0) {
        continue;
      }

      result.points[destination_index] = points[source_index];
      result.previous_points[destination_index] = previous_points[source_index];
      result.ids[destination_index] = ids[source_index];
      result.age[destination_index] = age[source_index];

      // initialize() already populated this correctly, but copying it
      // explicitly keeps this function independent of that behaviour.
      result.object_ids[destination_index] = object_ids[source_index];

      // Everything in the result is an inlier.
      result.inlier[destination_index] = 1;

      result.errors[destination_index] = errors[source_index];

      if (index_mapping != nullptr) {
        index_mapping->operator[](source_index) = destination_index;
      }

      ++destination_index;
    }

    ++destination_block_index;
  }

  result.checkInvariants();

  return result;
}

FeatureBlockContainer& FeatureBlockContainer::reduceToInliersInplace(
    IndexMapping* index_mapping) {
  FeatureBlockContainer filtered = reduceToInliers(index_mapping);
  *this = std::move(filtered);
  return *this;
}

cv::Mat FeatureBlockContainer::draw(const cv::Mat& image,
                                    bool show_intermediate) const {
  return drawBatchedFeatures(image, *this, show_intermediate);
}

void FeatureBlockContainer::copyFeatures(BlockView destination,
                                         const BlockView& source,
                                         size_t destination_offset) {
  copyBlock(destination.points() + destination_offset, source.points(),
            source.size());

  copyBlock(destination.previousPoints() + destination_offset,
            source.previousPoints(), source.size());

  copyBlock(destination.ids() + destination_offset, source.ids(),
            source.size());

  copyBlock(destination.age() + destination_offset, source.age(),
            source.size());

  copyBlock(destination.inlier() + destination_offset, source.inlier(),
            source.size());

  copyBlock(destination.errors() + destination_offset, source.errors(),
            source.size());
}

cv::Mat drawBatchedFeatures(const cv::Mat& image,
                            const FeatureBlockContainer& batchedFeatures,
                            bool show_intermediate) {
  cv::Mat canvas;

  // Ensure we are drawing on a 3-channel color image
  if (image.channels() == 1) {
    cv::cvtColor(image, canvas, cv::COLOR_GRAY2BGR);
  } else {
    canvas = image.clone();
  }

  static const cv::Scalar red(Color::red().bgra());
  static const cv::Scalar green(Color::green().bgra());
  static const cv::Scalar blue(Color::blue().bgra());

  for (size_t i = 0; i < batchedFeatures.size(); ++i) {
    const auto object_id = batchedFeatures.object_ids[i];
    const auto current_point = batchedFeatures.points[i];
    const auto inlier = batchedFeatures.inlier[i];
    const auto id = batchedFeatures.ids[i];

    const cv::Scalar object_color =
        dyno::Color::uniqueObjectId(object_id).bgra();

    const bool is_static = object_id == background_label;
    cv::Scalar color = is_static ? green : object_color;

    if (!inlier && show_intermediate) {
      // / Draw current feature location
      cv::circle(canvas, current_point, 4, red, 2);
    } else {
      // Draw optical-flow track
      if (batchedFeatures.age[i] > 1) {
        const auto previous_point = batchedFeatures.previous_points[i];
        cv::arrowedLine(canvas, previous_point, current_point, color, 1);
        cv::circle(canvas, current_point, 4, color);

        // if (is_static) {
        //   cv::putText(canvas, std::to_string(id),
        //               current_point + cv::Point2f(10, 5),
        //               cv::FONT_HERSHEY_COMPLEX, 0.35, cv::Scalar(255, 255,
        //               0), 1, cv::LINE_AA);
        // }
      }
      // show as new point
      else if (show_intermediate) {
        cv::circle(canvas, current_point, 6, blue, 1);
      }
    }
  }

  return canvas;
}

}  // namespace dyno2
