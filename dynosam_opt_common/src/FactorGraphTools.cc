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

#include "dynosam_opt_common/FactorGraphTools.hpp"

#include <gtsam/linear/LossFunctions.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam_unstable/slam/PoseToPointFactor.h>

#include <opencv4/opencv2/viz/types.hpp>

#include "dynosam_common/utils/Numerical.hpp"  //for saveMatrixAsUpperTriangular
#include "dynosam_opt_common/Symbols.hpp"
#include "gtsam/linear/GaussianConditional.h"

namespace dyno {

namespace factor_graph_tools {

gtsam::SharedNoiseModel robustifyHuber(double k,
                                       gtsam::SharedNoiseModel model) {
  return gtsam::noiseModel::Robust::Create(
      gtsam::noiseModel::mEstimator::Huber::Create(k), model);
}

void addBetweenFactor(FrameId from_frame, FrameId to_frame,
                      const gtsam::Pose3 from_pose_to,
                      gtsam::SharedNoiseModel noise_model,
                      gtsam::NonlinearFactorGraph& graph) {
  CHECK(noise_model);
  CHECK_EQ(noise_model->dim(), 6u);

  graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
      CameraPoseSymbol(from_frame), CameraPoseSymbol(to_frame), from_pose_to,
      noise_model);
}

SparsityStats computeCholeskySparsityStats(
    gtsam::GaussianFactorGraph::shared_ptr gaussian_fg,
    const std::optional<gtsam::Ordering>& ordering) {
  gtsam::Ordering order;
  if (ordering) {
    order = *ordering;
  } else {
    order = gtsam::Ordering::Colamd(*gaussian_fg);
  }

  auto [conditional, joint_factor] =
      gtsam::EliminateCholesky(*gaussian_fg, order);

  const gtsam::VerticalBlockMatrix::constBlock R = conditional->R();
  return SparsityStats(R);
}

SparsityStats computeHessianSparsityStats(
    gtsam::GaussianFactorGraph::shared_ptr gaussian_fg,
    const std::optional<gtsam::Ordering>& ordering) {
  if (ordering) {
    return SparsityStats(gaussian_fg->hessian(*ordering).first);
  }

  return SparsityStats(gaussian_fg->hessian().first);
}

SparsityStats computeJacobianSparsityStats(
    gtsam::GaussianFactorGraph::shared_ptr gaussian_fg,
    const std::optional<gtsam::Ordering>& ordering) {
  // NOTE: GaussianFactorGraph::jacobian(Ordering) SHOULD be the same as calling
  // gtsam::JacobianFactor(GaussianFactorGraph, ordering) and it is in the code
  // when this was tested but I guess this could change, leading to
  // consistencies with how the Jacobain factor is constructed in differnet
  // fg_tool functions
  if (ordering) {
    return SparsityStats(gaussian_fg->jacobian(*ordering).first);
  }

  return SparsityStats(gaussian_fg->jacobian().first);
}

std::pair<SparsityStats, cv::Mat> computeRFactor(
    gtsam::GaussianFactorGraph::shared_ptr gaussian_fg,
    const gtsam::Ordering ordering) {
  // TODO: why do we need to specify the ordering twice?
  // I think we should probaly construct the graph with native ordeing but then
  // elimate with provided ordering?
  // gtsam::JacobianFactor jacobian_factor(*gaussian_fg);

  // gtsam::GaussianConditional::shared_ptr conditional;
  // gtsam::JacobianFactor::shared_ptr joint_factor;

  // /// Not entirely sure the difference between the joint factor we get out
  // and
  // /// the inital jacobian factor we get in I believe the difference is in the
  // /// ordering which we do NOT apply when constructing the jacobian_factor.
  // I'm
  // /// not sure this is correct.
  // std::tie(conditional, joint_factor) = jacobian_factor.eliminate(ordering);
  // CHECK(conditional);

  // const gtsam::VerticalBlockMatrix::constBlock R = conditional->R();

  // // drawn image
  // size_t nnz = 0;
  // cv::Mat R_img(cv::Size(R.cols(), R.rows()), CV_8UC3,
  // cv::viz::Color::white()); for (int i = 0; i < R.rows(); ++i) {
  //   for (int j = 0; j < R.cols(); ++j) {
  //     // only draw if non zero
  //     if (std::fabs(R(i, j)) > 1e-15) {
  //       R_img.at<cv::Vec3b>(i, j) = (cv::Vec3b)cv::viz::Color::black();
  //       nnz++;
  //     }
  //   }
  // }

  // LOG(INFO) << "Number non-zero: " << nnz;
  // //only values above the diagonal so area of triangle
  // LOG(INFO) << "% non zeros " << ((double)nnz / (double)(R.cols() *
  // R.rows())/2.0) * 100.0;

  // // cv::imshow("R", R_img);
  // // cv::waitKey(0);
  // return std::make_pair(SparsityStats{}, R_img);

  // calc num non zero for j
  size_t nnz_J = 0;
  const gtsam::Matrix J = gaussian_fg->jacobian().first;
  for (int i = 0; i < J.rows(); ++i) {
    for (int j = 0; j < J.cols(); ++j) {
      // only draw if non zero
      if (std::fabs(J(i, j)) > 1e-15) {
        nnz_J++;
      }
    }
  }

  // auto [conditional, joint_factor] = gtsam::EliminateQR(*gaussian_fg,
  // gtsam::Ordering::Colamd(*gaussian_fg));
  auto [conditional, joint_factor] =
      gtsam::EliminateCholesky(*gaussian_fg, ordering);

  // gtsam::HessianFactor hessian_factor(*gaussian_fg);

  // // gtsam::GaussianConditional::shared_ptr conditional =
  // hessian_factor.eliminateCholesky(gtsam::Ordering(gaussian_fg->keys()));
  // gtsam::GaussianConditional::shared_ptr conditional =
  // hessian_factor.eliminateCholesky(ordering);
  // gtsam::JacobianFactor::shared_ptr joint_factor;

  // /// Not entirely sure the difference between the joint factor we get out
  // and
  // /// the inital jacobian factor we get in I believe the difference is in the
  // /// ordering which we do NOT apply when constructing the jacobian_factor.
  // I'm
  // /// not sure this is correct.
  // std::tie(conditional, joint_factor) = jacobian_factor.eliminate(ordering);
  // CHECK(conditional);

  const gtsam::VerticalBlockMatrix::constBlock R = conditional->R();

  // drawn image
  size_t nnz = 0;
  cv::Mat R_img(cv::Size(R.cols(), R.rows()), CV_8UC3, cv::viz::Color::white());
  for (int i = 0; i < R.rows(); ++i) {
    for (int j = 0; j < R.cols(); ++j) {
      // only draw if non zero
      if (std::fabs(R(i, j)) > 1e-15) {
        R_img.at<cv::Vec3b>(i, j) = (cv::Vec3b)cv::viz::Color::black();
        nnz++;
      }
    }
  }

  LOG(INFO) << "Number non-zero R: " << nnz;
  LOG(INFO) << "Number non-zero J: " << nnz_J;
  // lower better
  LOG(INFO) << "Fill in ratio: " << (double)nnz / (double)nnz_J;
  // only values above the diagonal so area of triangle
  LOG(INFO) << "% non zeros "
            << ((double)nnz / (double)(R.cols() * R.rows()) / 2.0) * 100.0;

  // cv::imshow("R", R_img);
  // cv::waitKey(0);
  return std::make_pair(SparsityStats{}, R_img);
}

cv::Mat drawBlockJacobians(gtsam::GaussianFactorGraph::shared_ptr gaussian_fg,
                           const gtsam::Ordering& ordering,
                           const DrawBlockJacobiansOptions& options) {
  const gtsam::JacobianFactor jacobian_factor(*gaussian_fg, ordering);
  const gtsam::Matrix J = jacobian_factor.jacobian().first;

  // each cv::mat will be the drawing of a column block (associated with one
  // variable) in the jacobian Each block will have dimensions determiend by dim
  // of the variable (determiend by the key) and number of rows of the full
  // jacobian matrix, J
  std::vector<std::pair<gtsam::Key, cv::Mat>> column_blocks;

  for (gtsam::Key key : ordering) {
    // this will have rows = J.rows(), cols = dimension of the variable
    const gtsam::VerticalBlockMatrix::constBlock Ablock =
        jacobian_factor.getA(jacobian_factor.find(key));
    const int var_dimensions = Ablock.cols();
    CHECK_EQ(Ablock.rows(), J.rows());

    // drawn image of each vertical block
    cv::Mat b_img(cv::Size(var_dimensions, Ablock.rows()), CV_8UC3,
                  cv::viz::Color::white());
    for (int i = 0; i < Ablock.rows(); ++i) {
      for (int j = 0; j < var_dimensions; ++j) {
        // only draw if non zero
        if (std::fabs(Ablock(i, j)) > 1e-15) {
          const auto colour = options.colour_selector(key);
          b_img.at<cv::Vec3b>(i, j) =
              cv::Vec3b(colour[0], colour[1], colour[2]);
        }
      }
    }

    column_blocks.push_back(std::make_pair(key, b_img));
  }

  CHECK(!options.desired_size.empty());  // cannot be empty

  const cv::Size& desired_size = options.desired_size;

  const int current_cols = J.cols();
  const int desired_cols = desired_size.width;
  const int desired_rows = desired_size.height;

  // define a ratio which which to scale each block (viz) individually
  // since we're going to add things like spacing we want to scale
  // up each block so its visible but dont want to scale up the line spacing
  // etc. the ratio ensures we get close to the final desired size of the TOTAL
  // image
  double ratio;
  // want positive ratio
  if (current_cols < desired_cols) {
    ratio = (double)desired_cols / (double)current_cols;
  } else {
    ratio = (double)current_cols / (double)desired_cols;
  }

  auto scale_and_draw_label = [&options, &ratio, &desired_rows](
                                  cv::Mat& current_block, gtsam::Key key) {
    // we all each jacobiab horizintally so only need to scale the columns
    const int scaled_cols = (ratio * (int)current_block.cols) + 10;

    // scale current block to the desired size with INTER_NEAREST - this keeps
    // the pixels looking clear and visible
    cv::resize(current_block, current_block,
               cv::Size(scaled_cols, desired_rows), 0, 0, cv::INTER_NEAREST);

    if (options.draw_label) {
      // draw text info
      cv::Mat text_box(cv::Size(current_block.cols, options.text_box_height),
                       CV_8UC3, cv::viz::Color::white());

      constexpr static double kFontScale = 0.5;
      constexpr static int kFontFace = cv::FONT_HERSHEY_DUPLEX;
      constexpr static int kThickness = 1;
      // draw text mid way in box
      cv::putText(text_box, options.label_formatter(key),
                  cv::Point(2, text_box.rows / 2), kFontFace, kFontScale,
                  cv::viz::Color::black(), kThickness);
      // add text box ontop of jacobian block
      cv::vconcat(text_box, current_block, current_block);
    }
  };  // end scale_and_drawinternal::_label

  if (column_blocks.size() == 1) {
    gtsam::Key key = column_blocks.at(0).first;
    cv::Mat current_block = column_blocks.at(0).second;
    scale_and_draw_label(current_block, key);
    return current_block;
  }

  // the final drawn image
  cv::Mat concat_column_blocks;

  for (size_t i = 1; i < column_blocks.size(); i++) {
    gtsam::Key key = column_blocks.at(i).first;
    cv::Mat current_block = column_blocks.at(i).second;
    scale_and_draw_label(current_block, key);

    if (i == 1) {
      cv::Mat first_block = column_blocks.at(0).second;
      gtsam::Key key_0 = column_blocks.at(0).first;
      scale_and_draw_label(first_block, key_0);
      const cv::Mat vert_line(cv::Size(2, first_block.rows), CV_8UC3,
                              cv::viz::Color::black());
      // concat first block with vertical loine
      cv::hconcat(first_block, vert_line, first_block);
      // concat current jacobian block with other blocks
      cv::hconcat(first_block, current_block, concat_column_blocks);
    } else {
      // concat current jacobian block with other blocks
      cv::hconcat(concat_column_blocks, current_block, concat_column_blocks);
    }

    // if not last, add vertical line
    if (i < column_blocks.size() - 1) {
      // concat blocks with vertial line
      const cv::Mat vert_line(cv::Size(2, concat_column_blocks.rows), CV_8UC3,
                              cv::viz::Color::black());
      cv::hconcat(concat_column_blocks, vert_line, concat_column_blocks);
    }
  }
  return concat_column_blocks;
}

// specalisation of values
// template <>
// void toGraphFileFormat<gtsam::Point3>(std::ostream& os,
//                                       const gtsam::Point3& t) {
//   os << t.x() << " " << t.y() << " " << t.z();
// }

// template <>
// void toGraphFileFormat<gtsam::Pose3>(std::ostream& os, const gtsam::Pose3& t)
// {
//   const gtsam::Point3 p = t.translation();
//   const auto q = t.rotation().toQuaternion();
//   os << p.x() << " " << p.y() << " " << p.z() << " " << q.x() << " " << q.y()
//      << " " << q.z() << " " << q.w();
// }

// // specalisation of factors
// template <>
// void toGraphFileFormat<gtsam::PriorFactor<gtsam::Pose3>>(
//     std::ostream& os, const gtsam::PriorFactor<gtsam::Pose3>& t) {
//   const gtsam::Pose3 measurement = t.prior();
//   toGraphFileFormat(os, measurement);

//   auto gaussianModel =
//       boost::dynamic_pointer_cast<gtsam::noiseModel::Gaussian>(t.noiseModel());
//   gtsam::Matrix Info = gaussianModel->information();
//   saveMatrixAsUpperTriangular(os, Info);
// }

// template <>
// void toGraphFileFormat<gtsam::BetweenFactor<gtsam::Pose3>>(
//     std::ostream& os, const gtsam::BetweenFactor<gtsam::Pose3>& t) {
//   const gtsam::Pose3 measurement = t.measured();
//   toGraphFileFormat(os, measurement);

//   auto gaussianModel =
//       boost::dynamic_pointer_cast<gtsam::noiseModel::Gaussian>(t.noiseModel());
//   gtsam::Matrix Info = gaussianModel->information();
//   saveMatrixAsUpperTriangular(os, Info);
// }

// template <>
// void toGraphFileFormat<LandmarkMotionTernaryFactor>(
//     std::ostream& os, const LandmarkMotionTernaryFactor& t) {
//   const gtsam::Point3 measurement(0, 0, 0);
//   toGraphFileFormat(os, measurement);

//   auto gaussianModel =
//       boost::dynamic_pointer_cast<gtsam::noiseModel::Gaussian>(t.noiseModel());
//   gtsam::Matrix Info = gaussianModel->information();
//   saveMatrixAsUpperTriangular(os, Info);
// }

// template <>
// void toGraphFileFormat<gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
//     std::ostream& os,
//     const gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>& t) {
//   const gtsam::Point3 measurement = t.measured();
//   toGraphFileFormat(os, measurement);

//   auto gaussianModel =
//       boost::dynamic_pointer_cast<gtsam::noiseModel::Gaussian>(t.noiseModel());
//   gtsam::Matrix Info = gaussianModel->information();
//   saveMatrixAsUpperTriangular(os, Info);
// }

}  // namespace factor_graph_tools

// void NonlinearFactorGraphManager::writeDynosamGraphFile(
//     const std::string& filename) const {
//   std::ofstream of(filename.c_str());

//   LOG(INFO) << "Writing dynosam graph file: " << filename;

//   using namespace factor_graph_tools;

//   // specalisations for dynosam factor graph
//   // gtsam does not have a write function for the base factor class so we
//   dont
//   // actually know the types contained within the NonLinearFactorGraph so we
//   // have to test them
//   auto write_factor = [](std::ofstream& os,
//                          const gtsam::NonlinearFactor::shared_ptr& nl_factor)
//                          {
//     // careful of mixing boost and std shared ptrs/casting here
//     auto prior_factor =
//         boost::dynamic_pointer_cast<gtsam::PriorFactor<gtsam::Pose3>>(
//             nl_factor);
//     if (prior_factor)
//       seralizeFactorToGraphFileFormat<gtsam::PriorFactor<gtsam::Pose3>>(
//           os, *prior_factor, "SE3_PRIOR_FACTOR");

//     // TODO: should give a different name for odom/smoothing factors?
//     // does not affect the graph structure but provides metadata?
//     auto between_factor =
//         boost::dynamic_pointer_cast<gtsam::BetweenFactor<gtsam::Pose3>>(
//             nl_factor);
//     if (between_factor)
//       seralizeFactorToGraphFileFormat<gtsam::BetweenFactor<gtsam::Pose3>>(
//           os, *between_factor, "SE3_BETWEEN_FACTOR");

//     auto motion_factor =
//         boost::dynamic_pointer_cast<LandmarkMotionTernaryFactor>(nl_factor);
//     if (motion_factor)
//       seralizeFactorToGraphFileFormat<LandmarkMotionTernaryFactor>(
//           os, *motion_factor, "SE3_MOTION_FACTOR");

//     auto pose_to_point_factor = boost::dynamic_pointer_cast<
//         gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(nl_factor);
//     if (pose_to_point_factor)
//       seralizeFactorToGraphFileFormat<
//           gtsam::PoseToPointFactor<gtsam::Pose3, Landmark>>(
//           os, *pose_to_point_factor, "POSE_TO_POINT_FACTOR");
//   };

//   auto write_value = [](std::ofstream& os, const gtsam::Values& value,
//                         gtsam::Key key) {
//     SymbolChar chr = DynoChrExtractor(key);

//     if (chr == InvalidDynoSymbol) {
//       throw std::runtime_error(
//           "Cannot write value to (dynosam) graph file as the associated key
//           is " "not a valid dynosam key!");
//     }

//     switch (chr) {
//       case kPoseSymbolChar:
//         seralizeValueToGraphFileFormat<gtsam::Pose3>(
//             os, value.at<gtsam::Pose3>(key), key, "SE3_POSE_VALUE");
//         break;
//       case kObjectMotionSymbolChar:
//         seralizeValueToGraphFileFormat<gtsam::Pose3>(
//             os, value.at<gtsam::Pose3>(key), key, "SE3_MOTION_VALUE");
//         break;
//       case kStaticLandmarkSymbolChar:
//         seralizeValueToGraphFileFormat<gtsam::Point3>(
//             os, value.at<gtsam::Point3>(key), key, "POINT3_STATIC_VALUE");
//         break;
//       case kDynamicLandmarkSymbolChar:
//         seralizeValueToGraphFileFormat<gtsam::Point3>(
//             os, value.at<gtsam::Point3>(key), key, "POINT3_DYNAMIC_VALUE");
//         break;
//       default:
//         LOG(FATAL) << "Should never reach here if DynoChrExtractor works!!";
//     }
//   };

//   gtsam::KeySet seen_keys;
//   for (const auto& factor : *this) {
//     write_factor(of, factor);
//     for (const auto key : *factor) {
//       // we have not seen this key before write it out to avoid duplication
//       if (seen_keys.find(key) == seen_keys.end()) {
//         write_value(of, values_, key);
//         seen_keys.insert(key);
//       }
//     }
//   }

//   std::flush(of);
// }

bool ISAM2Helper::marginalizeKeys(gtsam::ISAM2Result* result,
                                  gtsam::ISAM2* smoother,
                                  const gtsam::KeyVector& marginalize_keys) {
  CHECK_NOTNULL(smoother);

  // checks all keys are in the smoother
  gtsam::KeyVector keys_to_marginalize;
  for (const auto& key : marginalize_keys) {
    if (smoother->valueExists(key)) keys_to_marginalize.push_back(key);
  }

  if (keys_to_marginalize.empty()) {
    return false;
  }

  // create ordering constraints; forcing ISAM2 to put marginalizable variables
  // at the beginning
  boost::optional<gtsam::FastMap<gtsam::Key, int>> constrainedKeys = {};
  createOrderingConstraints(keys_to_marginalize, *smoother, constrainedKeys);

  if (!constrainedKeys) {
    return false;
  }
  // gather additional keys to re-eliminate
  std::unordered_set<gtsam::Key> additionalKeys =
      Base::gatherAdditionalKeysToReEliminate(*smoother, keys_to_marginalize);
  gtsam::KeyList additionalMarkedKeys(additionalKeys.begin(),
                                      additionalKeys.end());
  // update ISAM
  const auto isam_result = smoother->update(
      gtsam::NonlinearFactorGraph(), gtsam::Values(), gtsam::FactorIndices(),
      constrainedKeys, {}, additionalMarkedKeys);
  // marginalize leaf keys!
  gtsam::FastList<gtsam::Key> leafKeys(keys_to_marginalize.begin(),
                                       keys_to_marginalize.end());
  smoother->marginalizeLeaves(leafKeys);

  if (result) *result = isam_result;
  return true;
}

void ISAM2Helper::createOrderingConstraints(
    const gtsam::KeyVector& marginalizableKeys, const ISAM2& smoother,
    boost::optional<gtsam::FastMap<gtsam::Key, int>>& constrainedKeys) {
  if (marginalizableKeys.size() > 0) {
    // find the set difference to identify keys we dont want to marginalize
    gtsam::KeySet all_keys_sorted = smoother.getLinearizationPoint().keySet();
    gtsam::KeySet marginalizableKeys_sorted(marginalizableKeys.begin(),
                                            marginalizableKeys.end());

    gtsam::KeyVector other_variables;
    std::set_difference(
        all_keys_sorted.begin(), all_keys_sorted.end(),
        marginalizableKeys_sorted.begin(), marginalizableKeys_sorted.end(),
        std::inserter(other_variables, other_variables.begin()));

    constrainedKeys = FastMap<Key, int>();
    // Generate ordering constraints so that the marginalizable variables will
    // be eliminated first Set all variables to Group1
    for (gtsam::Key key : other_variables) {
      constrainedKeys->operator[](key) = 1;
    }
    // Set marginalizable variables to Group0
    for (gtsam::Key key : marginalizableKeys_sorted) {
      constrainedKeys->operator[](key) = 0;
    }
  }
}

}  // namespace dyno
