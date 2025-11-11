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

#include <glog/logging.h>
#include <gtsam/base/Matrix.h>

#include "dynosam_common/Types.hpp"

namespace dyno {

namespace internal {

struct HungarianAlgorithm {
 public:
  HungarianAlgorithm() = default;

  /**
   * @brief Solve the optical assignment problem given a set of workers (row)
   * and jobs (columns). We have to find an assignment of the jobs to the
   * workers, such that each job is assigned to one worker and each worker is
   * assigned one job, such that the total cost of assignment is minimum.
   *
   * DistMatrix is nonnegative [n * m] matrix (where n <= m), representing n
   * workers and m jobs. The element in the i-th row and j-th column represents
   * the cost of assigning the j-th job to the i-th worker.
   *
   * The result is a vector of size [n * 1], where each value is the index in
   * the column (j) that i is assigned to such that \hat{j} = assignment[i]
   *
   * @param DistMatrix const Eigen::MatrixXd&
   * @param assignment Eigen::VectorXi&
   * @return double
   */
  double solve(const Eigen::MatrixXd &cost_matrix,
               Eigen::VectorXi &assignment) const;

 private:
  void assignmentoptimal(int *assignment, double *cost, double *distMatrixIn,
                         int nOfRows, int nOfColumns) const;
  void buildassignmentvector(int *assignment, bool *starMatrix, int nOfRows,
                             int nOfColumns) const;
  void computeassignmentcost(int *assignment, double *cost, double *distMatrix,
                             int nOfRows) const;
  void step2a(int *assignment, double *distMatrix, bool *starMatrix,
              bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns,
              bool *coveredRows, int nOfRows, int nOfColumns, int minDim) const;
  void step2b(int *assignment, double *distMatrix, bool *starMatrix,
              bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns,
              bool *coveredRows, int nOfRows, int nOfColumns, int minDim) const;
  void step3(int *assignment, double *distMatrix, bool *starMatrix,
             bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns,
             bool *coveredRows, int nOfRows, int nOfColumns, int minDim) const;
  void step4(int *assignment, double *distMatrix, bool *starMatrix,
             bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns,
             bool *coveredRows, int nOfRows, int nOfColumns, int minDim,
             int row, int col) const;
  void step5(int *assignment, double *distMatrix, bool *starMatrix,
             bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns,
             bool *coveredRows, int nOfRows, int nOfColumns, int minDim) const;
};

}  // namespace internal

// template<
//     typename WORKERS,
//     typename JOBS,
//     typename WORKER = typename WORKERS::value_type,
//     typename JOB = typename JOBS::value_type>
// struct Assignment {
// public:
//     using Workers = WORKERS;
//     using Jobs = JOBS;

//     using Worker = WORKER;
//     using Job = JOB;

//     using ConstWorker = std::add_const_t<Worker>;
//     using ConstJob = std::add_const_t<Job>;

//     using CostExpr = std::function<double(ConstWorker&, ConstJob&)>;
//     using CostAssignment = std::function<void(Eigen::MatrixXd&)>;

//     static void optimalAssignment(Eigen::VectorXi& assignment, const Workers&
//     workers, const Jobs& jobs, const CostExpr& cost_expr) {
//         CostAssignment cost_assignment =[](Eigen::MatrixXd&) -> void {};
//         optimalAssignment(assignment, workers, jobs, cost_expr,
//         cost_assignment);
//     }
//     static void optimalAssignment(Eigen::VectorXi& assignment, const Workers&
//     workers, const Jobs& jobs, const CostExpr& cost_expr, const
//     CostAssignment& cost_assignment) {
//         Eigen::MatrixXd cost_matrix;
//         constructCostMatrix(cost_matrix, workers, jobs, cost_expr,
//         cost_assignment);

//         CHECK_EQ(cost_matrix.rows(), workers.size());
//         CHECK_EQ(cost_matrix.cols(), jobs.size());

//         internal::HungarianAlgorithm().solve(cost_matrix, assignment);

//     }

// private:
//     static void constructCostMatrix(Eigen::MatrixXd& cost_matrix, const
//     Workers& workers, const Jobs& jobs, const CostExpr& cost_expr, const
//     CostAssignment& cost_assignment) {
//         const auto n = workers.size();
//         const auto m = jobs.size();

//         cost_matrix.resize(n, m);

//         //for each row (worker to assign)
//         decltype(n) row = 0;
//         //for each job to construct a cost for
//         decltype(m) col = 0;
//         for(auto worker_itr = workers.begin(); worker_itr != workers.end();
//         worker_itr++) {
//             const auto worker = *worker_itr;
//             for(auto job_itr = jobs.begin(); job_itr != jobs.end();
//             job_itr++) {
//                 const auto job = *job_itr;
//                 const double cost = cost_expr(worker, job);
//                 cost_matrix(row, col) = cost;
//                 col++;
//             }
//             row++;
//         }

//         cost_assignment(cost_matrix);
//     }
// };

}  // namespace dyno
