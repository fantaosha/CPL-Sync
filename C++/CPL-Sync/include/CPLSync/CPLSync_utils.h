/** This file provides a convenient set of utility functions for reading in a
set of pose-graph SLAM measurements and constructing the corresponding data
matrices used in the CPL-Sync algorithm.
 *
 * Copyright (C) 2018 - 2019 by Taosha Fan (taosha.fan@gmail.com)
 */

#pragma once

#include <string>

#include <Eigen/Sparse>

#include "CPLSync/CPLSync_types.h"
#include "CPLSync/RelativePoseMeasurement.h"

namespace CPLSync {

measurements_t read_g2o_file(const std::string &filename, size_t &num_poses);

/** Given a vector of relative pose measurements, this function computes and
 * returns the corresponding rotational connection Laplacian */
ComplexSparseMatrix
construct_rotational_connection_Laplacian(const measurements_t &measurements);

/** Given a vector of relative pose measurements, this function computes and
 * returns the associated oriented incidence matrix A */
RealSparseMatrix
construct_oriented_incidence_matrix(const measurements_t &measurements);

/** Given a vector of relative pose measurements, this function computes and
 * returns the associated diagonal matrix of translational measurement
 * precisions */
RealDiagonalMatrix
construct_translational_precision_matrix(const measurements_t &measurements);

/** Given a vector of relative pose measurements, this function computes and
 * returns the associated matrix of raw translational measurements */
ComplexSparseMatrix
construct_translational_data_matrix(const measurements_t &measurements);

/** Given a vector of relative pose measurements, this function computes and
 * returns the B matrices defined in equation (69) of the tech report */
void construct_B_matrices(const measurements_t &measurements,
                          ComplexSparseMatrix &B1, ComplexSparseMatrix &B2,
                          ComplexSparseMatrix &B3);

/** Given a vector of relative pose measurements, this function constructs the
 * matrix M parameterizing the objective in the translation-explicit formulation
 * of the CPL-Sync problem (Problem 2) in the CPL-Sync tech report) */
ComplexSparseMatrix
construct_quadratic_form_data_matrix(const measurements_t &measurements);

/** Given the measurement matrix B3 defined in equation (69c) of the tech report
 * and the problem dimension d, this function computes and returns the
 * corresponding chordal initialization for the rotational states */
ComplexVector chordal_initialization(const ComplexSparseMatrix &B3);

/** Given the measurement matrices B1 and B2 and a matrix R of rotational state
 * estimates, this function computes and returns the corresponding optimal
 * translation estimates */
ComplexVector recover_translations(const ComplexSparseMatrix &B1,
                                   const ComplexSparseMatrix &B2,
                                   const ComplexVector &R);
} // namespace CPLSync
