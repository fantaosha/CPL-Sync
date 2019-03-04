#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

#include <Eigen/Geometry>
#include <Eigen/SPQRSupport>

#include "CPLSync/CPLSync_utils.h"

namespace CPLSync {

measurements_t read_g2o_file(const std::string &filename, size_t &num_poses) {

  // Preallocate output vector
  measurements_t measurements;

  // A single measurement, whose values we will fill in
  CPLSync::RelativePoseMeasurement measurement;

  // A string used to contain the contents of a single line
  std::string line;

  // A string used to extract tokens from each line one-by-one
  std::string token;

  // Preallocate various useful quantities
  Scalar dx, dy, dz, dtheta, dqx, dqy, dqz, dqw, I11, I12, I13, I14, I15, I16,
      I22, I23, I24, I25, I26, I33, I34, I35, I36, I44, I45, I46, I55, I56, I66;

  Scalar ss, cc;

  size_t i, j;

  // Open the file for reading
  std::ifstream infile(filename);

  num_poses = 0;

  while (std::getline(infile, line)) {
    // Construct a stream from the string
    std::stringstream strstrm(line);

    // Extract the first token from the string
    strstrm >> token;

    if (token == "EDGE_SE2") {
      // This is a 2D pose measurement

      /** The g2o format specifies a 2D relative pose measurement in the
       * following form:
       *
       * EDGE_SE2 id1 id2 dx dy dtheta, I11, I12, I13, I22, I23, I33
       *
       */

      // Extract formatted output
      strstrm >> i >> j >> dx >> dy >> dtheta >> I11 >> I12 >> I13 >> I22 >>
          I23 >> I33;

      // Fill in elements of this measurement

      // Pose ids
      measurement.i = i;
      measurement.j = j;

      // Raw measurements
      sincos(dtheta, &ss, &cc);
      measurement.t = Complex(dx, dy);
      measurement.R = Complex(cc, ss);

      Eigen::Matrix<Scalar, 2, 2> TranCov;
      TranCov << I11, I12, I12, I22;
      measurement.tau = 2 / TranCov.inverse().trace();

      measurement.kappa = 2 * I33;

    } else if (token == "VERTEX_SE2") {
      continue;
    } else {
      std::cout << "Error: unrecognized type: " << token << "!" << std::endl;
      assert(false);
    }

    // Update maximum value of poses found so far
    size_t max_pair = std::max<size_t>(measurement.i, measurement.j);

    num_poses = ((max_pair > num_poses) ? max_pair : num_poses);
    measurements.push_back(measurement);
  } // while

  infile.close();

  num_poses++; // Account for the use of zero-based indexing

  return measurements;
}

ComplexSparseMatrix
construct_rotational_connection_Laplacian(const measurements_t &measurements) {

  size_t num_poses = 0; // We will use this to keep track of the largest pose
  // index encountered, which in turn provides the number
  // of poses

  // Each measurement contributes 2*d elements along the diagonal of the
  // connection Laplacian, and 2*d^2 elements on a pair of symmetric
  // off-diagonal blocks

  size_t measurement_stride = 4;

  std::vector<Eigen::Triplet<Complex>> triplets;
  triplets.reserve(measurement_stride * measurements.size());

  size_t i, j, max_pair;

  for (const CPLSync::RelativePoseMeasurement &measurement : measurements) {
    i = measurement.i;
    j = measurement.j;

    // Elements of ith block-diagonal
    triplets.emplace_back(i, i, measurement.kappa);

    // Elements of jth block-diagonal
    triplets.emplace_back(j, j, measurement.kappa);

    // Elements of ij block
    triplets.emplace_back(i, j, -measurement.kappa * std::conj(measurement.R));

    // Elements of ji block
    triplets.emplace_back(j, i, -measurement.kappa * measurement.R);

    // Update num_poses
    max_pair = std::max<size_t>(i, j);

    if (max_pair > num_poses)
      num_poses = max_pair;
  }

  num_poses++; // Account for 0-based indexing

  // Construct and return a sparse matrix from these triplets
  ComplexSparseMatrix LGrho(num_poses, num_poses);
  LGrho.setFromTriplets(triplets.begin(), triplets.end());

  return LGrho;
}

RealSparseMatrix
construct_oriented_incidence_matrix(const measurements_t &measurements) {
  std::vector<Eigen::Triplet<Scalar>> triplets;
  triplets.reserve(2 * measurements.size());

  size_t num_poses = 0;
  size_t max_pair;
  for (size_t m = 0; m < measurements.size(); m++) {
    triplets.emplace_back(measurements[m].i, m, -1);
    triplets.emplace_back(measurements[m].j, m, 1);

    max_pair = std::max<size_t>(measurements[m].i, measurements[m].j);
    if (max_pair > num_poses)
      num_poses = max_pair;
  }
  num_poses++; // Account for zero-based indexing

  RealSparseMatrix A(num_poses, measurements.size());
  A.setFromTriplets(triplets.begin(), triplets.end());
  return A;
}

RealDiagonalMatrix
construct_translational_precision_matrix(const measurements_t &measurements) {

  // Allocate output matrix
  RealDiagonalMatrix Omega(measurements.size());

  RealDiagonalMatrix::DiagonalVectorType &diagonal = Omega.diagonal();

  for (size_t m = 0; m < measurements.size(); m++)
    diagonal[m] = measurements[m].tau;

  return Omega;
}

ComplexSparseMatrix
construct_translational_data_matrix(const measurements_t &measurements) {

  size_t num_poses = 0;

  std::vector<Eigen::Triplet<Complex>> triplets;
  triplets.reserve(measurements.size());

  size_t max_pair;
  for (size_t m = 0; m < measurements.size(); m++) {
    triplets.emplace_back(m, measurements[m].i, -measurements[m].t);

    max_pair = std::max<size_t>(measurements[m].i, measurements[m].j);
    if (max_pair > num_poses)
      num_poses = max_pair;
  }
  num_poses++; // Account for zero-based indexing

  ComplexSparseMatrix T(measurements.size(), num_poses);
  T.setFromTriplets(triplets.begin(), triplets.end());

  return T;
}

void construct_B_matrices(const measurements_t &measurements,
                          ComplexSparseMatrix &B1, ComplexSparseMatrix &B2,
                          ComplexSparseMatrix &B3) {
  // Clear input matrices
  B1.setZero();
  B2.setZero();
  B3.setZero();

  size_t num_poses = 0;

  std::vector<Eigen::Triplet<Complex>> triplets;

  // Useful quantities to cache

  size_t i, j; // Indices for the tail and head of the given measurement
  Scalar sqrttau;
  size_t max_pair;

  /// Construct the matrix B1 from equation (69a) in the tech report
  triplets.reserve(2 * measurements.size());

  for (size_t e = 0; e < measurements.size(); e++) {
    i = measurements[e].i;
    j = measurements[e].j;
    sqrttau = sqrt(measurements[e].tau);

    // Block corresponding to the tail of the measurement
    triplets.emplace_back(e, i,
                          -sqrttau); // Diagonal element corresponding to tail
    triplets.emplace_back(e, j,
                          sqrttau); // Diagonal element corresponding to head

    // Keep track of the number of poses we've seen
    max_pair = std::max<size_t>(i, j);
    if (max_pair > num_poses)
      num_poses = max_pair;
  }
  num_poses++; // Account for zero-based indexing

  B1.resize(measurements.size(), num_poses);
  B1.setFromTriplets(triplets.begin(), triplets.end());

  /// Construct matrix B2 from equation (69b) in the tech report
  triplets.clear();
  triplets.reserve(measurements.size());

  for (size_t e = 0; e < measurements.size(); e++) {
    i = measurements[e].i;
    sqrttau = sqrt(measurements[e].tau);
    triplets.emplace_back(e, i, -sqrttau * measurements[e].t);
  }

  B2.resize(measurements.size(), num_poses);
  B2.setFromTriplets(triplets.begin(), triplets.end());

  /// Construct matrix B3 from equation (69c) in the tech report
  triplets.clear();
  triplets.reserve(2 * measurements.size());

  for (size_t e = 0; e < measurements.size(); e++) {
    Scalar sqrtkappa = std::sqrt(measurements[e].kappa);
    auto const &R = measurements[e].R;

    i = measurements[e].i; // Tail of measurement
    j = measurements[e].j; // Head of measurement

    // Representation of the -sqrt(kappa) * Rt(i,j) \otimes I_d block
    triplets.emplace_back(e, i, -sqrtkappa * R);

    triplets.emplace_back(e, j, sqrtkappa);
  }

  B3.resize(measurements.size(), num_poses);
  B3.setFromTriplets(triplets.begin(), triplets.end());
}

ComplexSparseMatrix
construct_quadratic_form_data_matrix(const measurements_t &measurements) {

  size_t num_poses = 0;

  std::vector<Eigen::Triplet<Complex>> triplets;

  // Number of nonzero elements contributed to L(W^tau) by each measurement
  size_t LWtau_nnz_per_measurement = 4;

  // Number of nonzero elements contributed to V by each measurement
  size_t V_nnz_per_measurement = 2;

  // Number of nonzero elements contributed to L(G^rho) by each measurement
  size_t LGrho_nnz_per_measurement = 4;

  // Number of nonzero elements contributed to Sigma by each measurement
  size_t Sigma_nnz_per_measurement = 1;

  // Number of nonzero elements contributed to the entire matrix M by each
  // measurement
  size_t num_nnz_per_measurement =
      LWtau_nnz_per_measurement + 2 * V_nnz_per_measurement +
      LGrho_nnz_per_measurement + Sigma_nnz_per_measurement;

  /// Working space
  size_t i, j; // Indices for the tail and head of the given measurement
  size_t max_pair;

  triplets.reserve(num_nnz_per_measurement * measurements.size());

  // Scan through the set of measurements to determine the total number of poses
  // in this problem
  for (const CPLSync::RelativePoseMeasurement &measurement : measurements) {
    max_pair = std::max<size_t>(measurement.i, measurement.j);
    if (max_pair > num_poses)
      num_poses = max_pair;
  }
  num_poses++; // Account for zero-based indexing

  // Now scan through the measurements again, using knowledge of the total
  // number of poses to compute offsets as appropriate

  for (const CPLSync::RelativePoseMeasurement &measurement : measurements) {

    i = measurement.i; // Tail of measurement
    j = measurement.j; // Head of measurement

    // Add elements for L(W^tau)
    triplets.emplace_back(i, i, measurement.tau);
    triplets.emplace_back(j, j, measurement.tau);
    triplets.emplace_back(i, j, -measurement.tau);
    triplets.emplace_back(j, i, -measurement.tau);

    // Add elements for V (upper-right block)
    triplets.emplace_back(i, num_poses + i, measurement.tau * measurement.t);
    triplets.emplace_back(j, num_poses + i, -measurement.tau * measurement.t);

    // Add elements for V' (lower-left block)
    triplets.emplace_back(num_poses + i, i,
                          measurement.tau * std::conj(measurement.t));
    triplets.emplace_back(num_poses + i, j,
                          -measurement.tau * std::conj(measurement.t));

    // Add elements for L(G^rho)
    // Elements of ith block-diagonal
    triplets.emplace_back(num_poses + i, num_poses + i, measurement.kappa);

    // Elements of jth block-diagonal
    triplets.emplace_back(num_poses + j, num_poses + j, measurement.kappa);

    // Elements of ij block
    triplets.emplace_back(num_poses + i, num_poses + j,
                          -measurement.kappa * std::conj(measurement.R));

    // Elements of ji block
    triplets.emplace_back(num_poses + j, num_poses + i,
                          -measurement.kappa * measurement.R);

    // Add elements for Sigma
    triplets.emplace_back(num_poses + i, num_poses + i,
                          measurement.tau * std::norm(measurement.t));
  }

  ComplexSparseMatrix M(2 * num_poses, 2 * num_poses);
  M.setFromTriplets(triplets.begin(), triplets.end());

  return M;
}

ComplexVector chordal_initialization(const ComplexSparseMatrix &B3) {
  size_t num_poses = B3.cols();

  /// We want to find a minimizer of
  /// || B3 * r ||
  ///
  /// For the purposes of initialization, we can simply fix the first pose to
  /// the origin; this corresponds to fixing the first d^2 elements of r to
  /// vec(I_d), and slicing off the first d^2 columns of B3 to form
  ///
  /// min || B3red * rred + c ||, where
  ///
  /// c = B3(1:d^2) * vec(I_3)

  ComplexSparseMatrix B3red = B3.rightCols((num_poses - 1));
  // Must be in compressed format to use Eigen::SparseQR!
  B3red.makeCompressed();

  ComplexVector cR = B3.leftCols(1);

  Eigen::SPQR<ComplexSparseMatrix> QR(B3red);

  ComplexVector Rchordal(num_poses, 1);
  Rchordal(0) = 1;
  Rchordal.tail(num_poses - 1) = -QR.solve(cR);

  //Rchordal.array() /= Rchordal.array().abs();
  Rchordal.tail(num_poses - 1).rowwise().normalize();

  return Rchordal;
}

ComplexVector recover_translations(const ComplexSparseMatrix &B1,
                                   const ComplexSparseMatrix &B2,
                                   const ComplexVector &R) {
  size_t n = R.rows();

  /// We want to find a minimizer of
  /// || B1 * t + B2 * vec(R) ||
  ///
  /// For the purposes of initialization, we can simply fix the first pose to
  /// the origin; this corresponds to fixing the first d elements of t to 0,
  /// and
  /// slicing off the first d columns of B1 to form
  ///
  /// min || B1red * tred + c) ||, where
  ///
  /// c = B2 * vec(R)

  // Form the matrix comprised of the right (n-1) block columns of B1
  ComplexSparseMatrix B1red = B1.rightCols(n - 1);

  ComplexVector c = B2 * R;

  // Solve
  Eigen::SPQR<ComplexSparseMatrix> QR(B1red);
  ComplexVector tred = -QR.solve(c);

  // Allocate output matrix
  ComplexMatrix t = ComplexMatrix::Zero(n, 1);

  // Set rightmost n-1 columns
  t.bottomRows(n - 1) = tred;

  return t;
}
} // namespace CPLSync
