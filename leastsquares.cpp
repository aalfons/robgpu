/* =========================================================================
   Copyright (c) 2010-2013, Institute for Microelectronics,
                            Institute for Analysis and Scientific Computing,
                            TU Wien.
   Portions of this software are copyright by UChicago Argonne, LLC.

                            -----------------
                  ViennaCL - The Vienna Computing Library
                            -----------------

   Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at
               
   (A list of authors and contributors can be found in the PDF manual)

   License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */

/*
* 
*   Tutorial: Least Squares problem for matrices from ViennaCL or Boost.uBLAS (least-squares.cpp and least-squares.cu are identical, the latter being required for compilation using CUDA nvcc)
* 
*   See Example 2 at http://tutorial.math.lamar.edu/Classes/LinAlg/QRDecomposition.aspx for a reference solution.
*
*/

// activate ublas support in ViennaCL
#define VIENNACL_WITH_UBLAS 

// Must be set prior to any ViennaCL includes if you want to use ViennaCL algorithms on Eigen objects
#define VIENNACL_WITH_EIGEN 1


//
// include necessary system headers
//
#include <iostream>

//
// Boost includes
//
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>
//
// ViennaCL includes
//
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/matrix_proxy.hpp"
#include "viennacl/linalg/qr.hpp"
#include "viennacl/linalg/lu.hpp"

// 
// R includes
//
#include <Rcpp.h>

// 
// local includes
//
#include "leastsquares.h"

using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Rcpp::as;

int leastsquares (Rcpp::NumericMatrix Rcpp_X, Rcpp::NumericMatrix Rcpp_Y)
{
  typedef double               ScalarType;     //feel free to change this to 'double' if supported by your hardware
  typedef boost::numeric::ublas::matrix<ScalarType>              MatrixType;
  typedef boost::numeric::ublas::vector<ScalarType>              VectorType;
  typedef viennacl::matrix<ScalarType, viennacl::column_major>   VCLMatrixType;
  typedef viennacl::vector<ScalarType>                           VCLVectorType;

  //
  // Create vectors and matrices with data, cf. http://tutorial.math.lamar.edu/Classes/LinAlg/QRDecomposition.aspx
  //

  //
  // Setup the matrix in ViennaCL:
  //

  // copy from Rcpp to VCL
  Map<MatrixXd> Eigen_A(Rcpp_X.begin(), Rcpp_X.nrow(), Rcpp_X.ncol());
  Map<MatrixXd> Eigen_b(Rcpp_Y.begin(), Rcpp_Y.nrow(), 1);

  MatrixXd EigenMatrix_A(Eigen_A);
  VectorXd EigenMatrix_b(Eigen_b);

  // generate ublas matrix from Rcpp matrix
  VCLMatrixType vcl_A(Rcpp_X.nrow(), Rcpp_X.ncol());
  VCLVectorType vcl_b(Rcpp_X.nrow());

  viennacl::copy(EigenMatrix_A, vcl_A);
  viennacl::copy(EigenMatrix_b, vcl_b);
   
  //////////// Part 2: Use ViennaCL types for BLAS 3 computations, but use Boost.uBLAS for the panel factorization ////////////////
  
  // std::cout << "--- ViennaCL (hybrid implementation)  ---" << std::endl;
  std::vector<ScalarType> hybrid_betas = viennacl::linalg::inplace_qr(vcl_A);
  
  // compute modified RHS of the minimization problem: 
  // b := Q^T b
  viennacl::linalg::inplace_qr_apply_trans_Q(vcl_A, hybrid_betas, vcl_b);

  // Final step: triangular solve: Rx = b'.
  // We only need the upper part of A such that R is a square matrix
  viennacl::range vcl_range(0, Rcpp_X.ncol());

  viennacl::matrix_range<VCLMatrixType> vcl_R(vcl_A, vcl_range, vcl_range);

  viennacl::vector_range<VCLVectorType> vcl_b2(vcl_b, vcl_range);

  viennacl::linalg::inplace_solve(vcl_R, vcl_b2, viennacl::linalg::upper_tag());

  std::cout << "Result: b2" << vcl_b2 << std::endl;

  return EXIT_SUCCESS;
}

