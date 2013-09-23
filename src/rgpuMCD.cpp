
#include "robgpu_settings.h"

#include <stdio.h>
#include <Rdefines.h>
#include <Rcpp.h>

#include "rgpuMCD.h"
#include "gpuMCD.h"

using namespace Rcpp;

extern "C"
SEXP rgpuMCD(SEXP pX, SEXP pn, SEXP pp, SEXP pcovMat, SEXP pnsamp, SEXP psample_size, SEXP psample_index)
{
  // data input and information
  NumericMatrix rcpp_X(pX);
  IntegerVector rcpp_n(pn);
  IntegerVector rcpp_p(pp);

  // covariance matrix
  NumericMatrix rcpp_covMat(pcovMat);

  // subsample information
  IntegerVector rcpp_nsamp(pnsamp);
  NumericVector rcpp_sample_size(psample_size);

  // subsample indices
  IntegerMatrix rcpp_sample_index(psample_index);

  // store covariance determinant for each subsample
  NumericVector rcpp_sample_det(rcpp_nsamp[0]); 
  
  // compute MCD
  gpuMCD(rcpp_X.begin(), rcpp_n[0], rcpp_p[0], rcpp_covMat.begin(), pcovMat, rcpp_sample_det.begin(), rcpp_nsamp[0], rcpp_sample_size[0], rcpp_sample_index.begin(), 0);

  return wrap(rcpp_sample_det);
}

