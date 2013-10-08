
#include <Rcpp.h>

#include "colwisesum.h"

using namespace Rcpp;

extern "C"
SEXP rcolwisesum(SEXP _X)
{
  // create result vector
  NumericMatrix X(_X);
  NumericVector colsums(X.ncol());

  // compute colsums on GPU
  colwisesum(X.begin(), X.ncol(), X.nrow(), colsums.begin());

  return colsums;
}

