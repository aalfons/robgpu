
#include <Rcpp.h>

#include "colwisesd.h"

using namespace Rcpp;

extern "C"
SEXP rcolwisesd(SEXP _X)
{
  // create result vector
  NumericMatrix X(_X);
  NumericVector colsds(X.ncol());

  // compute colsums on GPU
  colwisesd(X.begin(), X.ncol(), X.nrow(), colsds.begin());

  return colsds;
}

