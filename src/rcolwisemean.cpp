
#include <Rcpp.h>

#include "colwisemean.h"

using namespace Rcpp;

extern "C"
SEXP rcolwisemean(SEXP _X)
{
  // create result vector
  NumericMatrix X(_X);
  NumericVector colmeans(X.ncol());

  // compute colsums on GPU
  colwisemean(X.begin(), X.ncol(), X.nrow(), colmeans.begin());

  return colmeans;
}

