
#include <Rcpp.h>
#include <stdio.h>

#include "rgcor.h"

using namespace Rcpp;

extern "C"
SEXP rcor(SEXP _X, SEXP _Y)
{
  // create Rcpp matrices
  NumericMatrix X(_X);
  NumericMatrix Y(_Y);

  // create result matrix
  NumericMatrix correlations(X.ncol(), Y.ncol());
  
  // compute colsums on GPU
  rgcor(X.begin(), X.ncol(), Y.begin(), Y.ncol(), X.nrow(), 
        correlations.begin(), 0);

  return correlations;
}

