
#include "robgpu_settings.h"

#include <Rcpp.h>

#include "compute_inverse.h"

RcppExport 
SEXP compute_inverse(SEXP pMatrix) 
{
  Rcpp::Function solve("solve");
  Rcpp::NumericMatrix res = solve(pMatrix);
  Rcpp::NumericMatrix resMat(pMatrix);
  
  int n = resMat.nrow();
  int p = resMat.ncol();
  
  for (int i = 0; i < p; i++) {
    for (int j = 0; j < n; j++) {
      resMat(j, i) = res(j, i);
    }
  }

  return R_NilValue;
}

