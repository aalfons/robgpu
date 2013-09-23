
#include "robgpu_settings.h"

#include <Rcpp.h>

#include "compute_det.h"

RcppExport 
double compute_det(SEXP pX) 
{
  Rcpp::Function det("det");
  Rcpp::NumericVector res;
  
  res = det(pX);

  return res[0];
}

