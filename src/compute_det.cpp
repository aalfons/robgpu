
#include "robgpu_settings.h"

#include <RcppArmadillo.h>
#include "compute_det.h"

// [[Rcpp::export]]
double compute_det(SEXP pX) 
{
  arma::mat X = Rcpp::as<arma::mat>(pX);

  return arma::det(X);
}

