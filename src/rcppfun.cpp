#include <Rcpp.h>

extern "C" SEXP rcppfun(SEXP fun, SEXP arg1, SEXP arg2) {
  Rcpp::Function cppfun(fun);
  Rcpp::NumericVector cor;

  SEXP res = cppfun(arg1, arg2);
  cor = Rcpp::NumericVector(res);

  printf("%.4f\n", cor[0]);
  
  return res;
}


