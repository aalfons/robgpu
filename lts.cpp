#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cublas.h>
#include <R.h>
#include <Rcpp.h>

#include "cuseful.h"
#include "lsfit.h"
#include "lts.h"
#include "qrdecomp.h"


void gpuLTSF(float * X, int rows, int cols, float * Y, int yCols,
	double tol, float * coeffs, float * resids, float * effects,
	int * rank, int * pivot, double * qrAux)
{

  // TODO call lsfit with subset of the data.
  Rcpp::Environment base("package:base");
  Rcpp::Function sample = base["sample"];
  
  Rcpp::IntegerVector sampleIdx((int) rows * 0.75);

  sampleIdx = sample(rows, (int) rows * 0.75); 

  for (unsigned int i = 0; i < sampleIdx.length(); i ++) {
    std::cout << sampleIdx[i] << " ";
  }
  std::cout << std::endl;

  // gpuLSFitF(X, rows, cols, Y, yCols, tol, coeffs, resids, effects,
  //    rank, pivot, qrAux);

}

void gpuLTSD(double *X, int n, int p, double *Y, int nY,
	   double tol, double *coeffs, double *resids, double *effects,
	   int *rank, int *pivot, double * qrAux) 
{
// NYI
}


