void gpuLTSF(float *X, int n, int p, float *Y, int nY,
	double tol, float *coeffs, float *resids, float *effects,
	int *rank, int *pivot, double * qrAux);
void gpuLTSD(double *X, int n, int p, double *Y, int nY,
	double tol, double *coeffs, double *resids, double *effects,
	int *rank, int *pivot, double * qrAux);

