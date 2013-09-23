
gpuMCD <- function(x, alpha, nsamp) {

	x <- as.matrix(x)
	p <- ncol(x)
	n <- nrow(x)

  # compute subsample size
  sample.size <- floor(alpha * n)

  # generate subsample index matrix for all samples
  sample.index <- t(sapply(1:nsamp, function(x) { sample(n, sample.size) })) - 1
  
  # 'allocate' result matrix
  covMat <- matrix(0, nrow = p, ncol = p)

  # call the c-level function to find the minimum covariance determinant
  cov.det <- .Call("rgpuMCD", 
			      x, n, p, covMat, nsamp, sample.size, sample.index,
			      PACKAGE = 'robgpu')

  
  # which sample has the minimum
  min.cov.det <- which(cov.det == min(cov.det))
 
  # compute the final covariance matrix 
  cov.det <- .Call("rgpuMCD", 
			      x, n, p, covMat, 1, sample.size, 
            as.matrix(sample.index[min.cov.det, ]),
			      PACKAGE = 'robgpu')




  return (list(cov = covMat, sample.det = cov.det))
}

