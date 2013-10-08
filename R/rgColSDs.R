#' GPU implementation of colwise standard deviation computation
#'
#' @param X a numeric matrix.
#' @return A vector containing the sd for each column of the input matrix.
#' 
#' @author Roland Boubela 
#' 
#' @examples
#' X <- matrix(runif(16), 4, 4)
#' rgColSDs(X)
#' 
#' @keywords gpu parallel
#' @import Rcpp

rgColSDs <- function(X) {

  res <- .Call("rcolwisesd", X, PACKAGE = "robgpu")
  
  return(res)
}

