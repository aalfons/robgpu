#' GPU implementation of colwise sum computation
#'
#' @param X a numeric matrix.
#' @return A vector containing the sum for each column of the input matrix.
#' 
#' @author Roland Boubela 
#' 
#' @examples
#' X <- matrix(runif(16), 4, 4)
#' rgColSums(X)
#' 
#' @keywords gpu parallel
#' @import Rcpp

rgColSums <- function(X) {

  res <- .Call("rcolwisesum", X, PACKAGE = "robgpu")
  
  return(res)
}

