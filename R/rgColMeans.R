#' GPU implementation of colwise mean computation
#'
#' @param X a numeric matrix.
#' @return A vector containing the mean for each column of the input matrix.
#' 
#' @author Roland Boubela 
#' 
#' @examples
#' X <- matrix(runif(16), 4, 4)
#' rgColMeans(X)
#' 
#' @keywords gpu parallel
#' @import Rcpp

# colwise mean on a GPU
rgColMeans <- function(X) {

  res <- .Call("rcolwisemean", X, PACKAGE = "robgpu")
  
  return(res)
}

