#' GPU implementation of the Pearson Correlation Coefficient
#'
#' @param X a numeric matrix.
#' @param Y a numeric matrix.
#' @return The correlation matrix.
#' 
#' @author Roland Boubela 
#' 
#' @examples
#' X <- matrix(runif(16), 4, 4)
#' Y <- matrix(runif(16), 4, 4)
#' rgCor(X, Y)
#' 
#' @keywords gpu parallel
#' @import Rcpp

rgCor <- function(X, Y) {
  
  res <- .Call('rcor', X, Y, PACKAGE = "robgpu")
  
  return(res)
}

