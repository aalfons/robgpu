\name{rgColSDs}
\alias{rgColSDs}
\title{GPU implementation of colwise standard deviation computation}
\usage{
  rgColSDs(X)
}
\arguments{
  \item{X}{a numeric matrix.}
}
\value{
  A vector containing the sd for each column of the input
  matrix.
}
\description{
  GPU implementation of colwise standard deviation
  computation
}
\examples{
X <- matrix(runif(16), 4, 4)
rgColSDs(X)
}
\author{
  Roland Boubela
}
\keyword{gpu}
\keyword{parallel}

