\name{rgColSums}
\alias{rgColSums}
\title{GPU implementation of colwise sum computation}
\usage{
  rgColSums(X)
}
\arguments{
  \item{X}{a numeric matrix.}
}
\value{
  A vector containing the sum for each column of the input
  matrix.
}
\description{
  GPU implementation of colwise sum computation
}
\examples{
X <- matrix(runif(16), 4, 4)
rgColSums(X)
}
\author{
  Roland Boubela
}
\keyword{gpu}
\keyword{parallel}

