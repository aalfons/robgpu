\name{rgColMeans}
\alias{rgColMeans}
\title{GPU implementation of colwise mean computation}
\usage{
  rgColMeans(X)
}
\arguments{
  \item{X}{a numeric matrix.}
}
\value{
  A vector containing the mean for each column of the input
  matrix.
}
\description{
  GPU implementation of colwise mean computation
}
\examples{
X <- matrix(runif(16), 4, 4)
rgColMeans(X)
}
\author{
  Roland Boubela
}
\keyword{gpu}
\keyword{parallel}

