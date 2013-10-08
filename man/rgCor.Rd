\name{rgCor}
\alias{rgCor}
\title{GPU implementation of the Pearson Correlation Coefficient}
\usage{
  rgCor(X, Y)
}
\arguments{
  \item{X}{a numeric matrix.}

  \item{Y}{a numeric matrix.}
}
\value{
  The correlation matrix.
}
\description{
  GPU implementation of the Pearson Correlation Coefficient
}
\examples{
X <- matrix(runif(16), 4, 4)
Y <- matrix(runif(16), 4, 4)
rgCor(X, Y)
}
\author{
  Roland Boubela
}
\keyword{gpu}
\keyword{parallel}

