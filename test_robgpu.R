
R

require(robgpu)
require(gputools)

Y <- matrix(c(-4, 2, 5, -1), ncol = 1, nrow = 4)
X <- matrix(c(2, -1, 1, 1, -5, 2, -3, 1, -4, 1, -1, 1), ncol = 3, byrow = T)

lsfit(X, Y, intercept = F)$coef
leastsquares(X, Y)


# with intercept

Y <- matrix(c(-4, 2, 5, -1), ncol = 1, nrow = 4)
X <- matrix(c(1, 2, -1, 1, 1, 1, -5, 2, 1, -3, 1, -4, 1, 1, -1, 1), ncol = 4, byrow = T)

lsfit(X, Y, intercept = F)$coef
leastsquares(X, Y)


# big thing

R

require(robgpu)

nvariables <- 1e3
nobs <- 1e4

set.seed(0123)
Y <- matrix(rnorm(nobs), ncol = 1)
X <- matrix(rnorm(nvariables * nobs), ncol = nvariables, byrow = T)

system.time(lsfit(X, Y, intercept = F)$coef)
system.time(gpuLTS(X, Y, intercept = F)$coef)
system.time(gpuLsfit(X, Y, intercept = F)$coef)




