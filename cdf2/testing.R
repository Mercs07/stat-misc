# calculating + plotting a bivariate cdf (empirical or model-based)

library(Rcpp)
library(viridis)
library(dplyr)
library(ggplot2)
library(microbenchmark)

# how shall we plot it?

plot_ecdf = function(e2){
  par(mar = c(2,2,1,1),mgp = c(3,0.67,0))
  image(x=e2$ux,y=e2$uy,z=e2$cdf,col=viridis::viridis(40),las=1)
}

# R versions

# 'naive' version - rather slow
ecdf1 = function(U,ff){
  stopifnot(nrow(U)==length(ff),ncol(U)==2)
  ux = sort(unique(U[,1])); uy = sort(unique(U[,2]))
  res = matrix(0,nrow=length(ux),ncol=length(uy))
  for(i in seq_along(ux)){
    for(j in seq_along(uy)){
      res[i,j] = sum(ff[U[,1]<=ux[i] & U[,2]<=uy[j]])
    }
  }
  list("ux"=ux,"uy"=uy,"cdf"=res)
}

# second try with R:
ecdf2 = function(U,ff){
  stopifnot(nrow(U)==length(ff),ncol(U)==2)
  ds = data.frame(x = U[,1],y=U[,2],f=ff) %>%
    dplyr::arrange(x,y)
  UX = unique(ds$x); UY = sort(unique(ds$y))
  ds$uj = sapply(ds$y,function(z)which(UY==z))
  nX = length(UX); nY = length(UY)
  xloc = c(1,1+cumsum(rle(ds$x)$lengths))
  res = matrix(0,nrow=nX,ncol=nY)
  cf = rep(0,nY)
  for(i in 1:nX){
    fi = rep(0,nY)
    xi = xloc[i]:(xloc[i+1]-1)
    fi[ds$uj[xi]] = ds$f[xi]
    cf = cf + cumsum(fi)
    res[i,] = cf
  }
  list("ux" = UX,"uy" = UY,"cdf" = res)
}

# third try in C++: exports cdF()
sourceCpp("cdf2.cpp")

N = 5000
xy1 = cbind(rpois(N,10),rnorm(N))
xy2 = cbind(rgamma(N,2,1.4),rnorm(N))
xy3 = cbind(xy1[,2],xy1[,1])

ff = runif(nrow(unique(xy1)),0,1)
ff = ff/sum(ff) # it's a distribution!

r1 = ecdf1(xy1,ff)
r2 = ecdf2(xy1,ff)
r3 = cdF(xy1,ff,0,TRUE)

all.equal(r1$cdf,r2$cdf)
all.equal(r1$cdf,r3$cdf)

plot_ecdf(r3)