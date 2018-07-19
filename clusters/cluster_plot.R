# checking the output of the naive k-means implementation in C++/Eigen

library(Rcpp)
library(RColorBrewer)

setwd("C:/Users/skm/Dropbox/CNN/src")

sourceCpp("clusters.cpp")

# comparing with kmeans - from the documentation it's not clear what the
# different 'algorithm' options are, or if they use different distance metrics

X = cbind(rnorm(500,0,2),rgamma(500,4,2))

nClass = 6
f1 = kmeans(X,nClass)
f2 = kmcluster(X,2,nClass) # '2' for Euclidean distance

pc = brewer.pal(nClass,"Dark2")

plot(X,col = scales::alpha(pc[f1$cluster],0.67),
     pch = 19,las=1,main = 'stats::kmeans result')
points(f1$centers,pch = 23,cex = 2,bg = pc)

plot(X,col = scales::alpha(pc[f2$classes],0.67),
     pch = 19,las=1,main = 'kmcluster (L2) result')
points(f2$centroids,pch = 23,cex = 2,bg = pc)

# using L-1 norm?
f3 = kmcluster(X,1,nClass)
plot(X,col = scales::alpha(pc[f3$classes],0.67),
     pch = 19,las=1,main = 'kmcluster (L1) result')
points(f3$centroids,pch = 23,cex = 2,bg = pc)