## statistics-related miscellany

contents and brief descriptions:

***

1. `clusters` contains some simple code implemeting [_k_-means clustering](https://en.wikipedia.org/wiki/K-means_clustering). using the [Eigen C++ library](http://eigen.tuxfamily.org/index.php). Hardly a hot new topic! Also provides a simple example of easy toggling between a standalone C(++) program and functioning as a callable from another language (in this case, R via Rcpp).

Compilation needs `-std=c++11` or more recent and an include `-I'path-to-eigen'`. (and ensure that `__using_Rcpp` is defined or not, depending on the use case).

2. `cdf2` is a (hopefully) efficient impementation of calculating and plotting a bivarate empirical cdf. The concept is not very profound, but the algorithm is kind of interesting, and could certainly be improved.


