/*
	A simple k-means clustering implementation
*/

#include <vector>
#include <algorithm>
#include <map>

#define __using_rcpp // for use when linking to R: comment out otherwise!

#ifdef __using_rcpp  // technically, the Rcpp attributes could go anywhere since they are just comments. But let's keep them organized
// R-specific includes
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(cpp11)]]

#else
// the joy of file IO
#include <cstring> // strcmp
#include <iostream>
#include <sstream> // std::ostringstream, not strictly needed
#include <fstream> // std::ofstream
#include <Eigen/Dense>

#endif

using VecRef = const Eigen::Ref<const Eigen::VectorXd>&;
using MatRef = const Eigen::Ref<const Eigen::MatrixXd>&;
using VeiRef = const Eigen::Ref<const Eigen::VectorXi>&;

using namespace Eigen;

// tabulate the entries of an iterable
template<class T>
std::map<T,size_t> table(const std::vector<T>& v){
	std::map<T,size_t> M;
	for(auto it = v.cbegin();it != v.cend();++it){
		if(M.find(*it)==M.end()) M[*it] = 1;
		else M[*it] += 1;
	}
	return M;
}

// version accepting an Eigen::Vector. 
// The template signature of a Matrix is <type,rows,cols>
template<class T>
std::map<T,size_t> table(const Eigen::Matrix<T,Eigen::Dynamic,1>& v){
	std::map<T,size_t> M;
	T val;
	for(int i=0;i<v.size();i++){
		val = v.coeff(i);
		if(M.find(val)==M.end()) M[val] = 1;
		else M[val] += 1;
	}
	return M;
}

// print current class tabulation
template<class T>
void print(const std::map<T,size_t>& m){
	std::cout << "Value | #\n---------" << std::endl;
	for(auto it = m.cbegin();it != m.cend();++it){
		std::cout << it->first << " | " << it->second  << std::endl;
	}
	std::cout << "---------" << std::endl;
}

// calculate distance between two points (an observation and a centroid)
inline double dist(VecRef v1,VecRef v2,unsigned int norm){
	const VectorXd diff = v1-v2;
	switch(norm){
		case 0:  // '0' maps to minimum distance between point and centroid
			return diff.array().abs().minCoeff();
		break;
		
		case 1: // L1-distance
			return diff.lpNorm<1>();
		break;
		
		case 2: // Euclidean distance (or its squared equivalent)
			return diff.lpNorm<2>();
		break;
		
		case 3: // max. distance
			return diff.lpNorm<Eigen::Infinity>();
		break;
		default:
			return diff.lpNorm<2>();
	}
}

// given the current centroids, update each observations' membership according to the criterion
// centroids are identified as their (row) index in the centroids matrix
VectorXi updateMembers(MatRef X,MatRef centroids,const unsigned int d){
	const int n = X.rows(), k = centroids.rows();
	VectorXd dists(k);
	VectorXd::Index ix;
	VectorXi classes(n);
	for(int i=0;i<n;i++){
		for(int j=0;j<k;j++){
			dists(j) = dist(X.row(i),centroids.row(j),d);
		}
		double minDist = dists.minCoeff(&ix); // compiler will say 'not used' but that's OK.
		classes(i) = ix;
	}
	return classes;
}

// given the current membership, update the centroids
MatrixXd updateCentroids(MatRef X,VeiRef classes){
	const int k = classes.maxCoeff()+1, n = X.rows(), p = X.cols(); // class labels are zero-indexed
	ArrayXXd psums(MatrixXd::Zero(k,p));
	ArrayXd counts(VectorXd::Zero(k));
	for(int i=0;i<n;i++){
		psums.row(classes(i)) += X.row(i).array();
		counts(classes(i)) += 1.0;
	}
	return psums.colwise()/counts;
}

VectorXi kcluster(MatRef X,const unsigned int d,const size_t k,int max_it){
	const size_t n = X.rows(), p = X.cols();
	VectorXi classes(n),oldClass(VectorXi::Zero(n));
	MatrixXd centroids(k,p);
	int num_changes = n, nit = 0;  // return when num_changes is zero or constant
	for(size_t i=0;i<n;i++) classes(i) = i%k; // initialize classes between 0 and k-1: for different 'random' initializations, shuffle input data
	while(num_changes > 0 && nit < max_it){
		nit++;
		oldClass.noalias() = classes;
		centroids = updateCentroids(X,classes);
		classes = updateMembers(X,centroids,d);
		num_changes = (oldClass.array() != classes.array()).count();
	}
	return classes;
}


#ifdef __using_rcpp

//[[Rcpp::export]]
Rcpp::List kmcluster(Rcpp::NumericMatrix x,unsigned int d,size_t k,size_t maxit = 50){
	const Eigen::Map<Eigen::MatrixXd> X(&x[0],x.nrow(),x.ncol());
	VectorXi classes(kcluster(X,d,k,static_cast<int>(maxit)));
	MatrixXd centroids = updateCentroids(X,classes);
	classes += MatrixXi::Constant(X.rows(),1,1); // bump up so that class labels are {1,...,k}
	return Rcpp::List::create(
		Rcpp::Named("classes") = classes,
		Rcpp::Named("centroids") = centroids
	);
}

#else


int main(int argc,const char *argv[]){
	if(argc < 3 || !strcmp(argv[1],"-h") || !strcmp(argv[1],"--help")){
		std::cout << "Usage:  kclus k N d [p], where\n  k = number of clusters,\n  N = sample size,\n  d = distance metric used,\n  " << 
			"and (optional) p = dimension of X" << std::endl;
		return 0;
	}
	const size_t k = strtoul(argv[1],NULL,0), N = strtoul(argv[2],NULL,0);
	const size_t d = (argc > 3) ? strtoul(argv[3],NULL,0) : 2;
	const size_t p = (argc > 4) ? strtoul(argv[4],NULL,0) : 4; // columns in X
	MatrixXd XX(MatrixXd::Random(N,p)); // copping out of actually impementing file parsing in C++
	VectorXi est_class = kcluster(XX,d,k);
	auto ctab = table(est_class);
	std::cout << "The estimated class distribution:" << std::endl;
	print(ctab);
	// write results to csv
	std::ofstream f;
	f.open("cluster_output.txt",std::fstream::out);
	f << "Class, ";
	for(size_t j=0;j<p;++j) f << "X" << j << ", ";
	f << "\n";
	for(size_t i=0;i<N;i++){
		std::ostringstream line;
		line << est_class(i);
		for(size_t j=0;j<p;j++) line << ", " << XX(i,j);
		line << "\n";
		f << line.str();
	}
	f.close();
	return 0;
}

#endif

