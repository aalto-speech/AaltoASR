#ifndef PCGMM_HH
#define PCGMM_HH

#include <vector>
#include <iostream>
#include <fstream>
#include <cassert>
#include <cfloat>
#include <cstdlib>
#include <cstdio>
#include <string>

#include "fmd.h"
#include "gmd.h"
#include "blas1pp.h"
#include "blas2pp.h"
#include "blas3pp.h"
#include "lasvd.h"
#include "laslv.h"
#include "lavli.h"
#include "spdmd.h"

#include "FeatureBuffer.hh"

class Pcgmm {

  class Gaussian {
  public:
    LaVectorDouble mu;
    LaVectorDouble lambda;
    void resize(const unsigned int fea_dim,
		const unsigned int basis_dim) {
      mu.resize(fea_dim,1);
      lambda.resize(basis_dim,1);
    };
  };
  
private:

  std::vector<LaGenMatDouble> mbasis;
  std::vector<LaVectorDouble> vbasis;
  std::vector<Gaussian> gaussians;

public:

  Pcgmm() {};
  ~Pcgmm() {};
  
  inline unsigned int fea_dim() { return gaussians[0].mu.size(); };
  inline int basis_dim() { return mbasis.size(); };
  inline unsigned int num_gaussians() { return gaussians.size(); };

  void precompute(const FeatureVec &feature);

  double gaussian_likelihood(const int k);

  void copy(const Pcgmm &orig);

  void calculate_precision(const LaVectorDouble &lambda,
			   LaGenMatDouble &precision);

  void calculate_precision(const LaVectorDouble &lambda,
			   LaVectorDouble &precision);

  void calculate_covariance(const LaVectorDouble &lambda,
			   LaGenMatDouble &covariance);

  void calculate_covariance(const LaVectorDouble &lambda,
			   LaVectorDouble &covariance);

  void read_gk(const std::string &filename);
  void write_gk(const std::string &filename);

  void reset_basis(const unsigned int basis_dim, 
		  const unsigned int dim);

  void resize(int fea_dim, int basis_dim) {
    for (unsigned int i=0; i<num_gaussians(); i++)
      gaussians.at(i).resize(fea_dim, basis_dim);
  };

  void initialize_basis_svd(const std::vector<int> &weights, 
			    const std::vector<LaGenMatDouble> &sample_covs, 
			    const unsigned int basis_dim);
  
  void train_precision_polak_ribiere(int state, 
				     LaGenMatDouble &sample_cov);

  void polak_ribiere_direction(const LaVectorDouble &old_grad,
			       const LaVectorDouble &new_grad,
			       const LaVectorDouble &old_direction,
			       LaVectorDouble &new_direction);
  
  void gradient(const LaVectorDouble &lambda,
		const LaGenMatDouble &sample_cov,
		LaVectorDouble &grad);
  
  void limit_line_search(const LaGenMatDouble &R,
			 const LaGenMatDouble &curr_prec_estimate, 
			 double &min_interval,
			 double &max_interval);

  double line_search_more_thuente(const LaGenMatDouble &P,
				  const LaGenMatDouble &R,
				  const LaGenMatDouble &sample_cov,
				  const double min_interval,
				  const double max_interval,
				  int iter = 30,
				  double trial_init = 1.1,
				  double trial_add = 0.1);
  
  inline double eval_aux_function(const LaVectorDouble &eigs,
				  double step,
				  double beta);
  
  inline double eval_aux_function_derivative(const LaVectorDouble &eigs,
					     double step,
					     double beta);
  
  void map_m2v(const LaGenMatDouble &m,
	       LaVectorDouble &v);

  void map_v2m(const LaVectorDouble &v, 
	       LaGenMatDouble &m);
  
  void generalized_eigenvalues(const LaGenMatDouble &A, 
			       const LaGenMatDouble &B, 
			       LaVectorDouble &eigs);
  
  void cholesky_factor(const LaGenMatDouble &A,
		       LaGenMatDouble &B);
  
  bool is_spd(const LaGenMatDouble &A);
};


#endif /* PCGMM_HH */
