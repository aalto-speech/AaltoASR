#ifndef PCGMM_HH
#define PCGMM_HH

#include <vector>
#include <iostream>
#include <fstream>
#include <cassert>
#include <cfloat>
#include <clocale>
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

#include "HCL_LineSearch_d.h"
#include "HCL_UMin_lbfgs_d.h"
#include "HCL_Rn_d.h"

#include "FeatureBuffer.hh"
#include "LinearAlgebra.hh"


class Pcgmm {

public:
  
  class Gaussian {
  public:
    LaVectorDouble mu;
    LaVectorDouble lambda;
    double bias;
    LaVectorDouble linear_weight;
    
    void resize(const unsigned int fea_dim,
		const unsigned int basis_dim) {
      mu.resize(fea_dim,1);
      linear_weight.resize(fea_dim,1);
      lambda.resize(basis_dim,1);
    };
  };
  
  std::vector<LaGenMatDouble> mbasis;
  std::vector<LaVectorDouble> vbasis;
  std::vector<Gaussian> gaussians;
  std::vector<double> likelihoods;
  LaVectorDouble quadratic_feas;
  
  Pcgmm() {};
  ~Pcgmm() {};
  
  inline unsigned int fea_dim() { return mbasis[0].rows(); };
  inline int basis_dim() { return mbasis.size(); };
  inline unsigned int num_gaussians() { return gaussians.size(); };
  inline void remove_gaussian(unsigned int pos)
  { 
    std::vector<Gaussian>::iterator it;
    if (pos<gaussians.size())
      gaussians.erase(it+pos);     
  };
  
  void precompute();

  void compute_likelihoods(const FeatureVec &feature,
			   std::vector<float> &lls);

  double gaussian_likelihood(const int k);
  
  void copy(const Pcgmm &orig);

  Gaussian &get_gaussian(int k) { return gaussians.at(k); }
  
  void calculate_precision(const LaVectorDouble &lambda,
			   LaGenMatDouble &precision);
  
  void calculate_precision(const LaVectorDouble &lambda,
			   LaVectorDouble &precision);

  void calculate_precision(const HCL_RnVector_d &lambda,
			   LaGenMatDouble &precision);
  
  void calculate_precision(const HCL_RnVector_d &lambda,
			   LaVectorDouble &precision);
  
  void calculate_covariance(const LaVectorDouble &lambda,
			    LaGenMatDouble &covariance);

  void calculate_covariance(const LaVectorDouble &lambda,
			    LaVectorDouble &covariance);

  void calculate_covariance(const HCL_RnVector_d &lambda,
			    LaVectorDouble &covariance);

  void calculate_covariance(const HCL_RnVector_d &lambda,
			    LaGenMatDouble &covariance);

  void read_gk(const std::string &filename);
  void write_gk(const std::string &filename);

  void read_basis(const std::string &filename);
  void write_basis(const std::string &filename);
  
  void reset(const unsigned int basis_dim, 
	     const unsigned int dim,
	     const unsigned int gaussians);

  void reset_basis(const unsigned int basis_dim, 
		   const unsigned int dim);

  void initialize_basis_pca(const std::vector<double> &weights, 
			    const std::vector<LaGenMatDouble> &sample_covs, 
			    const unsigned int basis_dim);
  
  void gradient_untied(const HCL_RnVector_d &lambda,
		       const LaGenMatDouble &sample_cov,
		       HCL_RnVector_d &grad,
		       bool affine);
  
  void limit_line_search(const LaGenMatDouble &R,
			 const LaGenMatDouble &curr_prec_estimate, 
			 LaVectorDouble &eigs,
			 double &max_interval);
  
  void optimize_lambda(const LaGenMatDouble &sample_cov,
		       LaVectorDouble &lambda);
  
  void optimize_basis(const std::vector<LaGenMatDouble> &sample_covs,
		      const std::vector<LaVectorDouble> &lambda,
		      const std::vector<double> &c);

  double G(const LaGenMatDouble &precision,
	   const LaGenMatDouble &sample_cov);
  
  double eval_linesearch_value(const LaVectorDouble &eigs,
			       double step,
			       double beta,
			       double c);
  
  double eval_linesearch_derivative(const LaVectorDouble &eigs,
				    double step,
				    double beta);

  double kullback_leibler_covariance(const LaGenMatDouble &sigma1,
				     const LaGenMatDouble &sigma2);
  
private:

  // For temporary stuff
  LaGenMatDouble matrix_t1;
  LaGenMatDouble matrix_t2;
  LaVectorDouble vector_t1;
  LaVectorDouble vector_t2;
};




// Define A HCL Functional for optimizing untied parameters
class PcgmmLambdaFcnl : public HCL_Functional_d {

public:
  
  PcgmmLambdaFcnl(HCL_RnSpace_d &vs,
		  int basis_dim,
		  Pcgmm &pcgmm,
		  const LaGenMatDouble &sample_cov,
		  bool affine);
  
  ~PcgmmLambdaFcnl();
  
  ostream & Write(ostream & o) const;
  
  HCL_VectorSpace_d & Domain() const;
  
  virtual double Value1(const HCL_Vector_d &x) const;
  
  virtual void Gradient1(const HCL_Vector_d & x,
			 HCL_Vector_d & g) const;
  
  virtual void HessianImage(const HCL_Vector_d & x,
			    const HCL_Vector_d & dx,
			    HCL_Vector_d & dy ) const;
  
  virtual double LineSearchValue(double mu) const;
  
  virtual double LineSearchDerivative(double mu) const;
  
  virtual void SetLineSearchStartingPoint(const HCL_Vector_d &base);
  
  virtual void SetLineSearchDirection(const HCL_Vector_d & dir);
  
  virtual double MaxStep(const HCL_Vector_d & x, 
			 const HCL_Vector_d & dir) const;

  
private:
  double m_fval_starting_point;
  double m_fval_temp;
  double m_max_step;
  double m_beta;
  double m_linevalue_const;
  double m_temp;
  bool m_affine;

  Pcgmm &m_pcgmm;
  
  LaVectorDouble m_eigs;
  LaGenMatDouble m_precision;
  LaGenMatDouble m_R;
  LaGenMatDouble m_sample_cov;
  
  HCL_RnSpace_d &m_vs;
  HCL_RnVector_d m_base;
  HCL_RnVector_d m_dir;
};


#endif /* PCGMM_HH */
