#ifndef SCGMM_HH
#define SCGMM_HH

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
#include "laexcp.h"

#include "HCL_LineSearch_d.h"
#include "HCL_UMin_lbfgs_d.h"
#include "HCL_Rn_d.h"

#include "FeatureBuffer.hh"
#include "LinearAlgebra.hh"


class Scgmm {

public:

  class Gaussian {
  public:
    LaVectorDouble lambda;
    double K_value;
    void resize(const unsigned int basis_dim) {
      lambda.resize(basis_dim,1);
    };
  };

  int m_fea_dim;
  int m_vec_dim;
  int m_exp_dim;
  std::vector<LaVectorDouble> basis_theta;
  std::vector<LaVectorDouble> basis_psi;
  std::vector<LaGenMatDouble> basis_P;
  std::vector<LaVectorDouble> basis_Pvec;
  std::vector<Gaussian> gaussians;
  std::vector<double> likelihoods;
  LaVectorDouble quadratic_feas;

  Scgmm() {};
  ~Scgmm() {};
  
  inline unsigned int fea_dim() { return m_fea_dim; };
  inline unsigned int exp_dim() { return m_exp_dim; };
  inline int basis_dim() { return basis_theta.size(); };
  inline unsigned int num_gaussians() { return gaussians.size(); };
  inline void remove_gaussian(unsigned int pos)
  { 
    std::vector<Gaussian>::iterator it;
    if (pos<gaussians.size())
      gaussians.erase(it+pos);     
  };
  
  void offline_computations();
  void precompute(const FeatureVec &feature);
  void compute_all_likelihoods(const FeatureVec &feature, std::vector<float> &lls);
  double compute_likelihood(const int k, const FeatureVec &feature);

  void copy(const Scgmm &orig);

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
			    LaGenMatDouble &covariance);
  
  void calculate_covariance(const HCL_RnVector_d &lambda,
			    LaVectorDouble &covariance);

  void calculate_psi(const LaVectorDouble &lambda,
		     LaVectorDouble &psi);
  
  void calculate_psi(const HCL_RnVector_d &lambda,
		     LaVectorDouble &psi);

  void calculate_mu(const LaVectorDouble &lambda,
		    LaVectorDouble &mu);

  void calculate_mu(const HCL_RnVector_d &lambda,
		    LaVectorDouble &mu);

  void calculate_theta(const LaVectorDouble &lambda,
		       LaVectorDouble &theta);

  void calculate_theta(const HCL_RnVector_d &lambda,
		       LaVectorDouble &theta);

  void read_gk(const std::string &filename);
  void write_gk(const std::string &filename);

  void read_basis(const std::string &filename);
  void write_basis(const std::string &filename);

  void reset(const unsigned int basis_dim, 
	     const unsigned int fea_dim,
	     const unsigned int num_gaussians);

  void reset_basis(const unsigned int basis_dim, 
		   const unsigned int fea_dim);

  void resize(int fea_dim, int basis_dim) {
    for (unsigned int i=0; i<num_gaussians(); i++)
      gaussians.at(i).resize(basis_dim);
  };

  void initialize_basis_pca(const std::vector<double> &weights,
			    const std::vector<LaGenMatDouble> &covs,
			    const std::vector<LaVectorDouble> &means,
			    const unsigned int basis_dim);

  double G(const LaVectorDouble &mean,
	   const LaGenMatDouble &precision,
	   const LaVectorDouble &sample_mean,
	   const LaGenMatDouble &sample_cov);

  double K(const LaGenMatDouble &sample_cov,
	   const LaVectorDouble &sample_mean);

  double K(const LaVectorDouble &theta);

  double H(const LaVectorDouble &theta,
	   const LaVectorDouble &f);

  void gradient_untied(const HCL_RnVector_d &lambda,
		       const LaVectorDouble &mean,
		       const LaGenMatDouble &secondmoment,
		       HCL_RnVector_d &grad,
		       bool affine);

  void limit_line_search(const LaGenMatDouble &R,
			 const LaGenMatDouble &curr_prec_estimate, 
			 LaVectorDouble &eigvals,
			 LaGenMatDouble &eigvecs,
			 double &max_interval);

  double eval_linesearch_value(const LaVectorDouble &eigs,
			       const LaVectorDouble &v,
			       const LaVectorDouble &dv,
			       double step,
			       double beta);

  double eval_linesearch_derivative(const LaVectorDouble &eigs,
				    const LaVectorDouble &v,
				    const LaVectorDouble &dv,
				    double step,
				    double beta);

  double kullback_leibler(const HCL_RnVector_d &lambda1,
			  const HCL_RnVector_d &lambda2);

  double kullback_leibler(const LaVectorDouble &mu1,
			  const LaGenMatDouble &sigma1,
			  const LaVectorDouble &mu2,
			  const LaGenMatDouble &sigma2);

  void f_to_gaussian_params(const LaVectorDouble &f,
			    LaVectorDouble &sample_mu,
			    LaGenMatDouble &sample_sigma);

  void gaussian_params_to_f(const LaVectorDouble &sample_mu,
			    const LaGenMatDouble &sample_sigma,
			    LaVectorDouble &f);
  
  void theta_to_gaussian_params(const LaVectorDouble &theta,
				LaVectorDouble &mu,
				LaGenMatDouble &sigma);
  
private:

  // The feature that has been used in precomputation
  FeatureVec precomputation_feature;  
};





class ScgmmLambdaFcnl : public HCL_Functional_d {

public:
  ScgmmLambdaFcnl(HCL_RnSpace_d &vs,
		  int basis_dim,
		  Scgmm &scgmm,
		  const LaGenMatDouble &sample_cov,
		  const LaVectorDouble &sample_mean,
		  bool aff=false);
  ~ScgmmLambdaFcnl();

  ostream & Write(ostream & o) const;
  HCL_VectorSpace_d & Domain() const;
  
  // RETURN NEGATIVE VALUES BECAUSE HCL DOES MINIMIZATION!
  virtual double Value1(const HCL_Vector_d &x) const;
  virtual void Gradient1(const HCL_Vector_d & x,
			 HCL_Vector_d & g) const;
  virtual void HessianImage(const HCL_Vector_d & x, const HCL_Vector_d & dx,
			    HCL_Vector_d & dy ) const;
  virtual double LineSearchValue(double mu) const;
  virtual double LineSearchDerivative(double mu) const;  
  virtual void SetLineSearchStartingPoint(const HCL_Vector_d &base);
  // ASSUMES: m_precision and m_psi are set == SetLineSearchStartingPoint() called
  virtual void SetLineSearchDirection(const HCL_Vector_d & dir);
  virtual double MaxStep(const HCL_Vector_d & x, const HCL_Vector_d & dir) const;

private:
  double m_fval_starting_point;
  double m_max_step;
  double m_beta;
  double m_temp;
  bool m_affine;

  Scgmm &m_scgmm;
  
  // Optimization related
  LaVectorDouble m_eigvals;
  LaGenMatDouble m_eigvecs;
  LaVectorDouble m_v;
  LaVectorDouble m_dv;
  LaGenMatDouble m_precision;
  LaVectorDouble m_psi;
  LaVectorDouble m_theta;
  LaGenMatDouble m_R;
  
  // Collected statistics
  LaVectorDouble m_f;
  LaVectorDouble m_sample_mean;
  LaGenMatDouble m_sample_cov;
  LaGenMatDouble m_sample_secondmoment;
  
  // Modifiers in the basis space
  HCL_RnSpace_d &m_vs;
  HCL_RnVector_d m_base;
  HCL_RnVector_d m_dir;
};


#endif /* SCGMM_HH */
