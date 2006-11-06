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

private:
  // Define A HCL Functional for optimizing untied parameters
  class ScgmmLambdaFcnl : public HCL_Functional_d {
  public:
    ScgmmLambdaFcnl(int basis_dim,
		    Scgmm *scgmm,
		    const LaGenMatDouble &sample_cov,
		    const LaVectorDouble &sample_mean)
    { 
      assert(sample_cov.rows()==sample_cov.cols());

      // Mean and Covariance
      m_sample_mean.copy(sample_mean);
      m_sample_cov.copy(sample_cov);      

      // Second moment
      m_sample_secondmoment.copy(sample_cov);
      Blas_R1_Update(m_sample_secondmoment, sample_mean, sample_mean, 1);

      scgmm->gaussian_params_to_f(sample_mean, sample_cov, m_f);     
      m_scgmm=scgmm;
      m_vs=new HCL_RnSpace_d(basis_dim);
      m_base=(HCL_RnVector_d*)m_vs->Member();
      m_dir=(HCL_RnVector_d*)m_vs->Member();
      m_precision.resize(sample_cov.rows(), sample_cov.cols());
      m_eigvals.resize(sample_cov.rows(),1);
      m_R.resize(sample_cov.rows(), sample_cov.cols());
    };
    
    ~ScgmmLambdaFcnl() {
      HCL_delete( m_base );
      HCL_delete( m_dir );
      HCL_delete( m_vs );
    }
    
    ostream & Write(ostream & o) const { 
      return o; 
    };
    
    HCL_VectorSpace_d & Domain() const { 
      return *m_vs;
    };

    // RETURN NEGATIVE VALUES BECAUSE HCL DOES MINIMIZATION!
    virtual double Value1(const HCL_Vector_d &x) const {
      double result;
      LaVectorDouble theta;
      m_scgmm->calculate_theta((HCL_RnVector_d&)x, theta);
      result = m_scgmm->H(theta, m_f);
      return -result;
    };
    
    virtual void Gradient1(const HCL_Vector_d & x,
			   HCL_Vector_d & g) const {
      m_scgmm->gradient_untied((HCL_RnVector_d&)x,
			       m_sample_mean,
			       m_sample_secondmoment,
			       (HCL_RnVector_d&)g);
      // RETURN NEGATIVE VALUES BECAUSE HCL DOES MINIMIZATION!
      g.Mul(-1);
    };
    
    virtual void HessianImage(const HCL_Vector_d & x,
			      const HCL_Vector_d & dx,
			      HCL_Vector_d & dy ) const {
      fprintf(stderr, "Warning, HessianImage not implemented");
    }
    
    // RETURN NEGATIVE VALUES BECAUSE HCL DOES MINIMIZATION!
    virtual double LineSearchValue(double mu) const {
      double result;
      result=m_scgmm->eval_linesearch_value(m_eigvals, m_v, m_dv, mu, m_beta)+m_fval_starting_point;
      return -result;
    };
    
    // RETURN NEGATIVE VALUES BECAUSE HCL DOES MINIMIZATION!
    virtual double LineSearchDerivative(double mu) const {
      double result=0;
      result=m_scgmm->eval_linesearch_derivative(m_eigvals, m_v, m_dv, mu, m_beta);
      return -result;
    };
    
    virtual void SetLineSearchStartingPoint(const HCL_Vector_d &base)
    {
      m_base->Copy(base);
      
      m_scgmm->calculate_precision(*m_base, m_precision);
      assert(LinearAlgebra::is_spd(m_precision));
      m_scgmm->calculate_psi(*m_base, m_psi);
      m_scgmm->calculate_theta(*m_base, m_theta);

      m_fval_starting_point=m_scgmm->H(m_theta, m_f);
    };


    // ASSUMES: m_precision and m_psi are set == SetLineSearchStartingPoint() called
    virtual void SetLineSearchDirection(const HCL_Vector_d & dir)
    {
      m_dir->Copy(dir);

      int d=m_scgmm->fea_dim();

      m_scgmm->calculate_precision(*m_dir, m_R);
      m_scgmm->limit_line_search(m_R, m_precision, m_eigvals, m_eigvecs, m_max_step);

      LaGenMatDouble t=LaGenMatDouble::zeros(d);
      LaGenMatDouble t2=LaGenMatDouble::zeros(d);
      LinearAlgebra::matrix_power(m_precision, t, -0.5);
      Blas_Mat_Mat_Mult(m_eigvecs, t, t2);

      m_v.resize(d,1);
      m_dv.resize(d,1);
      LaVectorDouble d_psi;
      m_scgmm->calculate_psi(*m_dir, d_psi);
      Blas_Mat_Vec_Mult(t2, m_psi, m_v);
      Blas_Mat_Vec_Mult(t2, d_psi, m_dv);

      LaVectorDouble d_theta;
      m_scgmm->calculate_theta(*m_dir, d_theta);
      m_beta = Blas_Dot_Prod(d_theta, m_f);
    };
    
    virtual double MaxStep(const HCL_Vector_d & x, 
			   const HCL_Vector_d & dir) const
    {
      for (int i=1; i<=x.Dim(); i++) {
	assert(((HCL_RnVector_d&)x)(i)==(*m_base)(i));
	assert(((HCL_RnVector_d&)dir)(i)==(*m_dir)(i));
      }
      return m_max_step;
    };
    
  private:
    double m_fval_starting_point;
    double m_max_step;
    double m_beta;
    double m_temp;

    Scgmm *m_scgmm;

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
    HCL_RnSpace_d *m_vs;
    HCL_RnVector_d *m_base;
    HCL_RnVector_d *m_dir;
  };

  
public:

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

  void precompute();
  
  void compute_likelihoods(const FeatureVec &feature,
			   std::vector<float> &lls);

  double gaussian_likelihood(const int k);
  
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
		       HCL_RnVector_d &grad);

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
  
  void optimize_lambda(const LaGenMatDouble &sample_cov,
		       const LaVectorDouble &sample_mean,
		       LaVectorDouble &lambda);

  void set_hcl_bfgs_cfg_file(const std::string &filename) {
    hcl_grad_set=true;
    hcl_grad_cfg=std::string(filename);
  }
  
  void set_hcl_line_cfg_file(const std::string &filename) {
    hcl_line_set=true;
    hcl_line_cfg=std::string(filename);
  }

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

  bool hcl_grad_set;
  bool hcl_line_set;
  std::string hcl_grad_cfg;
  std::string hcl_line_cfg;

  // For temporary stuff
  LaGenMatDouble matrix_t1;
  LaGenMatDouble matrix_t2;
  LaGenMatDouble matrix_t3;
  LaGenMatDouble matrix_t4;

  LaVectorDouble vector_t1;
  LaVectorDouble vector_t2;
  LaVectorDouble vector_t3;
  LaVectorDouble vector_t4;
};


#endif /* SCGMM_HH */
