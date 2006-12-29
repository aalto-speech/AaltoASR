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
  
private:
  // Define A HCL Functional for optimizing untied parameters
  class PcgmmLambdaFcnl : public HCL_Functional_d {
  public:
    PcgmmLambdaFcnl(int basis_dim,
		    Pcgmm *pcgmm,
		    const LaGenMatDouble &sample_cov) 
      : m_sample_cov(sample_cov) 
    { 
      assert(sample_cov.rows()==sample_cov.cols());

      m_pcgmm=pcgmm;
      m_vs=new HCL_RnSpace_d(basis_dim); 
      m_base=(HCL_RnVector_d*)m_vs->Member();
      m_dir=(HCL_RnVector_d*)m_vs->Member();
      m_precision.resize(sample_cov.rows(), sample_cov.cols());
      m_eigs.resize(sample_cov.rows(),1);
      m_R.resize(sample_cov.rows(), sample_cov.cols());
    };

    ~PcgmmLambdaFcnl() {
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

    virtual double Value1(const HCL_Vector_d &x) const {
      double result;
      LaGenMatDouble t;
      m_pcgmm->calculate_precision((HCL_RnVector_d&)x, t);
      assert(LinearAlgebra::is_spd(t));
      result=m_pcgmm->G(t, m_sample_cov);
      // RETURN NEGATIVE VALUES BECAUSE HCL DOES MINIMIZATION!
      return -result;
    };

    virtual void Gradient1(const HCL_Vector_d & x,
			   HCL_Vector_d & g) const {
      m_pcgmm->gradient_untied((HCL_RnVector_d&)x,
			       m_sample_cov,
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
      result=m_pcgmm->eval_linesearch_value(m_eigs, mu, m_beta, m_linevalue_const)
	+m_fval_starting_point;
      return -result;
    };

    // RETURN NEGATIVE VALUES BECAUSE HCL DOES MINIMIZATION!
    virtual double LineSearchDerivative(double mu) const {
      double result;
      result = m_pcgmm->eval_linesearch_derivative(m_eigs, mu, m_beta);
      return -result;
    };

    virtual void SetLineSearchStartingPoint(const HCL_Vector_d &base)
    {
      m_base->Copy(base);
      m_pcgmm->calculate_precision(*m_base, m_precision);
      assert(LinearAlgebra::is_spd(m_precision));
      m_fval_starting_point=m_pcgmm->G(m_precision, m_sample_cov);
      m_linevalue_const=-log(LinearAlgebra::determinant(m_precision));
    };

    virtual void SetLineSearchDirection(const HCL_Vector_d & dir)
    {
      m_dir->Copy(dir);
      m_pcgmm->calculate_precision(*m_dir, m_R);
      m_pcgmm->limit_line_search(m_R, m_precision, m_eigs, m_max_step);
      LaGenMatDouble t(m_pcgmm->fea_dim(), m_pcgmm->fea_dim());
      Blas_Mat_Mat_Mult(m_sample_cov, m_R, t, 1.0, 0.0);
      m_beta=t.trace();
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
    double m_fval_temp;
    double m_max_step;
    double m_beta;
    double m_linevalue_const;
    double m_temp;

    Pcgmm *m_pcgmm;

    LaVectorDouble m_eigs;
    LaGenMatDouble m_precision;
    LaGenMatDouble m_R;
    LaGenMatDouble m_sample_cov;

    HCL_RnSpace_d *m_vs;
    HCL_RnVector_d *m_base;
    HCL_RnVector_d *m_dir;

    // Temporary modifiers
    LaGenMatDouble m_matrix_t1;
  };


public:

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
		       HCL_RnVector_d &grad);
  
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

  void set_hcl_bfgs_cfg_file(const std::string &filename) {
    hcl_grad_set=true;
    hcl_grad_cfg=std::string(filename);
  }
  
  void set_hcl_line_cfg_file(const std::string &filename) {
    hcl_line_set=true;
    hcl_line_cfg=std::string(filename);
  }

  double kullback_leibler_covariance(const LaGenMatDouble &sigma1,
				     const LaGenMatDouble &sigma2);
  
private:

  bool hcl_grad_set;
  bool hcl_line_set;
  std::string hcl_grad_cfg;
  std::string hcl_line_cfg;

  // For temporary stuff
  LaGenMatDouble matrix_t1;
  LaGenMatDouble matrix_t2;
  LaVectorDouble vector_t1;
  LaVectorDouble vector_t2;
};


#endif /* PCGMM_HH */
