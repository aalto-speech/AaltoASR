#ifndef SUBSPACES_HH
#define SUBSPACES_HH

#include <vector>
#include <iostream>
#include <fstream>
#include <cassert>
#include <cfloat>
#include <cstdlib>
#include <cstdio>
#include <string>

#include "HCL_LineSearch_d.h"
#include "HCL_UMin_lbfgs_d.h"
#include "HCL_Rn_d.h"

#include "FeatureBuffer.hh"
#include "LinearAlgebra.hh"



class PrecisionSubspace {

private:
  int m_subspace_dim;
  int m_feature_dim;
  std::vector<Vector> m_vspace;
  std::vector<Matrix> m_mspace;
  
  bool m_computed;
  Vector m_quadratic_features;
  HCL_LineSearch_MT_d *m_ls;
  HCL_UMin_lbfgs_d *m_bfgs;
  std::string m_ls_cfg_file;
  std::string m_bfgs_cfg_file;

public:

  PrecisionSubspace();
  PrecisionSubspace(int subspace_dim, int feature_dim);
  ~PrecisionSubspace();
  void set_subspace_dim(int subspace_dim);
  void set_feature_dim(int feature_dim);
  int subspace_dim() const;
  int feature_dim() const;
  double dotproduct(const Vector &lambda) const;
  void precompute(const Vector &f);
  void reset_cache() { m_computed = false; }
  bool computed() { return m_computed; }
  void optimize_coefficients(const Matrix &sample_cov,
                             Vector &lambda);
  void copy(const PrecisionSubspace &orig);
  void compute_precision(const LaVectorDouble &lambda,
                         LaGenMatDouble &precision);
  void compute_precision(const LaVectorDouble &lambda,
                         LaVectorDouble &precision);
  void compute_precision(const HCL_RnVector_d &lambda,
                         LaGenMatDouble &precision);
  void compute_precision(const HCL_RnVector_d &lambda,
                         LaVectorDouble &precision);
  void compute_covariance(const LaVectorDouble &lambda,
                          LaGenMatDouble &covariance);
  void compute_covariance(const LaVectorDouble &lambda,
                          LaVectorDouble &covariance);
  void compute_covariance(const HCL_RnVector_d &lambda,
                          LaVectorDouble &covariance);
  void compute_covariance(const HCL_RnVector_d &lambda,
                          LaGenMatDouble &covariance);
  void read_subspace(std::ifstream &in);
  void write_subspace(std::ofstream &out);
  void reset(const unsigned int subspace_dim,
             const unsigned int feature_dim);
  void initialize_basis_pca(std::vector<double> &weights,
			    std::vector<LaGenMatDouble> &sample_covs,
			    unsigned int basis_dim);
  void gradient_untied(const HCL_RnVector_d &lambda,
		       const LaGenMatDouble &sample_cov,
		       HCL_RnVector_d &grad,
		       bool affine);
  void limit_line_search(const LaGenMatDouble &R,
			 const LaGenMatDouble &curr_prec_estimate,
			 LaVectorDouble &eigs,
			 double &max_interval);
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

  void set_hcl_optimization(HCL_LineSearch_MT_d *ls,
                            HCL_UMin_lbfgs_d *bfgs,
                            std::string ls_cfg_file,
                            std::string bfgs_cfg_file);

  void ls_set_defaults();
  void bfgs_set_defaults();
  void ls_set_config();
  void bfgs_set_config();
};




// Define A HCL Functional for optimizing untied parameters
class PcgmmLambdaFcnl : public HCL_Functional_d {
  
public:
  
  PcgmmLambdaFcnl(HCL_RnSpace_d &vs,
		  int basis_dim,
		  PrecisionSubspace &pcgmm,
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

  PrecisionSubspace &m_pcgmm;
  
  LaVectorDouble m_eigs;
  LaGenMatDouble m_precision;
  LaGenMatDouble m_R;
  LaGenMatDouble m_sample_cov;
  
  HCL_RnSpace_d &m_vs;
  HCL_RnVector_d m_base;
  HCL_RnVector_d m_dir;
};



class ExponentialSubspace {

private:
  int m_subspace_dim;
  int m_feature_dim;
  int m_vectorized_dim;
  int m_exponential_dim;
  std::vector<Vector> m_basis_theta;
  std::vector<Vector> m_basis_psi;
  std::vector<Matrix> m_basis_P;
  std::vector<Vector> m_basis_Pvec;
  
  bool m_computed;
  Vector m_quadratic_features;
  HCL_UMin_lbfgs_d *m_bfgs;
  HCL_LineSearch_MT_d *m_ls;
  std::string m_ls_cfg_file;
  std::string m_bfgs_cfg_file;
  
public:

  ExponentialSubspace();
  ExponentialSubspace(int subspace_dim, int feature_dim);
  ~ExponentialSubspace();
  
  void set_subspace_dim(int subspace_dim);
  void set_feature_dim(int feature_dim);
  int subspace_dim() const;
  int feature_dim() const;
  int exponential_dim() const;
  double dotproduct(const Vector &lambda) const;
  void precompute(const Vector &f);
  void reset_cache() { m_computed = false; }
  bool computed() { return m_computed; }
  void optimize_coefficients(const Vector &sample_mean,
                             const Matrix &sample_cov,
                             Vector &lambda);
  inline unsigned int feature_dim() { return m_feature_dim; };
  inline unsigned int exponential_dim() { return m_exponential_dim; };
  inline int subspace_dim() { return m_subspace_dim; };
  
  void copy(const ExponentialSubspace &orig);
  void compute_precision(const Vector &lambda,
                         Matrix &precision);
  void compute_precision(const Vector &lambda,
                         Vector &precision);
  void compute_precision(const HCL_RnVector_d &lambda,
                         Matrix &precision);
  void compute_precision(const HCL_RnVector_d &lambda,
                         Vector &precision);
  void compute_covariance(const Vector &lambda,
                          Matrix &covariance);
  void compute_covariance(const Vector &lambda,
                          Vector &covariance);
  void compute_covariance(const HCL_RnVector_d &lambda,
                          Matrix &covariance);
  void compute_covariance(const HCL_RnVector_d &lambda,
                          Vector &covariance);
  void compute_psi(const Vector &lambda,
                   Vector &psi);
  void compute_psi(const HCL_RnVector_d &lambda,
                   Vector &psi);
  void compute_mu(const Vector &lambda,
                  Vector &mu);
  void compute_mu(const HCL_RnVector_d &lambda,
                  Vector &mu);
  void compute_theta(const Vector &lambda,
                     Vector &theta);
  void compute_theta(const HCL_RnVector_d &lambda,
                     Vector &theta);
  void read_subspace(std::ifstream &in);
  void write_subspace(std::ofstream &out);
  void reset(const unsigned int subspace_dim,
	     const unsigned int feature_dim);
  void initialize_basis_pca(std::vector<double> &weights,
			    std::vector<Matrix> &covs,
			    std::vector<Vector> &means,
			    unsigned int basis_dim);
  double K(const Matrix &sample_cov,
	   const Vector &sample_mean);
  double K(const Vector &theta);
  double H(const Vector &theta,
	   const Vector &f);
  void gradient_untied(const HCL_RnVector_d &lambda,
		       const Vector &mean,
		       const Matrix &secondmoment,
		       HCL_RnVector_d &grad,
		       bool affine);
  void limit_line_search(const Matrix &R,
			 const Matrix &curr_prec_estimate,
			 Vector &eigvals,
			 Matrix &eigvecs,
			 double &max_interval);
  double eval_linesearch_value(const Vector &eigs,
			       const Vector &v,
			       const Vector &dv,
			       double step,
			       double beta);
  double eval_linesearch_derivative(const Vector &eigs,
				    const Vector &v,
				    const Vector &dv,
				    double step,
				    double beta);
  void f_to_gaussian_params(const Vector &f,
			    Vector &sample_mu,
			    Matrix &sample_sigma);
  void gaussian_params_to_f(const Vector &sample_mu,
			    const Matrix &sample_sigma,
			    Vector &f);
  void theta_to_gaussian_params(const Vector &theta,
				Vector &mu,
				Matrix &sigma);

  void set_hcl_optimization(HCL_LineSearch_MT_d *ls,
                            HCL_UMin_lbfgs_d *bfgs,
                            std::string ls_cfg_file,
                            std::string bfgs_cfg_file);

  void ls_set_defaults();
  void bfgs_set_defaults();
  void ls_set_config();
  void bfgs_set_config();
};



class ScgmmLambdaFcnl : public HCL_Functional_d {

public:
  ScgmmLambdaFcnl(HCL_RnSpace_d &vs,
		  int basis_dim,
		  ExponentialSubspace &scgmm,
		  const Matrix &sample_cov,
		  const Vector &sample_mean,
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

  ExponentialSubspace &m_es;
  
  // Optimization related
  Vector m_eigvals;
  Matrix m_eigvecs;
  Vector m_v;
  Vector m_dv;
  Matrix m_precision;
  Vector m_psi;
  Vector m_theta;
  Matrix m_R;
  
  // Collected statistics
  Vector m_f;
  Vector m_sample_mean;
  Matrix m_sample_cov;
  Matrix m_sample_secondmoment;
  
  // Modifiers in the basis space
  HCL_RnSpace_d &m_vs;
  HCL_RnVector_d m_base;
  HCL_RnVector_d m_dir;
};


#endif /* SUBSPACES_HH */
