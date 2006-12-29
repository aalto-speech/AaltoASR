#include "Pcgmm.hh"



void
Pcgmm::copy(const Pcgmm &orig)
{
  assert(orig.mbasis.size()==orig.vbasis.size());

  mbasis.resize(orig.mbasis.size());
  vbasis.resize(orig.vbasis.size());
  gaussians.resize(orig.gaussians.size());
  likelihoods.resize(orig.gaussians.size());
  quadratic_feas.resize(orig.quadratic_feas.size(),1);

  for (unsigned int i=0; i<orig.mbasis.size(); i++) {
    mbasis.at(i).copy(orig.mbasis.at(i));
    vbasis.at(i).copy(orig.vbasis.at(i));
  }

  for (unsigned int i=0; i<orig.gaussians.size(); i++) {
    gaussians.at(i).mu.copy(orig.gaussians.at(i).mu);
    gaussians.at(i).lambda.copy(orig.gaussians.at(i).lambda);
    gaussians.at(i).linear_weight.copy(orig.gaussians.at(i).linear_weight);
    gaussians.at(i).bias=orig.gaussians.at(i).bias;
  }
}


void 
Pcgmm::precompute()
{
  for (unsigned int i=0; i<num_gaussians(); i++) {
    LaGenMatDouble t,t2;
    calculate_precision(gaussians.at(i).lambda, t);
    Blas_Mat_Vec_Mult(t, gaussians.at(i).mu, gaussians.at(i).linear_weight);
    t.scale(1/(2*3.1416));
    LinearAlgebra::matrix_power(t, t2, 0.5);
    gaussians.at(i).bias = log(LinearAlgebra::determinant(t2))
      - 0.5*Blas_Dot_Prod(gaussians.at(i).mu,gaussians.at(i).linear_weight);
  }
}


void 
Pcgmm::compute_likelihoods(const FeatureVec &feature,
			   std::vector<float> &lls)
{
  lls.resize(num_gaussians());

  // Convert feature vector to lapackpp vector
  LaVectorDouble f=LaVectorDouble(feature.dim());
  for (int i=0; i<feature.dim(); i++)
    f(i)=feature[i];

  // Compute 'quadratic features'
  for (int i=0; i<basis_dim(); i++) {
    LaVectorDouble y=LaVectorDouble(feature.dim());
    Blas_Mat_Vec_Mult(mbasis.at(i), f, y);
    quadratic_feas(i)=-0.5*Blas_Dot_Prod(f,y);
  }

  // Compute likelihoods
  for (unsigned int i=0; i<num_gaussians(); i++) {
    likelihoods.at(i)=gaussians.at(i).bias
      +Blas_Dot_Prod(gaussians.at(i).linear_weight,f)
      +Blas_Dot_Prod(gaussians.at(i).lambda,quadratic_feas);
    likelihoods.at(i)=exp(likelihoods.at(i));
    lls.at(i)=likelihoods.at(i);
  }
}


double 
Pcgmm::gaussian_likelihood(const int k)
{
  return likelihoods.at(k);
}


void 
Pcgmm::initialize_basis_pca(const std::vector<double> &c,
			    const std::vector<LaGenMatDouble> &sample_covs, 
			    const unsigned int basis_dim)
{
  assert(c.size() == sample_covs.size());

  unsigned int num_covs=sample_covs.size();
  int d=sample_covs.at(0).rows();
  int d_vec=(int)(d*(d+1)/2);
  double c_sum=0;
  
  LaGenMatDouble m=LaGenMatDouble::zeros(d);
  LaGenMatDouble m_sqrt=LaGenMatDouble::zeros(d);
  LaGenMatDouble m_neg_sqrt=LaGenMatDouble::zeros(d);
  LaGenMatDouble identity=LaGenMatDouble::eye(d);

  // Reset current basis
  reset(basis_dim, d, num_gaussians());

  // Calculate c_sum
  for (unsigned int i=0; i<num_covs; i++)
    c_sum+=c.at(i);

  // Calculate mean covariance m and sqrts
  for (unsigned int i=0; i<num_covs; i++)
    Blas_Mat_Mat_Mult(identity, sample_covs.at(i), m, c.at(i)/c_sum, 1);
  LinearAlgebra::matrix_power(m, m_sqrt, 0.5);
  LinearAlgebra::matrix_power(m, m_neg_sqrt, -0.5);
  
  // Calculate sample precisions and project them so that the inner
  // product <U,V>_m == Tr(mUmV) is preserved in vector mapping
  std::vector<LaGenMatDouble> sample_precs;
  sample_precs.resize(num_covs);
  for (unsigned int i=0; i<num_covs; i++) {
    // Calculate precisions
    sample_precs.at(i).copy(sample_covs.at(i));
    LaVectorLongInt pivots(d);
    LUFactorizeIP(sample_precs.at(i), pivots);
    LaLUInverseIP(sample_precs.at(i), pivots);
    // Normalize by calculating m^(-1/2) * P * m^(-1/2)
    matrix_t1.resize(d,d);
    Blas_Mat_Mat_Mult(m_neg_sqrt, sample_precs.at(i), matrix_t1, 1.0, 0.0);
    Blas_Mat_Mat_Mult(matrix_t1, m_neg_sqrt, sample_precs.at(i), 1.0, 0.0);
  }
  
  // Map sample precisions to vectors
  std::vector<LaVectorDouble> sample_prec_vectors;
  sample_prec_vectors.resize(num_covs);
  for (unsigned int i=0; i<num_covs; i++)
    LinearAlgebra::map_m2v(sample_precs.at(i), sample_prec_vectors.at(i));

  // Calculate covariance C and \rho_0
  LaGenMatDouble C=LaGenMatDouble::zeros(d_vec, d_vec);
  vector_t1.resize(d_vec,1);
  matrix_t1.resize(d_vec,d_vec);
  vector_t1(LaIndex())=0;
  matrix_t1(LaIndex(),LaIndex())=0;
  mbasis.at(0)(LaIndex(),LaIndex())=0;

  for (unsigned int i=0; i<num_covs; i++) {
    assert(sample_prec_vectors.at(i).rows()==d_vec);
    Blas_Mat_Mat_Mult(identity, sample_precs.at(i), mbasis.at(0), c.at(i)/c_sum , 1);
    Blas_Add_Mult(vector_t1, c.at(i)/c_sum , sample_prec_vectors.at(i));
    Blas_R1_Update(C, sample_prec_vectors.at(i), sample_prec_vectors.at(i), c.at(i)/c_sum);
  }
  // Remove mean squared
  Blas_R1_Update(matrix_t1, vector_t1, vector_t1, -1);

  // PCA for C
  vector_t1.resize(C.rows(),1);
  LaEigSolveSymmetricVecIP(C, vector_t1);

  // Other basis matrices are top eigenvectors of C
  // after renormalization
  vector_t1.resize(d,1);
  matrix_t1.resize(d,d);
  matrix_t2.resize(d,d);

  // S_0
  Blas_Mat_Mat_Mult(m_sqrt, mbasis.at(0), matrix_t1, 1.0, 0.0);
  Blas_Mat_Mat_Mult(matrix_t1, m_sqrt, mbasis.at(0), 1.0, 0.0);
  LinearAlgebra::map_m2v(mbasis.at(0), vbasis.at(0));
  // S_i, i=1,2,3...
  for (unsigned int i=1; i<basis_dim; i++) {
    vector_t1.copy(C.col(C.cols()-i));
    LinearAlgebra::map_v2m(vector_t1, matrix_t1);
    Blas_Mat_Mat_Mult(m_sqrt, matrix_t1, matrix_t2, 1.0, 0.0);
    Blas_Mat_Mat_Mult(matrix_t2, m_sqrt, mbasis.at(i), 1.0, 0.0);
    LinearAlgebra::map_m2v(mbasis.at(i), vbasis.at(i));
  }
}


void
Pcgmm::reset(const unsigned int basis_dim, 
	     const unsigned int d,
	     const unsigned int g)
{
  gaussians.resize(g);
  likelihoods.resize(g);

  reset_basis(basis_dim, d);
}


void
Pcgmm::reset_basis(const unsigned int basis_dim, 
		   const unsigned int d)
{
  unsigned int d_vec=(unsigned int)d*(d+1)/2;
  mbasis.resize(basis_dim);
  vbasis.resize(basis_dim);
  quadratic_feas.resize(basis_dim,1);

  for (unsigned int i=0; i<gaussians.size(); i++)
    gaussians.at(i).resize(d, basis_dim);

  for (unsigned int i=0; i<basis_dim; i++) {
    mbasis.at(i).resize(d,d);
    vbasis.at(i).resize(d_vec,1);
    for (unsigned int j=0; j<d; j++)
      for (unsigned int k=0; k<d; k++)
	(mbasis.at(i))(j,k)=0;
    for (unsigned int l=0; l<d_vec; l++)
      vbasis.at(i)(l)=0;
  }
}


void
Pcgmm::read_gk(const std::string &filename)
{
  std::ifstream in(filename.c_str());
  if (!in) {
    fprintf(stderr, "Pcgmm::read_gk(): could not open %s\n", 
	    filename.c_str());
    assert(false);
  }
  
  int gauss=0, basis_dim=0, fea_dim=0;
  std::string cov_str;

  // Read header line
  in >> gauss >> fea_dim >> cov_str;
  
  if (!(cov_str == "pcgmm")) {
    throw std::string("Model type not pcgmm");
    assert(false);
  }
  
  // Read precision basis
  in >> basis_dim;
  reset(basis_dim, fea_dim, gauss);
  for (int b=0; b<basis_dim; b++) {
    for (int i=0; i<fea_dim; i++)
      for (int j=0; j<fea_dim; j++) {
	in >> mbasis[b](i,j);
      }
    LinearAlgebra::map_m2v(mbasis[b], vbasis[b]);
  }

  // Read gaussian parameters
  for (int g=0; g<gauss; g++) {
    for (int i=0; i<fea_dim; i++)
      in >> gaussians[g].mu(i);
    for (int i=0; i<basis_dim; i++)
      in >> gaussians[g].lambda(i);
  }

  assert(LinearAlgebra::is_spd(mbasis.at(0)));  
  precompute();
}


void
Pcgmm::write_gk(const std::string &filename)
{
  std::ofstream out(filename.c_str());

  // Find the largest lambda
  int max_lambda_dim=0;
  for (unsigned int g=0; g<num_gaussians(); g++)
    if (gaussians.at(g).lambda.size()>max_lambda_dim)
      max_lambda_dim=gaussians.at(g).lambda.size();

  // Write header line
  out << num_gaussians() << " " << fea_dim()
      << " pcgmm " << max_lambda_dim << std::endl;
  
  // Write precision basis
  for (int b=0; b<max_lambda_dim; b++) {
    for (unsigned int i=0; i<fea_dim(); i++)
      for (unsigned int j=0; j<fea_dim(); j++)
	out << mbasis[b](i,j) << " ";
    out << std::endl;
  }

  // Write gaussian parameters
  for (unsigned int g=0; g<num_gaussians(); g++) {
    for (unsigned int i=0; i<fea_dim(); i++)
      out << gaussians[g].mu(i) << " ";
    for (int i=0; i<gaussians[g].lambda.size(); i++)
      out << gaussians[g].lambda(i) << " ";
    out << std::endl;
  }
}


void 
Pcgmm::write_basis(const std::string &filename)
{
  std::ofstream out(filename.c_str());
  
  // Write header line
  out << fea_dim() << " pcgmm " << basis_dim() << std::endl;
  
  // Write precision basis
  for (int b=0; b<basis_dim(); b++) {
    for (unsigned int i=0; i<fea_dim(); i++)
      for (unsigned int j=0; j<fea_dim(); j++)
	out << mbasis[b](i,j) << " ";
    out << std::endl;
  }
}


void
Pcgmm::read_basis(const std::string &filename)
{
  std::ifstream in(filename.c_str());
  if (!in) {
    fprintf(stderr, "Pcgmm::read_gk(): could not open %s\n", 
	    filename.c_str());
    assert(false);
  }
  
  int basis_dim=0, fea_dim=0;
  std::string cov_str;

  // Read header line
  in >> fea_dim >> cov_str;
  
  if (!(cov_str == "pcgmm")) {
    throw std::string("Model type not pcgmm");
    assert(false);
  }
  
  // Read precision basis
  in >> basis_dim;
  reset_basis(basis_dim, fea_dim);
  for (int b=0; b<basis_dim; b++) {
    for (int i=0; i<fea_dim; i++)
      for (int j=0; j<fea_dim; j++) {
	in >> mbasis[b](i,j);
      }
    LinearAlgebra::map_m2v(mbasis[b], vbasis[b]);
  }

  assert(LinearAlgebra::is_spd(mbasis.at(0)));
}


void
Pcgmm::calculate_precision(const LaVectorDouble &lambda,
			   LaGenMatDouble &precision)
{
  assert(lambda.size() <= basis_dim());
  assert(LinearAlgebra::is_spd(mbasis.at(0)));
  
  precision.resize(fea_dim(),fea_dim());
  precision(LaIndex(0,fea_dim()-1),LaIndex(0,fea_dim()-1))=0;
  LaGenMatDouble identity = LaGenMatDouble::eye(fea_dim());
  for (int b=0; b<lambda.size(); b++)
    Blas_Mat_Mat_Mult(identity, mbasis[b], precision, lambda(b), 1);
}


void
Pcgmm::calculate_precision(const LaVectorDouble &lambda,
			   LaVectorDouble &precision)
{
  calculate_precision(lambda, matrix_t1);
  assert(LinearAlgebra::is_spd(matrix_t1));
  LinearAlgebra::map_m2v(matrix_t1, precision);
}


void
Pcgmm::calculate_precision(const HCL_RnVector_d &lambda,
			   LaGenMatDouble &precision)
{
  assert(lambda.Dim() <= basis_dim());
  assert(LinearAlgebra::is_spd(mbasis.at(0)));

  precision.resize(fea_dim(),fea_dim());
  precision(LaIndex(0,fea_dim()-1),LaIndex(0,fea_dim()-1))=0;
  LaGenMatDouble identity = LaGenMatDouble::eye(fea_dim());
  for (int b=0; b<lambda.Dim(); b++)
    Blas_Mat_Mat_Mult(identity, mbasis[b], precision, lambda(b+1), 1);
}


void
Pcgmm::calculate_precision(const HCL_RnVector_d &lambda,
			   LaVectorDouble &precision)
{
  calculate_precision(lambda, matrix_t1);
  assert(LinearAlgebra::is_spd(matrix_t1));
  LinearAlgebra::map_m2v(matrix_t1, precision);
}


void
Pcgmm::calculate_covariance(const LaVectorDouble &lambda,
			    LaGenMatDouble &covariance)
{
  LaVectorLongInt pivots(fea_dim());
  calculate_precision(lambda, covariance);
  LUFactorizeIP(covariance, pivots);
  LaLUInverseIP(covariance, pivots);
  assert(LinearAlgebra::is_spd(covariance));
}


void
Pcgmm::calculate_covariance(const LaVectorDouble &lambda,
			    LaVectorDouble &covariance)
{
  calculate_covariance(lambda, matrix_t1);
  assert(LinearAlgebra::is_spd(matrix_t1));
  LinearAlgebra::map_m2v(matrix_t1, covariance);
}


void 
Pcgmm::calculate_covariance(const HCL_RnVector_d &lambda,
			    LaVectorDouble &covariance)
{
  calculate_covariance(lambda, matrix_t1);
  assert(LinearAlgebra::is_spd(matrix_t1));
  LinearAlgebra::map_m2v(matrix_t1, covariance);
}


void
Pcgmm::calculate_covariance(const HCL_RnVector_d &lambda,
			    LaGenMatDouble &covariance)
{
  LaVectorLongInt pivots(fea_dim());
  calculate_precision(lambda, covariance);
  LUFactorizeIP(covariance, pivots);
  LaLUInverseIP(covariance, pivots);
  if (!LinearAlgebra::is_spd(covariance))
    lambda.Write(std::cout);
  assert(LinearAlgebra::is_spd(covariance));
}


void
Pcgmm::gradient_untied(const HCL_RnVector_d &lambda,
		       const LaGenMatDouble &sample_cov, 
		       HCL_RnVector_d &grad)
{
  assert(sample_cov.rows() == sample_cov.cols());
  assert(lambda.Dim() <= basis_dim());
  assert(lambda.Dim() == grad.Dim());
  
  LaGenMatDouble t;
  LaGenMatDouble curr_cov_estimate;
  calculate_covariance(lambda, curr_cov_estimate);

  for (unsigned int i=0; i<fea_dim(); i++)
    for (unsigned int j=0; j<fea_dim(); j++)
      curr_cov_estimate(i,j) -= sample_cov(i,j);
  
  t.resize(fea_dim(), fea_dim());
  for (int i=0; i<lambda.Dim(); i++) {
    Blas_Mat_Mat_Mult(mbasis.at(i), curr_cov_estimate, t, 1.0, 0.0);
    grad(i+1)=t.trace();
  }
}


double
Pcgmm::G(const LaGenMatDouble &precision,
	 const LaGenMatDouble &sample_cov)
{
  LaGenMatDouble C(precision.rows(), precision.cols());
  double t=log(LinearAlgebra::determinant(precision));
  Blas_Mat_Mat_Mult(sample_cov, precision, C, 1.0, 0.0);
  t-=C.trace();

  /* debug
  LaVectorLongInt pivots(precision.rows(),1);
  LaGenMatDouble mtemp(precision);
  LUFactorizeIP(mtemp, pivots);
  LaLUInverseIP(mtemp, pivots);
  std::cout << "G value: " << t << std::endl;
  std::cout << "Kullback-Leibler value: " 
	    << kullback_leibler_covariance(sample_cov, mtemp) 
	    << std::endl;
  */

  return t;
}


double
Pcgmm::eval_linesearch_value(const LaVectorDouble &eigs,
			     double step,
			     double beta,
			     double c)
{
  double f=0;
  for (int i=0; i<eigs.size(); i++)
    f += log(1+step*eigs(i));
  f -= step*beta;
  return f;
}


double
Pcgmm::eval_linesearch_derivative(const LaVectorDouble &eigs,
				  double step,
				  double beta)
{
  double d=-beta;
  for (int i=0; i<eigs.size(); i++)
    d += eigs(i)/(1+step*eigs(i));
  return d;
}


void
Pcgmm::limit_line_search(const LaGenMatDouble &R,
			 const LaGenMatDouble &curr_prec_estimate,
			 LaVectorDouble &eigs,
			 double &max_interval)
{
  assert(R.rows()==R.cols());
  assert(curr_prec_estimate.rows()==curr_prec_estimate.cols());
  assert(R.rows()==curr_prec_estimate.rows());

  LaGenMatDouble t;
  LinearAlgebra::generalized_eigenvalues(R, curr_prec_estimate, eigs, t);
  double min=DBL_MAX;
  double max=-DBL_MAX;

  // Limit the line search based 
  // on the eigenvalues of the pair (R, curr_prec..)
  for (int i=0; i<eigs.size(); i++) {
    if (eigs(i)<min)
      min=eigs(i);
    if (eigs(i)>max)
      max=eigs(i);
  }

  if (min>0)
    max_interval=DBL_MAX;
  else
    max_interval=0.99*(-1/min);
}


double
Pcgmm::kullback_leibler_covariance(const LaGenMatDouble &sigma1,
				   const LaGenMatDouble &sigma2)
{
  LaVectorLongInt pivots(sigma1.cols(),1);
  LaGenMatDouble t1(sigma2),t2(sigma2);
  LUFactorizeIP(t1, pivots);
  LaLUInverseIP(t1, pivots);

  double value=LinearAlgebra::determinant(sigma2)
    /LinearAlgebra::determinant(sigma1);
  value=log2(value);
  Blas_Mat_Mat_Mult(t1, sigma1, t2, 1.0, 0.0);
  value += t2.trace();
  value -= sigma1.cols();
  value *= 0.5;

  return value;
}


void
Pcgmm::optimize_lambda(const LaGenMatDouble &sample_cov,
		       LaVectorDouble &lambda)
{
  assert(LinearAlgebra::is_spd(mbasis.at(0)));
  
  // Start actual optimization process
  int d=lambda.size();  
  PcgmmLambdaFcnl f(d, this, sample_cov);

  HCL_RnVector_d *x=(HCL_RnVector_d*)f.Domain().Member();

  for (int i=0; i<d; i++)    
    (*x)(i+1)=lambda(i);
  
  HCL_LineSearch_MT_d ls;
  if (hcl_line_set)
    ls.Parameters().Merge(hcl_line_cfg.c_str());
  
  HCL_UMin_lbfgs_d bfgs(&ls);
  if (hcl_grad_set)
    bfgs.Parameters().Merge(hcl_grad_cfg.c_str());
  
  bfgs.Minimize(f, *x);
  
  for (int i=0; i<d; i++)
    lambda(i)=(*x)(i+1);

  HCL_delete(x);
}
