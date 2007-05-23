#include "Subspaces.hh"



void
PrecisionSubspace::copy(const PrecisionSubspace &orig)
{
  assert(orig.m_mspace.size()==orig.m_vspace.size());

  m_mspace.resize(orig.m_mspace.size());
  m_vspace.resize(orig.m_vspace.size());
  m_quadratic_features.resize(orig.m_quadratic_features.size(),1);

  for (unsigned int i=0; i<orig.m_mspace.size(); i++) {
    m_mspace.at(i).copy(orig.m_mspace.at(i));
    m_vspace.at(i).copy(orig.m_vspace.at(i));
  }
}


void 
PrecisionSubspace::initialize_basis_pca(const std::vector<double> &c,
                                        const std::vector<LaGenMatDouble> &sample_covs, 
                                        const unsigned int basis_dim)
{
  assert(c.size() == sample_covs.size());
  
  unsigned int num_covs=sample_covs.size();
  int d=sample_covs.at(0).rows();
  int d_vec=(int)(d*(d+1)/2);
  double c_sum=0;

  // Some meaning
  LaGenMatDouble m=LaGenMatDouble::zeros(d);
  LaGenMatDouble m_sqrt=LaGenMatDouble::zeros(d);
  LaGenMatDouble m_neg_sqrt=LaGenMatDouble::zeros(d);
  LaGenMatDouble identity=LaGenMatDouble::eye(d);

  // Pure temporary stuff    
  LaGenMatDouble matrix_t1;
  LaGenMatDouble matrix_t2;
  LaVectorDouble vector_t1;
  LaVectorDouble vector_t2;

  // Reset current basis
  reset(basis_dim, d);
  
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
  m_mspace.at(0)(LaIndex(),LaIndex())=0;

  for (unsigned int i=0; i<num_covs; i++) {
    assert(sample_prec_vectors.at(i).rows()==d_vec);
    Blas_Mat_Mat_Mult(identity, sample_precs.at(i), m_mspace.at(0), c.at(i)/c_sum , 1);
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
  Blas_Mat_Mat_Mult(m_sqrt, m_mspace.at(0), matrix_t1, 1.0, 0.0);
  Blas_Mat_Mat_Mult(matrix_t1, m_sqrt, m_mspace.at(0), 1.0, 0.0);
  LinearAlgebra::map_m2v(m_mspace.at(0), m_vspace.at(0));
  // S_i, i=1,2,3...
  for (unsigned int i=1; i<basis_dim; i++) {
    vector_t1.copy(C.col(C.cols()-i));
    LinearAlgebra::map_v2m(vector_t1, matrix_t1);
    Blas_Mat_Mat_Mult(m_sqrt, matrix_t1, matrix_t2, 1.0, 0.0);
    Blas_Mat_Mat_Mult(matrix_t2, m_sqrt, m_mspace.at(i), 1.0, 0.0);
    LinearAlgebra::map_m2v(m_mspace.at(i), m_vspace.at(i));
  }
}


void
PrecisionSubspace::reset(const unsigned int subspace_dim, 
                         const unsigned int feature_dim)
{
  unsigned int d_vec=(unsigned int)feature_dim*(feature_dim+1)/2;
  m_mspace.resize(subspace_dim);
  m_vspace.resize(subspace_dim);
  m_quadratic_features.resize(subspace_dim,1);
  
  for (unsigned int i=0; i<subspace_dim; i++) {
    m_mspace.at(i).resize(feature_dim,feature_dim);
    m_vspace.at(i).resize(d_vec,1);
    for (unsigned int j=0; j<feature_dim; j++)
      for (unsigned int k=0; k<feature_dim; k++)
	(m_mspace.at(i))(j,k)=0;
    for (unsigned int l=0; l<d_vec; l++)
      m_vspace.at(i)(l)=0;
  }
}


void 
PrecisionSubspace::write_basis(const std::string &filename)
{
  std::ofstream out(filename.c_str());
  
  // Write header line
  out << fea_dim() << " pcgmm " << basis_dim() << std::endl;
  
  // Write precision basis
  for (int b=0; b<basis_dim(); b++) {
    for (unsigned int i=0; i<fea_dim(); i++)
      for (unsigned int j=0; j<fea_dim(); j++)
	out << m_mspace[b](i,j) << " ";
    out << std::endl;
  }
}


void
PrecisionSubspace::read_basis(const std::string &filename)
{
  std::ifstream in(filename.c_str());
  if (!in) {
    fprintf(stderr, "PrecisionSubspace::read_gk(): could not open %s\n", 
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
  reset(basis_dim, fea_dim);
  for (int b=0; b<basis_dim; b++) {
    for (int i=0; i<fea_dim; i++)
      for (int j=0; j<fea_dim; j++) {
	in >> m_mspace[b](i,j);
      }
    LinearAlgebra::map_m2v(m_mspace[b], m_vspace[b]);
  }

  assert(LinearAlgebra::is_spd(m_mspace.at(0)));
}


void
PrecisionSubspace::calculate_precision(const LaVectorDouble &lambda,
                                       LaGenMatDouble &precision)
{
  assert(lambda.size() <= basis_dim());
  assert(LinearAlgebra::is_spd(m_mspace.at(0)));
  
  precision.resize(fea_dim(),fea_dim());
  precision(LaIndex(0,fea_dim()-1),LaIndex(0,fea_dim()-1))=0;
  LaGenMatDouble identity = LaGenMatDouble::eye(fea_dim());
  for (int b=0; b<lambda.size(); b++)
    Blas_Mat_Mat_Mult(identity, m_mspace[b], precision, lambda(b), 1);
}


void
PrecisionSubspace::calculate_precision(const LaVectorDouble &lambda,
                                       LaVectorDouble &precision)
{
  LaGenMatDouble precision_matrix;
  calculate_precision(lambda, precision_matrix);
  assert(LinearAlgebra::is_spd(precision_matrix));
  LinearAlgebra::map_m2v(precision_matrix, precision);
}


void
PrecisionSubspace::calculate_precision(const HCL_RnVector_d &lambda,
                                       LaGenMatDouble &precision)
{
  assert(lambda.Dim() <= basis_dim());
  assert(LinearAlgebra::is_spd(m_mspace.at(0)));
  
  precision.resize(fea_dim(),fea_dim());
  precision(LaIndex(0,fea_dim()-1),LaIndex(0,fea_dim()-1))=0;
  LaGenMatDouble identity = LaGenMatDouble::eye(fea_dim());
  for (int b=0; b<lambda.Dim(); b++)
    Blas_Mat_Mat_Mult(identity, m_mspace[b], precision, lambda(b+1), 1);
}


void
PrecisionSubspace::calculate_precision(const HCL_RnVector_d &lambda,
                                       LaVectorDouble &precision)
{
  LaGenMatDouble precision_matrix;
  calculate_precision(lambda, precision_matrix);
  assert(LinearAlgebra::is_spd(precision_matrix));
  LinearAlgebra::map_m2v(precision_matrix, precision);
}


void
PrecisionSubspace::calculate_covariance(const LaVectorDouble &lambda,
                                        LaGenMatDouble &covariance)
{
  LaVectorLongInt pivots(fea_dim());
  calculate_precision(lambda, covariance);
  LUFactorizeIP(covariance, pivots);
  LaLUInverseIP(covariance, pivots);
  assert(LinearAlgebra::is_spd(covariance));
}


void
PrecisionSubspace::calculate_covariance(const LaVectorDouble &lambda,
                                        LaVectorDouble &covariance)
{
  LaGenMatDouble covariance_matrix;
  calculate_covariance(lambda, covariance_matrix);
  assert(LinearAlgebra::is_spd(covariance_matrix));
  LinearAlgebra::map_m2v(covariance_matrix, covariance);
}


void 
PrecisionSubspace::calculate_covariance(const HCL_RnVector_d &lambda,
                                        LaVectorDouble &covariance)
{
  LaGenMatDouble covariance_matrix;
  calculate_covariance(lambda, covariance_matrix);
  assert(LinearAlgebra::is_spd(covariance_matrix));
  LinearAlgebra::map_m2v(covariance_matrix, covariance);
}


void
PrecisionSubspace::calculate_covariance(const HCL_RnVector_d &lambda,
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
PrecisionSubspace::gradient_untied(const HCL_RnVector_d &lambda,
                                   const LaGenMatDouble &sample_cov,
                                   HCL_RnVector_d &grad,
                                   bool affine)
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
    Blas_Mat_Mat_Mult(m_mspace.at(i), curr_cov_estimate, t, 1.0, 0.0);
    grad(i+1)=t.trace();
  }

  if (affine)
    grad(1)=0;
}


double
PrecisionSubspace::G(const LaGenMatDouble &precision,
                     const LaGenMatDouble &sample_cov)
{
  LaGenMatDouble C(precision.rows(), precision.cols());
  double t=log(LinearAlgebra::determinant(precision));
  Blas_Mat_Mat_Mult(sample_cov, precision, C, 1.0, 0.0);
  t-=C.trace();
  return t;
}


double
PrecisionSubspace::eval_linesearch_value(const LaVectorDouble &eigs,
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
PrecisionSubspace::eval_linesearch_derivative(const LaVectorDouble &eigs,
                                              double step,
                                              double beta)
{
  double d=-beta;
  for (int i=0; i<eigs.size(); i++)
    d += eigs(i)/(1+step*eigs(i));
  return d;
}


void
PrecisionSubspace::limit_line_search(const LaGenMatDouble &R,
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


PrecisionSubspace::PrecisionSubspace(int subspace_dim, int feature_dim)
{
  set_subspace_dim(subspace_dim);
  set_feature_dim(feature_dim);
}


PrecisionSubspace::~PrecisionSubspace()
{
}


void
PrecisionSubspace::set_subspace_dim(int subspace_dim)
{
  m_subspace_dim=subspace_dim;
  m_mspace.resize(subspace_dim);
  m_vspace.resize(subspace_dim);
  m_quadratic_features.resize(subspace_dim);
}


void
PrecisionSubspace::set_feature_dim(int feature_dim)
{
  m_feature_dim = feature_dim;
  for (int i=0; i<m_subspace_dim; i++) {
    m_mspace[i].resize(feature_dim,feature_dim);
    m_vspace[i].resize((feature_dim*feature_dim+1)/2, 1);
  }
}


int
PrecisionSubspace::subspace_dim() const
{
  return m_subspace_dim;
}


int
PrecisionSubspace::feature_dim() const
{
  return m_feature_dim;
}


double
PrecisionSubspace::dotproduct(const Vector &lambda) const
{
  assert(lambda.size() == m_quadratic_features.size());
  return Blas_Dot_Prod(lambda, m_quadratic_features);
}


void
PrecisionSubspace::precompute(const FeatureVec &f)
{
  if (!m_computed) {
    LaVectorDouble y=LaVectorDouble(m_feature_dim);
    for (int i=0; i<m_subspace_dim; i++) {
      Blas_Mat_Vec_Mult(m_mspace.at(i), *f.get_vector(), y);
      m_quadratic_features(i)=-0.5*Blas_Dot_Prod(*f.get_vector(),y);
    }
    m_computed=true;
  }
}


void
PrecisionSubspace::compute_precision(const Vector &lambda,
                                     Matrix &precision)
{
  assert(lambda.size() <= subspace_dim());
  assert(LinearAlgebra::is_spd(m_mspace.at(0)));
  
  precision.resize(feature_dim(),feature_dim());
  precision=0;
  LaGenMatDouble identity = LaGenMatDouble::eye(feature_dim());
  for (int b=0; b<lambda.size(); b++)
    Blas_Mat_Mat_Mult(identity, m_mspace[b], precision, lambda(b), 1);
}


void
PrecisionSubspace::optimize_coefficients(const Matrix sample_cov, Vector &lambda)
{
  
  
}


PcgmmLambdaFcnl::PcgmmLambdaFcnl(HCL_RnSpace_d &vs,
				 int basis_dim,
				 PrecisionSubspace &pcgmm,
				 const LaGenMatDouble &sample_cov,
				 bool affine)
  : 
  m_pcgmm(pcgmm),
  m_sample_cov(sample_cov),
  m_vs(vs),
  m_base(m_vs),
  m_dir(m_vs)
{ 
  assert(sample_cov.rows()==sample_cov.cols());

  m_affine=affine;
  m_precision.resize(sample_cov.rows(), sample_cov.cols());
  m_eigs.resize(sample_cov.rows(),1);
  m_R.resize(sample_cov.rows(), sample_cov.cols());
}


PcgmmLambdaFcnl::~PcgmmLambdaFcnl() {
}


ostream&
PcgmmLambdaFcnl::Write(ostream & o) const { 
  return o; 
}


HCL_VectorSpace_d &
PcgmmLambdaFcnl::Domain() const { 
  return m_vs;
}
  

double
PcgmmLambdaFcnl::Value1(const HCL_Vector_d &x) const {
  double result;
  LaGenMatDouble t;
  m_pcgmm.calculate_precision((HCL_RnVector_d&)x, t);
  assert(LinearAlgebra::is_spd(t));
  result=m_pcgmm.G(t, m_sample_cov);
  // RETURN NEGATIVE VALUES BECAUSE HCL DOES MINIMIZATION!
  return -result;
}


void
PcgmmLambdaFcnl::Gradient1(const HCL_Vector_d & x,
			   HCL_Vector_d & g) const {
  m_pcgmm.gradient_untied((HCL_RnVector_d&)x,
			  m_sample_cov,
			  (HCL_RnVector_d&)g,
			  m_affine);
  g.Mul(-1);
}


void
PcgmmLambdaFcnl::HessianImage(const HCL_Vector_d & x,
			      const HCL_Vector_d & dx,
			      HCL_Vector_d & dy ) const {
  fprintf(stderr, "Warning, HessianImage not implemented");
}


double
PcgmmLambdaFcnl::LineSearchValue(double mu) const {
  double result;
  result=m_pcgmm.eval_linesearch_value(m_eigs, mu, m_beta, m_linevalue_const)
    +m_fval_starting_point;
  return -result;
}


double
PcgmmLambdaFcnl::LineSearchDerivative(double mu) const {
  double result;
  result = m_pcgmm.eval_linesearch_derivative(m_eigs, mu, m_beta);
  return -result;
}


void
PcgmmLambdaFcnl::SetLineSearchStartingPoint(const HCL_Vector_d &base)
{
  m_base.Copy(base);
  m_pcgmm.calculate_precision(m_base, m_precision);
  assert(LinearAlgebra::is_spd(m_precision));
  m_fval_starting_point=m_pcgmm.G(m_precision, m_sample_cov);
  m_linevalue_const=-log(LinearAlgebra::determinant(m_precision));
}


void
PcgmmLambdaFcnl::SetLineSearchDirection(const HCL_Vector_d & dir)
{
  m_dir.Copy(dir);
  m_pcgmm.calculate_precision(m_dir, m_R);
  m_pcgmm.limit_line_search(m_R, m_precision, m_eigs, m_max_step);
  LaGenMatDouble t(m_pcgmm.feature_dim(), m_pcgmm.feature_dim());
  Blas_Mat_Mat_Mult(m_sample_cov, m_R, t, 1.0, 0.0);
  m_beta=t.trace();
}
 

double
PcgmmLambdaFcnl::MaxStep(const HCL_Vector_d & x, 
			 const HCL_Vector_d & dir) const
{
  for (int i=1; i<=x.Dim(); i++) {
    assert(((HCL_RnVector_d&)x)(i)==(m_base)(i));
    assert(((HCL_RnVector_d&)dir)(i)==(m_dir)(i));
  }
  return m_max_step;
}



ExponentialSubspace::ExponentialSubspace(int subspace_dim, int feature_dim)
{
  set_subspace_dim(subspace_dim);
  set_feature_dim(feature_dim);
}


ExponentialSubspace::~ExponentialSubspace()
{
}


void
ExponentialSubspace::set_subspace_dim(int dim)
{
  m_subspace_dim=dim;
  m_basis_theta.resize(dim);
  m_basis_psi.resize(dim);
  m_basis_P.resize(dim);
  m_basis_Pvec.resize(dim);
  m_quadratic_features.resize(dim);
}


void
ExponentialSubspace::set_feature_dim(int feature_dim)
{
  m_feature_dim = feature_dim;
  for (int i=0; i<m_subspace_dim; i++) {
    m_basis_theta[i].resize((feature_dim*feature_dim+3)/2, 1);
    m_basis_psi[i].resize(feature_dim, 1);
    m_basis_P[i].resize((feature_dim*feature_dim+3)/2, 1);
    m_basis_Pvec[i].resize((feature_dim*feature_dim+3)/2, 1);
  }
}


int
ExponentialSubspace::subspace_dim() const
{
  return m_subspace_dim;
}


int
ExponentialSubspace::feature_dim() const
{
  return m_feature_dim;
}


int
ExponentialSubspace::exponential_dim() const
{
  return m_feature_dim+(m_feature_dim*(m_feature_dim+1)/2);
}


double
ExponentialSubspace::dotproduct(const Vector &lambda) const
{
  assert(lambda.size() == m_quadratic_features.size());
  return Blas_Dot_Prod(lambda, m_quadratic_features);
}


void
ExponentialSubspace::precompute(const FeatureVec &f)
{
  if (!m_computed) {
    LaVectorDouble feature_exp=LaVectorDouble(m_feature_dim+(m_feature_dim*m_feature_dim+1)/2);
    
    // Combine to get the exponential feature vector
    for (unsigned int i=0; i<feature_dim(); i++)
      feature_exp(i)=(*f.get_vector())(i);
    LaGenMatDouble xxt=LaGenMatDouble::zeros(feature_dim(), feature_dim());
    LaVectorDouble xxt_vector=LaVectorDouble(exponential_dim()-feature_dim());
    Blas_R1_Update(xxt, *f.get_vector(), *f.get_vector(), -0.5);
    LinearAlgebra::map_m2v(xxt, xxt_vector);
    for (unsigned int i=feature_dim(); i<exponential_dim(); i++)
      feature_exp(i)=xxt_vector(i-feature_dim());

    // Compute quadratic features
    for (int i=0; i<subspace_dim(); i++)
      m_quadratic_features(i)=Blas_Dot_Prod(m_basis_theta.at(i), feature_exp);

    m_computed=true;
  }
}


void
ExponentialSubspace::compute_precision(const Vector &lambda,
                                       Matrix &precision)
{
  assert(lambda.size() <= subspace_dim());
  assert(LinearAlgebra::is_spd(m_basis_P.at(0)));
  
  precision.resize(feature_dim(),feature_dim());
  precision=0;
  LaGenMatDouble identity = LaGenMatDouble::eye(feature_dim());
  for (int b=0; b<lambda.size(); b++)
    Blas_Mat_Mat_Mult(identity, m_basis_P[b], precision, lambda(b), 1);
}


void
ExponentialSubspace::compute_psi(const Vector &lambda, Vector &psi)
{
  assert(lambda.size() <= subspace_dim());
  
  psi.resize(feature_dim());
  psi = 0;
  for (int b=0; b<lambda.size(); b++)
    Blas_Add_Mult(psi, lambda(b), m_basis_psi[b]);
}


void
ExponentialSubspace::optimize_coefficients(const Matrix sample_cov, Vector &lambda)
{
  
  
}


void
ExponentialSubspace::copy(const ExponentialSubspace &orig)
{
  m_basis_theta.resize(orig.m_basis_theta.size());
  m_basis_psi.resize(orig.m_basis_psi.size());
  m_basis_P.resize(orig.m_basis_P.size());
  m_basis_Pvec.resize(orig.m_basis_Pvec.size());

  for (unsigned int i=0; i<orig.m_basis_theta.size(); i++) {
    m_basis_theta.at(i).copy(orig.m_basis_theta.at(i));
    m_basis_psi.at(i).copy(orig.m_basis_psi.at(i));
    m_basis_P.at(i).copy(orig.m_basis_P.at(i));
    m_basis_Pvec.at(i).copy(orig.m_basis_Pvec.at(i));
  }
}


void
ExponentialSubspace::reset(const unsigned int subspace_dim,
                           const unsigned int feature_dim)
{
  m_feature_dim=feature_dim;
  m_vectorized_dim=feature_dim*(feature_dim+1)/2;
  m_exponential_dim=m_feature_dim+m_vectorized_dim;
  m_basis_theta.resize(subspace_dim);
  m_basis_psi.resize(subspace_dim);
  m_basis_P.resize(subspace_dim);
  m_basis_Pvec.resize(subspace_dim);
  m_quadratic_features.resize(subspace_dim,1);

  for (unsigned int i=0; i<subspace_dim; i++) {
    m_basis_theta.at(i).resize(m_exponential_dim, 1);
    m_basis_theta.at(i)(LaIndex())=0;
    m_basis_psi.at(i).resize(m_feature_dim, 1);
    m_basis_psi.at(i)(LaIndex())=0;
    m_basis_P.at(i).resize(m_feature_dim, m_feature_dim);
    m_basis_P.at(i)(LaIndex(), LaIndex())=0;
    m_basis_Pvec.at(i).resize(m_vectorized_dim, 1);
    m_basis_Pvec.at(i)(LaIndex())=0;
  }
}


void 
ExponentialSubspace::calculate_precision(const LaVectorDouble &lambda,
                                         LaGenMatDouble &precision)
{
  assert(lambda.size()<=subspace_dim());
  
  // Current precisision
  precision.resize(feature_dim(),feature_dim());
  precision(LaIndex(),LaIndex())=0;
  LaGenMatDouble identity = LaGenMatDouble::eye(feature_dim());
  for (int b=0; b<lambda.size(); b++)
    Blas_Mat_Mat_Mult(identity, m_basis_P[b], precision, lambda(b), 1);
}


void
ExponentialSubspace::calculate_precision(const LaVectorDouble &lambda,
                                         LaVectorDouble &precision)
{
  // Calculate precision in matrix form
  LaGenMatDouble mprecision;
  // Convert to vector
  calculate_precision(lambda, mprecision);
  LinearAlgebra::map_m2v(mprecision, precision);
}


void 
ExponentialSubspace::calculate_precision(const HCL_RnVector_d &lambda,
                                         LaGenMatDouble &precision)
{
  assert(lambda.Dim()<=subspace_dim());
  
  // Current precisision
  precision.resize(feature_dim(),feature_dim());
  precision(LaIndex(),LaIndex())=0;
  LaGenMatDouble identity = LaGenMatDouble::eye(feature_dim());
  for (int b=0; b<lambda.Dim(); b++)
    Blas_Mat_Mat_Mult(identity, m_basis_P[b], precision, lambda(b+1), 1);
}


void
ExponentialSubspace::calculate_precision(const HCL_RnVector_d &lambda,
                                         LaVectorDouble &precision)
{
  assert(lambda.Dim()<=subspace_dim());

  // Calculate precision in matrix form
  LaGenMatDouble mprecision;
  // Convert to vector
  calculate_precision(lambda, mprecision);
  LinearAlgebra::map_m2v(mprecision, precision);
}


void 
ExponentialSubspace::calculate_covariance(const LaVectorDouble &lambda,
                                          LaGenMatDouble &covariance)
{
  assert(lambda.size()<=subspace_dim());

  LaVectorLongInt pivots(feature_dim());
  // Calculate precision
  calculate_precision(lambda, covariance);
  // Invert precision -> covariance
  LUFactorizeIP(covariance, pivots);
  LaLUInverseIP(covariance, pivots);
}


void 
ExponentialSubspace::calculate_covariance(const LaVectorDouble &lambda,
                                          LaVectorDouble &covariance)
{
  assert(lambda.size()<=subspace_dim());

  // Calculate precision in matrix form
  LaGenMatDouble mcovariance;
  // Convert to vector
  calculate_covariance(lambda, mcovariance);
  LinearAlgebra::map_m2v(mcovariance, covariance);
}


void 
ExponentialSubspace::calculate_covariance(const HCL_RnVector_d &lambda,
                                          LaGenMatDouble &covariance)
{
  assert(lambda.Dim()<=subspace_dim());

  LaVectorLongInt pivots(feature_dim());
  // Calculate precision
  calculate_precision(lambda, covariance);
  // Invert precision -> covariance
  LUFactorizeIP(covariance, pivots);
  LaLUInverseIP(covariance, pivots);
}


void 
ExponentialSubspace::calculate_covariance(const HCL_RnVector_d &lambda,
                                          LaVectorDouble &covariance)
{
  assert(lambda.Dim()<=subspace_dim());

  // Calculate precision in matrix form
  LaGenMatDouble mcovariance;
  // Convert to vector
  calculate_covariance(lambda, mcovariance);
  LinearAlgebra::map_m2v(mcovariance, covariance);
}


void 
ExponentialSubspace::calculate_psi(const LaVectorDouble &lambda,
                                   LaVectorDouble &psi)
{
  assert(lambda.size()<=subspace_dim());

  psi.resize(feature_dim(),1);
  psi(LaIndex())=0;
  for (int b=0; b<lambda.size(); b++)
    Blas_Add_Mult(psi, lambda(b), m_basis_psi.at(b));
}


void 
ExponentialSubspace::calculate_psi(const HCL_RnVector_d &lambda,
                                   LaVectorDouble &psi)
{
  assert(lambda.Dim()<=subspace_dim());

  psi.resize(feature_dim(),1);
  psi(LaIndex())=0;
  for (int b=0; b<lambda.Dim(); b++)
    Blas_Add_Mult(psi, lambda(b+1), m_basis_psi.at(b));
}


void
ExponentialSubspace::calculate_mu(const LaVectorDouble &lambda,
                                  LaVectorDouble &mu)
{
  assert(lambda.size()<=subspace_dim());

  mu.resize(feature_dim(),1);
  LaGenMatDouble cov;
  LaVectorDouble psi;
  calculate_covariance(lambda, cov);
  calculate_psi(lambda, psi);
  Blas_Mat_Vec_Mult(cov, psi, mu);
}


void
ExponentialSubspace::calculate_mu(const HCL_RnVector_d &lambda,
                                  LaVectorDouble &mu)
{
  assert(lambda.Dim()<=subspace_dim());

  mu.resize(feature_dim(),1);
  LaGenMatDouble cov;
  LaVectorDouble psi;
  calculate_covariance(lambda, cov);
  calculate_psi(lambda, psi);
  Blas_Mat_Vec_Mult(cov, psi, mu);
}


void
ExponentialSubspace::calculate_theta(const LaVectorDouble &lambda,
                                     LaVectorDouble &theta)
{
  assert(lambda.size()<=subspace_dim());

  theta.resize(exponential_dim(),1);
  theta(LaIndex())=0;
  for (int b=0; b<lambda.size(); b++)
    Blas_Add_Mult(theta, lambda(b), m_basis_theta.at(b));
}


void
ExponentialSubspace::calculate_theta(const HCL_RnVector_d &lambda,
                                     LaVectorDouble &theta)
{
  assert(lambda.Dim()<=subspace_dim());

  theta.resize(exponential_dim(),1);
  theta(LaIndex())=0;
  for (int b=0; b<lambda.Dim(); b++)
    Blas_Add_Mult(theta, lambda(b+1), m_basis_theta.at(b));
}


void 
ExponentialSubspace::initialize_basis_pca(const std::vector<double> &c,
                                          const std::vector<LaGenMatDouble> &covs, 
                                          const std::vector<LaVectorDouble> &means, 
                                          const unsigned int subspace_dim)
{
  assert(c.size() == covs.size());

  int d=covs.at(0).rows();
  int d_vec=(int)(d*(d+1)/2);
  int d_exp=d+d_vec;
  double c_sum=0;

  // Somewhat meaningful
  LaVectorDouble total_mean=LaVectorDouble(d,1);
  LaVectorDouble total_psi=LaVectorDouble(d,1);
  LaGenMatDouble total_covariance=LaGenMatDouble::zeros(d);
  LaGenMatDouble total_precision=LaGenMatDouble::zeros(d);
  LaVectorDouble total_precision_vec=LaVectorDouble(d_vec,1);
  LaGenMatDouble total_precision_sqrt=LaGenMatDouble::zeros(d);
  LaGenMatDouble total_precision_negsqrt=LaGenMatDouble::zeros(d);
  LaVectorDouble transformed_mean=LaVectorDouble(d_exp);
  LaGenMatDouble transformed_covariance=LaGenMatDouble::zeros(d_exp);
  LaVectorLongInt pivots(d);
  //  std::vector<LaVectorDouble> transformed_psis;
  //  std::vector<LaVectorDouble> transformed_precisions;
  LaGenMatDouble transformed_parameters=LaGenMatDouble::zeros(d_exp, c.size());

  // Pure temporary stuff
  LaGenMatDouble matrix_t1;
  LaGenMatDouble matrix_t2;
  LaVectorDouble vector_t1;
  LaVectorDouble vector_t2;
  LaVectorDouble vector_t3;
  LaVectorDouble vector_t4;

  total_mean(LaIndex())=0;
  total_psi(LaIndex())=0;
  transformed_mean(LaIndex())=0;

  // Reset current basis
  reset(subspace_dim, d);

  // Calculate c_sum
  for (unsigned int i=0; i<c.size(); i++)
    c_sum+=c.at(i);

  // Calculate total mean
  for (unsigned int i=0; i<c.size(); i++)
    Blas_Add_Mult(total_mean, c.at(i)/c_sum, means.at(i));

  // Calculate total covariance
  LaGenMatDouble identity=LaGenMatDouble::eye(d);
  for (unsigned int i=0; i<c.size(); i++) {
    Blas_Mat_Mat_Mult(identity, covs.at(i), total_covariance, c.at(i)/c_sum, 1);
    Blas_R1_Update(total_covariance, means.at(i), means.at(i), c.at(i)/c_sum);
  }
  Blas_R1_Update(total_covariance, total_mean, total_mean, -1);

  // Total precision, psi and some powers of P
  total_precision.copy(total_covariance);
  LUFactorizeIP(total_precision, pivots);
  LaLUInverseIP(total_precision, pivots);
  Blas_Mat_Vec_Mult(total_precision, total_mean, total_psi, 1, 0);
  LinearAlgebra::matrix_power(total_precision, total_precision_sqrt, 0.5);
  LinearAlgebra::matrix_power(total_precision, total_precision_negsqrt, -0.5);
  LinearAlgebra::map_m2v(total_precision, total_precision_vec);

  // Transform the full covariance parameters
  matrix_t1.resize(d,d);
  matrix_t2.resize(d,d);
  vector_t1.resize(d,1);
  vector_t2.resize(d_exp,1);

  LaVectorDouble transformed_psi=LaVectorDouble(d,1);
  LaVectorDouble transformed_precision=LaVectorDouble(d_vec,1);
  for (unsigned int i=0; i<c.size(); i++) {
    matrix_t1.copy(covs.at(i));
    LUFactorizeIP(matrix_t1, pivots);
    LaLUInverseIP(matrix_t1, pivots);
    Blas_Mat_Vec_Mult(matrix_t1, means.at(i), vector_t1);
    LinearAlgebra::map_m2v(matrix_t1, transformed_precision);

    for (int j=0; j<d; j++)
      transformed_parameters(j,i)=vector_t1(j);
    
    for (int j=d; j<d_exp; j++)
      transformed_parameters(j,i)=transformed_precision(j-d);
  }
  
  // Remove average
  LaVectorDouble average=LaVectorDouble(d_exp,1);
  average(LaIndex())=0;
  for (unsigned int i=0; i<c.size(); i++)
    for (int j=0; j<d_exp; j++)
      average(j) += transformed_parameters(j,i)/c.size();
  for (unsigned int i=0; i<c.size(); i++)
    for (int j=0; j<d_exp; j++)
      transformed_parameters(j,i) -= average(j);

  // Do the singular value decomposition
  LaVectorDouble Sigma=LaVectorDouble(d_exp,1);
  LaGenMatDouble U=LaGenMatDouble(d_exp,d_exp);
  LaGenMatDouble VT=LaGenMatDouble(0, 0);
  LaSVD_IP(transformed_parameters, Sigma, U, VT);

  // Let's initialize the basis itself
  m_basis_theta.resize(subspace_dim);
  m_basis_psi.resize(subspace_dim);
  m_basis_P.resize(subspace_dim);  
  m_basis_Pvec.resize(subspace_dim);  

  vector_t1.resize(d_exp,1);
  vector_t2.resize(d,1);
  vector_t3.resize(d,1);
  vector_t4.resize(d,1);
  matrix_t1.resize(d,d);
  matrix_t2.resize(d,d);

  for (unsigned int i=0; i<subspace_dim; i++) {    
    m_basis_theta.at(i).resize(d_exp,1);
    m_basis_psi.at(i).resize(d,1);
    m_basis_P.at(i).resize(d,d);

    // First basis is total_theta
    if (i==0) {
      m_basis_psi.at(i).copy(total_psi);
      m_basis_P.at(i).copy(total_precision);
      LinearAlgebra::map_m2v(m_basis_P.at(i), m_basis_Pvec.at(i));
      LinearAlgebra::map_m2v(total_precision, vector_t4);
      for (int j=0; j<d; j++)
	m_basis_theta.at(i)(j)=total_psi(j);
      for (int j=d; j<d_exp; j++)
	m_basis_theta.at(i)(j)=vector_t4(j-d);
      vector_t4.resize(d, 1);
    }
    // Others are top singular vectors of the transformed_parameters
    else {
      // vector_t1 equals i:th eigenvector
      for (int j=0; j<d_exp; j++)
	vector_t1(j)=U(j, i-1);
      
      // matrix_t1 equals precision part of the i:th eigenvector
      LinearAlgebra::map_v2m(vector_t1(LaIndex(d, d_exp-1)), matrix_t1);

      // vector_t2 equals psi part of the i:th eigenvector
      vector_t2.copy(vector_t1(LaIndex(0, d-1)));

      m_basis_psi.at(i).copy(vector_t2);
      m_basis_P.at(i).copy(matrix_t1);
      LinearAlgebra::map_m2v(m_basis_P.at(i), m_basis_Pvec.at(i));

      // Theta_i
      for (int j=0; j<d; j++)
	m_basis_theta.at(i)(j)=m_basis_psi.at(i)(j);

      for (int j=d; j<d_exp; j++)
      	m_basis_theta.at(i)(j)=m_basis_Pvec.at(i)(j-d);
    }
  }
}


void
ExponentialSubspace::read_basis(const std::string &filename)
{
  std::ifstream in(filename.c_str());
  if (!in) {
    fprintf(stderr, "ExponentialSubspace::read_basis(): could not open %s\n", 
	    filename.c_str());
    assert(false);
  }
  
  int b_temp=0;
  std::string cov_str;

  // Read header line
  in >> m_feature_dim >> cov_str;
  m_exponential_dim=m_feature_dim+m_feature_dim*(m_feature_dim+1)/2;

  if (!(cov_str == "scgmm")) {
    throw std::string("Basis type not scgmm");
    assert(false);
  }

  in >> b_temp;
  reset(b_temp, m_feature_dim);

  // Read exponential basis
  for (int b=0; b<subspace_dim(); b++) {
    for (unsigned int i=0; i<exponential_dim(); i++)
      in >> m_basis_theta[b](i);
    for (unsigned int i=0; i<feature_dim(); i++)
      m_basis_psi[b](i)=m_basis_theta[b](i);
    for (unsigned int i=feature_dim(); i<exponential_dim(); i++)
      m_basis_Pvec[b](i-feature_dim())=m_basis_theta[b](i);

    LinearAlgebra::map_v2m(m_basis_Pvec[b], m_basis_P[b]);
  }
  assert(LinearAlgebra::is_spd(m_basis_P[0]));
}


void
ExponentialSubspace::write_basis(const std::string &filename)
{
  std::ofstream out(filename.c_str());

  // Write header line
  out << feature_dim() << " scgmm " << subspace_dim() << std::endl;
  
  // Write exponential basis
  for (int b=0; b<subspace_dim(); b++) {
    for (unsigned int i=0; i<exponential_dim(); i++)
      out << m_basis_theta[b](i) << " ";
    out << std::endl;
  }
}


double
ExponentialSubspace::K(const LaGenMatDouble &precision,
                       const LaVectorDouble &psi)
{
  double result;
  LaVectorDouble tvec(psi);
  LaGenMatDouble cov(precision);
  LaVectorLongInt pivots(cov.rows(),1);

  LUFactorizeIP(cov, pivots);
  LaLUInverseIP(cov, pivots);
  Blas_Mat_Vec_Mult(cov, psi, tvec);

  result = -psi.size()*log(2*3.1416);
  result += log(LinearAlgebra::determinant(precision));
  result -= Blas_Dot_Prod(psi, tvec);
  result *= 0.5;
  return result;
}


double
ExponentialSubspace::K(const LaVectorDouble &theta)
{
  int d_exp=theta.size();
  int d=int(-1.5 + 0.5*(pow(9+8*d_exp, 0.5)));

  LaVectorDouble psi(d,1);
  LaGenMatDouble precision(d,d);
  
  psi.copy(theta(LaIndex(0,d-1)));
  LinearAlgebra::map_v2m(theta(LaIndex(d,d_exp-1)), precision);
  
  assert(LinearAlgebra::is_spd(precision));

  return K(precision, psi);
}


double
ExponentialSubspace::H(const LaVectorDouble &theta,
	 const LaVectorDouble &f)
{
  double result;
  result = K(theta);
  result += Blas_Dot_Prod(theta, f);
  return result;
}


void
ExponentialSubspace::gradient_untied(const HCL_RnVector_d &lambda,
                                     const LaVectorDouble &sample_mean,
                                     const LaGenMatDouble &sample_secondmoment,
                                     HCL_RnVector_d &grad,
                                     bool affine)
{
  assert(lambda.Dim()<=subspace_dim());
  assert(lambda.Dim()==grad.Dim());
  assert(sample_mean.size()==sample_secondmoment.rows());
  assert(sample_mean.size()==sample_secondmoment.cols());

  int d=sample_mean.size();

  LaVectorDouble mean;
  LaGenMatDouble covariance;
  calculate_covariance(lambda, covariance);
  calculate_mu(lambda, mean);

  LaVectorDouble grad_psi(sample_mean);
  LaGenMatDouble grad_p(covariance);
  LaVectorDouble grad_p_vec;

  // grad_psi
  Blas_Add_Mult(grad_psi, -1, mean);
  
  // grad_p
  Blas_R1_Update(grad_p, mean, mean);
  // FIXME BLAS_ADD_MULT?
  for (int i=0; i<d; i++)
    for (int j=0; j<d; j++)
      grad_p(i,j) -= sample_secondmoment(i,j);
  Blas_Scale(0.5, grad_p);  

  // grad_p_vec
  LinearAlgebra::map_m2v(grad_p, grad_p_vec);

  // gradient in terms of lambdas
  for (int i=0; i<grad.Dim(); i++)
    grad(i+1)=Blas_Dot_Prod(m_basis_Pvec.at(i),grad_p_vec)+Blas_Dot_Prod(m_basis_psi.at(i), grad_psi);

  if (affine)
    grad(1)=0;
}


double
ExponentialSubspace::eval_linesearch_value(const LaVectorDouble &eigs,
                                           const LaVectorDouble &v,
                                           const LaVectorDouble &dv,
                                           double step,
                                           double beta)
{
  double f=0;
  LaVectorDouble t(v);
  Blas_Add_Mult(t,step,dv);

  f += step*beta;
  for (int i=0; i<eigs.size(); i++) {
    f += 0.5*log(1+step*eigs(i));
    f -= 0.5*t(i)*t(i)/(1+step*eigs(i));
    f += 0.5*v(i)*v(i);
  }    

  return f;
}


double
ExponentialSubspace::eval_linesearch_derivative(const LaVectorDouble &eigs,
                                                const LaVectorDouble &v,
                                                const LaVectorDouble &dv,
                                                double step,
                                                double beta)
{
  double d=0;
  LaVectorDouble t(v);
  Blas_Add_Mult(t,step,dv);

  d=beta;
  for (int i=0; i<eigs.size(); i++) {
    d += 0.5*eigs(i)/(1+step*eigs(i));
    d -= t(i)*dv(i)/(1+step*eigs(i));
    d += 0.5*t(i)*t(i)*eigs(i)/((1+step*eigs(i))*(1+step*eigs(i)));
  } 

  return d;
}


void
ExponentialSubspace::limit_line_search(const LaGenMatDouble &R,
                                       const LaGenMatDouble &curr_prec_estimate,
                                       LaVectorDouble &eigvals,
                                       LaGenMatDouble &eigvecs,
                                       double &max_interval)
{
  assert(R.rows()==R.cols());
  assert(curr_prec_estimate.rows()==curr_prec_estimate.cols());
  assert(R.rows()==curr_prec_estimate.rows());

  LinearAlgebra::generalized_eigenvalues(R, curr_prec_estimate, eigvals, eigvecs);
  double min=DBL_MAX;
  double max=-DBL_MAX;
  
  // Limit the line search based 
  // on the eigenvalues of the pair (R, curr_prec..)
  for (int i=0; i<eigvals.size(); i++) {
    if (eigvals(i)<min)
      min=eigvals(i);
    if (eigvals(i)>max)
      max=eigvals(i);
  }
  
  if (min>0)
    max_interval=DBL_MAX;
  else
    max_interval=0.95*(-1/min);
}


void
ExponentialSubspace::f_to_gaussian_params(const LaVectorDouble &f,
                                          LaVectorDouble &sample_mu,
                                          LaGenMatDouble &sample_sigma)
{
  int d=feature_dim();
  int d_exp=exponential_dim();
  LaVectorDouble t;

  sample_mu.resize(d,1);
  sample_sigma.resize(d,d);

  sample_mu.copy(f(LaIndex(0,d-1)));
  t.copy(f(LaIndex(d,d_exp-1)));

  Blas_Scale(-2, t);
  LinearAlgebra::map_v2m(t, sample_sigma);
  Blas_R1_Update(sample_sigma, sample_mu, sample_mu, -1);

  assert(LinearAlgebra::is_spd(sample_sigma));
}


void
ExponentialSubspace::gaussian_params_to_f(const LaVectorDouble &sample_mu,
                                          const LaGenMatDouble &sample_sigma,
                                          LaVectorDouble &f)
{
  assert(sample_mu.size()==sample_sigma.rows());
  assert(sample_sigma.rows()==sample_sigma.cols());
  assert(LinearAlgebra::is_spd(sample_sigma));

  int d=sample_mu.size();
  int d_vec=(int)(d*(d+1)/2);
  int d_exp=d+d_vec;

  LaVectorDouble t;
  LaGenMatDouble sample_secondmoment(sample_sigma);

  f.resize(d_exp,1);

  for (int i=0; i<d; i++)
    f(i)=sample_mu(i);

  Blas_R1_Update(sample_secondmoment, sample_mu, sample_mu, 1);
  LinearAlgebra::map_m2v(sample_secondmoment, t);
  Blas_Scale(-0.5, t);

  for (int i=d; i<d_exp; i++)
    f(i)=t(i-d);
}


void
ExponentialSubspace::theta_to_gaussian_params(const LaVectorDouble &theta,
                                              LaVectorDouble &mu,
                                              LaGenMatDouble &sigma)
{
  int d=feature_dim();
  int d_vec=(int)(d*(d+1)/2);
  int d_exp=d+d_vec;

  // psi
  LaVectorDouble psi(theta(LaIndex(0,d-1)));

  // sigma
  sigma.resize(d,d);  
  LaVectorDouble t(theta(LaIndex(d,d_exp-1)));
  LinearAlgebra::map_v2m(t, sigma);
  LaVectorLongInt pivots(feature_dim());
  LUFactorizeIP(sigma, pivots);
  LaLUInverseIP(sigma, pivots);

  // mu
  mu.resize(d,1);
  Blas_Mat_Vec_Mult(sigma, psi, mu);

  assert(LinearAlgebra::is_spd(sigma));
}



ScgmmLambdaFcnl::ScgmmLambdaFcnl(HCL_RnSpace_d &vs,
				 int subspace_dim,
				 ExponentialSubspace &es,
				 const LaGenMatDouble &sample_cov,
				 const LaVectorDouble &sample_mean,
				 bool affine)
  : 
  m_es(es),
  m_sample_mean(sample_mean),
  m_sample_cov(sample_cov),
  m_vs(vs),
  m_base(m_vs),
  m_dir(m_vs)
{ 
  assert(sample_cov.rows()==sample_cov.cols());
  assert(sample_mean.size()==sample_cov.rows());

  m_affine=affine;
  m_sample_secondmoment.copy(m_sample_cov);
  Blas_R1_Update(m_sample_secondmoment, m_sample_mean, m_sample_mean, 1);
  m_precision.resize(m_sample_cov.rows(), m_sample_cov.cols());
  m_eigvals.resize(m_sample_cov.rows(),1);
  m_R.resize(m_sample_cov.rows(), m_sample_cov.cols());
  m_es.gaussian_params_to_f(m_sample_mean, m_sample_cov, m_f);
}


ScgmmLambdaFcnl::~ScgmmLambdaFcnl() {
}


ostream & 
ScgmmLambdaFcnl::Write(ostream & o) const { 
  return o; 
}


HCL_VectorSpace_d & 
ScgmmLambdaFcnl::Domain() const { 
  return m_vs;
}


// RETURN NEGATIVE VALUES BECAUSE HCL DOES MINIMIZATION!
double
ScgmmLambdaFcnl::Value1(const HCL_Vector_d &x) const {
  double result;
  LaVectorDouble theta;
  m_es.calculate_theta((HCL_RnVector_d&)x, theta);
  result = m_es.H(theta, m_f);
  return -result;
}


void
ScgmmLambdaFcnl::Gradient1(const HCL_Vector_d & x,
			   HCL_Vector_d & g) const {
  m_es.gradient_untied((HCL_RnVector_d&)x,
			  m_sample_mean,
			  m_sample_secondmoment,
			  (HCL_RnVector_d&)g,
			  m_affine);
  // RETURN NEGATIVE VALUES BECAUSE HCL DOES MINIMIZATION!
  g.Mul(-1);
}


void
ScgmmLambdaFcnl::HessianImage(const HCL_Vector_d & x,
			      const HCL_Vector_d & dx,
			      HCL_Vector_d & dy ) const {
  fprintf(stderr, "Warning, HessianImage not implemented");
}


// RETURN NEGATIVE VALUES BECAUSE HCL DOES MINIMIZATION!
double
ScgmmLambdaFcnl::LineSearchValue(double mu) const {
  double result;
  result=m_es.eval_linesearch_value(m_eigvals, m_v, m_dv, mu, m_beta)+m_fval_starting_point;
  return -result;
}


// RETURN NEGATIVE VALUES BECAUSE HCL DOES MINIMIZATION!
double
ScgmmLambdaFcnl::LineSearchDerivative(double mu) const {
  double result=0;
  result=m_es.eval_linesearch_derivative(m_eigvals, m_v, m_dv, mu, m_beta);
  return -result;
}


void
ScgmmLambdaFcnl::SetLineSearchStartingPoint(const HCL_Vector_d &base)
{
  m_base.Copy(base);
  
  m_es.calculate_precision(m_base, m_precision);
  assert(LinearAlgebra::is_spd(m_precision));
  m_es.calculate_psi(m_base, m_psi);
  m_es.calculate_theta(m_base, m_theta);
  
  m_fval_starting_point=m_es.H(m_theta, m_f);
}


// ASSUMES: m_precision and m_psi are set == SetLineSearchStartingPoint() called
void
ScgmmLambdaFcnl::SetLineSearchDirection(const HCL_Vector_d & dir)
{
  m_dir.Copy(dir);
  
  int d=m_es.feature_dim();
  
  m_es.calculate_precision(m_dir, m_R);
  m_es.limit_line_search(m_R, m_precision, m_eigvals, m_eigvecs, m_max_step);
  
  LaGenMatDouble t=LaGenMatDouble::zeros(d);
  LaGenMatDouble t2=LaGenMatDouble::zeros(d);
  LinearAlgebra::matrix_power(m_precision, t, -0.5);
  Blas_Mat_Mat_Mult(m_eigvecs, t, t2, 1.0, 0.0);
  
  m_v.resize(d,1);
  m_dv.resize(d,1);
  LaVectorDouble d_psi;
  m_es.calculate_psi(m_dir, d_psi);
  Blas_Mat_Vec_Mult(t2, m_psi, m_v);
  Blas_Mat_Vec_Mult(t2, d_psi, m_dv);
  
  LaVectorDouble d_theta;
  m_es.calculate_theta(m_dir, d_theta);
  m_beta = Blas_Dot_Prod(d_theta, m_f);
}


double
ScgmmLambdaFcnl::MaxStep(const HCL_Vector_d & x, 
			 const HCL_Vector_d & dir) const
{
  for (int i=1; i<=x.Dim(); i++) {
    assert(((HCL_RnVector_d&)x)(i)==(m_base)(i));
    assert(((HCL_RnVector_d&)dir)(i)==(m_dir)(i));
  }
  return m_max_step;
}
