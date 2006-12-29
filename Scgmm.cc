#include "Scgmm.hh"


void
Scgmm::copy(const Scgmm &orig)
{
  basis_theta.resize(orig.basis_theta.size());
  basis_psi.resize(orig.basis_psi.size());
  basis_P.resize(orig.basis_P.size());
  basis_Pvec.resize(orig.basis_Pvec.size());
  gaussians.resize(orig.gaussians.size());
  likelihoods.resize(orig.gaussians.size());

  for (unsigned int i=0; i<orig.basis_theta.size(); i++) {
    basis_theta.at(i).copy(orig.basis_theta.at(i));
    basis_psi.at(i).copy(orig.basis_psi.at(i));
    basis_P.at(i).copy(orig.basis_P.at(i));
    basis_Pvec.at(i).copy(orig.basis_Pvec.at(i));
  }

  for (unsigned int i=0; i<orig.gaussians.size(); i++) {
    gaussians.at(i).lambda.copy(orig.gaussians.at(i).lambda);
    gaussians.at(i).K_value=orig.gaussians.at(i).K_value;
  }
}


void 
Scgmm::precompute()
{
  LaVectorDouble theta;
  
  // Calculate K for each gaussian
  for (unsigned int i=0; i<num_gaussians(); i++) {
    calculate_theta(gaussians.at(i).lambda, theta);    
    gaussians.at(i).K_value=K(theta);
  }
}


void 
Scgmm::compute_likelihoods(const FeatureVec &feature,
			   std::vector<float> &lls)
{
  lls.resize(num_gaussians());

  // Convert feature vector to exponential feature vector in lapackpp format
  LaVectorDouble feature_linear=LaVectorDouble(fea_dim());
  LaVectorDouble feature_exp=LaVectorDouble(exp_dim());
  LaVectorDouble feature_second_moment=LaVectorDouble(exp_dim()-fea_dim());
  LaGenMatDouble xxt=LaGenMatDouble::zeros(fea_dim(),fea_dim());
  for (unsigned int i=0; i<fea_dim(); i++) {
    feature_linear(i)=feature[i];
    feature_exp(i)=feature[i];
  }

  Blas_R1_Update(xxt, feature_linear, feature_linear, -0.5);
  LinearAlgebra::map_m2v(xxt, feature_second_moment);
  for (unsigned int i=fea_dim(); i<exp_dim(); i++)
    feature_exp(i)=feature_second_moment(i-fea_dim());

  // Compute 'quadratic features'
  for (int i=0; i<basis_dim(); i++)
    quadratic_feas(i)=Blas_Dot_Prod(basis_theta.at(i), feature_exp);
  
  // Compute likelihoods
  for (unsigned int i=0; i<num_gaussians(); i++) {
    likelihoods.at(i)=gaussians.at(i).K_value
      +Blas_Dot_Prod(gaussians.at(i).lambda,quadratic_feas);
    likelihoods.at(i)=exp(likelihoods.at(i));
    lls.at(i)=likelihoods.at(i);
  }
}


double 
Scgmm::gaussian_likelihood(const int k)
{
  return likelihoods.at(k);
}


void
Scgmm::reset(const unsigned int basis_dim, 
	     const unsigned int f,
	     const unsigned int g)  
{
  gaussians.resize(g);
  likelihoods.resize(g);  
  for (unsigned int i=0; i<g; i++) {
    gaussians.at(i).resize(basis_dim);
    likelihoods.at(i)=0;
  }
  
  reset_basis(basis_dim, f);
}


void
Scgmm::reset_basis(const unsigned int basis_dim, 
		   const unsigned int f)
{
  m_fea_dim=f;
  m_vec_dim=f*(f+1)/2;
  m_exp_dim=m_fea_dim+m_vec_dim;
  basis_theta.resize(basis_dim);
  basis_psi.resize(basis_dim);
  basis_P.resize(basis_dim);
  basis_Pvec.resize(basis_dim);
  quadratic_feas.resize(basis_dim,1);

  for (unsigned int i=0; i<basis_dim; i++) {
    basis_theta.at(i).resize(m_exp_dim, 1);
    basis_theta.at(i)(LaIndex())=0;
    basis_psi.at(i).resize(m_fea_dim, 1);
    basis_psi.at(i)(LaIndex())=0;
    basis_P.at(i).resize(m_fea_dim, m_fea_dim);
    basis_P.at(i)(LaIndex(), LaIndex())=0;
    basis_Pvec.at(i).resize(m_vec_dim, 1);
    basis_Pvec.at(i)(LaIndex())=0;
  }
}


void 
Scgmm::calculate_precision(const LaVectorDouble &lambda,
			   LaGenMatDouble &precision)
{
  assert(lambda.size()<=basis_dim());
  
  // Current precisision
  precision.resize(fea_dim(),fea_dim());
  precision(LaIndex(),LaIndex())=0;
  LaGenMatDouble identity = LaGenMatDouble::eye(fea_dim());
  for (int b=0; b<lambda.size(); b++)
    Blas_Mat_Mat_Mult(identity, basis_P[b], precision, lambda(b), 1);
}


void
Scgmm::calculate_precision(const LaVectorDouble &lambda,
			   LaVectorDouble &precision)
{
  // Calculate precision in matrix form
  LaGenMatDouble mprecision;
  // Convert to vector
  calculate_precision(lambda, mprecision);
  LinearAlgebra::map_m2v(mprecision, precision);
}


void 
Scgmm::calculate_precision(const HCL_RnVector_d &lambda,
			   LaGenMatDouble &precision)
{
  assert(lambda.Dim()<=basis_dim());
  
  // Current precisision
  precision.resize(fea_dim(),fea_dim());
  precision(LaIndex(),LaIndex())=0;
  LaGenMatDouble identity = LaGenMatDouble::eye(fea_dim());
  for (int b=0; b<lambda.Dim(); b++)
    Blas_Mat_Mat_Mult(identity, basis_P[b], precision, lambda(b+1), 1);
}


void
Scgmm::calculate_precision(const HCL_RnVector_d &lambda,
			   LaVectorDouble &precision)
{
  assert(lambda.Dim()<=basis_dim());

  // Calculate precision in matrix form
  LaGenMatDouble mprecision;
  // Convert to vector
  calculate_precision(lambda, mprecision);
  LinearAlgebra::map_m2v(mprecision, precision);
}


void 
Scgmm::calculate_covariance(const LaVectorDouble &lambda,
			    LaGenMatDouble &covariance)
{
  assert(lambda.size()<=basis_dim());

  LaVectorLongInt pivots(fea_dim());
  // Calculate precision
  calculate_precision(lambda, covariance);
  // Invert precision -> covariance
  LUFactorizeIP(covariance, pivots);
  LaLUInverseIP(covariance, pivots);
}


void 
Scgmm::calculate_covariance(const LaVectorDouble &lambda,
			    LaVectorDouble &covariance)
{
  assert(lambda.size()<=basis_dim());

  // Calculate precision in matrix form
  LaGenMatDouble mcovariance;
  // Convert to vector
  calculate_covariance(lambda, mcovariance);
  LinearAlgebra::map_m2v(mcovariance, covariance);
}


void 
Scgmm::calculate_covariance(const HCL_RnVector_d &lambda,
			    LaGenMatDouble &covariance)
{
  assert(lambda.Dim()<=basis_dim());

  LaVectorLongInt pivots(fea_dim());
  // Calculate precision
  calculate_precision(lambda, covariance);
  // Invert precision -> covariance
  LUFactorizeIP(covariance, pivots);
  LaLUInverseIP(covariance, pivots);
}


void 
Scgmm::calculate_covariance(const HCL_RnVector_d &lambda,
			    LaVectorDouble &covariance)
{
  assert(lambda.Dim()<=basis_dim());

  // Calculate precision in matrix form
  LaGenMatDouble mcovariance;
  // Convert to vector
  calculate_covariance(lambda, mcovariance);
  LinearAlgebra::map_m2v(mcovariance, covariance);
}


void 
Scgmm::calculate_psi(const LaVectorDouble &lambda,
		     LaVectorDouble &psi)
{
  assert(lambda.size()<=basis_dim());

  psi.resize(fea_dim(),1);
  psi(LaIndex())=0;
  for (int b=0; b<lambda.size(); b++)
    Blas_Add_Mult(psi, lambda(b), basis_psi.at(b));
}


void 
Scgmm::calculate_psi(const HCL_RnVector_d &lambda,
		     LaVectorDouble &psi)
{
  assert(lambda.Dim()<=basis_dim());

  psi.resize(fea_dim(),1);
  psi(LaIndex())=0;
  for (int b=0; b<lambda.Dim(); b++)
    Blas_Add_Mult(psi, lambda(b+1), basis_psi.at(b));
}


void
Scgmm::calculate_mu(const LaVectorDouble &lambda,
		    LaVectorDouble &mu)
{
  assert(lambda.size()<=basis_dim());

  mu.resize(fea_dim(),1);
  LaGenMatDouble cov;
  LaVectorDouble psi;
  calculate_covariance(lambda, cov);
  calculate_psi(lambda, psi);
  Blas_Mat_Vec_Mult(cov, psi, mu);
}


void
Scgmm::calculate_mu(const HCL_RnVector_d &lambda,
		    LaVectorDouble &mu)
{
  assert(lambda.Dim()<=basis_dim());

  mu.resize(fea_dim(),1);
  LaGenMatDouble cov;
  LaVectorDouble psi;
  calculate_covariance(lambda, cov);
  calculate_psi(lambda, psi);
  Blas_Mat_Vec_Mult(cov, psi, mu);
}


void
Scgmm::calculate_theta(const LaVectorDouble &lambda,
		       LaVectorDouble &theta)
{
  assert(lambda.size()<=basis_dim());

  theta.resize(exp_dim(),1);
  theta(LaIndex())=0;
  for (int b=0; b<lambda.size(); b++)
    Blas_Add_Mult(theta, lambda(b), basis_theta.at(b));
}


void
Scgmm::calculate_theta(const HCL_RnVector_d &lambda,
		       LaVectorDouble &theta)
{
  assert(lambda.Dim()<=basis_dim());

  theta.resize(exp_dim(),1);
  theta(LaIndex())=0;
  for (int b=0; b<lambda.Dim(); b++)
    Blas_Add_Mult(theta, lambda(b+1), basis_theta.at(b));
}


void 
Scgmm::initialize_basis_pca(const std::vector<double> &c,
			    const std::vector<LaGenMatDouble> &covs, 
			    const std::vector<LaVectorDouble> &means, 
			    const unsigned int basis_dim)
{
  assert(c.size() == covs.size());

  int d=covs.at(0).rows();
  int d_vec=(int)(d*(d+1)/2);
  int d_exp=d+d_vec;
  double c_sum=0;

  LaVectorDouble total_mean=LaVectorDouble(d,1);
  LaVectorDouble total_psi=LaVectorDouble(d,1);
  LaGenMatDouble total_covariance=LaGenMatDouble::zeros(d);
  LaGenMatDouble total_precision=LaGenMatDouble::zeros(d);
  LaGenMatDouble total_precision_sqrt=LaGenMatDouble::zeros(d);
  LaGenMatDouble total_precision_negsqrt=LaGenMatDouble::zeros(d);
  LaVectorDouble transformed_mean=LaVectorDouble(d_exp);
  LaGenMatDouble transformed_covariance=LaGenMatDouble::zeros(d_exp);
  LaVectorLongInt pivots(d);
  //  std::vector<LaVectorDouble> transformed_psis;
  //  std::vector<LaVectorDouble> transformed_precisions;
  LaGenMatDouble transformed_parameters=LaGenMatDouble::zeros(d_exp, c.size());
  total_mean(LaIndex())=0;
  total_psi(LaIndex())=0;
  transformed_mean(LaIndex())=0;

  // Reset current basis
  reset(basis_dim, d, num_gaussians());

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

  // Do the singular value decomposition
  LaVectorDouble Sigma=LaVectorDouble(d_exp,1);
  LaGenMatDouble U=LaGenMatDouble(d_exp,d_exp);
  LaGenMatDouble VT=LaGenMatDouble(0, 0);
  LaSVD_IP(transformed_parameters, Sigma, U, VT);

  // Let's initialize the basis itself
  basis_theta.resize(basis_dim);
  basis_psi.resize(basis_dim);
  basis_P.resize(basis_dim);  
  basis_Pvec.resize(basis_dim);  

  vector_t1.resize(d_exp,1);
  vector_t2.resize(d,1);
  vector_t3.resize(d,1);
  vector_t4.resize(d,1);
  matrix_t1.resize(d,d);
  matrix_t2.resize(d,d);

  for (unsigned int i=0; i<basis_dim; i++) {    
    basis_theta.at(i).resize(d_exp,1);
    basis_psi.at(i).resize(d,1);
    basis_P.at(i).resize(d,d);

    // First basis is total_theta
    if (i==0) {
      basis_psi.at(i).copy(total_psi);
      basis_P.at(i).copy(total_precision);
      LinearAlgebra::map_m2v(basis_P.at(i), basis_Pvec.at(i));
      LinearAlgebra::map_m2v(total_precision, vector_t4);
      for (int j=0; j<d; j++)
	basis_theta.at(i)(j)=total_psi(j);
      for (int j=d; j<d_exp; j++)
	basis_theta.at(i)(j)=vector_t4(j-d);
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

      basis_psi.at(i).copy(vector_t2);
      basis_P.at(i).copy(matrix_t1);
      LinearAlgebra::map_m2v(basis_P.at(i), basis_Pvec.at(i));

      // Theta_i
      for (int j=0; j<d; j++)
	basis_theta.at(i)(j)=basis_psi.at(i)(j);

      for (int j=d; j<d_exp; j++)
      	basis_theta.at(i)(j)=basis_Pvec.at(i)(j-d);
    }
  }
}


void
Scgmm::read_gk(const std::string &filename)
{
  std::ifstream in(filename.c_str());
  if (!in) {
    fprintf(stderr, "Scgmm::read_gk(): could not open %s\n", 
	    filename.c_str());
    assert(false);
  }
  
  int g_temp=0, b_temp=0;
  std::string cov_str;

  // Read header line
  in >> g_temp >> m_fea_dim >> cov_str;
  m_exp_dim=m_fea_dim+m_fea_dim*(m_fea_dim+1)/2;

  if (!(cov_str == "scgmm")) {
    throw std::string("Model type not scgmm");
    assert(false);
  }

  in >> b_temp;
  reset(b_temp, m_fea_dim, g_temp);

  // Read exponential basis
  for (int b=0; b<basis_dim(); b++) {
    for (int i=0; i<m_exp_dim; i++)
      in >> basis_theta[b](i);
    for (int i=0; i<m_fea_dim; i++)
      basis_psi[b](i)=basis_theta[b](i);
    LinearAlgebra::map_v2m(basis_theta[b](LaIndex(m_fea_dim, m_exp_dim-1)),basis_P[b]);
  }
    
  // Read gaussian parameters
  for (unsigned int g=0; g<num_gaussians(); g++) {
    for (int i=0; i<basis_dim(); i++)
      in >> gaussians.at(g).lambda(i);
  }

  precompute();
}


void
Scgmm::write_gk(const std::string &filename)
{
  std::ofstream out(filename.c_str());

  // Find the largest lambda
  int max_lambda_dim=0;
  for (unsigned int g=0; g<num_gaussians(); g++)
    if (gaussians.at(g).lambda.size()>max_lambda_dim)
      max_lambda_dim=gaussians.at(g).lambda.size();

  // Write header line
  out << num_gaussians() << " " << fea_dim()
      << " scgmm " << max_lambda_dim << std::endl;
  
  // Write exponential basis
  for (int b=0; b<max_lambda_dim; b++) {
    for (unsigned int i=0; i<exp_dim(); i++)
      out << basis_theta[b](i) << " ";
    out << std::endl;
  }

  // Write gaussian parameters
  for (unsigned int g=0; g<num_gaussians(); g++) {
    for (int i=0; i<gaussians[g].lambda.size(); i++)
      out << gaussians[g].lambda(i) << " ";
    out << std::endl;
  }
}


void
Scgmm::read_basis(const std::string &filename)
{
  std::ifstream in(filename.c_str());
  if (!in) {
    fprintf(stderr, "Scgmm::read_basis(): could not open %s\n", 
	    filename.c_str());
    assert(false);
  }
  
  int b_temp=0;
  std::string cov_str;

  // Read header line
  in >> m_fea_dim >> cov_str;
  m_exp_dim=m_fea_dim+m_fea_dim*(m_fea_dim+1)/2;

  if (!(cov_str == "scgmm")) {
    throw std::string("Basis type not scgmm");
    assert(false);
  }

  in >> b_temp;
  reset_basis(b_temp, m_fea_dim);

  // Read exponential basis
  for (int b=0; b<basis_dim(); b++) {
    for (int i=0; i<m_exp_dim; i++)
      in >> basis_theta[b](i);
    for (int i=0; i<m_fea_dim; i++)
      basis_psi[b](i)=basis_theta[b](i);
    for (int i=m_fea_dim; i<m_exp_dim; i++)
      basis_Pvec[b](i-m_fea_dim)=basis_theta[b](i);

    LinearAlgebra::map_v2m(basis_Pvec[b], basis_P[b]);
  }
  assert(LinearAlgebra::is_spd(basis_P[0]));
}


void
Scgmm::write_basis(const std::string &filename)
{
  std::ofstream out(filename.c_str());

  // Write header line
  out << fea_dim() << " scgmm " << basis_dim() << std::endl;
  
  // Write exponential basis
  for (int b=0; b<basis_dim(); b++) {
    for (unsigned int i=0; i<exp_dim(); i++)
      out << basis_theta[b](i) << " ";
    out << std::endl;
  }
}


double
Scgmm::K(const LaGenMatDouble &precision,
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
Scgmm::K(const LaVectorDouble &theta)
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
Scgmm::H(const LaVectorDouble &theta,
	 const LaVectorDouble &f)
{
  double result;
  result = K(theta);
  result += Blas_Dot_Prod(theta, f);
  
  /* debug
  std::cout << "K value: " << result << std::endl;
  LaVectorDouble mu, sample_mu;
  LaVectorDouble sigma, sample_sigma;
  f_to_gaussian_params(f, sample_mu, sample_sigma);
  theta_to_gaussian_params(theta, mu, sigma);
  
  std::cout << "H value: " << result << std::endl;
  std::cout << "Kullback-Leibler value: " 
	    << kullback_leibler(sample_mu, sample_sigma, mu, sigma)
	    << std::endl;
  */

  return result;
}


double
Scgmm::G(const LaVectorDouble &mean,
	 const LaGenMatDouble &precision,
	 const LaVectorDouble &sample_mean,
	 const LaGenMatDouble &sample_cov)
{
  LaGenMatDouble Sigma(sample_cov);
  LaGenMatDouble C(sample_cov);
  double g=log(LinearAlgebra::determinant(precision));
  LaVectorDouble t(mean);
  Blas_Add_Mult(t, -1, sample_mean);
  Blas_R1_Update(Sigma, t, t, 1);

  Blas_Mat_Mat_Mult(Sigma, precision, C, 1.0, 0.0);
  g-=C.trace();
  
  /* debug
  LaVectorLongInt pivots(precision.rows(),1);
  LaGenMatDouble mtemp(precision);
  LUFactorizeIP(mtemp, pivots);
  LaLUInverseIP(mtemp, pivots);
  std::cout << "G value: " << g << std::endl;
  std::cout << "Kullback-Leibler value: " 
	    << kullback_leibler(sample_mean,sample_cov,mean,mtemp)
	    << std::endl;
  */
  
  return g;
}  


void
Scgmm::gradient_untied(const HCL_RnVector_d &lambda,
		       const LaVectorDouble &sample_mean,
		       const LaGenMatDouble &sample_secondmoment,
		       HCL_RnVector_d &grad)
{
  assert(lambda.Dim()<=basis_dim());
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
    grad(i+1)=Blas_Dot_Prod(basis_Pvec.at(i),grad_p_vec)+Blas_Dot_Prod(basis_psi.at(i), grad_psi);
}


double
Scgmm::eval_linesearch_value(const LaVectorDouble &eigs,
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
Scgmm::eval_linesearch_derivative(const LaVectorDouble &eigs,
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
Scgmm::limit_line_search(const LaGenMatDouble &R,
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
    max_interval=0.99*(-1/min);
}


double 
Scgmm::kullback_leibler(const HCL_RnVector_d &lambda1,
			const HCL_RnVector_d &lambda2)
{
  LaVectorDouble mu1, mu2;
  LaGenMatDouble sigma1, sigma2;
  
  calculate_mu(lambda1, mu1);
  calculate_mu(lambda2, mu2);
  
  calculate_covariance(lambda1, sigma1);
  calculate_covariance(lambda2, sigma2);

  assert(LinearAlgebra::is_spd(sigma1));
  assert(LinearAlgebra::is_spd(sigma2));

  return kullback_leibler(mu1, sigma1, mu2, sigma2);
}


double 
Scgmm::kullback_leibler(const LaVectorDouble &mu1,
			const LaGenMatDouble &sigma1,
			const LaVectorDouble &mu2,
			const LaGenMatDouble &sigma2)
{
  assert(mu1.size()==mu2.size());
  assert(sigma1.rows()==sigma2.rows());
  assert(sigma1.cols()==sigma2.cols());
  assert(sigma1.rows()==sigma1.cols());
  assert(sigma1.rows()==mu1.size());
  
  LaVectorLongInt pivots(sigma1.cols(),1);
  LaGenMatDouble t1(sigma2),t2(sigma2);
  LaVectorDouble t3(mu2), t4(mu1);
  LUFactorizeIP(t1, pivots);
  LaLUInverseIP(t1, pivots);
  
  Blas_Mat_Mat_Mult(t1, sigma1, t2, 1.0, 0.0);
  Blas_Add_Mult(t3, -1, mu1); 
  Blas_Mat_Vec_Mult(t1, t3, t4);
  
  double value=LinearAlgebra::determinant(sigma2)
    /LinearAlgebra::determinant(sigma1);
  value = log2(value);  
  value += t2.trace();
  value += Blas_Dot_Prod(t3, t4);
  value -= sigma1.cols();
  value *= 0.5;

  return value;
}


void
Scgmm::f_to_gaussian_params(const LaVectorDouble &f,
			    LaVectorDouble &sample_mu,
			    LaGenMatDouble &sample_sigma)
{
  int d=fea_dim();
  int d_exp=exp_dim();
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
Scgmm::gaussian_params_to_f(const LaVectorDouble &sample_mu,
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
Scgmm::theta_to_gaussian_params(const LaVectorDouble &theta,
				LaVectorDouble &mu,
				LaGenMatDouble &sigma)
{
  int d=fea_dim();
  int d_vec=(int)(d*(d+1)/2);
  int d_exp=d+d_vec;

  // psi
  LaVectorDouble psi(theta(LaIndex(0,d-1)));

  // sigma
  sigma.resize(d,d);  
  LaVectorDouble t(theta(LaIndex(d,d_exp-1)));
  LinearAlgebra::map_v2m(t, sigma);
  LaVectorLongInt pivots(fea_dim());
  LUFactorizeIP(sigma, pivots);
  LaLUInverseIP(sigma, pivots);

  // mu
  mu.resize(d,1);
  Blas_Mat_Vec_Mult(sigma, psi, mu);

  assert(LinearAlgebra::is_spd(sigma));
}


void
Scgmm::optimize_lambda(const LaGenMatDouble &sample_cov,
		       const LaVectorDouble &sample_mean,
		       LaVectorDouble &lambda)  
{
  assert(sample_cov.rows()==sample_cov.cols());
  assert(sample_cov.rows()==sample_mean.size());

  int d=lambda.size();
  ScgmmLambdaFcnl f(d, this, sample_cov, sample_mean);

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
