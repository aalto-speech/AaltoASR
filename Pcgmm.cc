#include "Pcgmm.hh"


void
Pcgmm::copy(const Pcgmm &orig)
{
  assert(orig.mbasis.size()==orig.vbasis.size());

  mbasis.resize(orig.mbasis.size());
  vbasis.resize(orig.vbasis.size());
  gaussians.resize(orig.gaussians.size());
  quadratic_feas.resize(orig.quadratic_feas.size(),1);

  for (unsigned int i=0; i<orig.mbasis.size(); i++) {
    mbasis.at(i).copy(orig.mbasis.at(i));
    vbasis.at(i).copy(orig.mbasis.at(i));
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
    matrix_power(t, t2, 0.5);
    gaussians.at(i).bias = log(determinant(t))
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
Pcgmm::map_m2v(const LaGenMatDouble &m,
	       LaVectorDouble &v)
{
  assert(m.rows()==m.cols());

  int dim=m.rows(), pos=0;
  v.resize((int)(dim*(dim+1)/2),1);
  
  for (int i=0; i<dim; i++)
    for (int j=0; i+j<dim; j++) {
      // Multiply off-diagonal elements by sqrt(2) 
      // to preserve inner products
      if (j!=j+i)
	v(pos)=sqrt(2)*m(j,j+i);
      else
	v(pos)=m(j,j+i);
      ++pos;
    }
}


void
Pcgmm::map_v2m(const LaVectorDouble &v,
	       LaGenMatDouble &m)
{
  // Deduce the matrix dimensions; numel(v)=dim*(dim+1)/2
  int dim=(int)(0.5*sqrt(1+8*v.size())-0.5);
  int pos=0;
  float a=1/sqrt(2);

  assert(int(dim*(dim+1)/2)==v.size());

  m.resize(dim,dim);

  for (int i=0; i<dim; i++)
    for (int j=0; i+j<dim; j++)
      {
	// Divide off-diagonal elements by sqrt(2)
	if (j+i!=j) {
	  m(j,j+i) = a*v(pos);
	  m(j+i,j) = a*v(pos);
	}
	else
	  m(j,j) = v(pos);
	++pos;
      }
}


void 
Pcgmm::initialize_basis_svd(const std::vector<int> &c,
			    const std::vector<LaGenMatDouble> &sample_covs, 
			    const unsigned int basis_dim)
{
  assert(c.size() == sample_covs.size());

  int d=sample_covs.at(0).rows();
  int d_vec=(int)(d*(d+1)/2);
  int c_sum=0;

  // Reset current basis
  reset(basis_dim, d, num_gaussians());

  // Calculate c_sum
  for (unsigned int i=0; i<c.size(); i++)
    c_sum+=c.at(i);

  // Calculate mean covariance
  LaGenMatDouble m=LaGenMatDouble::zeros(d);
  LaGenMatDouble identity=LaGenMatDouble::eye(d);
  for (unsigned int i=0; i<c.size(); i++) {
    Blas_Mat_Mat_Mult(identity, sample_covs.at(i), m, c.at(i)/c_sum, 0);
  }
  
  // Calculate sample precisions and project them so that the inner
  // product <U,V>_m == Tr(mUmV) is preserved in vector mapping
  std::vector<LaGenMatDouble> sample_precs;
  sample_precs.resize(basis_dim);
  for (unsigned int i=0; i<basis_dim; i++) {
    // Calculate precisions
    sample_precs.at(i).copy(sample_covs.at(i));
    LaVectorLongInt pivots(d);
    LUFactorizeIP(sample_precs.at(i), pivots);
    LaLUInverseIP(sample_precs.at(i), pivots);
    // Left-side multiplication by m
    Blas_Mat_Mat_Mult(m, sample_precs.at(i), sample_precs.at(i), 1, 0);
  }

  // Map sample precisions to vectors
  std::vector<LaVectorDouble> sample_prec_vectors;
  sample_prec_vectors.resize(basis_dim);
  for (unsigned int i=0; i<basis_dim; i++) {
    map_m2v(sample_precs.at(i), sample_prec_vectors.at(i));
  }
  
  // Generate space V
  LaGenMatDouble V=LaGenMatDouble::zeros(d_vec, d_vec);
  for (unsigned int i=0; i<c.size(); i++)
    Blas_R1_Update(V, sample_prec_vectors.at(i), sample_prec_vectors.at(i), 0);
  
  // Singular value decomposition for V
  LaGenMatDouble left_singulars(d_vec,d_vec);
  LaGenMatDouble right_singulars(d_vec,d_vec);
  LaVectorDouble sigma(d_vec);
  LaSVD_IP(V, sigma, left_singulars, right_singulars);

  // First basis matrix is the mean precision matrix
  mbasis.at(0).copy(m);
  LaVectorLongInt pivots(d);
  LUFactorizeIP(mbasis.at(0), pivots);
  LaLUInverseIP(mbasis.at(0), pivots);
  map_m2v(mbasis.at(0), vbasis.at(0));

  // Other basis matrices are top singular vectors of V
  for (unsigned int i=1; i<basis_dim; i++) {
    vbasis.at(i).copy(left_singulars.row(i));
    map_v2m(vbasis.at(i), mbasis.at(i));
  }
}  


void 
Pcgmm::reset(const unsigned int basis_dim, 
	     const unsigned int d,
	     const unsigned int g)
{
  unsigned int d_vec=(unsigned int)d*(d+1)/2;
  mbasis.resize(basis_dim);
  vbasis.resize(basis_dim);
  quadratic_feas.resize(basis_dim,1);

  gaussians.resize(g);
  for (unsigned int i=0; i<g; i++)
    gaussians.at(i).resize(d, basis_dim);
  likelihoods.resize(g);

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
    map_m2v(mbasis[b], vbasis[b]);
  }

  // Read gaussian parameters
  for (int g=0; g<gauss; g++) {
    for (int i=0; i<fea_dim; i++)
      in >> gaussians[g].mu(i);
    for (int i=0; i<basis_dim; i++)
      in >> gaussians[g].lambda(i);
  }
  
  precompute();
}


void
Pcgmm::write_gk(const std::string &filename)
{
  std::ofstream out(filename.c_str());

  // Write header line
  out << num_gaussians() << " " << fea_dim()
      << " pcgmm " << basis_dim() << std::endl;
  
  // Write precision basis
  for (int b=0; b<basis_dim(); b++) {
    for (unsigned int i=0; i<fea_dim(); i++)
      for (unsigned int j=0; j<fea_dim(); j++)
	out << mbasis[b](i,j) << " ";
    out << std::endl;
  }

  // Write gaussian parameters
  for (unsigned int g=0; g<num_gaussians(); g++) {
    for (unsigned int i=0; i<fea_dim(); i++)
      out << gaussians[g].mu(i) << " ";
    for (int i=0; i<basis_dim(); i++)
      out << gaussians[g].lambda(i) << " ";
    out << std::endl;
  }
}


void
Pcgmm::calculate_precision(const LaVectorDouble &lambda,
			   LaGenMatDouble &precision)
{
  assert(lambda.size() == basis_dim());
  
  // Current precisision
  precision.resize(fea_dim(),fea_dim());
  LaGenMatDouble identity = LaGenMatDouble::eye(fea_dim());
  for (int b=0; b<lambda.size(); b++)
    Blas_Mat_Mat_Mult(identity, mbasis[b], precision, lambda(b), 1);
}


void
Pcgmm::calculate_precision(const LaVectorDouble &lambda,
			   LaVectorDouble &precision)
{
  // Calculate precision in matrix form
  LaGenMatDouble mprecision;
  // Convert to vector
  calculate_precision(lambda, mprecision);
  map_m2v(mprecision, precision);
}


void
Pcgmm::calculate_covariance(const LaVectorDouble &lambda,
			    LaGenMatDouble &covariance)
{
  LaVectorLongInt pivots(basis_dim());
  // Calculate precision
  calculate_precision(lambda, covariance);
  // Invert precision -> covariance
  LUFactorizeIP(covariance, pivots);
  LaLUInverseIP(covariance, pivots);
}


void
Pcgmm::calculate_covariance(const LaVectorDouble &lambda,
			    LaVectorDouble &covariance)
{
  // Calculate precision in matrix form
  LaGenMatDouble mcovariance;
  // Convert to vector
  calculate_covariance(lambda, mcovariance);
  map_m2v(mcovariance, covariance);
}


void
Pcgmm::gradient(const LaVectorDouble &lambda,
		const LaGenMatDouble &sample_cov, 
		LaVectorDouble &grad)
{
  assert(sample_cov.rows() == sample_cov.cols());
  assert(lambda.size() == basis_dim());

  grad.resize(basis_dim(),1);

  LaGenMatDouble curr_cov_estimate;
  calculate_covariance(lambda, curr_cov_estimate);

  for (unsigned int i=0; i<fea_dim(); i++)
    for (unsigned int j=0; j<fea_dim(); j++)
      curr_cov_estimate(i,j) -= sample_cov(i,j);

  for (int i=0; i<basis_dim(); i++) {
    LaGenMatDouble projection;
    Blas_Mat_Mat_Mult(curr_cov_estimate, mbasis.at(i), projection);
    grad(i)=projection.trace();
  }
}


void
Pcgmm::train_precision_polak_ribiere(int state,
				     LaGenMatDouble &sample_cov,
				     int iter)
{
  // Initialize
  double min_interval, max_interval, best_step;
  LaVectorDouble old_grad, new_grad, old_d, new_d, old_lambda, new_lambda;
  LaGenMatDouble R, curr_precision;

  old_grad.resize(basis_dim(),1);
  new_grad.resize(basis_dim(),1);
  old_d.resize(basis_dim(),1);
  new_d.resize(basis_dim(),1);
  old_lambda.resize(basis_dim(),1);
  new_lambda.resize(basis_dim(),1);
  calculate_precision(gaussians.at(state).lambda, curr_precision);

  for (int i=0; i<iter; i++) {
    old_grad.copy(new_grad);
    old_d.copy(new_d);

    gradient(gaussians.at(state).lambda, sample_cov, new_grad);
    polak_ribiere_direction(old_grad, new_grad, old_d, new_d);
    
    calculate_precision(new_d, R);
    limit_line_search(R, curr_precision, min_interval, max_interval);

    best_step=line_search_more_thuente(curr_precision, R, sample_cov,
				       min_interval, max_interval);
    best_step=std::max(best_step, 0.0000001);

    old_lambda.copy(new_lambda);
    Blas_Add_Mult(new_lambda, best_step, new_d);

    calculate_precision(new_lambda, curr_precision);
  }

  gaussians.at(state).lambda.copy(new_lambda);
}


void
Pcgmm::polak_ribiere_direction(const LaVectorDouble &old_grad,
			       const LaVectorDouble &new_grad,
			       const LaVectorDouble &old_direction,
			       LaVectorDouble &new_direction)
{
  assert(old_grad.size() == (int)basis_dim());
  assert(new_grad.size() == (int)basis_dim());
  assert(old_direction.size() == (int)basis_dim());

  new_direction.resize(basis_dim(),1);
  new_direction.copy(new_grad);

  LaVectorDouble diff_grad=LaVectorDouble(new_grad);
  Blas_Add_Mult(diff_grad, -1, old_grad);

  double n = Blas_Dot_Prod(new_grad, diff_grad);
  n /= Blas_Dot_Prod(old_grad, old_grad);

  Blas_Add_Mult(new_direction, n, old_direction);
}



double
Pcgmm::line_search_more_thuente(const LaGenMatDouble &P,
				const LaGenMatDouble &R,
				const LaGenMatDouble &W,
				const double min_interval,
				const double max_interval,
				int iter,
				double trial_init,
				double trial_add)
{
  // Calculate generalized eigenvalues of pair (R,P)
  LaVectorDouble eigs;
  generalized_eigenvalues(R, P, eigs);
  
  // Calculate trace of W*R
  LaGenMatDouble wr;
  Blas_Mat_Mat_Mult(W, R, wr);
  double trace = wr.trace();
  
  // Initialize bounds and trial values
  double a_lower=0, a_upper=DBL_MAX, a_trial=0,
    f_trial=0, f_trial_derivative=0, 
    f_lower=0, f_upper=0,
    t=0;
  
  for (int i=0; i<iter; i++) {
    // Calculate function values for the lower/upper limit
    f_lower = eval_aux_function(eigs, a_lower, trace);
    f_upper = eval_aux_function(eigs, a_upper, trace);

    // Calculate function/derivative values for the trial value
    a_trial = trial_init*(a_upper-a_lower)+a_lower;
    f_trial = eval_aux_function(eigs, a_trial, trace);
    f_trial_derivative = eval_aux_function_derivative(eigs, a_trial, trace);

    // Generate new trial values so long that the stopping criterion is met
    while (f_trial<=f_lower && (f_trial_derivative*(a_lower-a_trial)>0)) {
      a_trial = max_interval;
      t = a_trial + trial_add*(a_trial-a_lower);
      if (t < a_trial)
	a_trial = t;
      f_trial = eval_aux_function(eigs, a_trial, trace);
      f_trial_derivative = eval_aux_function_derivative(eigs, a_trial, trace);
      if (a_trial == max_interval) break;
    }

    // Deduce new upper and lower bounds
    if (f_trial>f_lower)
      a_upper = a_trial;
    else if (f_trial<=f_lower && f_trial_derivative*(a_lower-a_trial)>0)
      a_lower = a_trial;
    else if (f_trial<=f_lower && f_trial_derivative*(a_lower-a_trial)<0) {
      a_upper = a_lower;
      a_lower = a_trial;
    }

  }
  
  if (max_interval<a_trial)
    return max_interval;
  else
    return a_trial;
}


double
Pcgmm::eval_aux_function(const LaVectorDouble &eigs,
			 double step,
			 double trace)
{
  double f=0;
  for (int i=0; i<eigs.size(); i++)
    f += (log(1+step*eigs(i))-step*trace);
  return f;
}


double
Pcgmm::eval_aux_function_derivative(const LaVectorDouble &eigs,
				    double step,
				    double trace)
{
  double d=trace;
  for (int i=0; i<eigs.size(); i++)
    d += eigs(i)/(1+step*eigs(i));
  return d;
}


void
Pcgmm::limit_line_search(const LaGenMatDouble &R,
			 const LaGenMatDouble &curr_prec_estimate,
			 double &min_interval,
			 double &max_interval)
{
  assert(R.rows()==R.cols());
  assert(curr_prec_estimate.rows()==curr_prec_estimate.cols());
  assert(R.rows()==curr_prec_estimate.rows());

  LaVectorDouble eigs;
  generalized_eigenvalues(R, curr_prec_estimate, eigs);
  double min=DBL_MAX;
  double max=-DBL_MAX;

  // First limit the line search based 
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
    max_interval=-1/min;

  if (max<0)
    min_interval=-DBL_MAX;
  else
    min_interval=-1/max;

  // Then limit more to ensure positive defitiveness
  // of the resulting precision matrix estimate
  LaGenMatDouble t=LaGenMatDouble(curr_prec_estimate);
  LaGenMatDouble id=LaGenMatDouble::eye(R.rows());

  double min_lower=min_interval, min_upper=0, step=0;
  while (min_upper-min_lower>0.0001) {
    step=min_lower+(min_upper-min_lower)/2;
    Blas_Mat_Mat_Mult(id, R, t, step, 1.0);
    if (is_spd(t))
      min_upper=step;
    else
      min_lower=step;
    t.copy(curr_prec_estimate);
  }

  double max_lower=0, max_upper=max_interval;
  while (max_upper-max_lower>0.0001) {
    step=max_lower+(max_upper-max_lower)/2;
    Blas_Mat_Mat_Mult(id, R, t, step, 1.0);
    if (is_spd(t))
      max_lower=step;
    else
      max_upper=step;
    t.copy(curr_prec_estimate);
  }

  if (min_upper>min_interval)
    min_interval=min_upper;

  if (max_lower<max_interval)
    max_interval=max_lower;
}


void
Pcgmm::generalized_eigenvalues(const LaGenMatDouble &A,
			       const LaGenMatDouble &B,
			       LaVectorDouble &eigs)
{
  assert(A.rows()==A.cols());
  assert(B.rows()==B.cols());
  assert(A.rows()==B.rows());
  
  // Initialize
  LaGenMatDouble B_chol=LaGenMatDouble(B);
  eigs.resize(A.rows(),1);
  
  // Cholesky factorization for B
  cholesky_factor(B, B_chol);
  LaGenMatDouble B_chol_inv=LaGenMatDouble(B_chol);

  // Invert cholesky factorization of B
  LaVectorLongInt pivots(A.rows());
  LUFactorizeIP(B_chol_inv, pivots);
  LaLUInverseIP(B_chol_inv, pivots);
  
  // C = transpose(inv(chol(B)))*A*inv(chol(B))
  LaGenMatDouble C=LaGenMatDouble::zeros(A.rows());
  LaGenMatDouble t=LaGenMatDouble::zeros(A.rows());
  Blas_Mat_Trans_Mat_Mult(B_chol_inv, A, t);
  Blas_Mat_Mat_Mult(t, B_chol_inv, C, 1.0, 1.0);
  
  // Eigenvalues of C are the desired generalized eigenvalues for matrix pair (A,B)
  LaEigSolveSymmetricVecIP(C, eigs);
}


void
Pcgmm::cholesky_factor(const LaGenMatDouble &A,
		       LaGenMatDouble &B)
{
  assert(A.rows() == A.cols());

  B.resize(A.rows(), A.cols());
  B.copy(A);
  
  for (int j=0; j<B.rows(); j++) 
    {
      for (int k=0; k<j; k++)
	for (int i=j; i<B.rows(); i++)
	  B(i,j) = B(i,j)-B(i,k)*B(j,k);
      B(j,j) = sqrt(B(j,j));
      for (int k=j+1; k<B.rows(); k++)
	B(k,j) = B(k,j)/B(j,j);
    }
  
  for (int i=0; i<B.rows(); i++)
    for (int j=i+1; j<B.cols(); j++)
      B(i,j) = 0;
}


// FIXME: perhaps not the fastest spd check
bool
Pcgmm::is_spd(const LaGenMatDouble &A)
{
  assert(A.rows()==A.cols());

  LaGenMatDouble t=LaGenMatDouble(A);
  LaVectorDouble eigs=LaVectorDouble(A.rows());
  LaEigSolveSymmetricVecIP(t,eigs);

  for (int i=0; i<eigs.size(); i++)
    if (eigs(i)<0.0)
      return false;

  return true;
}


double
Pcgmm::determinant(const LaGenMatDouble &A)
{
  assert(A.rows()==A.cols());
  LaGenMatDouble chol;
  cholesky_factor(A, chol);
  double det=1;
  for (int i=0; i<chol.rows(); i++)
    det *= chol(i,i);
  det *=det;

  return det;
}


void 
Pcgmm::matrix_power(const LaGenMatDouble &A,
		    LaGenMatDouble &B,
		    double power)
{
  LaVectorDouble D = LaVectorDouble(A.rows(),1);
  LaGenMatDouble V = LaGenMatDouble(A);

  LaEigSolveSymmetricVecIP(V, D);

  LaGenMatDouble V_inverse = LaGenMatDouble(V);  
  LaVectorLongInt pivots(A.rows());
  LUFactorizeIP(V_inverse, pivots);
  LaLUInverseIP(V_inverse, pivots);

  B.resize(A.rows(), A.cols());
  for (int i=0; i<A.rows(); i++)
    for (int j=0; j<A.cols(); j++)
      V_inverse(i,j)=V_inverse(i,j) * pow(D(j),power);
  Blas_Mat_Mat_Mult(V, V_inverse, B);
}
