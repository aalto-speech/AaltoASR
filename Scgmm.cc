#include "Scgmm.hh"


void
Scgmm::copy(const Scgmm &orig)
{
  // FIXME
}

void 
Scgmm::precompute(const FeatureVec &feature)
{
  // FIXME
}

double 
Scgmm::gaussian_likelihood(const int k)
{
  // FIXME
  return 0;
}

void 
Scgmm::reset_basis(const unsigned int basis_dim, 
		   const unsigned int d)
{
  unsigned int d_vec=(unsigned int)d*(d+1)/2;
  mbasis.resize(basis_dim);
  vbasis.resize(basis_dim);

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
Scgmm::read_gk(const std::string &filename)
{
  std::ifstream in(filename.c_str());
  if (!in) {
    fprintf(stderr, "Scgmm::read_gk(): could not open %s\n", 
	    filename.c_str());
    assert(false);
  }
  
  int num_gaussians=0, basis_dim=0, fea_dim=0;
  std::string cov_str;

  // Read header line
  in >> num_gaussians >> fea_dim >> cov_str;
  
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
    //    map_m2v(mbasis[b], vbasis[b]);
  }

  // Read gaussian parameters
  for (int g=0; g<num_gaussians; g++) {
    for (int i=0; i<basis_dim; i++)
      in >> gaussians[g].lambda(i);
  }
}


void
Scgmm::write_gk(const std::string &filename)
{
  std::ofstream out(filename.c_str());

  // Write header line
  out << num_gaussians() << " " << fea_dim()
      << " pcgmm " << basis_dim() << std::endl;
  
  // Write precision basis
  for (unsigned int b=0; b<basis_dim(); b++) {
    for (unsigned int i=0; i<fea_dim(); i++)
      for (unsigned int j=0; j<fea_dim(); j++)
	out << mbasis[b](i,j) << " ";
    out << std::endl;
  }

  // Write gaussian parameters
  for (unsigned int g=0; g<num_gaussians(); g++) {
    for (unsigned int i=0; i<basis_dim(); i++)
      out << gaussians[g].lambda(i) << " ";
    out << std::endl;
  }
}


void Scgmm::gradient(const LaVectorDouble &lambda,
		     const LaGenMatDouble &sample_cov, 
		     LaVectorDouble &grad)
{

}


void Scgmm::polak_ribiere_direction(const LaVectorDouble &old_grad,
				    const LaVectorDouble &new_grad,
				    const LaVectorDouble &old_direction,
				    LaVectorDouble &new_direction)
{

}



double Scgmm::line_search_more_thuente(const LaGenMatDouble &P,
				const LaGenMatDouble &R,
				const LaGenMatDouble &W,
				const double min_interval,
				const double max_interval,
				int iter,
				double trial_init,
				double trial_add)
{
  return 0;
}


double Scgmm::eval_aux_function(const LaVectorDouble &eigs,
				double step,
				double trace)
{
  return 0;
}


double Scgmm::eval_aux_function_derivative(const LaVectorDouble &eigs,
					   double step,
					   double trace)
{
  return 0;
}


void Scgmm::limit_line_search(const LaGenMatDouble &R,
			      const LaGenMatDouble &curr_prec_estimate,
			      double &min_interval,
			      double &max_interval)
{
  
}


void Scgmm::generalized_eigenvalues(const LaGenMatDouble &A,
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


void Scgmm::cholesky_factor(const LaGenMatDouble &A, LaGenMatDouble &B)
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
bool Scgmm::is_spd(const LaGenMatDouble &A)
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
