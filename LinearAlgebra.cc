#include "LinearAlgebra.hh"


double
LinearAlgebra::determinant(const LaGenMatDouble &A)
{
  assert(A.rows()==A.cols());
  assert(is_spd(A));

  LaGenMatDouble chol;
  cholesky_factor(A, chol);
  double det=1;
  for (int i=0; i<chol.rows(); i++)
    det *= chol(i,i);
  det *=det;

  return det;
}


void 
LinearAlgebra::matrix_power(const LaGenMatDouble &A,
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
  LaGenMatDouble t=LaGenMatDouble::zeros(A.rows());
  LaGenMatDouble t2=LaGenMatDouble::zeros(A.rows());
  for (int i=0; i<t.rows(); i++)
    t(i,i)=pow(D(i),power);
  Blas_Mat_Mat_Mult(V, t, t2, 1.0, 0.0);
  Blas_Mat_Mat_Mult(t2, V_inverse, B, 1.0, 0.0);
}


void
LinearAlgebra::cholesky_factor(const LaGenMatDouble &A,
			       LaGenMatDouble &B)
{
  assert(A.rows() == A.cols());
  assert(is_spd(A));

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


void
LinearAlgebra::generalized_eigenvalues(const LaGenMatDouble &A,
				       const LaGenMatDouble &B,
				       LaVectorDouble &eigvals,
				       LaGenMatDouble &eigvecs)
{
  assert(A.rows()==A.cols());
  assert(B.rows()==B.cols());
  assert(A.rows()==B.rows());
  assert(is_spd(B));

  LaGenMatDouble B_negsqrt;
  LaGenMatDouble t(A);
  eigvals.resize(A.rows(),1);
  eigvecs.resize(A.rows(),A.cols());
  
  matrix_power(B, B_negsqrt, -0.5);
  Blas_Mat_Mat_Mult(B_negsqrt, A, t, 1.0, 0.0);
  Blas_Mat_Mat_Mult(t, B_negsqrt, eigvecs, 1.0, 0.0);
  
  LaEigSolveSymmetricVecIP(eigvecs, eigvals);
}


void
LinearAlgebra::map_m2v(const LaGenMatDouble &m,
		       LaVectorDouble &v)
{
  assert(m.rows()==m.cols());

  int dim=m.rows(), pos=0;
  v.resize((int)(dim*(dim+1)/2),1);
  
  for (int i=0; i<dim; i++)
    for (int j=0; j<=i; j++) {
      // Multiply off-diagonal elements by sqrt(2) 
      // to preserve inner products
      if (i==j)
	v(pos)=m(i,j);
      else
	v(pos)=sqrt(2)*m(i,j);
      ++pos;
    }
}


void
LinearAlgebra::map_v2m(const LaVectorDouble &v,
		       LaGenMatDouble &m)
{
  // Deduce the matrix dimensions; numel(v)=dim*(dim+1)/2
  int dim=(int)(0.5*sqrt(1+8*v.size())-0.5);
  int pos=0;
  float a=1/sqrt(2);

  assert(int(dim*(dim+1)/2)==v.size());

  m.resize(dim,dim);

  for (int i=0; i<dim; i++)
    for (int j=0; j<=i; j++)
      {
	// Divide off-diagonal elements by sqrt(2)
	if (i==j) 
	  m(j,j) = v(pos);
	else {
	  m(i,j) = a*v(pos);
	  m(j,i) = a*v(pos);
	}
	++pos;
      }
}


void
LinearAlgebra::map_mtlm2v(const Matrix &m,
			  LaVectorDouble &v)
{
  assert(m.nrows()==m.ncols());
  
  int dim=m.nrows(), pos=0;
  v.resize((int)(dim*(dim+1)/2),1);
  
  for (int i=0; i<dim; i++)
    for (int j=0; j<=i; j++) {
      // Multiply off-diagonal elements by sqrt(2) 
      // to preserve inner products
      if (i==j)
	v(pos)=m(i,j);
      else
	v(pos)=sqrt(2)*m(i,j);
      ++pos;
    }
}


void
LinearAlgebra::map_v2mtlm(const LaVectorDouble &v,
			  Matrix &m)
{
  // Deduce the matrix dimensions; numel(v)=dim*(dim+1)/2
  int dim=(int)(0.5*sqrt(1+8*v.size())-0.5);
  int pos=0;
  float a=1/sqrt(2);
  
  assert(int(dim*(dim+1)/2)==v.size());
  
  m.resize(dim,dim);
  
  for (int i=0; i<dim; i++)
    for (int j=0; j<=i; j++)
      {
	// Divide off-diagonal elements by sqrt(2)
	if (i==j) 
	  m(j,j) = v(pos);
	else {
	  m(i,j) = a*v(pos);
	  m(j,i) = a*v(pos);
	}
	++pos;
      }
}


void
LinearAlgebra::map_hclv2lapackm(const HCL_RnVector_d &v, 
				LaGenMatDouble &m)
{
  // Deduce the matrix dimensions; numel(v)=dim*(dim+1)/2
  int dim=(int)(0.5*sqrt(1+8*v.Dim())-0.5);
  int pos=0;
  float a=1/sqrt(2);
  
  assert(int(dim*(dim+1)/2)==v.Dim());
  
  m.resize(dim,dim);
  
  for (int i=0; i<dim; i++)
    for (int j=0; j<=i; j++)
      {
	// Divide off-diagonal elements by sqrt(2)
	if (i==j) 
	  m(j,j) = v(pos);
	else {
	  m(i,j) = a*v(pos);
	  m(j,i) = a*v(pos);
	}
	++pos;
      }
}


void
LinearAlgebra::map_lapackm2hclv(const LaGenMatDouble &m,
				HCL_RnVector_d &v)
{
  assert(m.rows()==m.cols());
  assert(m.rows()==v.Dim());

  int dim=m.rows(), pos=0;
  
  for (int i=0; i<dim; i++)
    for (int j=0; j<=i; j++) {
      // Multiply off-diagonal elements by sqrt(2) 
      // to preserve inner products
      if (i==j)
	v(pos)=m(i,j);
      else
	v(pos)=sqrt(2)*m(i,j);
      ++pos;
    }
}


double
LinearAlgebra::cond(const Matrix &m)
{
  assert(m.nrows()==m.ncols());

  LaGenMatDouble temp(m.nrows(), m.ncols());
  for (unsigned int i=0; i<m.nrows(); i++)
    for (unsigned int j=0; j<m.ncols(); j++)
      temp(i,j)=m(i,j);
  return cond(temp);
}


double
LinearAlgebra::cond(const LaGenMatDouble &m)
{
  assert(m.rows()==m.cols());

  double min=DBL_MAX;
  double max=-DBL_MAX;

  LaGenMatDouble a(m);
  LaVectorDouble eigs(m.rows());
  LaEigSolveSymmetricVecIP(a, eigs);

  for (int i=0; i<eigs.size(); i++) {
    if (eigs(i)<min)
      min=eigs(i);
    if (eigs(i)>max)
      max=eigs(i);
  }

  return max/min;
}


bool
LinearAlgebra::is_spd(const Matrix &m)
{
  assert(m.nrows()==m.ncols());

  LaGenMatDouble temp(m.nrows(), m.ncols());
  for (unsigned int i=0; i<m.nrows(); i++)
    for (unsigned int j=0; j<m.ncols(); j++)
      temp(i,j)=m(i,j);
  return is_spd(temp);
}


bool
LinearAlgebra::is_spd(const LaGenMatDouble &m)
{
  assert(m.rows()==m.cols());

  LaGenMatDouble t=LaGenMatDouble(m);
  LaVectorDouble eigs=LaVectorDouble(m.rows());
  LaEigSolveSymmetricVecIP(t,eigs);

  for (int i=0; i<eigs.size(); i++)
    if (eigs(i)<=0)
      return false;

  return true;
}


bool
LinearAlgebra::is_singular(const Matrix &m)
{
  assert(m.nrows()==m.ncols());

  LaGenMatDouble temp(m.nrows(), m.ncols());
  for (unsigned int i=0; i<m.nrows(); i++)
    for (unsigned int j=0; j<m.ncols(); j++)
      temp(i,j)=m(i,j);
  return is_singular(temp);
}


bool
LinearAlgebra::is_singular(const LaGenMatDouble &m)
{
  assert(m.rows()==m.cols());

  LaGenMatDouble a(m);
  LaVectorDouble eigs(m.rows());
  LaEigSolveSymmetricVecIP(a, eigs);

  for (int i=0; i<eigs.size(); i++)
    if (abs(eigs(i))<0.00000001)
      return true;
  
  return false;
}


void
LinearAlgebra::force_min_eig(Matrix &m, double min_eig)
{
  assert(m.nrows()==m.ncols());

  // Map to Lapack++ matrix
  LaGenMatDouble temp(m.nrows(), m.ncols());
  for (unsigned int i=0; i<m.nrows(); i++)
    for (unsigned int j=0; j<m.ncols(); j++)
      temp(i,j)=m(i,j);
  // Force minimum eigenvalues
  force_min_eig(temp, min_eig);
  // Map back to MTL matrix
  for (unsigned int i=0; i<m.nrows(); i++)
    for (unsigned int j=0; j<m.ncols(); j++)
      m(i,j)=temp(i,j);  
}


void
LinearAlgebra::force_min_eig(LaGenMatDouble &m, double min_eig)
{
  assert(m.rows()==m.cols());

  LaGenMatDouble eigv(m);
  LaVectorDouble eigs_vector(m.rows());
  LaGenMatDouble eigs = LaGenMatDouble::zeros(m.rows());
  LaGenMatDouble temp(m.rows(), m.cols());

  LaEigSolveSymmetricVecIP(eigv, eigs_vector);
  for (int i=0; i<m.rows(); i++)
    eigs(i,i)=std::max(min_eig, eigs_vector(i));

  // Calculate EIGV*EIGS*EIGV'
  Blas_Mat_Mat_Mult(eigv, eigs, temp, 1.0, 0.0);
  Blas_Mat_Mat_Trans_Mult(temp, eigv, m, 1.0, 0.0);
}

