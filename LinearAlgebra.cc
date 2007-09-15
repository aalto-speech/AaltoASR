#include "LinearAlgebra.hh"
#include "util.hh"


namespace LinearAlgebra {

  double
  spd_log_determinant(const Matrix &A)
  {
    assert(A.rows()==A.cols());
    //assert(is_spd(A));

    LaGenMatDouble chol;
    cholesky_factor(A, chol);
    double log_det=0;
    for (int i=0; i<chol.rows(); i++)
      log_det += log(chol(i,i));
    log_det *= 2;

    return log_det;
  }
  
  double
  spd_determinant(const Matrix &A)
  {
    assert(A.rows()==A.cols());
    //assert(is_spd(A));

    LaGenMatDouble chol;
    cholesky_factor(A, chol);
    double det=1;
    for (int i=0; i<chol.rows(); i++)
      det *= chol(i,i);
    det *=det;

    return det;
  }


  double
  determinant(const Matrix &A)
  {
    assert(A.rows()==A.cols());

    LaGenMatDouble B(A);
    LaVectorDouble eigvals(A.cols());
    LaEigSolveSymmetricVecIP(B, eigvals);
    double det=1;
    for (int i=0; i<eigvals.size(); i++)
      det *= eigvals(i);
    
    return det;
  }


  double
  log_determinant(const Matrix &A)
  {
    assert(A.rows()==A.cols());

    LaGenMatDouble B(A);
    LaVectorDouble eigvals(A.cols());
    LaEigSolveSymmetricVecIP(B, eigvals);
    double log_det=0;
    for (int i=0; i<eigvals.size(); i++)
      log_det += util::safe_log(eigvals(i));
    
    return log_det;
  }


  void 
  matrix_power(const Matrix &A,
               Matrix &B,
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
  cholesky_factor(const Matrix &A,
                  Matrix &B)
  {
    assert(A.rows() == A.cols());
    //assert(is_spd(A));

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
  generalized_eigenvalues(const Matrix &A,
                          const Matrix &B,
                          Vector &eigvals,
                          Matrix &eigvecs)
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
  generalized_eigenvalues(const Matrix &A,
                          const Matrix &B,
                          LaVectorComplex &eigvals,
                          LaGenMatComplex &eigvecs)
  {
    assert(A.rows()==A.cols());
    assert(B.rows()==B.cols());
    assert(A.rows()==B.rows());
    assert(is_spd(B));

    LaGenMatDouble B_negsqrt;
    LaGenMatDouble t(A);
    LaGenMatDouble t2(A);
    eigvals.resize(A.rows(),1);
    eigvecs.resize(A.rows(),A.cols());
  
    matrix_power(B, B_negsqrt, -0.5);
    Blas_Mat_Mat_Mult(B_negsqrt, A, t, 1.0, 0.0);
    Blas_Mat_Mat_Mult(t, B_negsqrt, t2, 1.0, 0.0);

    LaGenMatComplex c(t2);
    LaEigSolve(c, eigvals, eigvecs);
  }


  void
  map_m2v(const Matrix &m,
          Vector &v)
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
  map_v2m(const Vector &v,
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

/*
  void
  map_mtlm2v(const Matrix &m,
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
  map_v2mtlm(const Vector &v,
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
*/

  void
  map_hclv2lapackm(const HCL_RnVector_d &v, 
                   Matrix &m)
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
  map_lapackm2hclv(const Matrix &m,
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

/*
  double
  cond(const Matrix &m)
  {
  assert(m.nrows()==m.ncols());

  LaGenMatDouble temp(m.nrows(), m.ncols());
  for (unsigned int i=0; i<m.nrows(); i++)
  for (unsigned int j=0; j<m.ncols(); j++)
  temp(i,j)=m(i,j);
  return cond(temp);
  }
*/

  double
  cond(const Matrix &m)
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

/*
  bool
  is_spd(const Matrix &m)
  {
  assert(m.nrows()==m.ncols());

  LaGenMatDouble temp(m.nrows(), m.ncols());
  for (unsigned int i=0; i<m.nrows(); i++)
  for (unsigned int j=0; j<m.ncols(); j++)
  temp(i,j)=m(i,j);
  return is_spd(temp);
  }
*/

  bool
  is_spd(const Matrix &m)
  {
    assert(m.rows()==m.cols());

    LaGenMatDouble t=LaGenMatDouble(m);
    LaVectorDouble eigs=LaVectorDouble(m.rows());
    LaEigSolveSymmetricVecIP(t,eigs);

    for (int i=0; i<eigs.size(); i++) {
//      std::cout << eigs(i) << std::endl;
      if (eigs(i)<=0)
        return false;
    }
    return true;
  }

/*
  bool
  is_singular(const Matrix &m)
  {
  assert(m.nrows()==m.ncols());

  LaGenMatDouble temp(m.nrows(), m.ncols());
  for (unsigned int i=0; i<m.nrows(); i++)
  for (unsigned int j=0; j<m.ncols(); j++)
  temp(i,j)=m(i,j);
  return is_singular(temp);
  }
*/

  bool
  is_singular(const Matrix &m)
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

/*
  void
  force_min_eig(Matrix &m, double min_eig)
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
*/

  void
  force_min_eig(Matrix &m, double min_eig)
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


  void
  inverse(const Matrix &m,
          Matrix &inv)
  {
    inv.copy(m);
    LaVectorLongInt pivots(m.rows());
    LUFactorizeIP(inv, pivots);
    LaLUInverseIP(inv, pivots);
  }

}
