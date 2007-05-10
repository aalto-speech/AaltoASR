#ifndef LINEARALGEBRA_HH
#define LINEARALGEBRA_HH

#define LA_COMPLEX_SUPPORT
#include "lapackpp.h"
#include "HCL_Rn_d.h"

typedef LaGenMatDouble Matrix;
typedef LaVectorDouble Vector;


class LinearAlgebra {

public:
  
  // A assumed symmetric!
  static double determinant(const Matrix &A);
  
  static void matrix_power(const Matrix &A,
			   Matrix &B,
			   double power);
  
  static void cholesky_factor(const Matrix &A,
			      Matrix &B);
  
  // A assumed symmetric, B symmetric positive definite!
  static void generalized_eigenvalues(const Matrix &A, 
				      const Matrix &B, 
				      Vector &eigvals,
				      Matrix &eigvecs);

  // B assumed symmetric positive definite!
  static void generalized_eigenvalues(const Matrix &A,
				      const Matrix &B,
				      LaVectorComplex &eigvals,
				      LaGenMatComplex &eigvecs);
  
  static void map_m2v(const Matrix &m,
		      Vector &v);
  
  static void map_v2m(const Vector &v, 
		      Matrix &m);

  static void map_hclv2lapackm(const HCL_RnVector_d &hcl_v, 
			       Matrix &lapack_m);
  
  static void map_lapackm2hclv(const Matrix &lapack_m,
			       HCL_RnVector_d &hcl_v);

  static double cond(const Matrix &m);

  static bool is_spd(const Matrix &m);

  static bool is_singular(const Matrix &m);

  static void force_min_eig(Matrix &m, double min_eig);

  static void inverse(const Matrix &m, Matrix &inv);
};

#endif /* LINEARALGEBRA_HH */
