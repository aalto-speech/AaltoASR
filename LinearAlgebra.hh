#ifndef LINEARALGEBRA_HH
#define LINEARALGEBRA_HH

#define LA_COMPLEX_SUPPORT
#include "lapackpp.h"
#include "HCL_Rn_d.h"

typedef LaGenMatDouble Matrix;
typedef LaVectorDouble Vector;
typedef LaSymmMatDouble SymmetricMatrix;


/** Functions for matrix operations */
namespace LinearAlgebra {

  //!< A assumed symmetric positive definite!
  double spd_log_determinant(const Matrix &A);
  
  //!< A assumed symmetric positive definite!
  double spd_determinant(const Matrix &A);

    //!< A assumed symmetric!
  double log_determinant(const Matrix &A);
  
  //!< A assumed symmetric!
  double determinant(const Matrix &A);
  
  void matrix_power(const Matrix &A,
                    Matrix &B,
                    double power);

  // Computes the lower triangular Cholesky matrix
  //!< A assumed symmetric positive definite
  void cholesky_factor(const Matrix &A,
                       Matrix &B);
  
  //!< A assumed symmetric, B symmetric positive definite!
  void generalized_eigenvalues(const Matrix &A, 
                               const Matrix &B, 
                               Vector &eigvals,
                               Matrix &eigvecs);

  //!< B assumed symmetric positive definite!
  void generalized_eigenvalues(const Matrix &A,
                               const Matrix &B,
                               LaVectorComplex &eigvals,
                               LaGenMatComplex &eigvecs);
  
  void map_m2v(const Matrix &m,
               Vector &v);
  
  void map_v2m(const Vector &v, 
               Matrix &m);

  void map_hclv2lapackm(const HCL_RnVector_d &hcl_v, 
                        Matrix &lapack_m);
  
  void map_lapackm2hclv(const Matrix &lapack_m,
                        HCL_RnVector_d &hcl_v);

  double cond(const Matrix &m);

  bool is_spd(const Matrix &m);

  bool is_singular(const Matrix &m);

  void force_min_eig(Matrix &m, double min_eig);

  void inverse(const Matrix &m, Matrix &inv);

};

#endif /* LINEARALGEBRA_HH */
