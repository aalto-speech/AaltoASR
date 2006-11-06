#ifndef LINEARALGEBRA_HH
#define LINEARALGEBRA_HH

#include "fmd.h"
#include "gmd.h"
#include "blas1pp.h"
#include "blas2pp.h"
#include "blas3pp.h"
#include "lasvd.h"
#include "laslv.h"
#include "lavli.h"
#include "spdmd.h"

#include "mtl/mtl.h"

#include "HCL_Rn_d.h"

typedef mtl::matrix<float, mtl::rectangle<>, mtl::dense<>, 
		    mtl::row_major>::type Matrix;
typedef mtl::matrix<double, mtl::rectangle<>, mtl::dense<>, 
		    mtl::row_major>::type MatrixD;


class LinearAlgebra {

public:
  
  // A assumed symmetric!
  static double determinant(const LaGenMatDouble &A);
  
  static void matrix_power(const LaGenMatDouble &A,
			   LaGenMatDouble &B,
			   double power);
  
  static void cholesky_factor(const LaGenMatDouble &A,
			      LaGenMatDouble &B);
  
  // B assumed symmetric!
  static void generalized_eigenvalues(const LaGenMatDouble &A, 
				      const LaGenMatDouble &B, 
				      LaVectorDouble &eigvals,
				      LaGenMatDouble &eigvecs);
  
  static void map_m2v(const LaGenMatDouble &m,
		      LaVectorDouble &v);
  
  static void map_v2m(const LaVectorDouble &v, 
		      LaGenMatDouble &m);

  static void map_mtlm2v(const Matrix &m,
			 LaVectorDouble &v);
  
  static void map_v2mtlm(const LaVectorDouble &v, 
			 Matrix &m);
  
  static void map_hclv2lapackm(const HCL_RnVector_d &hcl_v, 
			       LaGenMatDouble &lapack_m);
  
  static void map_lapackm2hclv(const LaGenMatDouble &lapack_m,
			       HCL_RnVector_d &hcl_v);

  static double cond(const Matrix &m);
  static double cond(const LaGenMatDouble &m);

  static bool is_spd(const Matrix &m);
  static bool is_spd(const LaGenMatDouble &m);

  static bool is_singular(const Matrix &m);
  static bool is_singular(const LaGenMatDouble &m);

  static void force_min_eig(Matrix &m, double min_eig);
  static void force_min_eig(LaGenMatDouble &m, double min_eig);
};

#endif /* LINEARALGEBRA_HH */
