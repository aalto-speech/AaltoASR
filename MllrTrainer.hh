#ifndef MLLRTRAINER_HH
#define MLLRTRAINER_HH

#include <string>
#include <vector>

#include "FeatureGenerator.hh"
#include "HmmSet.hh"

#include "mtl/mtl.h"
#include "mtl/blais.h"
#include "mtl/matrix.h"
#include "mtl/scaled1D.h"
#include "mtl/lu.h"

#define max_iter 100

using namespace std;
using namespace mtl;

class MllrTrainer {

public:

  typedef matrix<double, rectangle<>, dense<>, row_major>::type d_matrix;
  typedef dense1D<double> d_vector;


  MllrTrainer(HmmSet &model, FeatureGenerator &feagen);
  ~MllrTrainer();

  /** find the kernel probabilities
   *  \param state current hmm state
   *  \param feature current feature
   */
  void find_probs(HmmState *state, const FeatureVec &feature);

  /** calculate the transformation matrix and set the transformation module
   *  \param name name of the transformation module
   */
  void calculate_transform(LinTransformModule *mllr_mod);

  /** Restore transformation to identity transform. */
  static void restore_identity(LinTransformModule *mllr_mod);

  void clear_stats();

private:

  /** update ksum and gsum
   *  \param state hmm-state index
   *  \param probs kernel probabilities
   *  \param feature the adjoining feature vector
   */
  void update_stats(HmmState *state, float *probs, const FeatureVec &feature);

  /** calculats parameter alpha
   *  \param Gi inverse of matrix G
   *  \param p cofactor vector
   *  \param k vector k
   *  \param beta
   */
  double calculate_alpha(d_matrix &Gi, d_vector &p, d_vector &k, double beta);

  /** evaluate quadratic form x * A * y^T
   *  \param x left-side vector (dim m)
   *  \param A (dim m x n)
   *  \param y right-side vector (dim n)
   */
  double quadratic(d_vector &x, d_matrix &A, d_vector &y);

private:
  HmmSet &m_model;
  FeatureGenerator &m_feagen;

  int m_dim; // model dimension / feature dimension

  d_vector **k_array; // holds statistics for vector k(i)
  d_matrix **G_array; // holds statistics for matrix G(i)

  double m_beta;  // holds statistics for variable beta

  int m_zero_prob_count;
};

#endif /* MLLRTRAINER_HH */
