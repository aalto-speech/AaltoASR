#ifndef MLLRTRAINER_HH
#define MLLRTRAINER_HH

#include <string>
#include <vector>

#include "FeatureGenerator.hh"
#include "HmmSet.hh"

#define max_iter 100

using namespace std;

class MllrTrainer {

public:

  MllrTrainer(HmmSet &model, FeatureGenerator &feagen);
  ~MllrTrainer();

  /** find the kernel probabilities
   * \param prior prior probability of the state
   * \param state current hmm state
   * \param feature current feature
   */
  void find_probs(double prior, HmmState *state, const FeatureVec &feature);

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
  void update_stats(HmmState *state, double *probs, const FeatureVec &feature);

  /** calculats parameter alpha
   *  \param Gi inverse of matrix G
   *  \param p cofactor vector
   *  \param k vector k
   *  \param beta
   */
  double calculate_alpha(Matrix &Gi, Vector &p, Vector &k, double beta);

  /** evaluate quadratic form x * A * y^T
   *  \param x left-side vector (dim m)
   *  \param A (dim m x n)
   *  \param y right-side vector (dim n)
   */
  double quadratic(Vector &x, Matrix &A, Vector &y);

private:
  HmmSet &m_model;
  FeatureGenerator &m_feagen;

  int m_dim; // model dimension / feature dimension

  std::vector<Vector*> k_array; // holds statistics for vector k(i)
  std::vector<Matrix*> G_array; // holds statistics for matrix G(i)

  double m_beta;  // holds statistics for variable beta

  int m_zero_prob_count;
};

#endif /* MLLRTRAINER_HH */
