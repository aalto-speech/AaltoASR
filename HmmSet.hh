#ifndef HMMSET_HH
#define HMMSET_HH

#include <vector>
#include <string>
#include <map>
#include <assert.h>

#include "FeatureModules.hh"
#include "Pcgmm.hh"
#include "Scgmm.hh"
#include "Distributions.hh"

#include "mtl/mtl.h"
#include "mtl/blais.h"

typedef mtl::matrix<float, mtl::rectangle<>, mtl::dense<>,
                    mtl::row_major>::type Matrix;
typedef mtl::dense1D<float> Vector;


//
// HmmState
//
struct HmmState {
  Mixture emission_pdf;
};


//
// HmmTransition
//
struct HmmTransition {
  HmmTransition() {}
  HmmTransition(int target, float prob) : target(target), prob(prob), bind_index(-1) {}
  int target; // Relative state index
  float prob;
  int bind_index;
};



//
// Hmm
//
class Hmm {
public:
  std::string label;

  void resize(int states);

  inline int num_states() const { return m_states.size(); }
  inline int &state(int index) { return m_states[index]; }
  inline std::vector<int> &transitions(int state);

private:
  std::vector<int> m_states;
  std::vector<std::vector<int> > m_transitions;
};

std::vector<int>&
Hmm::transitions(int state)
{
  return m_transitions[state + 1];
}


/// Set of hidden Markov models.
class HmmSet {
public:
  HmmSet();
  HmmSet(int dimension);
  HmmSet(const HmmSet &hmm_set);

  void reset();
  void copy(const HmmSet &hmm_set);

  Hmm &new_hmm(const std::string &label);
  Hmm &add_hmm(const std::string &label, int num_states);
  void clone_hmm(const std::string &source, const std::string &target);
  void untie_transitions(const std::string &label);
  inline HmmCovariance::Type covariance_type() const;
  void set_covariance_type(HmmCovariance::Type type);

  void set_dim(int dim);
  inline int dim() const;

  inline int num_hmms() const;
  inline Hmm &hmm(int hmm);
  inline Hmm &hmm(const std::string &label);
  inline int hmm_index(const std::string &label) const;

  void reserve_states(int states);
  inline int num_states() const;
  inline HmmState &state(int state);

  void reserve_kernels(int kernels);
  HmmKernel &add_kernel();
  void remove_kernel(int kernel);
  inline int num_kernels() const;
  inline HmmKernel &kernel(int kernel);

  void reserve_transitions(int transitions);
  HmmTransition &add_transition(int h, int source, int target, float prob,
                                int bind_index);
  int clone_transition(int index);
  inline int num_transitions() const;
  inline HmmTransition &transition(int t);

  // IO
  void read_gk(const std::string &filename);
  void read_mc(const std::string &filename);
  void read_ph(const std::string &filename);
  void read_all(const std::string &base);
  void write_gk(const std::string &filename);
  void write_mc(const std::string &filename);
  void write_ph(const std::string &filename);
  void write_all(const std::string &base);

  void compute_covariance_determinants(void);

  // Probs
  void reset_state_probs(); // FAST, unnormalized
  float state_prob(const int s, const FeatureVec&); // FAST, unnormalized 
//  float euclidean_distance(std::vector<float> &a, std::vector<float> &b);
  void compute_observation_log_probs(const FeatureVec&); // SLOW, accurate
//  void compute_obs_logprobs(const float *feature);
  void precompute(const FeatureVec &feature); // precomputation for subspace gmms
  float compute_kernel_likelihood(const int k, const FeatureVec &feature);
  
  // These work only for scaled probs...
  std::vector<float> obs_log_probs;
  std::vector<float> obs_kernel_likelihoods;

  // Exceptions
  struct DuplicateHmm : public std::exception {
    virtual const char *what() const throw()
      { return "HmmSet: duplicate hmm"; }
  };

  struct UnknownHmm : public std::exception {
    virtual const char *what() const throw()
      { return "HmmSet: unknown hmm"; }
  };

  struct ReadError : public std::exception {
    virtual const char *what() const throw()
      { return "HmmSet: read error"; }
  };

  struct WriteError : public std::exception {
    virtual const char *what() const throw()
      { return "HmmSet: write error"; }
  };

  struct OpenError : public std::exception {
    virtual const char *what() const throw()
      { return "HmmSet: open error"; }
  };

private:
  /// Dimensionality of the Gaussian densities.
  int m_dim;

  /// Type of covariance matrix.
  HmmCovariance::Type m_cov_type;

  std::map<std::string,int> m_hmm_map;

  std::vector<HmmKernel> m_kernels;
  std::vector<HmmTransition> m_transitions;
  std::vector<HmmState> m_states;
  std::vector<Hmm> m_hmms;

  /**
   * Undefined log-probabilities are marked with positive values.
   */
  std::vector<float> m_state_probs;

  /**
   * Undefined distances are marked with negative values.
   */
  std::vector<float> m_kernel_likelihoods;


  std::vector<int> m_valid_stateprobs;
  std::vector<int> m_valid_kernel_likelihoods;
  float m_viterbi_scale_coeff;

  // For computing full covariance kernels
  Vector m_kernel_temp_s;
  Vector m_kernel_temp_t;

public:
  Pcgmm pcgmm;
  Scgmm scgmm;
};

HmmCovariance::Type
HmmSet::covariance_type() const
{
  return m_cov_type;
}

int
HmmSet::dim() const
{
  return m_dim;
}

int
HmmSet::num_hmms() const
{
  return m_hmms.size();
}

int
HmmSet::num_states() const
{
  return m_states.size();
}

int
HmmSet::num_kernels() const
{
  return m_kernels.size();
}

int
HmmSet::num_transitions() const
{
  return m_transitions.size();
}

int
HmmSet::hmm_index(const std::string &label) const
{
  // Check if label exists
  std::map<std::string,int>::const_iterator it = m_hmm_map.find(label);
  if (it == m_hmm_map.end()) {
    fprintf(stderr, "HmmSet::hmm_index(): unknown hmm '%s'\n", label.c_str());
    throw UnknownHmm();
  }

  return (*it).second;
}

Hmm&
HmmSet::hmm(int hmm)
{
  return m_hmms[hmm];
}

Hmm&
HmmSet::hmm(const std::string &label)
{
  return m_hmms[hmm_index(label)];
}

HmmState&
HmmSet::state(int state)
{
  return m_states[state];
}

HmmKernel&
HmmSet::kernel(int kernel)
{
  return m_kernels[kernel];
}

HmmTransition&
HmmSet::transition(int transition)
{
  return m_transitions[transition];
}

void cholesky_factor(const Matrix &A, Matrix &B);

#endif /* HMMSET_HH */
