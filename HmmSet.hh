#ifndef HMMSET_HH
#define HMMSET_HH

#include <vector>
#include <string>
#include <map>
#include <assert.h>

#include "FeatureModules.hh"
#include "Distributions.hh"


//
// HmmState
//
class HmmState {
public:
  HmmState() {};
  HmmState(PDFPool *pool) : emission_pdf(pool) { };
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
  int dim() const { return m_pool.dim(); }

  Hmm &new_hmm(const std::string &label);
  Hmm &add_hmm(const std::string &label, int num_states);
  void clone_hmm(const std::string &source, const std::string &target);
  void untie_transitions(const std::string &label);

  inline int num_hmms() const;
  inline Hmm &hmm(int hmm);
  inline Hmm &hmm(const std::string &label);
  inline int hmm_index(const std::string &label) const;

  void reserve_states(int states);
  inline int num_states() const;
  inline HmmState &state(int state);

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

  // Probs
  void reset_state_probs(); // FAST, unnormalized
  float state_prob(const int s, const FeatureVec&); // FAST, unnormalized 
  void compute_observation_log_probs(const FeatureVec&); // SLOW, accurate
  
  // These work only for scaled probs...
  std::vector<float> obs_log_probs;

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
  std::map<std::string,int> m_hmm_map;
  std::vector<HmmTransition> m_transitions;
  std::vector<HmmState> m_states;
  std::vector<Hmm> m_hmms;

  /**
   * Undefined log-probabilities are marked with positive values.
   */
  std::vector<float> m_state_probs;
  std::vector<int> m_valid_stateprobs;

  PDFPool m_pool;
};


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


HmmTransition&
HmmSet::transition(int transition)
{
  return m_transitions[transition];
}


#endif /* HMMSET_HH */
