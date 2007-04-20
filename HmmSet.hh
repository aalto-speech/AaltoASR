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

  /** Get the transitions for this state
   * \param state the number of state for which transitions are wanted
   * \return a vector of transitions
   */
  inline std::vector<int> &transitions() { return m_transitions; }

  // NOTE: The formalism is changed to couple states and transitions
  // instead of phonemes and transitions, mavarjok 20.04.2007
  Mixture emission_pdf;
  std::vector<int> m_transitions;
};


//
// HmmTransition
//
struct HmmTransition {
  HmmTransition() {}
  HmmTransition(int target, double prob) : target(target), prob(prob), bind_index(-1) {}
  int target; // Relative state index
  double prob; // Transition probability
  int bind_index; // Link to the source HmmState
};



//
// Hmm
//
class Hmm {
public:

  // Label for this Hmm/phoneme
  std::string label;

  /** Set the number of states for this Hmm/phoneme
   * \param states the number of states
   */
  void resize(int states);

  /** 
   * \return the number of states
   */
  inline int num_states() const { return m_states.size(); }

  /** Get the state index to HmmSet for some state
   * \param index the index to a state of this Hmm
   * \return the HmmSet index to the desired state. see \ref HmmSet::state()
   */
  inline int &state(int index) { return m_states[index]; }

private:
  std::vector<int> m_states;
};



/// Set of hidden Markov models.
/// Keeps track of all the Hmms/phonemes, tied states,
/// transitions and mixtures in the system.
class HmmSet {
public:

  // Default constructor, creates only empty Set
  HmmSet();

  // Constructor, sets only feature dimensionality
  HmmSet(int dimension);

  // Copy constructor, calls copy(HmmSet)
  HmmSet(const HmmSet &hmm_set);

  // Resets all variables
  void reset();

  /** Copy the contents of another HmmSet
   * \param hmm_set the reference HmmSet
   */
    void copy(const HmmSet &hmm_set);
  
  /**
   * \return the dimensionality of the feature vectors for this HmmSet
   */
  int dim() const { return m_pool.dim(); }

  /** Creates a new Hmm. Doesn't add it to the HmmSet, see \ref add_hmm(). 
   *  Only label is set on creation, all other variables have their default values
   * \param label the phoneme label
   * \return reference to the newly created Hmm
   */
  Hmm &new_hmm(const std::string &label);

  /** Creates a new Hmm and adds it to this HmmSet
   * \param label the phoneme label
   * \param the number of states for this Hmm
   * \return reference to the newly created Hmm
   */
  Hmm &add_hmm(const std::string &label, int num_states);

  /**
   * \return the total number of Hmms/phonemes in this set
   */
  inline int num_hmms() const;

  /**
   * \param hmm the phoneme index
   * \return the corresponding Hmm
   */
  inline Hmm &hmm(int hmm);

  /**
   * \param label the phoneme label
   * \return the corresponding Hmm
   */
  inline Hmm &hmm(const std::string &label);

  /**
   * \param label the phoneme label
   * \return the corresponding Hmm index
   */
  inline int hmm_index(const std::string &label) const;

  /** Reserves space for tied states and sets pointers to the correct PDFPool
   * \param states the amount of tied states
   */
  void reserve_states(int states);

  /**
   * \return the number of tied states in the HmmSet
   */
  inline int num_states() const;

  /** Gives a reference to a tied state
   * \param state index of the tied state
   * \return reference to the state
   */ 
  inline HmmState &state(int state);

  /** Adds a new transition to the HmmSet
   * \param bind_index index of the source tied state
   * \param target relative state index of the untied target phoneme state (0-1)
   * \param prob state transition probability
   */
  HmmTransition &add_transition(int bind_index, int target, double prob);

  /** Makes a copy of a transition into \ref m_transitions
   * \param index index of the transition to be copied in \ref m_transitions
   * \return index of the transition copy
   */
  int clone_transition(int index);

  /** 
   * \return number of transitions in \ref m_transitions
   */
  inline int num_transitions() const;

  /** 
   * \param t the index of the transition in \ref m_transitions
   * \return the desired transition
   */
  inline HmmTransition &transition(int t);

  /** Reads the mixture base functions from a file. 
   *  Just a redirection to \ref PDFPool::read_gk()
   * \param filename the .gk file to be read
   */
  void read_gk(const std::string &filename);

  /** Reads the mixture coefficients from a file
   *  Calls \ref Mixture::read() for each line
   * \param filename the .mc file to be read
   */
  void read_mc(const std::string &filename);

  /** Reads the phonemes from a file
   *  Creates a Hmm for each phoneme, transitions and tied states
   * \param filename the .ph file to be read
   */
  void read_ph(const std::string &filename);
  
  /** Opens all (.gk/.mc/.ph) files with a common base filename
   * Calls \ref read_gk(), \ref read_mc(), \ref read_ph()
   * \param base the base filename
   */
  void read_all(const std::string &base);

  /** Writes the mixture base functions to a file. 
   *  Just a redirection to \ref PDFPool::write_gk()
   * \param filename the .gk file to be written
   */
  void write_gk(const std::string &filename);

  /** Writes the mixture coefficients to a file. 
   *  Calls \ref Mixture::write() for each line
   * \param filename the .gk file to be written
   */
  void write_mc(const std::string &filename);

  /** Writes the phonemes to a file.
   * \param filename the .ph file to be written
   */
  void write_ph(const std::string &filename);

  /** Writes all (.gk/.mc/.ph) files with a common base filename
   * Calls \ref write_gk(), \ref write_mc(), \ref write_ph()
   * \param base the base filename
   */
  void write_all(const std::string &base);


  /** Clears the state likelihood cache */
  void reset_cache();

  /** Compute a state likelihood, use cache
   * \param s the state index
   * \param f the feature
   * \return the state probability
   */
  double state_likelihood(const int s, const FeatureVec& f);

  /** Compute all state likelihoods to the cache
   * \param f the feature
   */
  void precompute_likelihoods(const FeatureVec &f);

  /** Prepares the HmmSet for parameter training. 
   * Should be called before \ref accumulate_ml() or \ref accumulate_from_dumped_statistics()
   */
  void start_accumulating();

  /** Accumulates maximum likelihood statistics with a new frame
   * \param f the feature in the current frame
   * \param state the hmm state index
   * \param transition indicates whether a state transition occurred (0-1)
   */
  void accumulate_ml(const FeatureVec &f, int state, int transition);

  /** Dumps the accumulated statistics to a file
   * \param base basename for the temporary files (base+gks/phs/mcs)
   */
  void dump_all_statistics(const std::string base) const;

  /** Accumulates the statistics from a dump file
   * \param base basename for the temporary files (base+gks/phs/mcs)
   */
  void accumulate_from_dumped_statistics(const std::string base);

  /** Stops parameter training. Sets parameters to the inferred ones.
   */
  void stop_accumulating();

  
  
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
  // Map with phoneme label as key and index to m_hmms as value
  std::map<std::string,int> m_hmm_map;
  // Container for all transitions in the system
  std::vector<HmmTransition> m_transitions;
  // Container for all Hmm states
  std::vector<HmmState> m_states;
  // Buffer of state likelihoods for the current feature
  std::vector<double> m_state_likelihoods;
  // List of the valid
  std::vector<int> m_valid_state_likelihoods;
  
  std::vector<Hmm> m_hmms;

  // For accumulating transition probabilities
  std::vector<HmmTransition> m_transition_accum;

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
