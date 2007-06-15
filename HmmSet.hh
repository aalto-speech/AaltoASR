#ifndef HMMSET_HH
#define HMMSET_HH

#include <vector>
#include <string>
#include <map>
#include <assert.h>

#include "FeatureGenerator.hh"
#include "FeatureModules.hh"
#include "Distributions.hh"

#define MAX_MLLT_ITER 7
#define MAX_MLLT_A_ITER 80

//
// HmmState
//
class HmmState {
public:
  HmmState() {};
  HmmState(int pdf_index) : emission_pdf(pdf_index) { };

  /** Get the transitions for this state
   * \param state the number of state for which transitions are wanted
   * \return a vector of transitions
   */
  inline std::vector<int> &transitions() { return m_transitions; }

  // NOTE: The formalism is changed to couple states and transitions
  // instead of phonemes and transitions, mavarjok 20.04.2007
  int emission_pdf; //!< Index to the emission PDF
  std::vector<int> m_transitions;
};


//
// HmmTransition
//
struct HmmTransition {
  HmmTransition() {}
  HmmTransition(int source, int target, double prob) : source_index(source), target_offset(target), prob(prob) {}
  int source_index;  //!< Index of the source state
  int target_offset; //!< Relative target offset
  double prob;       //!< Transition probability
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
   * \param label      the phoneme label
   * \param num_states the number of states for this Hmm
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


  /**
   * \return the number of states in the HmmSet
   */
  inline int num_states() const;

  /**
   * \return the number of emission PDFs in the HmmSet
   */
  inline int num_emission_pdfs() const;

  /** Gives a reference to a tied state
   * \param state index of the tied state
   * \return reference to the state
   */ 
  inline HmmState &state(int state);

  /**
   * \param state state index
   * \return the emission pdf index of the state
   */
  inline int emission_pdf_index(int state);
  
  /** Adds a new transition to the HmmSet
   * \param source index of the source state
   * \param target target state index within a HMM, relative to the current
   *               state (must be positive for left-to-right HMMs).
   * \param prob   state transition probability
   * \return index to the \ref m_transitions
   */
  int add_transition(int source, int target, double prob);

  /** Adds a new state to HmmSet
   * \param pdf_index PDF index of the state
   * \return index to \ref m_states
   */
  int add_state(int pdf_index);

  /** 
   * \return number of transitions in \ref m_transitions
   */
  inline int num_transitions() const;

  /** 
   * \param t the index of the transition in \ref m_transitions
   * \return the desired transition
   */
  inline HmmTransition &transition(int t);

  /** Adds a pdf to the pool
   * \param pdf Pointer to PDF object
   * \return Index of the pdf in the pool
   */
  int add_pool_pdf(PDF *pdf) { return m_pool.add_pdf(pdf); }

  /** Adds a mixture pdf to the model
   * \param pdf Pointer to Mixture object
   * \return Index of the mixture pdf in the model
   */
  int add_mixture_pdf(Mixture *pdf);

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

  /** Reads the HMM definition file. Handles both the legacy and the
   * current fileformat.
   * \param filename the .ph file to be read
   * \return true if the file was of legacy format (and no separate state
   * information needs to be loaded).
   */
  bool read_ph(const std::string &filename);

  /** Reads the HMM definitions in legacy file format, after the header.
   * \param in File stream
   */
  void read_legacy_ph(std::ifstream &in);
  
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

  void write_legacy_ph(const std::string &filename);

  /** Writes all (.gk/.mc/.ph) files with a common base filename
   * Calls \ref write_gk(), \ref write_mc(), \ref write_ph()
   * \param base the base filename
   */
  void write_all(const std::string &base);

  /** Clears the PDF likelihood cache */
  void reset_cache();

  /** Compute a state likelihood, use cache
   * \param s the state index
   * \param f the feature
   * \return the state probability
   */
  double state_likelihood(const int s, const FeatureVec& f) { return pdf_likelihood(m_states[s].emission_pdf, f); }

  /** Compute a PDF likelihood, use cache
   * \param p index of the PDF
   * \param f the feature
   * \return the PDF probability
   */
  double pdf_likelihood(const int p, const FeatureVec& f);

  /** Compute all PDF likelihoods to the cache
   * \param f the feature
   */
  void precompute_likelihoods(const FeatureVec &f);

  /** Prepares the HmmSet for parameter training. 
   * Should be called before \ref accumulate() or \ref accumulate_from_dump()
   */
  void start_accumulating();

  /** Accumulates emission pdf statistics with a new frame
   * \param f the feature in the current frame
   * \param state the hmm state index
   * \param gamma probability for this emission
   * \param pos 1 = update denominator stats, 0 = update numerator stats, default = false
   */
  void accumulate_distribution(const FeatureVec &f, int state, double gamma, int pos = 0);

  /** Accumulates state transition statistics
   * \param state the source state index
   * \param transition relative state index for this transition
   * \param prior prior probability for this transition
   */
  void accumulate_transition(int transition_index, double prior);

  /** Dumps the accumulated statistics to a file
   * \param base basename for the temporary files (base+gks/phs/mcs)
   */
  void dump_statistics(const std::string base) const;

  /** Dumps the state transition probabilities to a file
   * \param filename name of the dump file, preferably .phs
   */
  void dump_ph_statistics(const std::string filename) const;

  /** Dumps the mixture coefficient statistics
   * \param filename name of the dump file, preferably .mcs
   */
  void dump_mc_statistics(const std::string filename) const;

  /** Dumps the base distribution probabilities to a file
   * \param filename name of the dump file, preferably .gks
   */
  void dump_gk_statistics(const std::string filename) const;

  /** Accumulates the statistics from a dump file
   * \param base basename for the temporary files (base+gks/phs/mcs)
   */
  void accumulate_from_dump(const std::string base);

  /** Accumulates the state transition probabilities from a dump file
   * \param filename name of the dump file, preferably .phs
   */
  void accumulate_ph_from_dump(const std::string filename);

  /** Accumulates the mixture coefficient statistics from a dump file
   * \param filename name of the dump file, preferably .mcs
   */
  void accumulate_mc_from_dump(const std::string filename);

  /** Accumulates the base distribution statistics from a dump file
   * \param filename name of the dump file, preferably .gks
   */
  void accumulate_gk_from_dump(const std::string filename);

  /** Stops parameter training.
   */
  void stop_accumulating();

  /** Sets parameters according to the current accumulators.
   */
  void estimate_parameters(void);

  /** Estimates/updates the MLLT transform and Gaussian parameters
   * according to the current accumulators
   * \param fea_gen FeatureGenerator with the current MLLT transform
   * \param mllt_name Name of the MLLT module in feature configuration
   */
  void estimate_mllt(FeatureGenerator &fea_gen, const std::string &mllt_name);

  /** Sets transition parameters according to the current accumulators.
   */
  void estimate_transition_parameters();
  
  /** Sets the current estimation mode for this model
   * \param mode PDF::EstimationMode (ML/MMI)
   */
  void set_estimation_mode(PDF::EstimationMode mode);

  /** Gets the current estimation mode for this model
   */
  PDF::EstimationMode get_estimation_mode();

  /** Whether full statistics should be collected for all Gaussians
   * \param full_stats full statistics or not
   */
  void set_full_stats(bool full_stats);
  
  /** Sets the parameters for Gaussian estimation
   * \param minvar    Minimum variance
   * \param covsmooth Smoothing value for off-diagonal covariance terms
   * \param c1 the C1 constant
   * \param c2 the C2 constant
   */
  void set_gaussian_parameters(double minvar, double covsmooth, double c1, double c2) { m_pool.set_gaussian_parameters(minvar, covsmooth, c1, c2); }
  
  /** Splits every Gaussian in the pool with some constrains
   * OBS!! ASSUMES CURRENTLY CONTINUOUS-DENSITY HMMS!!
   * \param minocc    Minimum occupancy count needed for splitting a gaussian
   * \param maxg      Maximum number of Gaussians for any state containing
   *                  the Gaussian
   * \return          Amount of new Gaussians created to the pool
  */ 
  int split_gaussians(double minocc, int maxg);

  
private:

  /** Helper function for loading legacy ph-files
   * \param pdf_index pdf index to look for
   * \return state index with the required pdf, or -1 if not found
   */
  int get_state_with_pdf(int pdf_index);

public:
  
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
  /// Map with phoneme label as key and index to m_hmms as value
  std::map<std::string,int> m_hmm_map;
  
  /// Container for all transitions in the system
  std::vector<HmmTransition> m_transitions;
  
  /// Container for all Hmm states
  std::vector<HmmState> m_states;

  /// Container for the emission PDFs
  std::vector<Mixture*> m_emission_pdfs;
  
  /// Buffer of PDF likelihoods for the current feature
  std::vector<double> m_pdf_likelihoods;
  
  /// List of the PDFs with valid likelihoods in the cache
  std::vector<int> m_valid_pdf_likelihoods;
  
  std::vector<Hmm> m_hmms;

  // For accumulating transition probabilities
  std::vector<HmmTransition> m_transition_accum;
  // For marking which transitions have been accumulated
  std::vector<bool> m_accumulated;

  PDFPool m_pool;
  PDF::EstimationMode m_mode;
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
HmmSet::num_emission_pdfs() const
{
  return m_emission_pdfs.size();
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


int
HmmSet::emission_pdf_index(int state)
{
  return m_states[state].emission_pdf;
}


HmmTransition&
HmmSet::transition(int transition)
{
  return m_transitions[transition];
}


#endif /* HMMSET_HH */
