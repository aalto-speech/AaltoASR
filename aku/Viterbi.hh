#ifndef VITERBI_HH
#define VITERBI_HH

#include <vector>
#include <deque>

#include "FeatureGenerator.hh"
#include "PhnReader.hh"
#include "Lattice.hh"
#include "HmmSet.hh"


namespace aku {

//            |
//            |
//            |
//  position  |      Lattice
//            |
//            |
//            |
//            +-----------------------
//                    frame

/// Search the best path through the Lattice.
/**
 * This class can be used to compute the best path through a feature
 * stream and given transcription using a moving window.
 *
 * INVARIANTS:
 * - cell.from = -1 for every cell at frame 0
 * - cell.from < 0 and cell.log_prob = -INF for every cell outside range
 * - inactive frames have ranges (-1,-1)
 * - cell.transition_index is -1 for inactive frames (just for debugging)
 *
 * NOTES:
 * - unused cells are tested by: cell.from < 0
 * - active cells may have cell.log_prob < -INF, because observation
 *   probabilities may be so low.
 * - debug with sanity_check() method weird errors occur
 *
 * FIXME:
 * - should we floor all log_probs to -INF?  at least we have a
 *   problem if best_log_prob is -INF and all cells are under the beam
 *   we currently floor them!
 * - Not sure if the HMMs may contain loops or skips.
 *   Self transitions are perhaps only transitions used in calculations?
 * - After moving and filling, it is possible that the best path does match?
 * - HMM's are perhaps modified: Self transition is moved as first transition?
 * */
class Viterbi {
public:

  // The states and transitions of the given transcription are
  // precomputed on the Y-axis of viterbi-lattice.  The transitions
  // are relative to the position on the Y-axis.
  struct TranscriptionTransition {
    int target;
    int transition_index;
    float log_prob;
  };

  /// States of the given transcription. 
  struct TranscriptionState {
    TranscriptionState() { }
    TranscriptionState(int state, const std::string &label, 
		       const std::string &comment,
                       int hmm_index, int hmm_state_index)
      : state(state), label(label), hmm_index(hmm_index),
        hmm_state_index(hmm_state_index), comment(comment), printed(true) { }

    /// The index of the state of the HmmSet.
    int state;

    /// Label of the phoneme
    std::string label;

    int hmm_index; // HMM index in HmmSet
    int hmm_state_index; // HMM state index

    /// Comment in the original transcription.
    std::string comment;

    /// Relative transitions from this state.
    std::vector<TranscriptionTransition> transitions;

    /// HMM labels which should also be trained
    std::vector<std::string> additional_labels;

    /** Are the comment and label printed already?  (true by default)
     * 
     * This is used by align only.  Viterbi should not use this
     * variable at all.  This is an ugly hack.  FIXME!
     **/
    mutable bool printed;
  };

  Viterbi(HmmSet &model, FeatureGenerator &fea_gen, PhnReader *phn_reader);

  /// Reset the Viterbi lattice.
  void reset();

  /// Resize the lattice.  FIXME: destroys the contents?
  void resize(int frames, int positions, int block);

  /// Fill the probabilities of the lattice, and compute the best path.
  void fill();

  /// Add hmm to transcription, use if you don't wan't to read the transcription.
  void add_hmm_to_transcription(int hmm_index, std::string &comment,
                                std::vector<std::string> &additional_hmms);

  /// Move the lattice forward by repositioning the lattice.
  void move(int frame, int position);

  // Options
  inline void set_prob_beam(float beam) { m_prob_beam = beam; }
  inline void set_state_beam(int beam) { m_state_beam = beam; }
  inline void set_feature_frame(int feature_frame);
  inline void set_last_frame(int frame) { m_last_frame = frame; }
  inline void set_force_end(bool force) { m_force_end = force; }
  inline void set_last_window(bool last) { m_last_window = last; }
  inline void set_print_all_states(bool print) { m_print_all_states = print; }

  /// The first frame not filled yet because (1) lattice border reached, or (2) end of audio file reached.
  inline int last_frame() const { return m_last_frame; }

  inline int feature_frame() const { return m_feature_frame; }

  /// The first position not filled yet because (1) lattice border reached, or (2) end of transcription reached.
  inline int last_position() const { return m_last_position; }

  /// Access the transcription state on the given position.
  inline const TranscriptionState &transcription(int position) const;

  /// Return the length of the transcription.
  inline int transcription_length() const { return m_transcription.size(); }

  /// The first frame not filled yet.
  inline int current_frame() const { return m_current_frame; }

  /// The state index of the best path at the given frame.
  inline int best_state(int frame) const;

  /// The position of the best path at the given frame.
  inline int best_position(int frame) const;

  /// Accumulated log-probability of the best path since the last reset
  double best_path_log_prob(void) { return m_accumulated_log_prob + m_final_log_prob; }

  /// The transition index to the best path at the given frame.
  /**
   * Note that the last state in the lattice does not have a
   * transition.  It is marked with -1.
   **/
  inline int best_transition(int frame) const;

  /// Return a lattice cell.
  inline Lattice::Cell &at(int frame, int position);

  /// Return the active position range for the given frame.
  inline Lattice::Range &range(int frame);

  /// Show the lattice.
  void debug_show_lattice();

  struct RangeError : public std::exception {
    virtual const char *what() const throw()
      { return "Viterbi: range error"; }
  };

  struct EndPruned : public std::exception {
    virtual const char *what() const throw()
      { return "Viterbi: end pruned (fix the forced end of transcription)"; }
  };

  struct PathError : public std::exception {
    virtual const char *what() const throw()
      { return "Viterbi: path error (try a longer window or fix transcription)"; }
  };

private:
  /// Struct for storing the best path.
  struct Path {
    int position;
    int transition_index;
  };

  void compute_kernel_distances();	// For each frame
  void compute_observation_probs();	// For each frame
  void fill_transcription();
  void fill_transition_probs();
  void fill_observation_probs(const FeatureVec &fea_vec);
  void compute_best_path();

  void sanity_check(); // Only for debugging purposes

  /// The model used in computations.
  HmmSet &m_model;

  /// Tool for accessing the feature stream.
  FeatureGenerator &m_fea_gen;

  /// Tool for reading the transcription.
  PhnReader *m_phn_reader;

  /// The memory-resident part of the lattice.
  Lattice m_lattice;

  /// Current frame, i.e. the first frame which is not computed yet.
  int m_current_frame;

  /// The corresponding frame in feature stream.
  int m_feature_frame;

  /// The first frame outside the lattice.  
  /** 
   * The value may be smaller than lattice size, if the end of audio
   * file is reached. 
   **/
  int m_last_frame;

  /// The first position outside the lattice or end of transcription.
  /** 
   * The value may be smaller than the lattice size if the end of
   * transcription is reached.
   **/
  int m_last_position;

  /// The transcription corresponding to the lattice position.
  std::deque<TranscriptionState> m_transcription; 

  /// The best path through the lattice.
  std::vector<Path> m_best_path;

  /// The best log-probability at the current frame.
  float m_best_log_prob;

  /// Accumulated log-probabilities which have been reduced in window moving
  double m_accumulated_log_prob;

  /// The final log-probability of the current window
  double m_final_log_prob;
  
  /// The position of the best log-probability.
  int m_best_position;

  /// The width of the log-prob beam pruning.
  float m_prob_beam;

  /** The width of state beam pruning.
   *
   * Only this many states are preserved before and after the best
   * state.
   **/
  int m_state_beam;

  /// Force the best path end in the upper right corner of the lattice at
  /// the last window.  
  bool m_force_end;

  /// true if this is the last window
  bool m_last_window;

  bool m_print_all_states;

  /// Work space for probability computation.
  std::vector<float> m_state_prob;

};

Lattice::Cell&
Viterbi::at(int frame, int position)
{
  return m_lattice.at(frame, position);
}

Lattice::Range&
Viterbi::range(int frame)
{
  return m_lattice.range(frame);
}

void
Viterbi::set_feature_frame(int feature_frame)
{
  m_feature_frame = feature_frame;
}

const Viterbi::TranscriptionState&
Viterbi::transcription(int position) const
{
  if (position >= m_last_position) {
    throw RangeError();
  }
  return m_transcription[position];
}

int
Viterbi::best_state(int frame) const
{
  if (frame >= m_current_frame) {
    throw RangeError();
  }

  return m_transcription[m_best_path[frame].position].state;
}

int
Viterbi::best_position(int frame) const
{
  if (frame >= m_current_frame) {
    throw RangeError();
  }
  return m_best_path[frame].position;
}

int
Viterbi::best_transition(int frame) const
{
  if (frame >= m_current_frame) {
    throw RangeError();
  }
  return m_best_path[frame].transition_index;
}

}

#endif /* VITERBI_HH */
