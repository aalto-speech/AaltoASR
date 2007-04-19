#ifndef SEGMENTATOR_HH
#define SEGMENTATOR_HH

#include <map>
#include <vector>

/** Virtual base class for generating or reading segmentations of
 * training utterances.
 */
class Segmentator {
public:

  /** Structure for representing states and their probabilities */
  struct StateProbPair {
    int state_index;
    double prob;
    StateProbPair(int index, double p) : state_index(index), prob(p) { }
  };

  /** Structure for representing state pairs for transitions */
  struct StatePair {
    int from;
    int to;
    StatePair() { }
    StatePair(int f, int t) : from(f), to(t) { }
  };

  struct StatePairLessThan {
    bool operator()(const StatePair &s1, const StatePair &s2) const
    {
      if (s1.from == s2.from)
        return (s1.to < s2.to);
      return (s1.from < s2.from);
    }
  };

  /** Type specification for the map used in \ref transition_probs() */
  typedef std::map<StatePair, double, StatePairLessThan> TransitionMap;

  virtual ~Segmentator() { }

  /** Opens the reference transcription/lattice and the audio file
   * \param ref_file reference transcription/lattice file name
   */
  virtual void open(std::string ref_file) = 0;

  /** Close the reference file and free any allocated memory */
  virtual void close() = 0;

  /** Sets frame limits for the segmentation. The frame numbers are
   * referenced as they are for \ref FeatureGenerator.
   * \param first_frame Frame number for the first frame
   * \param last_frame  Frame number for the last frame (excluded)
   */
  virtual void set_frame_limits(int first_frame, int last_frame) = 0;

  /** Defines whether transitions probabilities are collected or not (default).
   * \param collect If true, the transition probabilities are collected and
   *                can be retrieved using \ref transition_probs()
   */
  virtual void set_collect_transition_probs(bool collect) = 0;
  
  /** Precomputes necessary statistics for generating the segmentation
   * for an utterance. */
  virtual void init_utterance_segmentation(void) = 0;
  
  /** Returns the current frame number, as referenced for
   * \ref FeatureGenerator. */
  virtual int current_frame(void) = 0;

  /** Computes the state probability statistics for the next frame.
   * \return true if a new frame is available, false if EOF was encountered.
   */
  virtual bool next_frame(void) = 0;

  /** Resets the segmentation to the first frame. If the probabilities
   * may have changed from previous call to \ref init_utterance_segmentation()
   * (e.g. features have changed), the segmentation needs to be initialized
   * again.
   */
  virtual void reset(void) = 0;

  /** Returns true if EOF was encountered during previous call to
   * \ref next_frame() */
  virtual bool eof(void) = 0;

  /** Returns a reference to a vector of possible states and their
   * probabilities */
  virtual const std::vector<StateProbPair>& state_probs(void) = 0;

  /** Returns a reference to a map of state pairs (transitions) with
      probabilities as the values */
  virtual const TransitionMap& transition_probs(void) = 0;
};

#endif // SEGMENTATOR_HH
