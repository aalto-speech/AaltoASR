#ifndef SEGMENTATOR_HH
#define SEGMENTATOR_HH


/** Structure for representing states and their probabilities */
struct StateProbPair {
  int state_num;
  double prob;
};

/** Virtual base class for generating or reading segmentations of
 * training utterances.
 */
class Segmentator {
public:

  /** Precomputes necessary statistics for generating the segmentation
   * for an utterance. */
  void init_utterance_segmentation(void) = 0;
  
  /** Returns the current frame number, as referenced for
   * \ref FeatureGenerator. */
  int current_frame(void) = 0;

  /** Computes the state probability statistics for the next frame. */
  void next_frame(void) = 0;

  /** Returns true if the previous frame read was the last one. After
   * that, \ref next_frame() should not be called */
  bool eof(void) = 0;

  /** Returns a reference to a vector of possible states and their
   * probabilities */
  const std::vector<StateProbPair>& state_probs(void) = 0;
};

#endif // SEGMENTATOR_HH
