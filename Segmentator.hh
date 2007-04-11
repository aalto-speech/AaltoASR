#ifndef SEGMENTATOR_HH
#define SEGMENTATOR_HH


/** Virtual base class for generating or reading segmentations of
 * training utterances.
 */
class Segmentator {
public:

  /** Structure for representing states and their probabilities */
  struct StateProbPair {
    int state_index;
    double prob;
  };

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
   * \param last_frame Frame number for the last frame (excluded)
   */
  virtual void set_frame_limits(int first_frame, int last_frame) = 0;
  
  /** Precomputes necessary statistics for generating the segmentation
   * for an utterance. */
  virtual void init_utterance_segmentation(void) = 0;
  
  /** Returns the current frame number, as referenced for
   * \ref FeatureGenerator. */
  virtual int current_frame(void) = 0;

  /** Computes the state probability statistics for the next frame. */
  virtual void next_frame(void) = 0;

  /** Resets the segmentation to the first frame.
   * init_utterance_segmentation needs to be called again.
   */
  virtual void reset(void) = 0;

  /** Returns true if the previous frame read was the last one. After
   * that, \ref next_frame() should not be called */
  virtual bool eof(void) = 0;

  /** Returns a reference to a vector of possible states and their
   * probabilities */
  virtual const std::vector<StateProbPair>& state_probs(void) = 0;
};

#endif // SEGMENTATOR_HH
