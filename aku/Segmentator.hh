#ifndef SEGMENTATOR_HH
#define SEGMENTATOR_HH

#include <string>
#include <map>
#include <vector>

namespace aku {

/** Virtual base class for generating or reading segmentations of
 * training utterances.
 */
class Segmentator {
public:

  /** Structure for representing PDFs and their prior probabilities */
  // struct IndexProbPair {
  //   int index;
  //   double prob;
  //   IndexProbPair(int i, double p) : index(i), prob(p) { }
  // };

  typedef std::map<int, double> IndexProbMap;

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
   * for an utterance.
   * \return true if successful, false if segmentation failed */
  virtual bool init_utterance_segmentation(void) = 0;
  
  /** Returns the current frame number, as referenced for
   * \ref FeatureGenerator. */
  virtual int current_frame(void) = 0;

  /** Computes the PDF probability statistics for the next frame.
   * \note Resets the \ref HmmSet cache of state/PDF probabilities
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

  /** \return true if the total likelihood of the utterance is computed
   * by the Segmentator class. If so, it will be available through
   * \ref get_total_log_likelihood() after the segmentation.
   */
  virtual bool computes_total_log_likelihood(void) = 0;

  /** If \ref computes_total_likelihood() returns true, this method can
   * be used to fetch the total likelihood of the utterance after the
   * segmentation.
   * \return total log likelihood.
   */
  virtual double get_total_log_likelihood(void) { return 0; }

  /** Returns a reference to a vector of possible PDFs and their
   * probabilities */
  virtual const IndexProbMap& pdf_probs(void) = 0;

  /** Returns a reference to a vector of possible transitions and their
   * probabilities */
  virtual const IndexProbMap& transition_probs(void) = 0;

  /** Returns the label of the most probable arc */
  virtual const std::string& highest_prob_label(void) = 0;
};

}

#endif // SEGMENTATOR_HH
