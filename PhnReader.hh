#ifndef PHNREADER_HH
#define PHNREADER_HH

#include <string>
#include <vector>
#include <stdio.h>
#include "HmmSet.hh"
#include "Segmentator.hh"


namespace aku {

/** A class for reading .phn phonetic label files
 *
 * Format of the phn file.  Optional fields marked with brackets.  The
 * first optional two fields exist if the first char on the line is digit.
 *
 * [start_sample1 end_sample1] label1 [here are possible comments...]
 * [start_sample2 end_sample2] label2 [some other comments...]
 * ...
 *
 * \note
 * sample numbers refer to 16kHz files, if files in other sample rates
 * are used, the sample numbers for phn files are still computed as
 * 16000*time(seconds).
*/

class PhnReader : public Segmentator {
public:

  /// Structure for the information on one line of phn transcription.
  class Phn {
  public:
    Phn();

    int start;
    int end;
    int state;
    std::vector<std::string> label;
    std::string speaker;
    std::string comment;
  };

  PhnReader(HmmSet *model);
  virtual ~PhnReader();

  virtual void open(std::string ref_file);
  virtual void close();
  virtual void reset(void);
  virtual bool eof(void) { return m_eof_flag; }
  virtual bool computes_total_log_likelihood(void) { return false; }

  virtual bool init_utterance_segmentation(void);
  virtual int current_frame(void) { return m_current_frame; }
  virtual bool next_frame(void);
  virtual const Segmentator::IndexProbMap& pdf_probs(void) { return m_cur_pdf; }
  virtual const Segmentator::IndexProbMap& transition_probs(void) { return m_transition_info; }
  virtual const std::string& highest_prob_label(void) { return m_cur_label; }

  /** Sets the frame rate for converting phn sample numbers to frame numbers.
   * \param frame_rate frames per second
   */
  void set_frame_rate(float frame_rate) { m_samples_per_frame = 16000/frame_rate; }
  
  /** Limits lines to be read from transcription
   * \param first_sample if reference defined, 
   * the first sample corresponding to the defined first_line is set
   */
  void set_line_limits(int first_line, int last_line, 
                       int *first_sample = NULL);

  virtual void set_frame_limits(int first_frame, int last_frame);

  virtual void set_collect_transition_probs(bool collect) { m_collect_transitions = collect; }
  
  void set_state_num_labels(bool l) { m_state_num_labels = l; }
  void set_relative_sample_numbers(bool r) { m_relative_sample_numbers = r; }

  int first_frame(void) { return m_first_frame; }

  /** Reads one line from phn transcription.
   * \note \ref next_frame() is a wrapper for next_phn_line() and should
   * be used whenever possible.
   * \warning Do not mix calls to \ref next_frame() and next_phn_line()!!!
   * \param phn Reference to a \ref Phn structure to be filled
   * \return false if EOF (or frame/line limits) were encountered,
   *         true otherwise.
   */
  bool next_phn_line(Phn &phn);

private:
  float m_samples_per_frame;

  /// first line to be included (1-N); if no limits, m_first_line = 0
  int m_first_line;

  /// last line to be excluded (1-N); if no limits, m_last_line = 0
  int m_last_line;

  /// first frame to be included (0-N)
  int m_first_frame;

  /// last frame to be excluded (0-N); if no limit, m_last_frame = 0
  int m_last_frame;

  /// current line (1-N)
  int m_current_line;

  /// current frame, advanced in \ref next_frame()
  int m_current_frame;

  /// current phn line
  Phn m_cur_phn;

  std::string m_line;
  FILE *m_file;

  HmmSet *m_model;

  /// true if eof has been detected, or line/frame limits have been reached
  bool m_eof_flag;

  /// true for speakered phns
  bool m_speaker_phns;
  
  /// true if labels are state numbers instead of HMM labels
  bool m_state_num_labels;

  /// true if phn sample numbers are relative to the first frame
  bool m_relative_sample_numbers;

  /// true if transitions are to be collected
  bool m_collect_transitions;

  /// A vector which holds the current pdf and its probability
  Segmentator::IndexProbMap m_cur_pdf;

  /// A map which holds the information about transitions
  Segmentator::IndexProbMap m_transition_info;

  /// String with the current label
  std::string m_cur_label;
};

}

#endif /* PHNREADER_HH */
