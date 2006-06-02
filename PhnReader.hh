#ifndef PHNREADER_HH
#define PHNREADER_HH

#include <string>
#include <vector>
#include <stdio.h>

/* 

Format of the phn file.  Optional fields marked with brackets.  The
first optional two fields exist if the first char on the line is digit.

[start_sample1 end_sample1] label1 [here are possible comments...]
[start_sample2 end_sample2] label2 [some other comments...]
...

NOTE: sample numbers refer to 16kHz files, if files in other sample rates
      are used, the sample numbers for phn files are still computed as
      16000*time(seconds).
*/

class PhnReader {
public:
  
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

  PhnReader();

  void open(std::string filename);
  void reset_file(void);
  void close();

  /** Limits lines to be read from transcription
   * \param first_sample if reference defined, 
   * the first sample corresponding to the defined first_line is set
   * \param last_sample if reference defined, 
   * the last sample corresponding to the defined last_line is set
   */
  void set_line_limit(int first_line, int last_line, 
		      int *first_sample);

  void set_sample_limit(int first_sample, int last_sample);

  /** Sets phn type: normal / speakered (sphn = speakered phn)
   * speakered phns have speaker ID between label and comment
   */   
  void set_speaker_phns(bool sphn);

  void set_state_num_labels(bool l) { m_state_num_labels = l; }

  bool next(Phn &phn);

private:

  /// first line to be included (1-N); if no limits, m_first_line = 0
  int m_first_line;

  /// last line to be included (1-N); if no limits, m_last_line = 0
  int m_last_line;

  /// first sample to be included (0-N)
  int m_first_sample;

  /// last sample to be included (0-N); if no samplelimit, m_last_sample = 0
  int m_last_sample;

  /// current line (1-N)
  int m_current_line;

  std::string m_line;
  FILE *m_file;
  
  /// true for speakered phns
  bool m_speaker_phns;

  /// true if labels are state numbers instead of HMM labels
  bool m_state_num_labels;
};

#endif /* PHNREADER_HH */
