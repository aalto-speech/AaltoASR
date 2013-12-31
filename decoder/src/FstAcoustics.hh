#ifndef FSTACOUSTICS
#define FSTACOUSTICS

/* This class repackages the acoustics handling from the original Aalto decoder for the 
 FST decoder*/

#include "NowayHmmReader.hh"
#include "LnaReaderCircular.hh"
#include "OneFrameAcoustics.hh"

class FstAcoustics {
public:
  FstAcoustics(const char * hmm_fname, const char * dur_fname);
  ~FstAcoustics();

  struct OpenError : public std::exception {
    virtual const char *what() const throw()
    { return "FstAcoustics: open error"; }
  };

  struct InvalidFormat : public std::exception {
    virtual const char *what() const throw()
    { return "FstAcoustics: invalid format"; }
  };

  // FIXME: These functions are direct copies from Toolbox, code duplication !
  void hmm_read(const char *file); // Read acu_model.ph
  void duration_read(const char *dur_file, std::vector<float> *a_table_ptr=nullptr, 
                     std::vector<float> *b_table_ptr=nullptr); // Read acu_model.dur
  virtual void lna_open(const char *file, int size); 
  virtual void lna_open_fd(const int fd, int size);
  void lna_close();

  bool next_frame() {
    return m_acoustics->go_to(m_frame++);
  }
  float log_prob(int emission_pdf_idx) {
    return m_acoustics->log_prob(emission_pdf_idx);
  }
  float num_models() {
    return m_acoustics->num_models();
  }

  float duration_logprob(int emission_pdf_idx, int duration); 

private:
  int m_frame;
  Acoustics *m_acoustics;

  std::vector<float> m_a_table;
  std::vector<float> m_b_table;

  NowayHmmReader *m_hmm_reader;
  LnaReaderCircular m_lna_reader;
};

#endif
