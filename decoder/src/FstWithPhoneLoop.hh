#ifndef FSTWITHPHONELOOP
#define FSTWITHPHONELOOP

#include "FstSearch.hh"

class FstWithPhoneLoop: public SearchModelReader {
friend class FstSearch;
public:
  FstWithPhoneLoop(const char *grammar_fst_name, const char * phone_loop_fst_name,
                   const char *hmm_path, const char * dur_path = NULL);

  struct ValueError : public std::exception {
    virtual const char *what() const throw()
    { return "FstWithPhoneLoop: invalid value"; }
  };

  void set_duration_scale(float d) {
    m_duration_scale=d;
    m_grammar_fst.set_duration_scale(d);
    m_phone_fst.set_duration_scale(d);
  }
  void set_transition_scale(float t) {
    m_transition_scale=t;
    m_grammar_fst.set_transition_scale(t);
    m_phone_fst.set_transition_scale(t);
  }

  void set_beam(float b) {
    throw ValueError();
  }

  void set_token_limit(int t) {
    throw ValueError();
  }

  void set_grammar_beam(float b) {
    m_beam=b;
    m_grammar_fst.set_beam(b);
  }

  void set_grammar_token_limit(int t) {
    m_token_limit=t;
    m_grammar_fst.set_token_limit(t);
  }

  void set_phone_beam(float b) {
    m_beam=b;
    m_phone_fst.set_beam(b);
  }

  void set_phone_token_limit(int t) {
    m_token_limit=t;
    m_phone_fst.set_token_limit(t);
  }

  void init_search() {
    m_grammar_fst.init_search();
    m_phone_fst.init_search();
  }

  void run();
  bytestype get_best_final_hypo_string_and_confidence(float &);

  void lna_open(const char *file, int size) {
    SearchModelReader::lna_open(file, size);
    m_grammar_fst.set_acoustics(m_acoustics);
    m_phone_fst.set_acoustics(m_acoustics);
  };
  
  void lna_open_fd(const int fd, int size) {
    SearchModelReader::lna_open_fd(fd, size);
    m_grammar_fst.set_acoustics(m_acoustics);
    m_phone_fst.set_acoustics(m_acoustics);
  };

private:
  FstSearch m_grammar_fst;
  FstSearch m_phone_fst;
};

#endif
