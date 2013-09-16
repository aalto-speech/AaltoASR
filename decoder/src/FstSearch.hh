#ifndef FSTSEARCH_HH
#define FSTSEARCH_HH

#include "Fst.hh"
#include "NowayHmmReader.hh"
#include "LnaReaderCircular.hh"
#include "OneFrameAcoustics.hh"

typedef std::string bytestype;

class SearchModelReader {
public:
  SearchModelReader(const char * hmm_path = NULL, const char * dur_path = NULL);
  ~SearchModelReader();

  struct OpenError : public std::exception {
    virtual const char *what() const throw()
    { return "FstSearch: open error"; }
  };
  struct InvalidFormat : public std::exception {
    virtual const char *what() const throw()
    { return "NowayHmmReader: invalid format"; }
  };

  // FIXME: These functions are direct copies from Toolbox, code duplication !
  void hmm_read(const char *file);
  void duration_read(const char *dur_file, std::vector<float> *a_table_ptr=NULL, 
                     std::vector<float> *b_table_ptr=NULL);
  virtual void lna_open(const char *file, int size);
  virtual void lna_open_fd(const int fd, int size);
  void lna_close();

  float duration_logprob(int emission_pdf_idx, int duration);

  // Setters
  void set_acoustics(Acoustics *a) {m_acoustics = a;}
  void set_dur_tables(const std::vector<float> &a_table, const std::vector<float> &b_table) {
    m_a_table = a_table; // Sightly inefficient, copy the vectors
    m_b_table = b_table;
  }
  virtual void set_duration_scale(float d) {m_duration_scale=d;}
  virtual void set_beam(float b) {m_beam=b;}
  virtual void set_token_limit(int t) {m_token_limit=t;}
  virtual void set_transition_scale(float t) {m_transition_scale=t;}

protected:
  float m_duration_scale;
  float m_beam;
  int m_token_limit;
  float m_transition_scale;
  int m_frame;
  Acoustics *m_acoustics;

  std::vector<float> m_a_table;
  std::vector<float> m_b_table;

private:
  bool m_delete_on_exit;

  NowayHmmReader *m_hmm_reader;
  LnaReaderCircular *m_lna_reader;
};

class FstSearch: public SearchModelReader {
friend class FstWithPhoneLoop;
public:
  FstSearch(const char * search_fst_fname, const char * hmm_path= NULL, const char * dur_path = NULL);

  void init_search();
  void run();
  bytestype get_best_final_hypo_string() {float foo; return get_best_final_hypo_string_and_logprob(foo);}
  bytestype get_best_final_hypo_string_and_logprob(float &logprob);

protected:
  void propagate_tokens();

private:
  struct Token {
    Token(): logprob(0.0f), node_idx(-1), state_dur(0) {};
    float logprob;
    std::vector<std::string> unemitted_words;
    int node_idx;
    int state_dur;

    std::string str();
  };

  float propagate_token(Token &, float beam_prune_threshold=-999999999.0f);

  Fst m_fst;
  std::vector<Token> m_active_tokens;
  std::vector<Token> m_new_tokens;
  std::vector<int> m_node_best_token;
};

#endif
