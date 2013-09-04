#ifndef FSTSEARCH_HH
#define FSTSEARCH_HH

#include "Fst.hh"
#include "NowayHmmReader.hh"
#include "LnaReaderCircular.hh"
#include "OneFrameAcoustics.hh"
//#include "Hmm.hh"

class FstSearch {
public:
  FstSearch(const char * search_fst_fname, const char * hmm_path, const char * dur_path = NULL);
  ~FstSearch();

  struct OpenError : public std::exception {
    virtual const char *what() const throw()
    { return "FstSearch: open error"; }
  };
  struct InvalidFormat : public std::exception {
    virtual const char *what() const throw()
      { return "NowayHmmReader: invalid format"; }
  };

  struct Token {
    Token(): logprob(0.0f), node_idx(-1), state_dur(0) {};
    //Token(Token &t): logprob(t.logprob), node_idx(t.node_idx), state_dur(t.state_dur) {};
    float logprob;
    std::vector<std::string> unemitted_words;
    int node_idx;
    int state_dur;

    inline std::string str() {
      std::ostringstream os;
      os << "Token " << node_idx << " " << logprob << " dur " << state_dur << " '";
      for (auto s: unemitted_words) {
        os << " " << s;
      }
      os << " '";
      return os.str();
    }

  };

  std::string run();
  float propagate_token(Token &, float beam_prune_threshold=-999999999.0f);
  float duration_logprob(int emission_pdf_idx, int duration);

  // FIXME: These functions are direct copies from Toolbox, code duplication !
  void hmm_read(const char *file);
  void duration_read(const char *dur_file);
  void lna_open(const char *file, int size);
  void lna_open_fd(const int fd, int size);
  void lna_close();

  float duration_scale;
  float beam;
  int token_limit;

private:
  Fst m_fst;
  std::vector<Token> m_active_tokens;
  std::vector<Token> m_new_tokens;
  std::vector<int> m_node_best_token;

  std::vector<float> m_a_table;
  std::vector<float> m_b_table;

  int m_frame;
  NowayHmmReader *m_hmm_reader;
  //std::map<std::string,int> *m_hmm_map;
  std::vector<Hmm> *m_hmms;
  Acoustics *m_acoustics;
  LnaReaderCircular *m_lna_reader;
  OneFrameAcoustics m_one_frame_acoustics;
};

#endif
