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

  struct Token {
    float logprob;
    std::vector<int> unemitted_words;
  };


  // FIXME: These functions are direct copies from Toolbox, code duplication !
  void hmm_read(const char *file);
  void duration_read(const char *dur_file);
  void lna_open(const char *file, int size);
  void lna_open_fd(const int fd, int size);
  void lna_close();

private:
  Fst m_fst;
  std::vector<Token> *m_active_tokens;
  std::vector<Token> m_token_buffer1;
  std::vector<Token> m_token_buffer2;
  std::vector<Token *> m_node_best_token;

  NowayHmmReader *m_hmm_reader;
  std::map<std::string,int> *m_hmm_map;
  std::vector<Hmm> *m_hmms;
  Acoustics *m_acoustics;
  LnaReaderCircular *m_lna_reader;
  OneFrameAcoustics m_one_frame_acoustics;

};

#endif
