#ifndef FSTSEARCH_HH
#define FSTSEARCH_HH

/* A simple decoder for fst networks. Assumes, that the network has state indices encoded.
   To create a search network, you should do something like (the example is given from mitfst tools)

fst_compose ../work/L.fst ../work/lat.fst -| fst_compose -t ../work/C.fst - - | fst_compose ../work/H.fst - -  | fst_optimize -A - ../work/final.fst

L.fst: Lexicon fst (word to monophones)
lat.fst: grammar (words that can follow each other)
C.fst: context grammar (which triphones can follow each other)
H.fst: triphone to state numbers (expands to the number of states specified in the ph file as well, from hmm2fsm)
final.fst: The fst that the search uses
 */

#include "FstAcoustics.hh"
#include "Fst.hh"

typedef std::string bytestype;

struct FstToken {
  FstToken(): logprob(0.0f), node_idx(-1), state_dur(0) {};
  float logprob;
  std::vector<std::string> unemitted_words;
  int node_idx;
  int state_dur;
  
  std::string str() const;
};

template <typename T>
class FstSearch_base {
  friend class FstConfidenceWithPhoneLoop;
public:
  FstSearch_base(const char *search_fst_name, FstAcoustics *fst_acu = nullptr);
  FstSearch_base(const char *search_fst_fname, const char *hmm_path, const char *dur_path);
  ~FstSearch_base();

  virtual void init_search();
  virtual void run();
  bytestype get_result() {float foo; return get_result_and_logprob(foo);}
  bytestype get_result_and_logprob(float &logprob);
  float get_best_token_logprob() {return m_new_tokens.size()>0? m_new_tokens[0].logprob: -9999999.0f;};
  float get_best_final_token_logprob();
  bytestype tokens_at_final_states();
  bytestype best_tokens(int n=10);

  void lna_open(const char *file, int size) {m_fst_acoustics->lna_open(file, size);}
  void lna_open_fd(const int fd, int size)  {m_fst_acoustics->lna_open_fd(fd, size);}
  void lna_close() {m_fst_acoustics->lna_close();}

  void set_duration_scale(float d) {m_duration_scale=d;}
  void set_beam(float b) {m_beam=b;}
  void set_token_limit(int t) {m_token_limit=t;}
  void set_transition_scale(float t) {m_transition_scale=t;}
  void set_acoustics(FstAcoustics *fsta) {m_fst_acoustics = fsta;}

  float get_duration_scale() {return m_duration_scale;}
  float get_beam() {return m_beam;}
  int get_token_limit() {return m_token_limit;}
  float get_transition_scale() {return m_transition_scale;}
  
  int verbose;

protected:
  void propagate_tokens();
  std::vector<T> m_new_tokens;

  float m_duration_scale;
  float m_beam;
  int m_token_limit;
  float m_transition_scale;
  bool m_one_token_per_node;

  Fst m_fst;
  FstAcoustics *m_fst_acoustics;
  bool m_delete_acoustics;

  std::vector<T> m_active_tokens;
  std::vector<int> m_node_best_token;

private:
  float propagate_token(T &, float beam_prune_threshold=-999999999.0f);
};

typedef FstSearch_base<FstToken> FstSearch;
#include "FstSearch_tmpl.hh"
#endif
