#ifndef TOOLBOX_HH
#define TOOLBOX_HH

#include "NowayHmmReader.hh"
#include "NowayLexiconReader.hh"
#include "LnaReaderCircular.hh"
#include "ArpaNgramReader.hh"
#include "Expander.hh"
#include "Search.hh"

typedef HypoStack HypoStack;

class Toolbox {
public:
  Toolbox();
  
  void hmm_read(const char *file);
  void lex_read(const char *file);
  void ngram_read(const char *file);

  void lna_open(const char *file, int models, int size);
  void lna_close();

  void init(int frames, int hypos) 
    { 
      m_search.init_search(frames, hypos); 
    }
  bool expand(int frame) { return m_search.expand_stack(frame); }
  int earliest_frame() { return m_search.earliest_frame(); }
  int last_frame() { return m_search.last_frame(); }
  HypoStack &stack(int frame) { return m_search.stack(frame); }

  void set_hypo_limit(int hypo_limit) { m_search.set_hypo_limit(hypo_limit); } 
  void set_word_limit(int word_limit) { m_search.set_word_limit(word_limit); }
  void set_lm_scale(double lm_scale) { m_search.set_lm_scale(lm_scale); }
  void set_token_limit(int limit) { m_expander.set_token_limit(limit); }
  void set_max_state_duration(int duration) { m_expander.set_max_state_duration(duration); }

  void print_hypo(Hypo &hypo);

private:
  NowayHmmReader m_hmm_reader;
  const std::map<std::string,int> &m_hmm_map;
  const std::vector<Hmm> &m_hmms;

  NowayLexiconReader m_lexicon_reader;
  Lexicon &m_lexicon;
  const Vocabulary &m_vocabulary;

  LnaReaderCircular m_lna_reader;

  ArpaNgramReader m_ngram_reader;
  const Ngram &m_ngram;

  Expander m_expander;
  Search m_search;
};

#endif /* TOOLBOX_HH */
