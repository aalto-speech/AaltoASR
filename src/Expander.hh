#ifndef EXPANDER_HH
#define EXPANDER_HH

#include <vector>

#include "Lexicon.hh"
#include "Hmm.hh"
#include "Acoustics.hh"

class Expander {
public:
  Expander(const std::vector<Hmm> &hmms, Lexicon &lexicon,
	   Acoustics &m_acoustics);
  void sort_best_tokens(int tokens);
  void keep_best_tokens(int tokens);
  Lexicon::Token *token_to_state(const Lexicon::Token *source_token,
				 Lexicon::State &source_state,
				 Lexicon::State &target_state,
				 double new_log_prob);
  void move_all_tokens();
  void clear_tokens();
  void create_initial_tokens(int start_frame);
  void expand(int start_frame, int frames);
  void set_token_limit(int limit) { m_token_limit = limit; }
  void debug_print_history(Lexicon::Token *token);
  void debug_print_tokens();
  std::vector<Lexicon::Token*> &tokens() { return m_tokens; }

private:
  int m_token_limit;
  std::vector<Lexicon::Token*> m_tokens;

  const std::vector<Hmm> &m_hmms;
  Lexicon &m_lexicon;
  Acoustics &m_acoustics;
};

#endif /* EXPANDER_HH */
