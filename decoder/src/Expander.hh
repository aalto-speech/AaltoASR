#ifndef EXPANDER_HH
#define EXPANDER_HH

#include <vector>

#include "Lexicon.hh"
#include "Hmm.hh"
#include "Acoustics.hh"

class Expander {
public:

  class Word {
  public:
    Word() : 
      word_id(-1), 
      best_length(-1), 
      best_avg_log_prob(0), 
      active(false),
      first_length(-1),
      last_length(-1) { }
    int word_id;
    int best_length;
    float best_avg_log_prob;
    bool active;

    // Log-probs for each length [first..last] are stored in log_probs
    int first_length;
    int last_length;
    std::vector<float> log_probs; // positive elements are unused

    inline bool active_length(int length) { return log_probs[length] < 0; }
    inline void clear_length(int length) { log_probs[length] = 1; }
    inline float best_log_prob() { return log_probs[best_length]; }
  };

  struct WordCompare {
    inline bool operator()(const Word *a, const Word *b) {
      return a->best_avg_log_prob > b->best_avg_log_prob;
    }
  };

  // Actions
  Expander(const std::vector<Hmm> &hmms,
	   Acoustics &m_acoustics);

  ~Expander();

  /** Search the best words.  FIXME: the name should be better
   *
   * If the number of frames has changed since the last call, the
   * initialization might be quite slow.  FIXME
   */
  void expand(int start_frame, int frames);

  // Options
  void set_forced_end(bool forced_end) { m_forced_end = forced_end; }
  void set_token_limit(int limit) { m_token_limit = limit; m_token_pool.reserve(limit); }
  void set_beam(float beam) { m_beam = beam; }
  float get_beam() { return m_beam; }
  void set_max_state_duration(int duration) { m_max_state_duration = duration;}
  void sort_words(int top = 0);

  void set_duration_scale(float scale) { m_duration_scale = scale; }
  void set_transition_scale(float scale) { m_transition_scale = scale; }

  void set_post_durations(bool durations) { m_post_durations = durations; }
  void set_rabiner_post_mode(int mode) { m_rabiner_post_mode = mode; }
  void set_lexicon(Lexicon *l) {m_lexicon = l;}

  // Info
  inline std::vector<Lexicon::Token*> &tokens() { return m_tokens; }

  /**
   * Returns the list of the best words.
   *
   * It is ok to reorder this vector freely, as long as all words are
   * preserved.  Expander uses this vector for cleaning in the
   * beginning of the next expand.
   **/
  inline const std::vector<Word*> &words() { return m_active_words; }
  inline Word* word(int index) { return &m_words[index]; }

  // Debug
  void debug_print_history(Lexicon::Token *token);
  void debug_print_timit(Lexicon::Token *token);
  void debug_print_tokens();

private:
  void check_words(); // FIXME: remove
  void check_best(int info, bool tmp = false); // FIXME: remove

  void sort_best_tokens(int tokens);
  void keep_best_tokens(int tokens);
  void move_all_tokens();
  void clear_tokens();
  void create_initial_tokens(int start_frame);
  Lexicon::Token *token_to_state(const Lexicon::Token *source_token,
				 Lexicon::State &source_state,
				 Lexicon::State &target_state,
				 float new_log_prob,
                                 float new_dur_log_prob,
                                 float aco_log_prob, // Added to the log probs
				 bool update_best,
                                 bool same_state, bool silence,
                                 const HmmState &target_hmm_state);

  Lexicon::Token* acquire_token(void);
  Lexicon::Token* acquire_token(const Lexicon::Token *source_token);
  void release_token(Lexicon::Token *token);
  
  const std::vector<Hmm> &m_hmms;
  Lexicon *m_lexicon;
  Acoustics &m_acoustics;

  // Options
  bool m_forced_end;
  int m_token_limit;
  float m_beam;
  int m_max_state_duration;
  float m_duration_scale;
  float m_transition_scale;
  bool m_post_durations;
  int m_rabiner_post_mode;

  std::vector<Lexicon::Token*> m_token_pool;

  // State
  std::vector<Lexicon::Token*> m_tokens;
  std::vector<Word> m_words;
  std::vector<Word*> m_active_words;
  int m_frame; // Current frame relative to the start frame.
  int m_frames; // Max frames per word
  float m_beam_best;
  float m_beam_best_tmp;
};

#endif /* EXPANDER_HH */
