#ifndef TOKENPASSSEARCH_HH
#define TOKENPASSSEARCH_HH

#include "TPLexPrefixTree.hh"
#include "TreeGram.hh"
#include "Acoustics.hh"

#define MAX_WC_COUNT 200
#define MAX_LEX_TREE_DEPTH 60

class TokenPassSearch {
public:
  TokenPassSearch(TPLexPrefixTree &lex, Vocabulary &vocab,
                  Acoustics &acoustics);

  // Resets search and creates the initial token
  void reset_search(int start_frame);
  void set_end_frame(int end_frame) { m_end_frame = end_frame; }

  // Proceeds decoding one frame
  bool run(void);

  // Print the best path
  void print_guaranteed_path(void);
  void print_best_path(bool only_not_printed);

  // Options
  void set_global_beam(float beam) { m_global_beam = beam; if (m_word_end_beam > m_global_beam) m_word_end_beam = beam; }
  void set_word_end_beam(float beam) { m_word_end_beam = beam; }
  void set_eq_depth_beam(float beam) { m_eq_depth_beam = beam; }
  void set_eq_word_count_beam(float beam) { m_eq_wc_beam = beam; }
  void set_similar_word_history_span(int n) { m_similar_word_hist_span = n; }
  void set_lm_scale(float lm_scale) { m_lm_scale = lm_scale; }
  void set_duration_scale(float dur_scale) { m_duration_scale = dur_scale; }
  void set_transition_scale(float trans_scale) { m_transition_scale = trans_scale; }
  void set_max_num_tokens(int tokens) { m_max_num_tokens = tokens; }
  void set_verbose(int verbose) { m_verbose = verbose; }
  void set_word_boundary(const std::string &word);
  void set_lm_lookahead(int order) { m_lm_lookahead = order; }
  void set_insertion_penalty(float ip) { m_insertion_penalty = ip; }

  void set_ngram(TreeGram *ngram);
  void set_lookahead_ngram(TreeGram *ngram);

private:
  void propagate_tokens(void);
  void propagate_token(TPLexPrefixTree::Token *token);
  void move_token_to_node(TPLexPrefixTree::Token *token,
                          TPLexPrefixTree::Node *node,
                          float transition_score);
  void prune_tokens(void);

#ifdef PRUNING_MEASUREMENT
  void analyze_tokens(void);
#endif
  
  TPLexPrefixTree::Token* find_similar_word_history(
    TPLexPrefixTree::WordHistory *wh, TPLexPrefixTree::Token *token_list);
  inline bool is_similar_word_history(TPLexPrefixTree::WordHistory *wh1,
                                      TPLexPrefixTree::WordHistory *wh2);
  float compute_lm_log_prob(TPLexPrefixTree::WordHistory *word_hist);

  float get_lm_lookahead_score(TPLexPrefixTree::WordHistory *word_hist,
                                TPLexPrefixTree::Node *node, int depth);
  float get_lm_bigram_lookahead(int prev_word_id,
                                TPLexPrefixTree::Node *node, int depth);
  float get_lm_trigram_lookahead(int w1, int w2,
                                 TPLexPrefixTree::Node *node, int depth);

  void clear_active_node_token_lists(void);

  inline float get_token_log_prob(float am_score, float lm_score) {
    return (am_score + m_lm_scale * lm_score); }
    
  TPLexPrefixTree::Token* acquire_token(void);
  void release_token(TPLexPrefixTree::Token *token);

  void save_token_statistics(int count);
  //void print_token_path(TPLexPrefixTree::PathHistory *hist);
  
private:
  TPLexPrefixTree &m_lexicon;
  TPLexPrefixTree::Node *m_root;
  Vocabulary &m_vocabulary;
  Acoustics &m_acoustics;

  std::vector<TPLexPrefixTree::Token*> *m_active_token_list;
  std::vector<TPLexPrefixTree::Token*> *m_new_token_list;

  std::vector<TPLexPrefixTree::Token*> *m_word_end_token_list;

  std::vector<TPLexPrefixTree::Token*> m_token_pool;

  std::vector<TPLexPrefixTree::Node*> m_active_node_list;

  class LMLookaheadScoreList {
  public:
    int index;
    std::vector<float> lm_scores;
  };
  HashCache<LMLookaheadScoreList*> lm_lookahead_score_list;
  
  int m_end_frame;
  int m_frame; // Current frame

  float m_best_log_prob; // The best total_log_prob in active tokens
  float m_worst_log_prob;
  float m_best_we_log_prob;

  // Ngram
  TreeGram *m_ngram;
  std::vector<int> m_lex2lm;
  TreeGram::Gram m_history_lm; // Temporary variable
  TreeGram *m_lookahead_ngram;

  // Options
  float m_global_beam;
  float m_word_end_beam;
  int m_similar_word_hist_span;
  float m_lm_scale;
  float m_duration_scale;
  float m_transition_scale; // Temporary scaling used for self transitions
  int m_max_num_tokens;
  int m_verbose;
  int m_word_boundary_id;
  int m_word_boundary_lm_id;
  int m_lm_lookahead; // 0=none, 1=bigram, 2=trigram
  int m_max_lookahead_score_list_size;
  int m_max_node_lookahead_buffer_size;
  float m_insertion_penalty;

  float m_current_glob_beam;
  float m_current_we_beam;
  float m_eq_depth_beam;
  float m_eq_wc_beam;
  
  int filecount;

  float m_wc_llh[MAX_WC_COUNT];
  float m_depth_llh[MAX_LEX_TREE_DEPTH/2];
  int m_min_word_count;

  bool m_lm_lookahead_initialized;

  int lm_la_cache_count[MAX_LEX_TREE_DEPTH];
  int lm_la_cache_miss[MAX_LEX_TREE_DEPTH];
  int lm_la_word_cache_count;
  int lm_la_word_cache_miss;
};

#endif // TOKENPASSSEARCH_HH
