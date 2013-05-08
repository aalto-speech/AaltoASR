#ifndef TOKENPASSSEARCH_HH
#define TOKENPASSSEARCH_HH

#include <stdexcept>
#include <vector>
#include <cmath>

#include "config.hh"
#include "fsalm/LM.hh"
#include "WordGraph.hh"
#include "TPLexPrefixTree.hh"
#include "NGram.hh"
#include "Acoustics.hh"
#include "LMHistory.hh"

// Visual studio math.h doesn't have log1p function varjokal 17.3.2010
#ifdef _MSC_VER
#include <boost/math/special_functions/log1p.hpp>
using namespace boost::math;
#endif

#define MAX_WC_COUNT 200
#define MAX_LEX_TREE_DEPTH 60

typedef std::vector<LMHistory*> HistoryVector;

class TokenPassSearch
{
public:
  struct CannotGenerateWordGraph: public std::runtime_error
  {
    CannotGenerateWordGraph(const std::string & message) :
      std::runtime_error(message)
    {
    }
  };

  struct WordGraphNotGenerated: public std::runtime_error
  {
    WordGraphNotGenerated() :
      std::runtime_error(
			 "Word graph was requested but it has not been generated.")
    {
    }
  };

  struct InvalidSetup: public std::runtime_error
  {
    InvalidSetup(const std::string & message) :
      std::runtime_error(message)
    {
    }
  };

  struct IOError: public std::runtime_error
  {
    IOError(const std::string & message) :
      std::runtime_error(message)
    {
    }
  };

  TokenPassSearch(TPLexPrefixTree &lex, Vocabulary &vocab,
                  Acoustics *acoustics);
  ~TokenPassSearch();

  /// \brief Resets the search and creates the initial token.
  ///
  /// Clears the active token list and adds one token that refers to
  /// \ref m_lexicon.start_node().
  ///
  void reset_search(int start_frame);
  void set_end_frame(int end_frame) { m_end_frame = end_frame; }

  /// \brief Proceeds decoding one frame.
  ///
  /// \exception CannotGenerateWordGraph If it is not possible to generate the
  /// word graph with the current parameters.
  ///
  bool run(void);

  /// \brief Prints the best path from the word_history structure, including
  /// the probabilities.
  ///
  /// If \a get_best_path is true, finds the active token that is in the
  /// NODE_FINAL state i.e. at the end of a word, with the highest probability,
  /// and writes the words and probabilities in the word_history of that token
  /// into a file, separated by spaces. Otherwise finds the common path to all
  /// tokens, and prints the rest that has not been printed yet.
  ///
  /// \exception WordGraphNotGenerated If word graph has not been generated.
  ///
  void write_word_history(FILE *file = stdout, bool get_best_path = true);

  /// \brief Writes the LM history of an active token into a file.
  ///
  /// If \a get_best_path is true, finds the active token that is in the
  /// NODE_FINAL state i.e. at the end of a word, with the highest probability,
  /// and writes the words in the lm_history of that token into a file,
  /// separated by spaces. Otherwise finds the common path to all tokens, and
  /// prints the rest that has not been printed yet.
  ///
  void print_lm_history(FILE *file = stdout, bool get_best_path = true);

  /// \brief Writes the best state history into a file.
  ///
  /// Finds the active token that is in the NODE_FINAL state i.e. at the end
  /// of a word, with the highest probability. Then writes its state history
  /// into a file.
  ///
  /// \param file The file where the result will be written, stoudt by default.
  ///
  void print_state_history(FILE *file = stdout);

  /// \brief Writes the best state history into a text string.
  ///
  /// Finds the active token that is in the NODE_FINAL state i.e. at the end
  /// of a word, with the highest probability. Then writes its state history
  /// into a text string.
  ///
  /// \return The best state history in a text string.
  ///
  std::string state_history_string();

  /// \brief Writes the best state history into a vector.
  ///
  /// Finds the active token that is in the NODE_FINAL state i.e. at the end
  /// of a word, with the highest probability. Then writes the
  /// TPLexPrefixTree::StateHistory objects in the state_history of that
  /// token into a vector.
  ///
  /// \param stack The vector where the result will be written.
  ///
  void get_state_history(std::vector<TPLexPrefixTree::StateHistory*> &stack);

  /// \brief Writes the LM history of an active token into a vector.
  ///
  /// If \a use_best_token is true, finds the active token that is in the
  /// NODE_FINAL state i.e. at the end of a word, with the highest
  /// probability. Otherwise finds any active token. Then writes the
  /// LMHistory objects in the lm_history of that token into
  /// a vector.
  ///
  /// \param vec The vector where the result will be written.
  ///
  void get_path(HistoryVector &vec, bool use_best_token, LMHistory *limit);

  // Options
  void set_acoustics(Acoustics *acoustics) { m_acoustics = acoustics; }
  void set_global_beam(float beam) { m_global_beam = beam; if (m_word_end_beam > m_global_beam) m_word_end_beam = beam; }
  void set_word_end_beam(float beam) { m_word_end_beam = beam; }
  void set_eq_depth_beam(float beam) { m_eq_depth_beam = beam; }
  void set_eq_word_count_beam(float beam) { m_eq_wc_beam = beam; }
  void set_fan_in_beam(float beam) { m_fan_in_beam = beam; }
  void set_fan_out_beam(float beam) { m_fan_out_beam = beam; }
  void set_state_beam(float beam) { m_state_beam = beam; }
  
  void set_similar_lm_history_span(int n) { m_similar_lm_hist_span = n; }
  void set_lm_scale(float lm_scale) { m_lm_scale = lm_scale; }
  void set_duration_scale(float dur_scale) { m_duration_scale = dur_scale; }
  void set_transition_scale(float trans_scale) { m_transition_scale = trans_scale; }
  void set_max_num_tokens(int tokens) { m_max_num_tokens = tokens; }

#ifdef ENABLE_MULTIWORD_SUPPORT
  void set_split_multiwords(bool value)
  {
    m_split_multiwords = value;
  }
#endif

  void set_print_probs(bool value) { m_print_probs = value; }
  void set_print_text_result(int print) { m_print_text_result = print; }
  void set_print_state_segmentation(int print) 
  { 
    m_print_state_segmentation = print; 
    m_keep_state_segmentation = print;
  }
  void set_keep_state_segmentation(int value)
  {
    m_keep_state_segmentation = value;
  }
  void set_verbose(int verbose) { m_verbose = verbose; }

  /// \brief Sets the word that represents word boundary.
  ///
  /// This function has to be called before calling set_ngram() or
  /// set_fsa_lm().
  ///
  /// \param word Word boundary. For word models, an empty string should be
  /// given.
  ///
  /// \exception invalid_argument If \a word is non-empty, but not in
  /// vocabulary.
  ///
  void set_word_boundary(const std::string &word);

  /// \brief Enables or disables lookahead language model.
  ///
  /// Can be enabled only before reading the lexicon.
  ///
  /// \param order 0=None, 1=Only in first subtree nodes, 2=Full.
  ///
  void set_lm_lookahead(int order) { m_lm_lookahead = order; }

  void set_insertion_penalty(float ip) { m_insertion_penalty = ip; }

  void set_require_sentence_end(bool s) { m_require_sentence_end = s; }

  /// \brief Remove :[0-9]+ from the end of each word? This allows the
  /// dictionary to have pronunciation IDs that do not affect decoding.
  ///
  void set_remove_pronunciation_id(bool remove) { m_remove_pronunciation_id = remove; }

  /// This function has to be called before calling set_ngram() or
  /// set_fsa_lm().
  ///
  /// \exception invalid_argument If \a start or \a end is not in vocabulary.
  ///
  void set_sentence_boundary(const std::string &start,
                             const std::string &end);

  void clear_hesitation_words();

  void add_hesitation_word(const std::string & word);

  /// \brief Sets the word classes for class-based language models.
  ///
  /// This function has to be called before calling set_ngram() or
  /// set_fsa_lm().
  ///
  void set_word_classes(const WordClasses * classes);

  /// \brief Sets an n-gram language model.
  ///
  /// \return The number of vocabulary entries that were not found in the
  /// language model.
  ///
  int set_ngram(NGram *ngram);

  /// \brief Sets a finite-state automaton language model.
  ///
  /// \return The number of vocabulary entries that were not found in the
  /// language model.
  ///
  int set_fsa_lm(fsalm::LM *lm);

  /// \brief Sets a lookahead n-gram language model.
  ///
  /// This recreates the word repository that was created after calling
  /// set_ngram(). This was the simplest thing I could think of, without
  /// making changes to the class interface, since we don't know already at
  /// set_ngram() whether the user will set a lookahead model later.
  ///
  /// \return The number of vocabulary entries that were not found in the
  /// language model.
  ///
  int set_lookahead_ngram(NGram *ngram);

  /// \brief If set to true, generates a word graph of the hypotheses during
  /// decoding (requires memory).
  ///
  void set_generate_word_graph(bool value)
  {
    m_generate_word_graph = value;
  }

  /// \brief Returns the value of the word graph generation flag.
  ///
  bool get_generate_word_graph() const
  {
    return m_generate_word_graph;
  }

  /// \brief Enables or disables word pair approximation when building a word
  /// graph.
  ///
  /// Enabled by default.
  ///
  void set_use_word_pair_approximation(bool value)
  {
    m_use_word_pair_approximation = value;
  }

  void set_use_lm_cache(bool value)
  {
    m_use_lm_cache = value;
  }

  int frame(void)
  {
    return m_frame;
  }

  /// \brief Writes nodes and arcs from word_graph to a Standard Lattice
  /// Format file.
  ///
  /// \exception WordGraphNotGenerated If word graph has not been generated.
  /// \exception IOError If unable to write the file.
  ///
  void write_word_graph(const std::string &file_name);
  void write_word_graph(FILE *file);

  void debug_ensure_all_paths_contain_history(LMHistory *limit);

  /// \brief Returns the logarithmic AM probability of an active token.
  ///
  /// If \a use_best_token is true, finds the active token that is in the
  /// NODE_FINAL state i.e. at the end of a word, with the highest
  /// probability. Otherwise finds any active token. Then returns the
  /// logarithm of the acoustic model probability.
  ///
  /// \return The logarithm of the AM probability.
  ///
  float get_am_log_prob(bool use_best_token) const;

  /// \brief Returns the logarithmic LM probability of an active token.
  ///
  /// If \a use_best_token is true, finds the active token that is in the
  /// NODE_FINAL state i.e. at the end of a word, with the highest
  /// probability. Otherwise finds any active token. Then returns the
  /// logarithm of the language model probability.
  ///
  /// \return The logarithm of the LM probability.
  ///
  float get_lm_log_prob(bool use_best_token) const;

  /// \brief Returns the logarithmic probability of an active token.
  ///
  /// If \a use_best_token is true, finds the active token that is in the
  /// NODE_FINAL state i.e. at the end of a word, with the highest
  /// probability. Otherwise finds any active token. Then returns the
  /// logarithm of the total (acoustic and language model) probability.
  ///
  /// \return The logarithm of the total probability.
  ///
  float get_total_log_prob(bool use_best_token) const;

  /// \brief For unit testing.
  const std::vector<LMHistory::Word> & get_word_repository() const;
  const WordClasses * get_word_classes() const;
  const Vocabulary & get_vocabulary() const;
  const NGram * get_ngram() const;

private:
  /// \brief Creates a lookup table for LMHistory::Word structures.
  ///
  /// \return The number of vocabulary entries that were not found in the
  /// language model.
  ///
  int create_word_repository();

  /// \brief Finds out the language model ID and class membership log
  /// probability of a word.
  ///
  /// If word classes are set using set_word_classes(), assumes that the
  /// language model is based on classes, and gives the language model ID of
  /// the corresponding class.
  ///
  /// If the word does not exist in the class definitions, tries to find it
  /// from the language model as it is, so that the language model can mix
  /// classes and regular words.
  ///
  /// \param lm_id Will be set to the language model ID, or -1 if the word or
  /// class does not exist in the language model.
  /// \param cm_log_prob Will be set to the class membership log probability,
  /// or 0 if the word is not a class definition.
  ///
  void find_word_from_lm(int word_id, std::string word, int & lm_id,
			 float & cm_log_prob) const;

  /// \brief Finds out the ID of a word in the lookahead LM.
  ///
  /// If word classes are set using set_word_classes(), assumes that the
  /// lookahead language model is based on classes, and gives the lookahead
  /// LM ID of the corresponding class.
  ///
  /// If the word does not exist in the class definitions, tries to find it
  /// from the lookahead language model as it is, so that the lookahead LM can
  /// mix classes and regular words.
  ///
  /// \return lookahead_lm_id Will be set to the lookahead LM ID, or 0 if the
  /// word or class does not exist in the lookahead LM.
  ///
  int find_word_from_lookahead_lm(int word_id, std::string word) const;

  /// \brief Finds the globally best token that is in the NODE_FINAL state,
  /// i.e. at the end of a word.
  ///
  /// \return A reference to the active token in NODE_FINAL state with the
  /// highest probability. If none is found, returns the best token not in
  /// NODE_FINAL state.
  ///
  const TPLexPrefixTree::Token & get_best_final_token() const;

  /// \brief Returns the first token in the active token list.
  ///
  const TPLexPrefixTree::Token & get_first_token() const;

  void add_sentence_end_to_hypotheses(void);

  /// \brief Propagates all the tokens in the active token list to the
  /// following nodes.
  ///
  void propagate_tokens(void);

  /// \brief Adds the sentence end symbol to every token.
  ///
  /// Adds sentence end to the LMHistory of every token, and to the
  /// WordHistory of every token that is in a final node. If there are no
  /// tokens in a final node, LMHistory will have sentence end, but
  /// WordHistory will not.
  ///
  void update_final_tokens();

  void copy_word_graph_info(TPLexPrefixTree::Token *src_token,
			    TPLexPrefixTree::Token *tgt_token);

  /// \brief Adds an arc to the word graph from the previous endpoint of
  /// \a new_token to a node that represents the last word in its word
  /// history.
  ///
  /// If there already exists a node for the last word in the word history at
  /// this frame with the same lex_node_id, uses the old node as the target
  /// node of the new arc.
  ///
  /// If there already exists an arc between the source node and the target
  /// node, just updates the arc probability.
  ///
  void build_word_graph_aux(TPLexPrefixTree::Token *new_token,
			    TPLexPrefixTree::WordHistory *word_history);
  void build_word_graph(TPLexPrefixTree::Token *new_token);

  /// \brief Moves the token towards all the arcs leaving the token's node.
  ///
  void propagate_token(TPLexPrefixTree::Token *token);

  /// \brief Appends a word to the LMHistory of a token.
  ///
  void append_to_word_history(TPLexPrefixTree::Token & token,
			      const LMHistory::Word & word);

  /// \brief Moves token to a connected node.
  ///
  /// Adds new tokens to \ref m_new_token_list.
  ///
  /// \param token The token to move.
  /// \param node A nodes that is connected to token's node
  ///
  void move_token_to_node(TPLexPrefixTree::Token *token,
                          TPLexPrefixTree::Node *node,
                          float transition_score);

  /// \brief Copes new tokens from \ref m_new_token_list to
  /// \ref m_active_token_list.
  ///
  void prune_tokens(void);

#ifdef PRUNING_MEASUREMENT
  void analyze_tokens(void);
#endif

  TPLexPrefixTree::Token*
  find_similar_fsa_token(int fsa_lm_node, TPLexPrefixTree::Token *token_list);

  /// \brief Finds a token from \a token_list that has similar LMHistory to
  /// \a wh up to m_similar_lm_hist_span words or classes.
  ///
  /// First checks the hash code and if they match, verifies using
  /// is_similar_lm_history().
  ///
  /// Note: Doesn't work if the sentence end is the first one in the word
  /// history!
  ///
  TPLexPrefixTree::Token* find_similar_lm_history(LMHistory *wh,
						  int lm_hist_code, TPLexPrefixTree::Token *token_list);

  /// \brief Checks if wh1 and wh2 are similar up to m_similar_lm_hist_span
  /// words or classes.
  ///
  /// Note: Doesn't work if the sentence end is the first one in the word
  /// history!
  ///
  bool is_similar_lm_history(LMHistory *wh1, LMHistory *wh2);

  /// \brief Computes a hash code from the m_similar_lm_hist_span last words
  /// in the LMHistory. The code will be used for recombination of similar
  /// histories.
  ///
  int compute_lm_hist_hash_code(LMHistory *wh) const;

  // language model scoring

#ifdef ENABLE_MULTIWORD_SUPPORT
  /// \brief Collects words from the LM history into an n-gram and returns its
  /// language model probability.
  ///
  /// Splits multiwords into their components, and if the last word is a
  /// multiword, sums the LM log probs of each component, since LM
  /// probabilities are not applied until the whole multiword has been
  /// decoded.
  ///
  float split_and_compute_ngram_score(LMHistory * history);
#endif

  /// \brief Creates m_history_ngram from at most \a words_needed words from
  /// \a history. The last word will be the final word of \a history. The
  /// number of words added is smaller if the beginning of history is reached
  /// sooner, or a sentence start is encountered.
  ///
  void create_history_ngram(LMHistory * history, int words_needed);

  /// \brief Collects words from the LM history into an n-gram and returns its
  /// language model probability.
  ///
  float compute_ngram_score(LMHistory * history);

  /// \brief Returns the probability for the n-gram in the LM history from the
  /// n-gram LM or its cache.
  ///
  float get_ngram_score(LMHistory *lm_hist, int lm_hist_code);

  /// \brief Moves a token to the next FSA language model node, and adds the
  /// transition probability to the LM log probability of the token.
  ///
  void advance_fsa_lm(TPLexPrefixTree::Token & token);

  /// \brief Updated lm_log_prob and lm_hist_code on token after adding a new
  /// word to the end of its lm_history.
  ///
  void update_lm_log_prob(TPLexPrefixTree::Token & token);

  /// \brief Computes the lookahead score as the maximum of possible word ends
  /// to the given LMHistory.
  ///
  /// Note! Doesn't work if the sentence end is the first one in the word
  /// history. Returns 0 in that case.
  ///
  float get_lm_lookahead_score(LMHistory *lm_hist,
                                TPLexPrefixTree::Node *node, int depth);

  /// \brief Computes bi-gram probabilities for every word pair starting with
  /// \a prev_word_id, using the lookahead LM, and returns the maximum.
  ///
  float get_lm_bigram_lookahead(int prev_word_id,
                                TPLexPrefixTree::Node *node, int depth);

  /// \brief Computes tri-gram probabilities for every word triplet starting
  /// with \a w1 \a w2, using the lookahead LM, and returns the maximum.
  ///
  float get_lm_trigram_lookahead(int w1, int w2,
                                 TPLexPrefixTree::Node *node, int depth);

  void clear_active_node_token_lists(void);

  inline float get_token_log_prob(float am_score, float lm_score)
  {
    return (am_score + m_lm_scale * lm_score);
  }

  TPLexPrefixTree::Token* acquire_token(void);
  LMHistory* acquire_lmhist(const LMHistory::Word *, LMHistory *);
  void release_token(TPLexPrefixTree::Token *token);
  void release_lmhist(LMHistory *);

  void save_token_statistics(int count);
  //void print_token_path(TPLexPrefixTree::PathHistory *hist);

  // Help variables to cope with memory leaks
  // Vector of pointers to memory blocks for m_token_pool, this is only for freeing up memory at the destructor
  std::vector<TPLexPrefixTree::Token *> m_token_dealloc_table;
  std::vector<LMHistory *> m_lmhist_dealloc_table;

public:

  /** The word graph of best hypotheses created during the recognition. */
  WordGraph word_graph;

private:
  TPLexPrefixTree &m_lexicon;
  Vocabulary &m_vocabulary;
#ifdef ENABLE_WORDCLASS_SUPPORT
  const WordClasses * m_word_classes;
#endif
  Acoustics *m_acoustics;

  typedef std::vector<TPLexPrefixTree::Token *> token_list_type;
  token_list_type * m_active_token_list;
  token_list_type * m_new_token_list;
  token_list_type * m_word_end_token_list;
  token_list_type m_token_pool;
  std::vector<LMHistory*> m_lmh_pool;

  std::vector<TPLexPrefixTree::Node*> m_active_node_list;

  class LMLookaheadScoreList
  {
  public:
    int index;
    std::vector<float> lm_scores;
  };
  HashCache<LMLookaheadScoreList*> lm_lookahead_score_list;

  class LMScoreInfo
  {
  public:
    float lm_score;
    std::vector<int> lm_hist;
  };
  HashCache<LMScoreInfo*> m_lm_score_cache;

  int m_end_frame;
  int m_frame; // Current frame

  float m_best_log_prob; // The best total_log_prob in active tokens
  float m_worst_log_prob;
  float m_best_we_log_prob;

  struct WordGraphInfo
  {
    struct Item
    {
      Item() : lex_node_id(-1), graph_node_id(-1) { }
      int lex_node_id;
      int graph_node_id;
    };
    WordGraphInfo() : frame(-1) { }
    int frame;
    std::vector<Item> items;
  };

  /// Records for each word, the frame number when it was last inserted into
  /// the word graph, and the node ID of each occurrence of the word at this
  /// time instance.
  std::vector<WordGraphInfo> m_recent_word_graph_info;

  TPLexPrefixTree::Token *m_best_final_token;

  /// The language model.
  NGram *m_ngram;
  fsalm::LM *m_fsa_lm;

  /// This is a repository of LMHistory::Word structures, indexed by
  /// dictionary word ID.
  std::vector<LMHistory::Word> m_word_repository;

  /// A null word (IDs -1) starts every LM history.
  LMHistory::Word m_null_word;

#ifdef ENABLE_MULTIWORD_SUPPORT
  /// Should the decoder split multiwords into their components before
  /// computing LM probabilities?
  bool m_split_multiwords;
#endif

  NGram::Gram m_history_ngram; // Temporary variable used by compute_ngram_score().
  NGram *m_lookahead_ngram;

  // Options
  float m_print_probs;
  int m_print_text_result;
  bool m_print_state_segmentation;
  bool m_keep_state_segmentation;
  float m_global_beam;
  float m_word_end_beam;
  int m_similar_lm_hist_span;
  float m_lm_scale;
  float m_duration_scale;
  float m_transition_scale; // Temporary scaling used for self transitions
  int m_max_num_tokens;
  int m_verbose;
  int m_word_boundary_id;
  int m_lm_lookahead; // 0=none, 1=bigram, 2=trigram
  int m_max_lookahead_score_list_size;
  int m_max_node_lookahead_buffer_size;
  float m_insertion_penalty;

  int m_sentence_start_id;
  int m_sentence_end_id;
  std::vector<int> m_hesitation_ids;
  bool m_use_sentence_boundary;
  bool m_generate_word_graph;
  bool m_require_sentence_end;
  bool m_remove_pronunciation_id;
  bool m_use_word_pair_approximation;
  bool m_use_lm_cache;

  float m_current_glob_beam;
  float m_current_we_beam;
  float m_eq_depth_beam;
  float m_eq_wc_beam;
  float m_fan_in_beam;
  float m_fan_out_beam;
  float m_state_beam;

  int filecount;

  float m_wc_llh[MAX_WC_COUNT];
  float m_depth_llh[MAX_LEX_TREE_DEPTH / 2];
  int m_min_word_count;

  float m_fan_in_log_prob;
  float m_fan_out_log_prob;
  float m_fan_out_last_log_prob;

  bool m_lm_lookahead_initialized;

  int lm_la_cache_count[MAX_LEX_TREE_DEPTH];
  int lm_la_cache_miss[MAX_LEX_TREE_DEPTH];
  int lm_la_word_cache_count;
  int lm_la_word_cache_miss;

  // FIXME: remove debug
public:
  void debug_print_best_lm_history();
  void debug_print_token_lm_history(FILE * file,
				    const TPLexPrefixTree::Token & token);
  void debug_print_token_word_history(FILE * file,
				      const TPLexPrefixTree::Token & token);
};

#endif // TOKENPASSSEARCH_HH
