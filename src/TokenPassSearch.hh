#ifndef TOKENPASSSEARCH_HH
#define TOKENPASSSEARCH_HH

#include <stdexcept>
#include <vector>

#include "config.hh"
#include "fsalm/LM.hh"
#include "WordGraph.hh"
#include "TPLexPrefixTree.hh"
#include "TreeGram.hh"
#include "Acoustics.hh"

#define MAX_WC_COUNT 200
#define MAX_LEX_TREE_DEPTH 60

typedef std::vector<TPLexPrefixTree::LMHistory*> HistoryVector;

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

	/// \brief Resets the search and creates the initial token.
	///
	/// Clears the active token list and adds one token that refers to
	/// \ref m_lexicon.start_node().
	///
	void reset_search(int start_frame);

	void set_end_frame(int end_frame)
	{
		m_end_frame = end_frame;
	}

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
	/// TPLexPrefixTree::LMHistory objects in the lm_history of that token into
	/// a vector.
	///
	/// \param vec The vector where the result will be written.
	///
	void get_path(HistoryVector &vec, bool use_best_token,
			TPLexPrefixTree::LMHistory *limit);

	// Options
	void set_acoustics(Acoustics *acoustics)
	{
		m_acoustics = acoustics;
	}
	void set_global_beam(float beam)
	{
		m_global_beam = beam;
		if (m_word_end_beam > m_global_beam)
			m_word_end_beam = beam;
	}
	void set_word_end_beam(float beam)
	{
		m_word_end_beam = beam;
	}
	void set_eq_depth_beam(float beam)
	{
		m_eq_depth_beam = beam;
	}
	void set_eq_word_count_beam(float beam)
	{
		m_eq_wc_beam = beam;
	}
	void set_fan_in_beam(float beam)
	{
		m_fan_in_beam = beam;
	}
	void set_fan_out_beam(float beam)
	{
		m_fan_out_beam = beam;
	}
	void set_state_beam(float beam)
	{
		m_state_beam = beam;
	}

	void set_similar_lm_history_span(int n)
	{
		m_similar_lm_hist_span = n;
	}
	void set_lm_scale(float lm_scale)
	{
		m_lm_scale = lm_scale;
	}
	void set_duration_scale(float dur_scale)
	{
		m_duration_scale = dur_scale;
	}
	void set_transition_scale(float trans_scale)
	{
		m_transition_scale = trans_scale;
	}
	void set_max_num_tokens(int tokens)
	{
		m_max_num_tokens = tokens;
	}
	void set_split_multiwords(bool value)
	{
		m_split_multiwords = value;
	}
	void set_print_probs(bool value)
	{
		m_print_probs = value;
	}
	void set_print_text_result(int print)
	{
		m_print_text_result = print;
	}
	void set_print_state_segmentation(int print)
	{
		m_print_state_segmentation = print;
		m_keep_state_segmentation = print;
	}
	void set_keep_state_segmentation(int value)
	{
		m_keep_state_segmentation = value;
	}
	void set_verbose(int verbose)
	{
		m_verbose = verbose;
	}

	/// \brief Sets the word that represents word boundary.
	///
	/// \param word Word boundary. For word models, an empty string should be
	/// given.
	///
	/// \exception invalid_argument If \a word is non-empty, but not in
	/// vocabulary.
	///
	void set_word_boundary(const std::string &word);

	void set_lm_lookahead(int order)
	{
		m_lm_lookahead = order;
	}
	void set_insertion_penalty(float ip)
	{
		m_insertion_penalty = ip;
	}

	void set_require_sentence_end(bool s)
	{
		m_require_sentence_end = s;
	}

	/// \exception invalid_argument If \a start or \a end is not in vocabulary.
	///
	void
	set_sentence_boundary(const std::string &start, const std::string &end);

	/// \brief Sets an n-gram language model.
	///
	/// \return The number of vocabulary entries that were not found in the
	/// language model.
	///
	int set_ngram(TreeGram *ngram);

	/// \brief Sets a finite-state automaton language model.
	///
	/// \return The number of vocabulary entries that were not found in the
	/// language model.
	///
	int set_fsa_lm(fsalm::LM *lm);

	/// \brief Sets a lookahead n-gram language model.
	///
	/// \return The number of vocabulary entries that were not found in the
	/// language model.
	///
	int set_lookahead_ngram(TreeGram *ngram);

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

	void debug_ensure_all_paths_contain_history(
			TPLexPrefixTree::LMHistory *limit);

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

private:
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
	void build_word_graph_aux(TPLexPrefixTree::Token *new_token,
			TPLexPrefixTree::WordHistory *word_history);
	void build_word_graph(TPLexPrefixTree::Token *new_token);

	/// \brief Moves the token towards all the arcs leaving the token's node.
	///
	void propagate_token(TPLexPrefixTree::Token *token);

	/// \brief Moves token to a connected node.
	///
	/// Adds new tokens to \ref m_new_token_list.
	///
	/// \param token The token to move.
	/// \param node A nodes that is connected to token's node
	///
	void move_token_to_node(TPLexPrefixTree::Token *token,
			TPLexPrefixTree::Node *node, float transition_score);

	/// \brief Updated lm_log_prob and lm_hist_code on token after advancing to
	/// a new node.
	///
	void update_lm_log_prob(TPLexPrefixTree::Token & token);

	/// \brief Copes new tokens from \ref m_new_token_list to
	/// \ref m_active_token_list.
	///
	void prune_tokens(void);

#ifdef PRUNING_MEASUREMENT
	void analyze_tokens(void);
#endif

	TPLexPrefixTree::Token*
	find_similar_fsa_token(int fsa_lm_node, TPLexPrefixTree::Token *token_list);

	TPLexPrefixTree::Token* find_similar_lm_history(
			TPLexPrefixTree::LMHistory *wh, int lm_hist_code,
			TPLexPrefixTree::Token *token_list);
	bool is_similar_lm_history(TPLexPrefixTree::LMHistory *wh1,
			TPLexPrefixTree::LMHistory *wh2);
	int compute_lm_hist_hash_code(TPLexPrefixTree::LMHistory *wh) const;

#ifdef ENABLE_MULTIWORD_SUPPORT
	/// \brief Creates m_history_ngram from at most \a words_needed words from
	/// \a history. The number of words added is smaller if the beginning of
	/// history is reached sooner, or a sentence start is encountered.
	///
	/// The last word will be the final word of \a history. If it's a multiword,
	/// \a final_components specifies how many of its components will be
	/// considered. This is needed to be able to compile a history that ends in
	/// any of the components of a final multiword.
	///
	void split_and_create_history_ngram(
			TPLexPrefixTree::LMHistory * history,
			int final_components,
			int words_needed);

	/// \brief Collects words from the LM history into an n-gram and returns its
	/// language model probability.
	///
	/// Splits multiwords into their components, and if the last word is a
	/// multiword, sums the LM log probs of each component, since LM
	/// probabilities are not applied until the whole multiword has been
	/// decoded.
	///
	float split_and_compute_lm_log_prob(
			TPLexPrefixTree::LMHistory * history);
#endif

	/// \brief Creates m_history_ngram from at most \a words_needed words from
	/// \a history. The last word will be the final word of \a history. The
	/// number of words added is smaller if the beginning of history is reached
	/// sooner, or a sentence start is encountered.
	///
	void create_history_ngram(
			TPLexPrefixTree::LMHistory * history,
			int words_needed);

	/// \brief Collects words from the LM history into an n-gram and returns its
	/// language model probability.
	///
	float compute_lm_log_prob(TPLexPrefixTree::LMHistory * history);

	float get_lm_score(TPLexPrefixTree::LMHistory *lm_hist, int lm_hist_code);
	float get_lm_lookahead_score(TPLexPrefixTree::LMHistory *lm_hist,
			TPLexPrefixTree::Node *node, int depth);
	float get_lm_bigram_lookahead(int prev_word_id,
			TPLexPrefixTree::Node *node, int depth);
	float get_lm_trigram_lookahead(int w1, int w2, TPLexPrefixTree::Node *node,
			int depth);

	void clear_active_node_token_lists(void);

	inline float get_token_log_prob(float am_score, float lm_score)
	{
		return (am_score + m_lm_scale * lm_score);
	}

	TPLexPrefixTree::Token* acquire_token(void);
	void release_token(TPLexPrefixTree::Token *token);

	void save_token_statistics(int count);
	//void print_token_path(TPLexPrefixTree::PathHistory *hist);

public:

	/** The word graph of best hypotheses created during the recognition. */
	WordGraph word_graph;

private:
	TPLexPrefixTree &m_lexicon;
	Vocabulary &m_vocabulary;
	Acoustics *m_acoustics;

	typedef std::vector<TPLexPrefixTree::Token *> token_list_type;
	token_list_type * m_active_token_list;
	token_list_type * m_new_token_list;
	token_list_type * m_word_end_token_list;
	token_list_type m_token_pool;

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
			Item() :
				lex_node_id(-1), graph_node_id(-1)
			{
			}
			int lex_node_id;
			int graph_node_id;
		};
		WordGraphInfo() :
			frame(-1)
		{
		}
		int frame;
		std::vector<Item> items;
	};
	std::vector<WordGraphInfo> m_recent_word_graph_info;
	TPLexPrefixTree::Token *m_best_final_token;

	/// The language model.
	TreeGram *m_ngram;
	fsalm::LM *m_fsa_lm;

	/// A mapping between word IDs in the dictionary and word IDs in the LM.
	std::vector<int> m_lex2lm;

#ifdef ENABLE_MULTIWORD_SUPPORT
	/// A mapping from a multiword in the dictionary to each of its components
	/// in the LM.
	std::vector< std::vector<int> > m_multiword_lex2lm;

	/// Should the decoder split multiwords into their components before
	/// computing LM probabilities?
	bool m_split_multiwords;
#endif

	/// A mapping between word IDs in the dictionary and word IDs in the lookahead LM.
	std::vector<int> m_lex2lookaheadlm;

	TreeGram::Gram m_history_ngram; // Temporary variable used by compute_lm_log_prob().
	TreeGram *m_lookahead_ngram;

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
	int m_word_boundary_lm_id;
	int m_lm_lookahead; // 0=none, 1=bigram, 2=trigram
	int m_max_lookahead_score_list_size;
	int m_max_node_lookahead_buffer_size;
	float m_insertion_penalty;

	int m_sentence_start_id;
	int m_sentence_start_lm_id;
	int m_sentence_end_id;
	int m_sentence_end_lm_id;
	bool m_use_sentence_boundary;
	bool m_generate_word_graph;
	bool m_require_sentence_end;
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
	void debug_print_token_lm_history(FILE * file, const TPLexPrefixTree::Token & token);
	void debug_print_token_word_history(FILE * file, const TPLexPrefixTree::Token & token);
};

#endif // TOKENPASSSEARCH_HH
