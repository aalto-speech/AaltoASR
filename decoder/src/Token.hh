#ifndef TOKEN_HH
#define TOKEN_HH

#include <vector>

#include "LMHistory.hh"
#include "history.hh"
#include "TPLexPrefixTree.hh"

/// The search network is built around a popular idea of a lexical prefix tree.
/// As suggested by Demuynck et al. (2000), the traditional phone-level tree can
/// be made even more efficient by utilizing the HMM level state tying, which
/// has been implemented here.
///
/// The search network is built of nodes (TPLexPrefixTree::Node), which are
/// linked to each other with arcs (TPLexPrefixTree::Arc). Nodes can either
/// correspond to one HMM state, or be dummy nodes without any acoustic
/// probabilities associated with them. In decoding, these dummy nodes are
/// passed immediately, they merely mediate the tokens (TPLexPrefixTree::Token)
/// used to represent the active search network. A node can also have a word
/// identity (word_id) associated with it, which leads to insertion of the word
/// into the word history of the token passing that node.
///
class Token {
public:
  struct WordHistory {
    inline WordHistory(int word_id, int frame, WordHistory *previous);

    int word_id;
    int end_frame;
    int lex_node_id; // FIXME: debug info (node where the history was created)
    int graph_node_id; // FIXME: debug info (word graph node)
    float lm_log_prob;
    float am_log_prob;
    float cum_lm_log_prob;
    float cum_am_log_prob;
    bool printed;

    WordHistory *previous;
    int reference_count;
  };

  struct StateHistory {
    inline StateHistory(int hmm_model, int start_time, StateHistory *previous);

    int hmm_model;
    int start_time;
    float log_prob;

    StateHistory *previous;
    int reference_count;
  };

  TPLexPrefixTree::Node *node;
  Token *next_node_token;
  float am_log_prob;
  float lm_log_prob;
  float cur_am_log_prob; // Used inside nodes
  float cur_lm_log_prob; // Used for LM lookahead
  float total_log_prob;
  LMHistory *lm_history;
  int lm_hist_code; // Hash code for word history (up to LM order)
  int fsa_lm_node;
  int recent_word_graph_node;
  WordHistory *word_history;
  int word_start_frame;

#ifdef PRUNING_MEASUREMENT
  float meas[6];
#endif

  int word_count;

  StateHistory *state_history;

  unsigned char depth;
  unsigned char dur;

  Token():
    node(nullptr),
    next_node_token(nullptr),
    am_log_prob(0.0f),
    lm_log_prob(0.0f),
    cur_am_log_prob(0.0f),
    cur_lm_log_prob(0.0f),
    total_log_prob(0.0f),
    lm_history(nullptr),
    lm_hist_code(0),
    fsa_lm_node(0),
    recent_word_graph_node(0),
    word_history(nullptr),
    word_start_frame(0),
    word_count(0),
    state_history(nullptr),
    depth(0),
    dur(0)
  {}

  /// \brief Writes the state history into a vector.
  ///
  /// Writes the Token::StateHistory objects from the state_history linked list
  /// into a vector. The order will be from the oldest to newest.
  ///
  /// \param result Vector where the result will be written.
  ///
  void get_state_history(std::vector<StateHistory *> & result) const;

  /// \brief Writes the word history into a vector.
  ///
  /// Writes the LMHistory objects from the state_history linked list into a
  /// vector. The order will be from the oldest to newest.
  ///
  /// \param result Vector where the result will be written.
  /// \param limit If set to other than nullptr, prints only the history after
  /// this node.
  ///
  void get_lm_history(std::vector<LMHistory *> & result,
                      LMHistory * limit = nullptr) const;
};

Token::WordHistory::WordHistory(int word_id, int end_frame,
                                WordHistory *previous)
  : word_id(word_id), end_frame(end_frame),
    lex_node_id(0), graph_node_id(0), lm_log_prob(0), am_log_prob(0), cum_lm_log_prob(0), cum_am_log_prob(0),
    printed(false), previous(previous), reference_count(0)
{
  if (previous) {
    hist::link(previous);
    cum_am_log_prob = previous->cum_am_log_prob;
    cum_lm_log_prob = previous->cum_lm_log_prob;
  }
}

Token::StateHistory::StateHistory(int hmm_model, int start_time,
                                  StateHistory *previous)
  : hmm_model(hmm_model),
    start_time(start_time),
    previous(previous),
    reference_count(0)
{
  if (previous)
    hist::link(previous);
}

#endif
