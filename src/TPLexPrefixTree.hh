#ifndef TPLEXPREFIXTREE_HH
#define TPLEXPREFIXTREE_HH

#include <cstddef>  // NULL
#include <vector>
#include <cassert>
#include <cmath>

#include "config.hh"
#include "HashCache.hh"
#include "SimpleHashCache.hh"

#include "history.hh"
#include "Hmm.hh"
#include "Vocabulary.hh"
#include "WordClasses.hh"

// Constructs and maintains the lexical prefix tree.

// NOTE!
// - Assumes that the transitions from the hmm states with identical
//   mixture models are the same, although with different destinations.
// - HMMs must have left-to-right topology, skip states are allowed,
//   except from the source state, see the next note.
// - If there are transitions from HMM source state to other states than to
//   the first real state and to the sink state, the tree construction might
//   not work with shared HMM states.


// Node flags
#define NODE_NORMAL               0x00
#define NODE_USE_WORD_END_BEAM    0x01
#define NODE_AFTER_WORD_ID        0x02
#define NODE_FAN_OUT              0x04
#define NODE_FAN_OUT_FIRST        0x08
#define NODE_FAN_IN               0x10
#define NODE_FAN_IN_FIRST         0x20
#define NODE_INSERT_WORD_BOUNDARY 0x40
#define NODE_FAN_IN_CONNECTION    0x80
#define NODE_LINKED               0x0100
#define NODE_SILENCE_FIRST        0x0200
#define NODE_FIRST_STATE_OF_WORD  0x0400
#define NODE_FINAL                0x0800
#define NODE_DEBUG_PRUNED	  0x4000
#define NODE_DEBUG_PRINTED	  0x8000


class TPLexPrefixTree {
public:

  class Node;

  typedef std::vector<Node *> node_vector;
  typedef std::map<std::string, node_vector> string_to_nodes_map;

  class LMHistoryWord {
  public:
    LMHistoryWord(int word_id, int lm_id, const WordClasses * classes);

    /// \brief Returns the word ID in the dictionary.
    int word_id() const { return m_word_id; }

    /// \brief Returns the word ID in the language model.
    int lm_id() const { return m_lm_id; }

#ifdef ENABLE_WORDCLASS_SUPPORT
    /// \brief Returns the ID of the word class.
    int class_id() const
    { return m_class_id; }
#endif

#ifdef ENABLE_MULTIWORD_SUPPORT
    /// \brief Sets the language model word ID of each component of a multiword.
    void set_component_lm_ids(const std::vector<int> & x);

    /// \brief Returns the number of components in a multiword (1 for regular
    /// words).
    int num_components() const
    { return m_component_lm_ids.size(); }

    /// \brief Returns the word ID of component \a index in the language model.
    int component_lm_id(int index) const
    { return m_component_lm_ids[index]; }
#endif

  private:
    /// Word ID in the dictionary.
    int m_word_id;

    /// Word ID in the language model.
    int m_lm_id;

#ifdef ENABLE_WORDCLASS_SUPPORT
    /// Word class ID.
    int m_class_id;
#endif

#ifdef ENABLE_MULTIWORD_SUPPORT
    /// Word IDs in the language model of the component words.
    std::vector<int> m_component_lm_ids;
#endif
  };

  struct LMHistory {
    LMHistory(const LMHistoryWord & last_word, LMHistory  *previous);

    const LMHistoryWord & last() const { return m_last_word; }

  public:
    LMHistory *previous;
    int reference_count;
    bool printed;
    int word_start_frame;
    int word_first_silence_frame; // "end frame", initialized to -1

  private:
    LMHistoryWord m_last_word;
  };

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

  class Token {
  public:
    Node *node;
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
  };

  class Arc {
  public:
    float log_prob;
    Node *next;
  };

  class Node {
  public:
    inline Node() : word_id(-1), state(NULL), token_list(NULL), flags(NODE_NORMAL) { }
    inline Node(int wid) : word_id(wid), state(NULL), token_list(NULL), flags(NODE_NORMAL) { }
    inline Node(int wid, HmmState *s) : word_id(wid), state(s), token_list(NULL), flags(NODE_NORMAL) { }
    int word_id; // -1 if none
    int node_id;
    HmmState *state;
    Token *token_list;
    std::vector<Arc> arcs;

    unsigned short flags;

    std::vector<int> possible_word_id_list;
    SimpleHashCache<float> lm_lookahead_buffer;
  };

  struct NodeArcId {
    Node *node;
    int arc_index;
  };

  TPLexPrefixTree(std::map<std::string,int> &hmm_map, std::vector<Hmm> &hmms);

  /// \brief Deletes all the nodes from \ref m_nodes.
  ///
  ~TPLexPrefixTree();

  /// \brief Returns a pointer to the root node.
  ///
  /// The pointer is granted to remain valid until the next call to
  /// initialize_lex_tree().
  ///
  inline TPLexPrefixTree::Node *root() { return m_root_node; }

  /// \brief Returns a pointer to the start node.
  ///
  /// The pointer is granted to remain valid until the next call to
  /// initialize_lex_tree().
  ///
  inline TPLexPrefixTree::Node *start_node() { return m_start_node; }

  inline int words() const { return m_words; }

  void set_verbose(int verbose) { m_verbose = verbose; }

  /// \brief Enables or disables lookahead language model.
  ///
  /// Can be enabled only before reading the lexicon.
  ///
  /// \param lm_lookahead 0=None, 1=Only in first subtree nodes, 2=Full.
  ///
  void set_lm_lookahead(int lm_lookahead);

  void set_cross_word_triphones(bool cw_triphones) { m_cross_word_triphones = cw_triphones; }
  void set_silence_is_word(bool b) { m_silence_is_word = b; }
  void set_ignore_case(bool b) { m_ignore_case = b; }

  void initialize_lex_tree(void);
  void add_word(std::vector<Hmm*> &hmm_list, int word_id);
  void finish_tree(void);
  
  void prune_lookahead_buffers(int min_delta, int max_depth);
  void set_lm_lookahead_cache_sizes(int cache_size);

  void set_word_boundary_id(int id) { m_word_boundary_id = id; }
  void set_optional_short_silence(bool state) { m_optional_short_silence = state; }
  void set_sentence_boundary(int sentence_start_id, int sentence_end_id);

  void clear_node_token_lists(void);

  void print_node_info(int node);
  void print_lookahead_info(int node, const Vocabulary &voc);
  void debug_prune_dead_ends(Node *node);
  void debug_add_silence_loop();
  
  
private:
  void expand_lexical_tree(Node *source, Hmm *hmm, HmmTransition &t,
                           float cur_trans_log_prob,
                           int word_end,
                           node_vector &hmm_state_nodes,
                           node_vector &sink_nodes,
                           std::vector<float> &sink_trans_log_probs,
                           unsigned short flags);
  void post_process_lex_branch(Node *node, std::vector<int> *lm_la_list);
  bool post_process_fan_triphone(Node *node, std::vector<int> *lm_la_list,
                                 bool fan_in);

  /// \brief Deletes all the nodes from \ref m_nodes, then creates new root,
  /// end, and start nodes.
  ///
  void initialize_nodes();

  /// \brief Creates fan in HMMs
  ///
  /// Assumes triphone models. Labels must be of form a-b+c.
  ///
  void create_cross_word_network(void);

  void add_hmm_to_fan_network(int hmm_id,
                              bool fan_out);
  void link_fan_out_node_to_fan_in(Node *node, const std::string &key);
  void link_node_to_fan_network(const std::string &key,
                                std::vector<Arc> &source_arcs,
                                bool fan_out,
                                bool ignore_length,
                                float out_transition_log_prob);
  void add_single_hmm_word_for_cross_word_modeling(Hmm *hmm, int word_id);
  void link_fan_in_nodes(void);
  void create_lex_tree_links_from_fan_in(Node *fan_in_node,
                                         const std::string &key);

  void analyze_cross_word_network(void);
  void count_fan_size(Node *node, unsigned short flag,
                      int *num_nodes, int *num_arcs);
  void count_prefix_tree_size(Node *node, int *num_nodes,
                              int *num_arcs);
  
  void free_cross_word_network_connection_points(void);
  Node* get_short_silence_node(void);
  Node* get_fan_out_entry_node(HmmState *state, const std::string &label);
  Node* get_fan_out_last_node(HmmState *state, const std::string &label);
  Node* get_fan_in_entry_node(HmmState *state, const std::string &label);
  Node* get_fan_in_last_node(HmmState *state, const std::string &label);

  /// \brief Finds the node with given state model, creating a new node if
  /// necessary.
  ///
  Node* get_fan_state_node(HmmState *state, node_vector & nodes);

  /// \brief Returns a reference to the entry of \ref nmap with given key,
  /// creating a new node_vector if necessary.
  ///
  node_vector & get_fan_node_list(
    const std::string &key,
    string_to_nodes_map &nmap);

  void add_fan_in_connection_node(Node *node, const std::string &prev_label);
  float get_out_transition_log_prob(Node *node);
  void prune_lm_la_buffer(int delta_thr, int depth_thr,
                          Node *node, int last_size, int cur_depth);

private:
  int m_words; // Largest word_id in the nodes plus one
  Node *m_root_node;
  Node *m_end_node;
  Node *m_start_node;
  Node *m_silence_node;
  Node *m_last_silence_node;
  node_vector m_nodes;
  int m_verbose;
  int m_lm_lookahead; // 0=None, 1=Only in first subtree nodes,
                      // 2=Full
  bool m_cross_word_triphones;
  int m_lm_buf_count;

  bool m_silence_is_word;
  bool m_ignore_case;
  bool m_optional_short_silence;
  HmmState *m_short_silence_state;
  int m_word_boundary_id;

  std::map<std::string,int> &m_hmm_map;
  std::vector<Hmm> &m_hmms;

  string_to_nodes_map m_fan_out_entry_nodes;
  string_to_nodes_map m_fan_out_last_nodes;
  string_to_nodes_map m_fan_in_entry_nodes;
  string_to_nodes_map m_fan_in_last_nodes;
  string_to_nodes_map m_fan_in_connection_nodes;
  std::vector<NodeArcId> m_silence_arcs;
};

//////////////////////////////////////////////////////////////////////

inline TPLexPrefixTree::LMHistoryWord::LMHistoryWord(
		int word_id_arg, int lm_id_arg, const WordClasses * classes)
  : m_word_id(word_id_arg), m_lm_id(lm_id_arg)
{
#ifdef ENABLE_WORDCLASS_SUPPORT
  if (classes != NULL) {
	  m_class_id = classes->get_membership(m_word_id).class_id;
  }
  else {
    m_class_id = -1;
  }
#endif
#ifdef ENABLE_MULTIWORD_SUPPORT
  m_component_lm_ids.push_back(lm_id_arg);
#endif
}

#ifdef ENABLE_MULTIWORD_SUPPORT
inline void TPLexPrefixTree::LMHistoryWord::set_component_lm_ids(
		const std::vector<int> & x)
{
  m_component_lm_ids = x;
}
#endif

inline TPLexPrefixTree::LMHistory::LMHistory(
		const LMHistoryWord & last_word, LMHistory * previous)
  : previous(previous),
    reference_count(0),
    printed(false),
    word_start_frame(0),
    word_first_silence_frame(-1),
    m_last_word(last_word)
{
  if (previous)
    hist::link(previous);
}

TPLexPrefixTree::WordHistory::WordHistory(int word_id, int end_frame, 
					  WordHistory *previous)
  : word_id(word_id), end_frame(end_frame), 
    lm_log_prob(0), am_log_prob(0), cum_lm_log_prob(0), cum_am_log_prob(0),
    printed(false), previous(previous), reference_count(0)
{
  if (previous) {
    hist::link(previous);
    cum_am_log_prob = previous->cum_am_log_prob;
    cum_lm_log_prob = previous->cum_lm_log_prob;
  }
}

TPLexPrefixTree::StateHistory::StateHistory(int hmm_model, int start_time,
                                            StateHistory *previous)
  : hmm_model(hmm_model),
    start_time(start_time),
    previous(previous),
    reference_count(0)
{
  if (previous)
    hist::link(previous);
}

#endif /* TPLEXPREFIXTREE_HH */
