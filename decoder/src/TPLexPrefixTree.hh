#ifndef TPLEXPREFIXTREE_HH
#define TPLEXPREFIXTREE_HH

#include <cstddef>  // NULL
#include <vector>
#include <cassert>
#include <cmath>

#include "config.hh"
#include "HashCache.hh"
#include "SimpleHashCache.hh"
#include "LMHistory.hh"

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
#define NODE_DEBUG_PRUNED         0x4000
#define NODE_DEBUG_PRINTED        0x8000


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
/// Cross-word triphone contexts are handled by building a separate network, to
/// which the lexical prefix tree is linked. The left context of the first
/// triphone of a word, and the right context of the last triphone of a word, is
/// silence, so the cross word network is linked to the second triphone and the
/// second to last triphone of every word. The part of the cross word network
/// linked to the beginning (second triphone) of a word is called fan-in. The
/// part where the end (second to last triphone) of a word is linked to, is
/// called fan-out.
///
/// Every triphone is defined in the acoustic models, and they are not tied at
/// the triphone level. Instead, each triphone has a set of HMM states
/// (currently three states in a left-to-right topology), and these states are
/// shared among all triphones. The state tying has been performed using a
/// decision tree.
///
class TPLexPrefixTree {
public:
  class Node;

  typedef std::vector<Node *> node_vector;
  typedef std::map<std::string, node_vector> string_to_nodes_map;

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

    Token():
      node(NULL),
      next_node_token(NULL),
      am_log_prob(0.0f),
      lm_log_prob(0.0f),
      cur_am_log_prob(0.0f),
      cur_lm_log_prob(0.0f),
      total_log_prob(0.0f),
      lm_history(NULL),
      lm_hist_code(0),
      fsa_lm_node(0),
      recent_word_graph_node(0),
      word_history(NULL),
      word_start_frame(0),
      word_count(0),
      state_history(NULL),
      depth(0),
      dur(0)
    { }
  };


  class Arc {
  public:
    float log_prob;
    Node *next;

    Arc():
      log_prob(0.0f),
      next(NULL)
    {}
  };

  class Node {
  public:
    inline Node() : word_id(-1), node_id(0), state(NULL), token_list(NULL), flags(NODE_NORMAL) { }
    inline Node(int wid) : word_id(wid), state(NULL), token_list(NULL), flags(NODE_NORMAL) { }
    inline Node(int wid, HmmState *s) : word_id(wid), state(s), token_list(NULL), flags(NODE_NORMAL) { }
    int word_id; // -1 for nodes without word identity.
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
    NodeArcId() :
      node(NULL),
      arc_index(0)
    {}
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

  /// \brief Sets the language model scale factor, to be used
  /// for scaling the pronunciation log probabilities.
  ///
  void set_lm_scale(double lm_scale);

  void set_cross_word_triphones(bool cw_triphones) { m_cross_word_triphones = cw_triphones; }
  void set_silence_is_word(bool b) { m_silence_is_word = b; }
  void set_ignore_case(bool b) { m_ignore_case = b; }

  void initialize_lex_tree(void);

  /// \brief Adds a word to the lexical prefix tree.
  ///
  /// The tree is constructed by adding words to the tree one at a time.
  /// Currently the construction procedure assumes triphone models.
  ///
  /// The construction algorithm starts from the (dummy) root node, and find the
  /// path in the tree which is common to the given state sequence. When a node
  /// is reached from which the path can not be continued, the rest of the
  /// states are inserted as tree nodes starting from the last common node.
  ///
  /// For the first phoneme of a word, the left context is silence, so in
  /// decoding the root node is accessed only after silence. Other contexts are
  /// handled by fan-in triphones, which are linked to the second phonemes of
  /// the words.
  ///
  /// The states are added up to the second last phoneme of the word. From
  /// there an arc is made to a dummy node containing the word identity of the
  /// new word. This node is then linked to proper fan-out triphones, which are
  /// created on demand.
  ///
  void add_word(std::vector<Hmm*> &hmm_list, int word_id, double prob);

  void finish_tree(void);
  
  void prune_lookahead_buffers(int min_delta, int max_depth);
  void set_lm_lookahead_cache_sizes(int cache_size);

  void set_word_boundary_id(int id) { m_word_boundary_id = id; }
  void set_optional_short_silence(bool state) { m_optional_short_silence = state; }
  void set_sentence_boundary(int sentence_start_id, int sentence_end_id);

  void clear_node_token_lists(void);

  void print_node_info(int node, const Vocabulary &voc);
  void print_lookahead_info(int node, const Vocabulary &voc);
  void debug_prune_dead_ends(Node *node);
  void debug_add_silence_loop();
  
  
private:
  /// \brief Creates a transition from source node, if it doesn't exist already.
  ///
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
  /// The construction of the search network starts by creating the fan-in
  /// triphones. This means that the HMM state sequences of every triphone are
  /// inserted into the search network.
  ///
  /// Assumes triphone models. Labels must be of form a-b+c.
  ///
  void create_cross_word_network(void);

  /// \brief Adds the HMM state sequences of a triphone into the search network.
  ///
  /// The triphones are organized by their central phoneme and the left context,
  /// so that if these are equal, the fan-in triphones are allowed to share
  /// their first nodes. The last nodes, however, are grouped differently, now
  /// according to the central phoneme and the right context. This way the
  /// number of arcs are minimized when linking other nodes to or from the
  /// fan-in triphones.
  ///
  /// Only works with strict left to right HMMs with one source and sink state.
  ///
  void add_hmm_to_fan_network(int hmm_id,
                              bool fan_out);

  void link_fan_out_node_to_fan_in(Node *node, const std::string &key);

  /// \brief If \a fan_out is true, creates an arc to every fan-out entry
  /// node in the group specified by \a key, creating the entry nodes if
  /// necessary. Otherwise creates an arc to every fan-in entry node.
  ///
  /// The entry triphones are organized so that triphones belonging to
  /// the same phoneme and having the same left context are grouped
  /// together and are allowed to share their common states.
  ///
  void link_node_to_fan_network(const std::string &key,
                                std::vector<Arc> &source_arcs,
                                bool fan_out,
                                bool ignore_length,
                                float out_transition_log_prob);

  /// \brief Create another instance of a null node with word identity after the
  /// fan-in network, and link it back to fan-in.
  ///
  /// When linking a word to the cross word network, single phoneme words must
  /// be handled separately. Another implementation of the word has to be added
  /// inside the cross-word network. This is done by adding a dummy node with
  /// the word identity after every fan-in triphone whose central phoneme
  /// corresponds to the given word. This dummy node is then linked back to the
  /// fan-in triphones, determined by the right context of the originating
  /// fan-in triphone.
  ///
  void add_single_hmm_word_for_cross_word_modeling(Hmm *hmm, int word_id, double prob);

  /// \brief Creates arcs from the last states of the fan-in triphones to
  /// the second triphone of each word.
  ///
  /// At this point, cross word network is ready and words have been linked
  /// to the fan-out layer. Also single HMM words have been linked back
  /// to the fan-in layer. What is left is to link the last states of
  /// the fan-in layer back to the beginning of the lexical prefix tree.
  ///
  void link_fan_in_nodes();

  /// \brief Creates arcs from a final fan-in node in the cross word network, to
  /// the second triphones of each word.
  ///
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

  /// \brief Returns a node for the last HMM state of a fan-in triphone.
  ///
  /// If the node doesn't exist, creates it, and saves in m_fan_in_last_nodes.
  ///
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

  /// \brief Marks a node as a connection point for linking back to the
  /// beginning (second triphone) of a word from the cross word network.
  ///
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
  double m_lm_scale;
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

TPLexPrefixTree::WordHistory::WordHistory(int word_id, int end_frame, 
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
