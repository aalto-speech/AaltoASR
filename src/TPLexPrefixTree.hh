#ifndef TPLEXPREFIXTREE_HH
#define TPLEXPREFIXTREE_HH

#include <vector>
#include <assert.h>

#include "HashCache.hh"
#include "SimpleHashCache.hh"

#include "Hmm.hh"
#include "Vocabulary.hh"

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


class TPLexPrefixTree {
public:

  class Node;

  class WordHistory {
  public:
    inline WordHistory(int word_id, int lm_id, class WordHistory *prev);
    inline void link() { m_reference_count++; }
    inline static void unlink(class WordHistory *hist);
    inline int get_num_references(void) { return m_reference_count; }

    int word_id;
    int lm_id; // Word ID in LM
    WordHistory *prev_word;
    bool printed; // TokenPassSearch::print* functions may set this true
    int word_start_frame;
  private:
    int m_reference_count;
  };

  class StateHistory {
  public:
    inline StateHistory(int hmm_model, int start_time,
                        class StateHistory *prev);

    inline void link() { m_reference_count++; }
    inline static void unlink(class StateHistory *hist);
    
    int hmm_model;
    int start_time;
    StateHistory *prev;
  private:
    int m_reference_count;
  };

  class PathHistory {
  public:
    inline PathHistory(float ll, float dll, int depth, class PathHistory *p);

    inline void link() { m_reference_count++; }
    inline static void unlink(class PathHistory *hist);
    
    float ll;
    float dll;
    int depth;
    PathHistory *prev;
  private:
    int m_reference_count;
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
    WordHistory *prev_word;
    int word_hist_code; // Hash code for word history (up to LM order)
    int word_start_frame;

#ifdef PRUNING_MEASUREMENT
    float meas[6];
#endif

    int word_count;
    StateHistory *state_history;
    
    //PathHistory *token_path;
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
  inline TPLexPrefixTree::Node *root() { return m_root_node; }
  inline TPLexPrefixTree::Node *start_node() { return m_start_node; }
  inline int words() const { return m_words; }

  void set_verbose(int verbose) { m_verbose = verbose; }
  void set_lm_lookahead(int lm_lookahead) { m_lm_lookahead = lm_lookahead; }
  void set_cross_word_triphones(bool cw_triphones) { m_cross_word_triphones = cw_triphones; }
  void set_silence_is_word(bool b) { m_silence_is_word = b; }

  void initialize_lex_tree(void);
  void add_word(std::vector<Hmm*> &hmm_list, int word_id);
  void finish_tree(void);
  
  void prune_lookahead_buffers(int min_delta, int max_depth);
  void set_lm_lookahead_cache_sizes(int cache_size);

  void set_word_boundary_id(int id) { m_word_boundary_id = id; }
  void set_optional_short_silence(bool state) { m_optional_short_silence = state; }
  void set_sentence_boundary(int sentence_end_id);

  void clear_node_token_lists(void);

  void print_node_info(int node);
  void print_lookahead_info(int node, const Vocabulary &voc);
  
private:
  void expand_lexical_tree(Node *source, Hmm *hmm, HmmTransition &t,
                           float cur_trans_log_prob,
                           int word_end,
                           std::vector<Node*> &hmm_state_nodes,
                           std::vector<Node*> &sink_nodes,
                           std::vector<float> &sink_trans_log_probs,
                           unsigned short flags);
  void post_process_lex_branch(Node *node, std::vector<int> *lm_la_list);
  bool post_process_fan_triphone(Node *node, std::vector<int> *lm_la_list,
                                 bool fan_in);

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
  Node* get_fan_state_node(HmmState *state, std::vector<Node*> *nodes);
  std::vector<Node*>* get_fan_node_list(
    const std::string &key,
    std::map< std::string, std::vector<Node*>* > &nmap);
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
  std::vector<Node*> node_list;
  int m_verbose;
  int m_lm_lookahead; // 0=None, 1=Only in first subtree nodes,
                      // 2=Full
  bool m_cross_word_triphones;
  int m_lm_buf_count;

  bool m_silence_is_word;

  bool m_optional_short_silence;
  HmmState *m_short_silence_state;
  int m_word_boundary_id;

  std::map<std::string,int> &m_hmm_map;
  std::vector<Hmm> &m_hmms;

  std::map< std::string, std::vector<Node*>* > m_fan_out_entry_nodes;
  std::map< std::string, std::vector<Node*>* > m_fan_out_last_nodes;
  std::map< std::string, std::vector<Node*>* > m_fan_in_entry_nodes;
  std::map< std::string, std::vector<Node*>* > m_fan_in_last_nodes;
  std::map< std::string, std::vector<Node*>* > m_fan_in_connection_nodes;
  std::vector<NodeArcId> m_silence_arcs;
};

//////////////////////////////////////////////////////////////////////

TPLexPrefixTree::WordHistory::WordHistory(int word_id, int lm_id,
                                          class WordHistory *prev)
  : word_id(word_id),
    lm_id(lm_id),
    prev_word(prev),
    printed(false),
    word_start_frame(0),
    m_reference_count(0)
{
  if (prev)
    prev->link();
}

void
TPLexPrefixTree::WordHistory::unlink(class WordHistory *hist)
{
  if (hist != NULL)
  {
    while (hist->m_reference_count == 1) {
      WordHistory *prev = hist->prev_word;
      delete hist;
      hist = prev;
      if (hist == NULL)
        return;
    }
    hist->m_reference_count--;
  }
}


TPLexPrefixTree::PathHistory::PathHistory(float ll, float dll, int depth,
                                          class PathHistory *p)
  : ll(ll),
    dll(dll),
    depth(depth),
    prev(p),
    m_reference_count(0)
{
  if (p)
    p->link();
}

void
TPLexPrefixTree::PathHistory::unlink(class PathHistory *hist)
{
  if (hist != NULL)
  {
    while (hist->m_reference_count == 1) {
      PathHistory *prev = hist->prev;
      delete hist;
      hist = prev;
      if (hist == NULL)
        return;
    }
    hist->m_reference_count--;
  }
}

TPLexPrefixTree::StateHistory::StateHistory(int hmm_model, int start_time,
                                            class StateHistory *prev)
  : hmm_model(hmm_model),
    start_time(start_time),
    prev(prev),
    m_reference_count(0)
{
  if (prev)
    prev->link();
}

void
TPLexPrefixTree::StateHistory::unlink(class StateHistory *hist)
{
  if (hist != NULL)
  {
    while (hist->m_reference_count == 1) {
      StateHistory *prev = hist->prev;
      delete hist;
      hist = prev;
      if (hist == NULL)
        return;
    }
    hist->m_reference_count--;
  }
}

#endif /* TPLEXPREFIXTREE_HH */
