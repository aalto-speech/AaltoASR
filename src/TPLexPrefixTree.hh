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
    unsigned char printed;
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

    float avg_ac_log_prob;

#ifdef PRUNING_MEASUREMENT
    float meas[6];
#endif

    int word_count;
    
    //PathHistory *token_path;
    unsigned char depth;
    unsigned char dur;

    unsigned char mode; // 0 = root, 1 = normal, 2 = after word_id
  };

  class Arc {
  public:
    float log_prob;
    Node *next;
  };

  class Node {
  public:
    inline Node() : word_id(-1), state(NULL), token_list(NULL) { }
    inline Node(int wid) : word_id(wid), state(NULL), token_list(NULL) { }
    inline Node(int wid, HmmState *s) : word_id(wid), state(s), token_list(NULL) { }
    int word_id; // -1 if none
    int node_id;
    HmmState *state;
    Token *token_list;
    std::vector<Arc> arcs;

    std::vector<int> possible_word_id_list;
    SimpleHashCache<float> lm_lookahead_buffer;
  };

  TPLexPrefixTree();
  TPLexPrefixTree(const TPLexPrefixTree &lexicon) { assert(false); } // never copy lexicon
  inline TPLexPrefixTree::Node *root() { return m_root_node; }
  inline int words() const { return m_words; }

  void set_verbose(int verbose) { m_verbose = verbose; }
  void set_lm_lookahead(int lm_lookahead) { m_lm_lookahead = lm_lookahead; }

  void add_word(std::vector<Hmm*> &hmm_list, int word_id);
  void finish_tree(void);
  void prune_lookahead_buffers(int min_delta, int max_depth);
  void set_lm_lookahead_cache_sizes(int root_size);

  void clear_node_token_lists(void);

  void print_node_info(int node);
  void print_lookahead_info(int node, const Vocabulary &voc);
  
private:
  void expand_lexical_tree(Node *source, Hmm *hmm, HmmTransition &t,
                           float cur_trans_log_prob,
                           int word_end,
                           std::vector<Node*> &hmm_state_nodes,
                           std::vector<Node*> &sink_nodes,
                           std::vector<float> &sink_trans_log_probs);
  void post_process_lex_branch(Node *node, std::vector<int> *lm_la_list);
  void prune_lm_la_buffer(int delta_thr, int depth_thr,
                          Node *node, int last_size, int cur_depth);
  void set_node_lm_lookahead_cache_size(Node *node, int cache_size,
                                        int last_branches);
  
private:
  int m_words; // Largest word_id in the nodes plus one
  Node *m_root_node;
  Node *m_end_node;
  std::vector<Node*> word_end_list;
  std::vector<Node*> node_list;
  int m_verbose;
  int m_lm_lookahead; // 0=None, 1=Only in first subtree nodes,
                      // 2=Full
  int m_lm_buf_count;
};

//////////////////////////////////////////////////////////////////////

TPLexPrefixTree::WordHistory::WordHistory(int word_id, int lm_id,
                                          class WordHistory *prev)
  : word_id(word_id),
    lm_id(lm_id),
    prev_word(prev),
    printed(0),
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

#endif /* TPLEXPREFIXTREE_HH */
