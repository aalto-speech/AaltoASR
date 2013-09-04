#ifndef HTKLATTICEGRAMMAR_HH
#define HTKLATTICEGRAMMAR_HH

#include <stdio.h>
#include "NGram.hh"
#include <set>

#define IMPOSSIBLE_LOGPROB -10000

#define IMPOSSIBLE_LOGPROB -10000

/* We inherit the interface to NGram for simplicity. Internally,
   this class has nothing to do with n-grams.
*/

class HTKLatticeGrammar: public NGram {
public:
  HTKLatticeGrammar() : m_lineno(0) {
    m_type = HTK_LATTICE_GRAMMAR;
    m_order = 999999; // Always keep the full history
    
  }

  void read(FILE *file, bool binary=false);
  void write(FILE *file, bool binary=false);
  
  // Wrappers
  float log_prob_bo(const std::vector<int> &gram){
    Gram g;
    for (int i=0; i<gram.size();i++) 
      g.push_back(gram[i]);
    if (match_begin(g)) return 0;
    return IMPOSSIBLE_LOGPROB;
  };

  inline float log_prob_i(const std::vector<int> &gram) {assert(false);return 0;};

  float log_prob_bo(const std::vector<unsigned short> &gram){
    Gram g;
    for (int i=0; i<gram.size();i++) 
      g.push_back(gram[i]);
    if (match_begin(g)) return 0;
    return IMPOSSIBLE_LOGPROB;
  };

  float log_prob_i(const std::vector<unsigned short> &gram){assert(false);return 0;};

  float log_prob_bo(const Gram &gram){
    if (match_begin(gram)) return 0;
    return IMPOSSIBLE_LOGPROB;
  }; 
  float log_prob_i(const Gram &gram){assert(false);return 0;};

  bool match_begin(const std::string &string_in);

  /* These are for lookaheads, not implemented */
  void fetch_bigram_list(int prev_word_id, 
			 std::vector<float> &result_buffer);
  void fetch_trigram_list(int w1, int w2, 
			  std::vector<float> &result_buffer) {
    assert(false);
  }

private:
  void read_error(std::string &);
  int m_lineno;

  bool read_clean_line(std::string &str, FILE *file, bool b);

  struct Node {
    std::vector<int> arcs_out;
  };

  struct Arc {
    Arc() : source(-1), target(-1), widx(-1) {}
    int source;
    int target;
    int widx;
  };

  std::vector<Node> m_nodes;
  std::vector<Arc> m_arcs;

  int m_start_node_idx, m_end_node_idx, m_null_idx;
  
  bool match_begin(const Gram &g);

  // This table is for LM lookahead and lists all allowable bigrams
  std::vector<std::set<int> > m_bigram_idxlist;
  void pregenerate_bigram_idxlist();
  void follow_and_insert_bigram(Arc *, int, std::set<int> &);
};
#endif
