#ifndef TREEGRAM_HH
#define TREEGRAM_HH

#include <vector>
#include <deque>
#include <stdio.h>

#include "Vocabulary.hh"

class TreeGram : private Vocabulary {
public:
  typedef std::deque<int> Gram;

  struct Node {
    Node(int word, float log_prob, float back_off, int child_index);
    int word;
    float log_prob;
    float back_off;
    int child_index;
  };

  enum Type { BACKOFF, INTERPOLATED };

  TreeGram();
  void set_type(Type type) { m_type = type; }
  void reserve_nodes(int nodes); 
  void set_interpolation(const std::vector<float> &interpolation);

  /// Adds a new gram to the language model.
  // 
  // The grams must be inserted in sorted order.  The only exception
  // is the OOV 1-gram, which can be updated any time, and which
  // exists by default with very small log-prob and zero back-off.
  void add_gram(const Gram &gram, float log_prob, float back_off);

private:
  int binary_search(int word, int first, int last);
  void print_gram(FILE *file, const Gram &gram);
  int find_child(int word, int node_index);
  void find_path(const Gram &gram);
  void check_order(const Gram &gram);

  // Data for storing the model
  Type m_type;
  int m_order;
  std::vector<float> m_interpolation;
  std::vector<int> m_order_count;
  std::vector<Node> m_nodes;

  // Data for creating the model
  std::vector<int> m_index_stack;
  Gram m_last_gram;
};

#endif /* TREEGRAM_HH */
