#ifndef TREEGRAM_HH
#define TREEGRAM_HH

#include <vector>
#include <deque>
#include <stdio.h>

#include "Vocabulary.hh"

class TreeGram : public Vocabulary {
public:
  typedef std::deque<int> Gram;

  struct Node {
    Node() : word(-1), log_prob(0), back_off(0), child_index(-1) {}
    Node(int word, float log_prob, float back_off, int child_index)
      : word(word), log_prob(log_prob), back_off(back_off), 
	child_index(child_index) {}
    int word;
    float log_prob;
    float back_off;
    int child_index;
  };

  class Iterator {
  public:
    Iterator(TreeGram *gram = NULL);
    void reset(TreeGram *gram);

    // Move to the next node in depth-first order
    bool next();

    // Move to the next node on the given order
    bool next_order(int order);

    // Return the node from the index stack. (default: the last one)
    const Node &node(int order = 0);

    // Order of the current node (1 ... n)
    int order() { return m_index_stack.size(); }

  private:
    TreeGram *m_gram;
    std::vector<int> m_index_stack;
  };

  enum Type { BACKOFF=0, INTERPOLATED=1 };

  TreeGram();
  void set_type(Type type) { m_type = type; }
  void reserve_nodes(int nodes); 
  void set_interpolation(const std::vector<float> &interpolation);

  /// Adds a new gram to the language model.
  // 
  // The grams must be inserted in sorted order.  The only exception
  // is the OOV 1-gram, which can be updated any time.  It exists by
  // default with very small log-prob and zero back-off.
  void add_gram(const Gram &gram, float log_prob, float back_off);
  void read(FILE *file);
  void write(FILE *file, bool reflip);

  float log_prob(const Gram &gram);
  int order() { return m_order; }
  int last_order() { return m_last_order; }
  int gram_count(int order) { return m_order_count.at(order-1); }

  /* Don't use this function, unles you really need to*/
  int find_child(int word, int node_index);
private:
  int binary_search(int word, int first, int last);
  void print_gram(FILE *file, const Gram &gram);
  void find_path(const Gram &gram);
  void check_order(const Gram &gram);
  void flip_endian();
  void fetch_gram(const Gram &gram, int first);

  Type m_type;
  int m_order;
  std::vector<int> m_order_count;	// number of grams in each order
  std::vector<float> m_interpolation;	// interpolation weights
  std::vector<Node> m_nodes;		// storage for the nodes
  std::vector<int> m_fetch_stack;	// indices of the gram requested
  int m_last_order;			// order of the last hit

  // For creating the model
  std::vector<int> m_insert_stack;	// indices of the last gram inserted
  Gram m_last_gram;			// the last ngram added to the model

};

#endif /* TREEGRAM_HH */
