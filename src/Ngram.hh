#ifndef NGRAM_HH
#define NGRAM_HH

#include <algorithm>
#include <vector>

#include "Vocabulary.hh"

// Binary search
template <class Vector, class Value>
inline int
find(Vector &nodes, const Value &value, int first, int end)
{
  int middle;
  int half;

  int len = end - first;

  while (len > 3) { // FIXME: magic value
    half = len / 2;
    middle = first + half;

    // Equal
    if (nodes[middle] == value)
      return middle;

    // First half
    if (nodes[middle] > value) {
      end = middle;
      len = end - first;
    }

    // Second half
    else {
      first = middle + 1;
      len = end - first;
    }
  }

  while (first < end) {
    if (nodes[first] == value)
      return first;
    first++;
  }

  return -1;
}

class Ngram : private Vocabulary {
public:
  friend class ArpaNgramReader;

  class Node {
  public:
    unsigned short word; 
    float log_prob;
    float back_off;
    int first;
    
    inline Node() : word(0), log_prob(0), back_off(0), first(-1) { }
    inline Node(unsigned short word, float log_prob, float back_off)
      : word(word), log_prob(log_prob), back_off(back_off), first(-1) { }
    inline bool operator<(int value) const { return word < value; }
    inline bool operator>(int value) const { return word > value; }
    inline bool operator==(int value) const { return word == value; }
  };

  template<class RandomAccessIterator>
  float log_prob(RandomAccessIterator begin, RandomAccessIterator end) const;

  inline Ngram() : m_order(0) { }
  inline int order() const { return m_order; }
  inline int size() const { return Vocabulary::size(); }
  inline const std::string &word(unsigned int index) const
    { return Vocabulary::word(index); }
  inline Node *node(int index) { return &m_nodes[index]; }
  inline const Node *node(int index) const { return &m_nodes[index]; }
  inline int nodes() const { return m_nodes.size(); }
  inline int index(const std::string &word) const 
    { return Vocabulary::index(word); }
  inline const Node *child(int word, const Node *node = NULL) const
    {
      if (node == NULL)
	return &m_nodes[word];

      if (node->first < 0)
	return NULL;

      int end = (node + 1)->first;
      if (end < 0)
	end = m_nodes.size();

      int index = find(m_nodes, word, node->first, end);
      if (index < 0)
	return NULL;
      return &m_nodes[index];
    }

private:
  int m_order;
  std::vector<Node> m_nodes;
};

template<class RandomAccessIterator>
float
Ngram::log_prob(RandomAccessIterator begin, RandomAccessIterator end) const
{
  RandomAccessIterator it;
  double log_prob = 0;

  for (; begin != end; begin++) {
    const Node *node = this->node(*begin);
    assert(node != NULL); // We must have a unigram for each word.

    for (it = begin+1; it != end; it++) {
      const Node *next = this->child(*it, node);
      if (!next)
	break;
      node = next;
    }
    
    // Full ngram found
    if (it == end) {
      log_prob += node->log_prob;
      return log_prob;
    }

    // Backoff found
    if (it + 1 == end)
      log_prob += node->back_off;
  }

  assert(false);
  return 0;
}

#endif /* NGRAM_HH */
