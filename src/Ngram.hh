#ifndef NGRAM_HH
#define NGRAM_HH

#include <algorithm>
#include <vector>

template <class Vector, class Value>
inline int
find(Vector &nodes, const Value &value, int first, int last)
{
  int middle;
  int half;

  int len = last - first;
  while (len > 1) {
    half = len / 2;
    middle = first + half;

    // Equal
    if (nodes[middle] == value)
      return middle;

    // First half
    if (nodes[middle] > value) {
      last = middle;
      len = last - first;
    }

    // Second half
    else {
      first = middle + 1;
      len = last - first;
    }
  }
  
  return -1;
}

class Ngram {
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

  inline Node *node(int word, Node *node = NULL)
    {
      if (node == NULL)
	return &m_nodes[word];

      if (node->first < 0)
	return NULL;

      int last = (node + 1)->first - 1;
      int index = find(m_nodes, word, node->first, last);
      if (index < 0)
	return NULL;
      return &m_nodes[index];
    }

private:
  std::vector<Node> m_nodes;
};

#endif /* NGRAM_HH */
