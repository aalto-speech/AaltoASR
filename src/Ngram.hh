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
    short word; 
    float log_prob;
    int first;
    int last;

    inline bool operator<(int value) const { return word < value; }
    inline bool operator>(int value) const { return word > value; }
    inline bool operator==(int value) const { return word == value; }
  };

  inline Node *unigram(int word) { return &m_nodes[word]; }

private:
  std::vector<Node> m_nodes;
};

#endif /* NGRAM_HH */
