#ifndef GRAMSORTER_HH
#define GRAMSORTER_HH

#include <deque>
#include <vector>
#include <algorithm>

class GramSorter {
public:
  typedef std::deque<int> Gram;

  // Structure for storing the probabilities of a single gram.
  struct Data {
    float log_prob;
    float back_off;
  };

  // Class for comparing grams.
  class Sorter {
  public:
    Sorter(const std::vector<int> &grams, int order) : 
      m_grams(grams), m_order(order) { }
    bool operator()(const int &a, const int &b) { 
      return GramSorter::lessthan(&m_grams[a * m_order], 
				  &m_grams[b * m_order],
				  m_order);
    }
  private:
    const std::vector<int> &m_grams;
    int m_order;
  };

  // Methods
  GramSorter(int order, int grams);
  void reset(int order, int grams);
  void add_gram(const Gram &gram, float log_prob, float back_off);
  void sort();

  int num_grams() { return m_indices.size(); }

  Gram gram(int index) { 
    Gram gram(m_order);
    int start = m_indices[index] * m_order;
    for (int i = 0; i < m_order; i++)
      gram[i] = m_grams[start + i];
    return gram;
  }
  Data data(int index) { return m_data.at(m_indices.at(index)); }

private:
  static bool lessthan(const int *i1, const int *i2, int order) {
    for (int i = 0; i < order; i++) {
      if (i1[i] < i2[i])
	return true;
      if (i1[i] > i2[i])
	return false;
    }
    return false;
  }

  int m_order;
  bool m_sorted; // are the nodes sorted already?

  // A vector containing all inserted grams.  For example, if the
  // structure contains 3-grams, the vector contains: 
  // a1 a2 a3 b1 b2 b3 c1 c2 c3
  std::vector<int> m_grams;

  // A vector containing the data (probabilities) for each gram.
  std::vector<Data> m_data;

  // A vector of indices of the grams.  Used for sorting.
  std::vector<int> m_indices;
};

#endif /* GRAMSORTER_HH */
