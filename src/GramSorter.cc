#include <stdio.h>
#include "GramSorter.hh"

GramSorter::GramSorter(int order, int grams)
  : m_sorted(true)
{
  reset(order, grams);
}

void
GramSorter::reset(int order, int grams)
{
  m_order = order;
  m_sorted = true;

  m_grams.clear();
  m_data.clear();
  m_indices.clear();

  if (grams > 0) {
    fprintf(stderr, "GramSorter: reserving %d grams for order %d...",
	    grams, order);
    m_grams.reserve(grams * order);
    m_data.reserve(grams);
    m_indices.reserve(grams);
  }

  fprintf(stderr, "done\n");
}

void 
GramSorter::add_gram(const Gram &gram, float log_prob, float back_off)
{
  // Check the length of the gram
  if (gram.size() != m_order) {
    fprintf(stderr, "GramSorter: got %d-gram while expecting %d-grams\n",
	    gram.size(), m_order);
    exit(1);
  }

  // Add gram to the node vector.
  assert(gram.size() == m_order);
  m_indices.push_back(m_indices.size());
  for (int i = 0; i < gram.size(); i++)
    m_grams.push_back(gram[i]);

  m_data.push_back(Data());
  m_data.back().log_prob = log_prob;
  m_data.back().back_off = back_off;
  assert(m_indices.size() * m_order == m_grams.size());

  // Check if the grams have been inserted in sorted order so far.
  if (m_indices.size() > 1 && m_sorted) {
    int i1 = (m_indices.size() - 1) * m_order;
    int i2 = (m_indices.size() - 2) * m_order;
    if (lessthan(&m_grams[i2], &m_grams[i1], m_order)) {
      fprintf(stderr, "GramSorter: %d-grams not sorted, sorting soon\n", 
	      m_order);
      m_sorted = false;
    }
  }
}

void
GramSorter::sort()
{
  if (!m_sorted) {
    fprintf(stderr, "GramSorter: sorting %d-grams now\n", m_order);
    std::sort(m_indices.begin(), m_indices.end(), Sorter(m_grams, m_order));
    fprintf(stderr, "GramSorter: sorted %d-grams\n", m_order);
  }
  else
    fprintf(stderr, "GramSorter: %d-grams were sorted already\n", m_order);
}


