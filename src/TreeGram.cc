#include "TreeGram.hh"

TreeGram::TreeGram()
  : m_type(BACKOFF),
    m_order(0)
{
  m_nodes.push_back(Node(0, -99, 0, -1));
}

void
TreeGram::reserve_nodes(int nodes)
{
  m_nodes.clear();
  m_nodes.reserve(nodes);
}

void
TreeGram::set_interpolation(const std::vector<float> &interpolation)
{
  m_interpolation = interpolation;
}

void
TreeGram::print_gram(FILE *file, const Gram &gram)
{
  for (int i = 0; i < gram.size(); i++) {
    fprintf(file, "%d ", gram[i]);
  }
  fputc('\n', file);
}

void
TreeGram::check_order(const Gram &gram)
{
  // Order must be the same or the next.
  if (gram.size() < m_last_gram.size() ||
      gram.size() > m_last_gram.size() + 1)
  {
    fprintf(stderr, "TreeGram::check_order(): "
	    "trying to insert %d-gram after %d-gram\n",
	    gram.size(), m_last_gram.size());
    print_gram(stderr, gram);
    exit(1);
  }

  // Unigrams must be in the correct places
  if (gram.size() == 1) {
    if (gram[0] != m_nodes.size()) {
      fprintf(stderr, "TreeGram::check_order(): "
	      "trying to insert 1-gram %d to node %d\n",
	      gram[0], m_nodes.size());
      exit(1);
    }
  }

  // With the same order, the grams must inserted in sorted order.
  if (gram.size() == m_last_gram.size()) {
    int i;
    for (i = 0; i < gram.size(); i++) {

      // Skipping grams
      if (gram[i] > m_last_gram[i])
	break;

      // Not in order
      if (gram[i] < m_last_gram[i]) {
	fprintf(stderr, "TreeGram::check_order(): "
		"gram not in sorted order\n");
	print_gram(stderr, gram);
	exit(1);
      }
    }

    // Duplicate?
    if (i == gram.size()) {
      fprintf(stderr, "TreeGram::check_order(): "
	      "duplicate gram\n");
      print_gram(stderr, gram);
      exit(1);
    }
  }
}

// Note that 'last' is not included in the range.
int
TreeGram::binary_search(int word, int first, int last)
{
  int middle;
  int half;

  int len = last - first;

  while (len > 5) { // FIXME: magic threshold to do linear search
    half = len / 2;
    middle = first + half;

    // Equal
    if (m_nodes[middle].word == word)
      return middle;

    // First half
    if (m_nodes[middle].word > word) {
      last = middle;
      len = last - first;
    }

    // Second half
    else {
      first = middle + 1;
      len = last - first;
    }
  }

  while (first < last) {
    if (m_nodes[first].word == word)
      return first;
    first++;
  }

  return -1;
}

// Returns unigram if node_index < 0
int
TreeGram::find_child(int word, int node_index)
{
  if (word < 0 || word >= m_words.size()) {
    fprintf(stderr, "TreeGram::find_child(): "
	    "index %d out of vocabulary size %d\n", word, m_words.size());
    exit(1);
  }

  if (node_index < 0)
    return word;

  assert(node_index < m_nodes.size() - 1);

  int first = m_nodes[node_index].child_index;
  int last = m_nodes[node_index + 1].child_index; // not included
  if (first < 0 || last < 0)
    return -1;

  return binary_search(word, first, last);
}

/// Finds the path to the current gram.
//
// PRECONDITIONS:
// - m_index_stack contains the indices of 'm_last_gram'
//
// POSTCONDITIONS:
// - m_index_stack contains the indices of 'gram' without the last word
void
TreeGram::find_path(const Gram &gram)
{
  int prev = -1;
  int order = 0;
  int index;

  assert(gram.size() > 1);

  // The beginning of the path can be found quickly by using the index
  // stack.
  while (1) {
    assert(order < m_last_gram.size() - 1);
    if (gram[order] != m_last_gram[order])
      break;
    order++;
  }
  m_index_stack.resize(order);

  // The rest of the path must be searched.
  order--;
  if (order < 0)
    prev = -1;
  else
    prev = m_index_stack[order];

  while (order < m_last_gram.size() - 1) {
    index = find_child(gram[order], prev);
    if (index < 0) {
      fprintf(stderr, "prefix not found\n");
      print_gram(stderr, gram);
      exit(1);
    }

    m_index_stack.push_back(index);

    prev == index;
    order++;
  }
}

void
TreeGram::add_gram(const Gram &gram, float log_prob, float back_off)
{
  assert(m_nodes.size() > 0);
  check_order(gram);

  // Update order counts
  if (gram.size() == m_last_gram.size() + 1) {
    m_order_count.push_back(0);
    m_order++;
  }
  assert(m_order_count.size() == gram.size());
  m_order_count[gram.size()]++;

  // Handle unigrams separately
  if (gram.size() == 1) {

    // OOV can be updated anytime.
    if (gram[0] == 0) {
      m_nodes[0].log_prob = log_prob;
      m_nodes[0].back_off = back_off;
    }

    // Unigram which is not OOV
    else
      m_nodes.push_back(Node(gram[0], log_prob, back_off, -1));
  }

  // 2-grams or higher
  else {
    // Update the path
    find_path(gram);

    if (m_nodes[m_index_stack.back()].child_index < 0)
      m_nodes[m_index_stack.back()].child_index = m_nodes.size();

    m_nodes.push_back(Node(gram.back(), log_prob, back_off, -1));
    m_nodes[m_index_stack.back() + 1].child_index = m_nodes.size();
  }

  m_last_gram = gram;
  assert(m_order == m_last_gram.size());
}
