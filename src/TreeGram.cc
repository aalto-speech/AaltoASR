#include <errno.h>
#include <assert.h>

#include "Endian.hh"
#include "TreeGram.hh"
#include "tools.hh"

static std::string format_str("cis-binlm2\n");

TreeGram::TreeGram()
  : m_type(BACKOFF),
    m_order(0),
    m_last_order(0)
{
}

void
TreeGram::reserve_nodes(int nodes)
{
  m_nodes.clear();
  m_nodes.reserve(nodes);
  m_nodes.push_back(Node(0, -99, 0, -1));
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
  // UNK can be updated anytime
  if (gram.size() == 1 && gram[0] == 0)
    return;

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
// - m_insert_stack contains the indices of 'm_last_gram'
//
// POSTCONDITIONS:
// - m_insert_stack contains the indices of 'gram' without the last word
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
  m_insert_stack.resize(order);

  // The rest of the path must be searched.
  order--;
  if (order < 0)
    prev = -1;
  else
    prev = m_insert_stack[order];

  while (order < m_last_gram.size() - 1) {
    index = find_child(gram[order], prev);
    if (index < 0) {
      fprintf(stderr, "prefix not found\n");
      print_gram(stderr, gram);
      exit(1);
    }

    m_insert_stack.push_back(index);

    prev = index;
    order++;
  }
}

void
TreeGram::add_gram(const Gram &gram, float log_prob, float back_off)
{
  if (m_nodes.empty()) {
    fprintf(stderr, "TreeGram::add_gram(): "
	    "nodes must be reserved before calling this function\n");
    exit(1);
  }

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

    if (m_nodes[m_insert_stack.back()].child_index < 0)
      m_nodes[m_insert_stack.back()].child_index = m_nodes.size();

    m_nodes.push_back(Node(gram.back(), log_prob, back_off, -1));
    m_nodes[m_insert_stack.back() + 1].child_index = m_nodes.size();
  }

  m_last_gram = gram;
  assert(m_order == m_last_gram.size());
}

void 
TreeGram::write(FILE *file, bool reflip) 
{
  fputs(format_str.c_str(), file);

  // Write type and interpolation weights
  if (m_type == BACKOFF)
    fputs("backoff\n", file);
  else if (m_type == INTERPOLATED)
    fputs("interpolated\n", file);

  fprintf(file, "%d\n", num_words());
  for (int i = 0; i < num_words(); i++)
    fprintf(file, "%s\n", word(i).c_str());
  fprintf(file, "%d %d\n", m_order, m_nodes.size());

  // Use correct endianity
  if (Endian::big) 
    flip_endian(); 

  // Write possible interpolation weights
  if (m_type == INTERPOLATED) {
    assert(m_interpolation.size() == m_order);
    fwrite(&m_interpolation[0], m_order * sizeof(m_interpolation[0]), 1, file);
  }

  // Write nodes
  fwrite(&m_nodes[0], m_nodes.size() * sizeof(TreeGram::Node), 1, file);
  
  if (ferror(file)) {
    fprintf(stderr, "TreeGram::write(): write error: %s\n", strerror(errno));
    exit(1);
  }

  // Restore to original endianess
  if (Endian::big && reflip)
    flip_endian();
}

void 
TreeGram::read(FILE *file) 
{
  std::string line;
  int words;
  bool ret;

  // Read the header
  ret = read_string(&line, format_str.length(), file);
  if (!ret || line != format_str) {
    fprintf(stderr, "TreeGram::read(): invalid file format\n");
    exit(1);
  }
  
  // Read LM type
  read_line(&line, file);
  chomp(&line);
  if (line == "backoff")
    m_type = BACKOFF;
  else if (line == "interpolated")
    m_type = INTERPOLATED;
  else {
    fprintf(stderr, "TreeGram::read(): invalid type: %s\n", line.c_str());
    exit(1);
  }

  // Read the number of words
  if (!read_line(&line, file)) {
    fprintf(stderr, "TreeGram::read(): unexpected end of file\n");
    exit(1);
  }
  words = atoi(line.c_str());
  if (words < 1) {
    fprintf(stderr, "TreeGram::read(): invalid number of words: %s\n", 
	    line.c_str());
    exit(1);
  }
  
  // Read the vocabulary
  for (int i=0; i < words; i++) {
    if (!read_line(&line, file)) {
      fprintf(stderr, "TreeGram::read(): "
	      "read error while reading vocabulary\n");
      exit(1);
    }
    chomp(&line);
    add_word(line);
  }

  // Read the order and the number of nodes
  int m_nodes_size;
  fscanf(file, "%d %d\n", &m_order, &m_nodes_size);
  reserve_nodes(m_nodes_size);

  // Read the possible interpolation weights
  if (m_type == INTERPOLATED) {
    m_interpolation.resize(m_order);
    fread(&m_interpolation[0], sizeof(float) * m_order, 1, file);
  }

  // Read the nodes
  size_t block_size = m_nodes.size() * sizeof(TreeGram::Node);
  size_t blocks_read = fread(&m_nodes[0], block_size, 1, file);
  if (blocks_read != 1) {
      fprintf(stderr, "TreeGram::read(): "
	      "read error while reading ngrams\n");
      exit(1);
  }

  if (Endian::big) 
    flip_endian();
}

void 
TreeGram::flip_endian() 
{
  assert(sizeof(m_nodes[0].word == 4));
  assert(sizeof(m_nodes[0].log_prob == 4));
  assert(sizeof(m_nodes[0].back_off == 4));
  assert(sizeof(m_nodes[0].child_index == 4));

  if (m_type == INTERPOLATED) {
    assert(m_interpolation.size() == m_order);
    Endian::convert(&m_interpolation[0], 4 * m_order);
  }

  for (int i = 0; i < m_nodes.size(); i++) {
    Endian::convert(&m_nodes[i].word, 4);
    Endian::convert(&m_nodes[i].log_prob, 4);
    Endian::convert(&m_nodes[i].back_off, 4);
    Endian::convert(&m_nodes[i].child_index, 4);
  }
}

// Fetch the node indices of the requested gram to m_fetch_stack as
// far as found in the tree structure.
void
TreeGram::fetch_gram(const Gram &gram, int first)
{
  int prev = -1;
  m_fetch_stack.clear();
  
  while (m_fetch_stack.size() < gram.size()) {
    int node = find_child(gram[first], prev);
    if (node < 0)
      break;
    m_fetch_stack.push_back(node);
    first++;
    node = prev;
  }
}

float
TreeGram::log_prob(const Gram &gram)
{
  assert(gram.size() > 0);

  float log_prob = 0;
  
  // Denote by (w(1) w(2) ... w(N)) the ngram that was requested.  The
  // log-probability of the back-off model is computed as follows:

  // Iterate n = 1..N:
  // - If (w(n) ... w(N)) not found, add the possible (w(n) ... w(N-1) backoff
  // - Otherwise, add the log-prob and return.
  int n = 0;
  while (1) {
    assert(n < gram.size());
    fetch_gram(gram, n);
    assert(m_fetch_stack.size() > 0);

    // Full gram found?
    if (m_fetch_stack.size() == gram.size()) {
      log_prob += m_nodes[m_fetch_stack.back()].log_prob;
      m_last_order = gram.size() - n;
      break;
    }
    
    // Back-off found?
    if (m_fetch_stack.size() == gram.size() - 1)
      log_prob += m_nodes[m_fetch_stack.back()].back_off;

    n++;
  }

  return log_prob;
}

TreeGram::Iterator::Iterator(TreeGram *gram)
  : m_gram(gram)
{
  if (gram)
    reset(gram);
}

void
TreeGram::Iterator::reset(TreeGram *gram)
{
  assert(gram);
  m_gram = gram;
  m_index_stack.clear();
  m_index_stack.reserve(gram->m_order);
}

bool
TreeGram::Iterator::next()
{
  bool backtrack = false;

  // Start the search
  if (m_index_stack.empty()) {
    m_index_stack.push_back(0);
    return true;
  }

  // Go to the next node.  Backtrack if necessary.
  while (1) {
    assert(!m_index_stack.empty());
    int index = m_index_stack.back();
    TreeGram::Node *node = &m_gram->m_nodes[index];

    // End of the structure?
    if (index == m_gram->m_nodes.size() - 1)
      return false;

    // If not backtracking, try diving deeper
    if (!backtrack) {
      // Do we have children?
      if (node->child_index > 0 && (node+1)->child_index > 0) {
	m_index_stack.push_back(node->child_index);
	return true;
      }
    }
    backtrack = false;

    // No children, try siblings 
    if (m_index_stack.size() == 1) {
      // Unigram level: we have always siblings
      m_index_stack.back()++;
      return true;
    }
    else {
      // Higher order
      m_index_stack.pop_back();
      TreeGram::Node *parent = &m_gram->m_nodes[m_index_stack.back()];

      // Do we have more siblings?
      index++;
      if (index < (parent+1)->child_index) {
	m_index_stack.push_back(index);
	return true;
      }

      // No more siblings, backtrack.
      backtrack = true;
    }
  }
}

bool
TreeGram::Iterator::next_order(int order)
{
  if (order < 1 || order > m_gram->m_order) {
    fprintf(stderr, "TreeGram::Iterator::next_order(): invalid order %d\n", 
	    order);
    exit(1);
  }

  while (1) {
    if (!next())
      return false;

    if (m_index_stack.size() == order)
      return true;
  }
}

const TreeGram::Node&
TreeGram::Iterator::node(int order)
{
  assert(m_gram);
  assert(!m_index_stack.empty());
  assert(order <= m_index_stack.size());
  assert(order >= 0);

  if (order == 0)
    return m_gram->m_nodes[m_index_stack.back()];
  else
    return m_gram->m_nodes[m_index_stack[order-1]];
}
