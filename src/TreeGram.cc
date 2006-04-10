#include <errno.h>
#include <assert.h>
#include <math.h>

// BEGIN fwrite-hack
//#include <unistd.h>
// END fwrite-hack

#include <memory>
#include "Endian.hh"
#include "TreeGram.hh"
#include "str.hh"
#include "def.hh"
#include "TreeGramArpaReader.hh"
#include "ClusterMap.hh"

static std::string format_str("cis-binlm2\n");

void
TreeGram::reserve_nodes(int nodes)
{
  m_nodes.clear();
  m_nodes.reserve(nodes);
  m_nodes.push_back(Node(0, -99, 0, -1));
  m_order_count.clear();
  m_order_count.push_back(1);
  m_order = 1;
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
    fprintf(file, "%s(%d) ", word(gram[i]).c_str(), gram[i]);
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

  while (len > 5) { // magic threshold to do linear search
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

  // Note that (node_index + 1) is used later, so the last node_index
  // must not pass.  Actually, we could return -1 for all largest
  // order grams.
  if (node_index >= m_nodes.size() - 1)
    return -1;

  int first = m_nodes[node_index].child_index;
  int last = m_nodes[node_index + 1].child_index; // not included
  if (first < 0 || last < 0)
    return -1;

  return binary_search(word, first, last);
}

TreeGram::Iterator
TreeGram::iterator(const Gram &gram)
{
  Iterator iterator;

  fetch_gram(gram, 0);
  iterator.m_index_stack = m_fetch_stack;
  iterator.m_gram = this;

  return iterator;
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
  while (order < gram.size() - 1) {
    if (gram[order] != m_last_gram[order])
      break;
    order++;
  }
  m_insert_stack.resize(order);

  // The rest of the path must be searched.
  if (order == 0)
    prev = -1;
  else
    prev = m_insert_stack[order-1];

  while (order < gram.size()-1) {
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

  // Initialize new order count
  if (gram.size() > m_order_count.size()) {
    //fprintf(stderr,"init order %d\n", gram.size());
    m_order_count.push_back(0);
    m_order++;
  }
  assert(m_order_count.size() == gram.size());

  // Update order counts, but only if we do not have UNK-unigram
  if (gram.size() > 1 || gram[0] != 0)
    m_order_count[gram.size()-1]++;

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
    // Fill the insert_stack with the indices of the current gram up
    // to n-1 words.
    find_path(gram);

    // Update the child range start of the parent node.
    if (m_nodes[m_insert_stack.back()].child_index < 0) {
      //fprintf(stderr,"changing m_nodes[%d].child_index = %d\n",
      //	      m_insert_stack.back(), m_nodes.size());
      m_nodes[m_insert_stack.back()].child_index = m_nodes.size();
    }
    // Insert the new node.
    m_nodes.push_back(Node(gram.back(), log_prob, back_off, -1));
    //fprintf(stderr,"Adding %d\n", m_nodes.size());

    // Update the child range end of the parent node.  Note, that this
    // must be done after insertion, because in extreme case, we might
    // update the inserted node.
    m_nodes[m_insert_stack.back() + 1].child_index = m_nodes.size();
    //fprintf(stderr,"changing m_nodes[%d].child_index = %d\n",
    //	      m_insert_stack.back()+1, m_nodes.size());
    m_insert_stack.push_back(m_nodes.size() - 1);
  }

  if (m_nodes.back().child_index != -1) {
    fprintf(stderr,"TreeGram: Warning, hope you will call finalize()...\n");
  }
  m_last_gram = gram;
  assert(m_order == m_last_gram.size());
}

void 
TreeGram::write(FILE *file, bool binary) { 
  if (!binary) {
    TreeGramArpaReader areader;
    areader.write(file, this);
    return;
  }
  write_real(file, true);
}

void 
TreeGram::write_real(FILE *file, bool reflip) 
{
  fputs(format_str.c_str(), file);

  // Write type
  if (m_type == BACKOFF)
    fputs("backoff\n", file);
  else if (m_type == INTERPOLATED)
    fputs("interpolated\n", file);

  // Write vocabulary 
  fprintf(file, "%d\n", num_words());
  for (int i = 0; i < num_words(); i++)
    fprintf(file, "%s\n", word(i).c_str());

  // Order, number of nodes and order counts
  fprintf(file, "%d %d\n", m_order, m_nodes.size());
  for (int i = 0; i < m_order; i++)
    fprintf(file, "%d\n", m_order_count[i]);

  // Use correct endianity
  if (Endian::big) 
    flip_endian(); 

  // BEGIN fwrite-hack

  // Write nodes
#if 0
  fprintf(stderr, "FIXME: using write() workaround for fwrite().\n"
          "Fix me when cluster machines have been upgraded to SuSE 9.1\n");

  // The workaround for SuSE 9.0 cluster machines.  In 9.0, the
  // fwrite() system call can not write buffers of 2^31 bytes or more.
  size_t bytes_to_write = m_nodes.size() * sizeof(TreeGram::Node);
  fflush(file);
  int fd = fileno(file);
  ssize_t ret = 
    ::write(fd, &m_nodes[0], bytes_to_write);
  if (ret < 0) {
    perror("TreeGram::write(): write() system call failed");
    abort();
  }
  if ((size_t)ret != bytes_to_write) {
    fprintf(stderr, "TreeGram::write(): "
	    "write() system call wrote only %lu of %lu bytes\n",
	    (size_t)ret, bytes_to_write);
    abort();
  }
#else
  // The original code
    // unistd.h can be removed from the start of the file, when the
    // workaround is disabled.  The header file is needed only for the
    // write() system call.
    fwrite(&m_nodes[0], m_nodes.size() * sizeof(TreeGram::Node), 1, file);
#endif
  // END fwrite-hack
  
  if (ferror(file)) {
    fprintf(stderr, "TreeGram::write(): write error: %s\n", strerror(errno));
    exit(1);
  }

  // Restore to original endianess
  if (Endian::big && reflip)
    flip_endian();
}

void 
TreeGram::read(FILE *file, bool binary) 
{
  if (!binary) {
    TreeGramArpaReader areader;
    areader.read(file, this);
    return;
  }

  std::string line;
  int words;
  bool ret;

  // Read the header
  ret = str::read_string(&line, format_str.length(), file);
  if (!ret || line != format_str) {
    fprintf(stderr, "TreeGram::read(): invalid file format\n");
    exit(1);
  }
  
  // Read LM type
  str::read_line(&line, file, true);
  if (line == "backoff")
    m_type = BACKOFF;
  else if (line == "interpolated")
    m_type = INTERPOLATED;
  else {
    fprintf(stderr, "TreeGram::read(): invalid type: %s\n", line.c_str());
    exit(1);
  }

  // Read the number of words
  if (!str::read_line(&line, file)) {
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
  clear_words();
  for (int i=0; i < words; i++) {
    if (!str::read_line(&line, file, true)) {
      fprintf(stderr, "TreeGram::read(): "
	      "read error while reading vocabulary\n");
      exit(1);
    }
    add_word(line);
  }

  // Read the order and the number of nodes
  int number_of_nodes;
  fscanf(file, "%d %d\n", &m_order, &number_of_nodes);

  // Read the counts for each order
  int sum = 0;
  m_order_count.resize(m_order);
  for (int i = 0; i < m_order; i++) {
    fscanf(file, "%d\n", &m_order_count[i]);
    sum += m_order_count[i];
  }
  if (sum != number_of_nodes) {
    fprintf(stderr, "TreeGram::read(): "
	    "the sum of order counts %d does not match number of nodes %d\n",
	    sum, number_of_nodes);
    exit(1);
  }

  // Read the nodes
  m_nodes.clear();
  m_nodes.resize(number_of_nodes);
  size_t block_size = number_of_nodes * sizeof(TreeGram::Node);
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
  assert(first >= 0 && first < gram.size());

  int prev = -1;
  m_fetch_stack.clear();
  
  int i = first;
  while (m_fetch_stack.size() < gram.size() - first) {
    int node = find_child(gram[i], prev);
    if (node < 0)
      break;
    m_fetch_stack.push_back(node);
    i++;
    prev = node;
  }
}

void
TreeGram::fetch_bigram_list(int prev_word_id, std::vector<int> &next_word_id,
                            std::vector<float> &result_buffer)
{
  assert(m_type==BACKOFF);
  float back_off_w;
  int i;
  int child_index, next_child_index, cl_idx=-1;
  float *lm_buf = new float[m_words.size()];
  
  if (clmap) cl_idx=clmap->get_cluster(2,prev_word_id);

  // Get backoff weight
  if (clmap) back_off_w = m_nodes[cl_idx].back_off;
  else back_off_w = m_nodes[prev_word_id].back_off;
  // Fill the unigram probabilities
  for (i = 0; i < m_words.size(); i++)
    lm_buf[i] = back_off_w + m_nodes[i].log_prob;

  // Fill the bigram probabilities
  if (clmap) prev_word_id=cl_idx;
  child_index = m_nodes[prev_word_id].child_index;
  next_child_index = m_nodes[prev_word_id+1].child_index;
  if (child_index != -1 && next_child_index > child_index)
  {
    for (i = child_index; i < next_child_index; i++)
      lm_buf[m_nodes[i].word] = m_nodes[i].log_prob;
  }

  // Map to result_buffer
  for (i = 0; i < next_word_id.size(); i++)
    result_buffer[i] = lm_buf[next_word_id[i]];
  delete [] lm_buf;
}

void
TreeGram::fetch_trigram_list(int w1, int w2, std::vector<int> &next_word_id,
                             std::vector<float> &result_buffer)
{
  assert(m_type==BACKOFF);
  int bigram_index;

  // Check if bigram (w1,w2) exists
  bigram_index = find_child(w2, w1);
  if (bigram_index == -1)
  {
    // No bigram (w1,w2), only condition to w2
    fetch_bigram_list(w2, next_word_id, result_buffer);
  }
  else
  {
    float *lm_buf = new float[m_words.size()];
    int child_index, next_child_index;
    float bigram_back_off_w,w2_back_off_w, temp;
    int i;
    
    // Get backoff weights
    bigram_back_off_w = m_nodes[bigram_index].back_off;
    w2_back_off_w = m_nodes[w2].back_off;
    
    // Fill the unigram probabilities
    temp = bigram_back_off_w + w2_back_off_w;
    for (i = 0; i < m_words.size(); i++)
      lm_buf[i] = temp + m_nodes[i].log_prob;
    
    // Fill bigram (w2, next_word_id) probabilities
    child_index = m_nodes[w2].child_index;
    next_child_index = m_nodes[w2+1].child_index;
    if (child_index != -1 && next_child_index > child_index)
    {
      for (i = child_index; i < next_child_index; i++)
        lm_buf[m_nodes[i].word] = bigram_back_off_w + m_nodes[i].log_prob;
    }

    // Fill trigram probabilities
    child_index = m_nodes[bigram_index].child_index;
    next_child_index = m_nodes[bigram_index+1].child_index;
    if (child_index != -1 && next_child_index > child_index)
    {
      for (i = child_index; i < next_child_index; i++)
        lm_buf[m_nodes[i].word] = m_nodes[i].log_prob;
    }
    
    // Map to result_buffer
    for (i = 0; i < next_word_id.size(); i++)
      result_buffer[i] = lm_buf[next_word_id[i]];
    delete lm_buf;
  }
}

float
TreeGram::log_prob_bo(const Gram &gram)
{
  // Please keep this version lean and mean. Other version can bloat as much
  // as they like

  float log_prob = 0.0;
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
    if (m_fetch_stack.size() == gram.size() - n) {
      log_prob += m_nodes[m_fetch_stack.back()].log_prob;
      m_last_order = gram.size() - n;
      break;
    }
    
    // Back-off found?
    if (m_fetch_stack.size() == gram.size() -n -1)
      log_prob += m_nodes[m_fetch_stack.back()].back_off;
    
    n++;
  }
  return log_prob;
}

float
TreeGram::log_prob_bo_cl(const Gram &gram)
{
  Gram clgram(gram);
  clmap->wg2cg(clgram);
  return(log_prob_bo(clgram));

#if 0
  while (1) {
    assert(n < clgram.size());
    fetch_gram(clgram, n);
    assert(m_fetch_stack.size() > 0);
    
    // Full gram found?
    if (m_fetch_stack.size() == clgram.size() - n) {
      log_prob += m_nodes[m_fetch_stack.back()].log_prob;
      m_last_order = clgram.size() - n;
      break;
    }
    
    // Back-off found?
    if (m_fetch_stack.size() == clgram.size() -n -1)
      log_prob += m_nodes[m_fetch_stack.back()].back_off;
    
    n++;
  }
  return log_prob;
#endif
}

float
TreeGram::log_prob_i(const Gram &gram) {
  float prob=0.0;
  float bo;
  m_last_order=0;

  const int looptill=std::min(gram.size(),(size_t) m_order);
  for (int n=1;n<=looptill;n++) {
    fetch_gram(gram,gram.size()-n);
    if (m_fetch_stack.size() < n-1 || n>m_order) {
      continue;
      //return(safelogprob(prob)); 
    }
    
    if (m_fetch_stack.size()==n-1) {
      bo = pow(10,m_nodes[m_fetch_stack.back()].back_off);
      prob*=bo;
      continue;
    }
    
    if (n>1) {
      bo = pow(10,m_nodes[m_fetch_stack[m_fetch_stack.size()-2]].back_off);
      prob=bo*prob;
    }
    m_last_order=n;
    prob += pow(10,m_nodes[m_fetch_stack.back()].log_prob);
  }
  return(safelogprob(prob));
}

float
TreeGram::log_prob_i_cl(const Gram &gram) {
  float prob=0.0;
  float bo;
  m_last_order=0;
  Gram clgram(gram);

  clgram=gram;
  clmap->wg2cg(clgram);
  
  const int looptill=std::min(gram.size(),(size_t) m_order);
  
  for (int n=1;n<=looptill;n++) {
    clgram.back()=clmap->get_fcluster(n,gram.back());
    fetch_gram(clgram,clgram.size()-n);
    if (m_fetch_stack.size() < n-1 || n>m_order) {
      //return(safelogprob(prob)); 
      continue;
    }
    
    if (m_fetch_stack.size()==n-1) {
      bo = pow(10,m_nodes[m_fetch_stack.back()].back_off);
      prob *= bo;
      continue;
    }
    
    if (n>1) {
      bo = pow(10,m_nodes[m_fetch_stack[m_fetch_stack.size()-2]].back_off);
      prob *= bo;
    }
    m_last_order=n;
    prob += pow(10,m_nodes[m_fetch_stack.back()].log_prob 
		+ clmap->get_full_emprob(n,gram.back()));
  }
  return(safelogprob(prob));
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
    //fprintf(stderr,"Node->child %d\n", node->child_index);

    // If not backtracking, try diving deeper
    if (!backtrack) {
      // Do we have children?
      if (node->child_index > 0 
	  && (node+1)->child_index 
	  > node->child_index) {
	m_index_stack.push_back(node->child_index);
	return true;
      }
    }

    // No children, try siblings 
    backtrack = false;

    // Unigrams
    if (m_index_stack.size() == 1) {

      // If last unigram, there is no siblings, and we are at the end
      // of the structure?
      if (index == m_gram->m_order_count[0] - 1)
	return false;

      // Next unigram
      m_index_stack.back()++;
      return true;
    }

    // Higher order
    else {
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

bool
TreeGram::Iterator::move_in_context(int delta)
{
  // First order
  if (m_index_stack.size() == 1) {
    assert(m_index_stack.back() < m_gram->m_order_count[0]);
    if (m_index_stack.back() + delta < 0 ||
	m_index_stack.back() + delta >= m_gram->m_order_count[0])
      return false;
    m_index_stack.back() += delta;
    return true;
  }

  // Higher orders
  Node &parent = m_gram->m_nodes[m_index_stack[m_index_stack.size() - 2]];
  Node &next_parent = m_gram->m_nodes[m_index_stack[m_index_stack.size() - 2] 
				      + 1];
  assert(parent.child_index > 0);
  assert(next_parent.child_index > 0);
  assert(m_index_stack.back() >= parent.child_index);
  assert(m_index_stack.back() < next_parent.child_index);

  if (m_index_stack.back() + delta < parent.child_index ||
      m_index_stack.back() + delta >= next_parent.child_index)
    return false;
  m_index_stack.back() += delta;
  return true;
}

bool
TreeGram::Iterator::up()
{
  if (m_index_stack.size() == 1)
    return false;
  m_index_stack.pop_back();
  return true;
}

bool
TreeGram::Iterator::down()
{
  Node &node = m_gram->m_nodes[m_index_stack.back()];
  Node &next = m_gram->m_nodes[m_index_stack.back() + 1];
  if (node.child_index < 0 || 
      next.child_index < 0 ||
      node.child_index == next.child_index)
    return false;
  m_index_stack.push_back(node.child_index);
  return true;
}

void TreeGram::print_debuglist() {
  for (int i=0;i<m_nodes.size();i++) {
    fprintf(stderr,"%d: %d %.4f %.4f %d\n", i, m_nodes[i].word, m_nodes[i].log_prob, m_nodes[i].back_off, m_nodes[i].child_index);
  }
}

void TreeGram::finalize() {
  if (m_nodes.back().child_index == -1) return;
  Node node;
  m_nodes.push_back(node);
}
