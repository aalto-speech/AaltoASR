#include <string>
#include <cerrno>
#include <iostream>

#include "ArpaNgramReader.hh"

// FIXME: NGram assumes sorted nodes, so check the order when reading

void
ArpaNgramReader::set_oov(const std::string &word)
{
  if (m_ngram.m_nodes.size() > 0) {
    fprintf(stderr, "ArpaNgramReader::set_oov(): LM not empty");
    exit(1);
  }
  m_ngram.set_oov(word);
}

bool
ArpaNgramReader::getline(std::string *str, bool chomp)
{
  char *ptr;
  char buf[4096];

  str->clear();
  while (1) {
    ptr = fgets(buf, 4096, m_file);
    if (!ptr)
      break;

    str->append(buf);
    if ((*str)[str->length() - 1] == '\n') {
      break;
    }
  }

  if (ferror(m_file) || str->length() == 0)
    return false;

  if (chomp && (*str)[str->length() - 1] == '\n')
    str->resize(str->length() - 1);

  return true;
}

void
ArpaNgramReader::regcomp(regex_t *preg, const char *regex, int cflags)
{
  int result = ::regcomp(preg, regex, cflags);
  if (result != 0) {
    char errbuf[4096];
    regerror(result, preg, errbuf, 4096);
    std::cerr << errbuf << std::endl;
    throw RegExpError();
  }
}


bool
ArpaNgramReader::regexec(const regex_t *preg, const char *string)
{
  int result = ::regexec(preg, string, m_matches.size(), &m_matches[0], 0);
  return result == 0;
}

void
ArpaNgramReader::split(std::string &str, std::vector<int> &points) 
{
  bool spaces = true;

  points.clear();
  for (int i = 0; i < m_str.length(); i++) {
    if (spaces) {
      if (str[i] != ' ' && str[i] != '\t') {
	spaces = false;
	points.push_back(i);
      }
    }
    
    else {
      if (str[i] == ' ' || str[i] == '\t') {
	str[i] = '\0';
	spaces = true;
      }
    }
  }
}

ArpaNgramReader::ArpaNgramReader()
  : m_matches(4),
    m_file(NULL)
{
  int cflags = REG_EXTENDED;

  regcomp(&m_r_count, "^ngram ([0-9]+)=([0-9]+)$", cflags);
  regcomp(&m_r_order, "^\\\\([0-9]+)-grams:$", cflags);
}

ArpaNgramReader::~ArpaNgramReader()
{
  regfree(&m_r_count);
  regfree(&m_r_order);
}

float 
ArpaNgramReader::str2float(const char *str)
{
  char *endptr;
  float value = strtod(str, &endptr);
  if (endptr == str)
    throw InvalidFloat();
  if (errno == ERANGE)
    throw RangeError();
  return value;
}

void
ArpaNgramReader::reset_stacks(int first)
{
  std::fill(&m_word_stack[first], &m_word_stack[m_word_stack.size()], -1);
  std::fill(&m_index_stack[first], &m_index_stack[m_index_stack.size()], -1);
}

void
ArpaNgramReader::read_header()
{
  // Skip header
  while (getline(&m_str)) {
    m_lineno++;
    if (m_str == "\\data\\")
      break;
  }
  if (ferror(m_file))
    throw ReadError();
}

void
ArpaNgramReader::read_counts()
{
  int total_ngrams = 0;

  m_counts.clear();
  while (getline(&m_str)) {
    m_lineno++;
    if (!regexec(&m_r_count, m_str.c_str()))
      break;
    m_ngram.m_order = atoi(&m_str.c_str()[start(1)]);
    int value = atoi(&m_str.c_str()[start(2)]);
    total_ngrams += value;
    m_counts.push_back(value);
    if (m_counts.size() != m_ngram.m_order)
      throw InvalidOrder();
  }
  if (ferror(m_file))
    throw ReadError();

  m_ngram.m_nodes.clear();
  m_ngram.m_nodes.reserve(total_ngrams + 1); // Room for UNK also
  m_words.reserve(m_ngram.m_order);
  m_word_stack.resize(m_ngram.m_order, -1);
  m_index_stack.resize(m_ngram.m_order - 1, -1); // The last index is not used
  m_points.reserve(m_ngram.m_order + 2); // Words + log_prob and back_off
}

void
ArpaNgramReader::read_ngram(int order)
{
  // Divide line in words
  split(m_str, m_points);
  if (m_points.size() < order + 1 || m_points.size() > order + 2)
    throw InvalidNgram();

  // Parse log_prob and back_off
  float log_prob = atof(&m_str[m_points[0]]);
  float back_off = 0;
  if (m_points.size() == order + 2)
    back_off = atof(&m_str[m_points[order + 1]]);

  // Parse words using vocabulary and build the internal LM
  // vocabulary.  Note, that Ngram::add_word() does not insert duplicates,
  // so this is safe.
  m_words.clear();
  for (int i = 0; i < order; i++) {
    int word_id = m_ngram.add_word(&m_str[m_points[i + 1]]);
    m_words.push_back(word_id);
  }

  // Read unigrams.  We do it this way, because the order is not sure.
  // The UNK may be anywhere in the LM (at least SRI does not always
  // put UNK in the beginning of unigrams).
  if (order == 1) {
    // No duplicates!
    if (m_words[0] != 0 && m_words[0] != m_ngram.m_nodes.size()) {
      fprintf(stderr, "ArpaNgramReader::read_ngram(): duplicate unigram '%s' "
	      "on line %d\n", m_ngram.word(m_words[0]).c_str(), m_lineno);
      exit(1);
    }

    if (m_words[0] == 0)
      m_ngram.m_nodes[m_words[0]] = 
	Ngram::Node(m_words[0], log_prob, back_off);    
    else
      m_ngram.m_nodes.push_back(Ngram::Node(m_words[0], log_prob, back_off));
  }

  // 2-grams or larger.  Find the path to the current n-gram.
  else {
    // The reader assumes that words in n-grams (n > 1) are sorted
    // in the same order as unigrams.
    if (m_words[0] < m_word_stack[0]) {
      fprintf(stderr, "%d: invalid sort order: ", m_lineno);
      for (int i = 0; i < m_words.size(); i++) {
	fprintf(stderr, "%s ", m_ngram.word(m_words[i]).c_str());
      }
      exit(1);
    }

    // If the first word in the stack changes, reset stacks.
    if (m_word_stack[0] != m_words[0]) {
      m_word_stack[0] = m_words[0];
      m_index_stack[0] = m_words[0];
      reset_stacks(1);
    }

    // Handle the words 2..order-1.
    for (int o = 1; o < order-1; o++) {
      int word = m_words[o];
      
      // The reader assumes that words in n-grams (n > 1) are sorted
      // in the same order as unigrams.
      if (word < m_word_stack[o]) {
	fprintf(stderr, "%d: invalid sort order: ", m_lineno);
	for (int i = 0; i < m_words.size(); i++) {
	  fprintf(stderr, "%s ", m_ngram.word(m_words[i]).c_str());
	}
	exit(1);
      }

      // Path differs
      if (m_word_stack[o] != word) {
	reset_stacks(o+1);

	// The rest of the ngram (excluding the last word) must be
	// already in the structure.  Find the path.
	Ngram::Node *root = &m_ngram.m_nodes[m_index_stack[o - 1]];
	Ngram::Node *root_next = root + 1;
	int index;

	if (m_index_stack[o] < 0)
	  index = root->first;
	else
	  index = m_index_stack[o] + 1; 

	while (m_ngram.m_nodes[index].word != word) {
	  assert(index < root_next->first);
	  if (index == root_next->first) {
	    fprintf(stderr, "%d: ", m_lineno);
	    for (int i = 0; i < m_words.size(); i++)
	      fprintf(stderr, "%s ", m_ngram.word(m_words[i]).c_str());
	    fputc('\n', stderr);
	    throw UnknownPrefix();
	  }
	  index++;
	}

	m_word_stack[o] = word;
	m_index_stack[o] = index;
      }
    }

    // Handle the last word.
    int word = m_words.back();

    // Only unigrams can have UNK as last word.
    if (word == 0) {
      fprintf(stderr, "%d: only unigrams can contain UNK as last word: ", 
	      m_lineno);
      exit(1);
    }

    // The reader assumes that words in n-grams (n > 1) are sorted
    // in the same order as unigrams.
    if (word < m_word_stack[order-1]) {
      fprintf(stderr, "%d: invalid sort order: ", m_lineno);
      for (int i = 0; i < m_words.size(); i++) {
	fprintf(stderr, "%s ", m_ngram.word(m_words[i]).c_str());
      }
      exit(1);
    }

    // No duplicates?
    if (m_word_stack[order - 1] == word)
      throw Duplicate();

    // Insert the ngram, and update the 'first' field of the previous
    // order.
    m_word_stack[order - 1] = word;
    m_ngram.m_nodes.push_back(Ngram::Node(m_words.back(), log_prob, back_off));

    Ngram::Node *previous = &m_ngram.m_nodes[m_index_stack[order - 2]];
    if (previous->first < 0)
      previous->first = m_ngram.m_nodes.size() - 1;
  }
}

void
ArpaNgramReader::read_ngrams(int order)
{
  bool header = false;
  int last_index_of_previous_order = -1;

  // Calculate the initial value for last_index_of_previous_order.  It
  // is (the index of the first node of the previous order) minus one.
  if (order > 1)
    last_index_of_previous_order = 
      m_ngram.m_nodes.size() - m_counts[order - 2] - 1;

  reset_stacks();

  for (int ngrams_read = 0; ngrams_read < m_counts[order - 1];) {
    if (!getline(&m_str))
      throw ReadError();
    m_lineno++;

    // Skip empty lines
    if (m_str.length() == 0)
      continue;

    // Command
    if (m_str[0] == '\\') {
      if (!header && regexec(&m_r_order, m_str.c_str())) {
	int new_order = atoi(&m_str.c_str()[start(1)]);
	if (new_order != order)
	  throw InvalidOrder();
	header = true;
      }

      else
	throw InvalidCommand();
    }

    // Ngram
    else {
      read_ngram(order);

      // Fix the 'first' fields in the n-grams of previous order.
      if (order > 1) {
	for (int i = last_index_of_previous_order + 1; 
	     i < m_index_stack[order - 2]; i++) 
	{
	  assert(m_ngram.m_nodes[i].first == -1);
	  m_ngram.m_nodes[i].first = m_ngram.m_nodes.size() - 1;
	}
	last_index_of_previous_order = m_index_stack[order - 2];
      }

      ngrams_read++;
    }
  }

  // Fix the rest of the 'first' fields of previous order.
  if (order > 1) {
    for (int i = last_index_of_previous_order + 1;
	 i < m_ngram.m_nodes.size() - m_counts[order - 1];
	 i++) {
      assert(m_ngram.node(i)->first == -1);
      m_ngram.node(i)->first = m_ngram.m_nodes.size();
    }
  }
}

// FIXME: ugly code
void
ArpaNgramReader::read(FILE *file)
{
  m_file = file;
  m_str.reserve(4096); // Just for efficiency
  m_lineno = 0;

  read_header();
  read_counts();

  // Ensure the we have UNK.  It may be overwritten by the LM.
  m_ngram.m_nodes.push_back(Ngram::Node(0, -99, 0)); // FIXME: magic num

  // Read unigrams.  If UNK was not in the file, we must update the
  // count, because we have inserted the UNK in the model.
  read_ngrams(1);
  if (m_ngram.m_nodes.size() != m_counts[0]) {
    fprintf(stderr, "warning: %d unigrams resulted in %d nodes\n",
	    m_counts[0], m_ngram.num_words());
    assert(m_counts[0] + 1 == m_ngram.num_words());
    fprintf(stderr, "warning: UNK was not in LM, so we inserted it\n");
    m_counts[0]++;
  }

  // Read the rest.
  for (int order = 2; order <= m_ngram.m_order; order++)
    read_ngrams(order);

  if (ferror(m_file))
    throw ReadError();

  // FIXME: remove debug!
  debug_sanity_check();
}

void
ArpaNgramReader::debug_sanity_check()
{
  fprintf(stderr, "WARNING: time consuming sanity check in ArpaNgramReader\n");

  // Calculate the starting points of the orders.
  std::vector<int> starts;
  starts.push_back(0);
  for (int o = 2; o <= m_ngram.order(); o++) {
    starts.push_back(starts[o-2] + m_counts[o-2]);
  }
  starts.push_back(m_ngram.m_nodes.size());

  // Print counts
  for (int o = 1; o <= m_ngram.order(); o++)
    fprintf(stderr, "%d-grams: %d (start %d)\n", o, m_counts[o-1], 
	    starts[o-1]);

  int prev_first = 0;
  int n = 0;
  for (int o = 1; o < m_ngram.order(); o++) {
    for (int i = 0; i < m_counts[o-1]; i++) {
      int first = m_ngram.m_nodes[n].first;
      if (first < prev_first || first <= 0) {
	fprintf(stderr, "invalid first field (%d) in node %d (order %d)\n", 
		first, n, o);
	exit(1);
      }

      if (first < starts[o] || first > starts[o + 1]) {
	fprintf(stderr, "first (%d) out of range (%d-%d) in node %d "
		"(order %d)\n", first, starts[o], starts[o + 1], n, o);
	exit(1);
      }
      prev_first = first;
      n++;
    }
  }

  for (int i = 0; i < m_counts[m_ngram.order()-1]; i++) {
    if (m_ngram.m_nodes[n].first != -1) {
      fprintf(stderr, "first not -1 (was %d) in node %d (order %d)\n", 
	      m_ngram.m_nodes[n].first, n, m_ngram.order());
      exit(1);
    }
    n++;
  }

  if (n != m_ngram.m_nodes.size()) {
    fprintf(stderr, "size mismatch: %d vs %d\n", n, m_ngram.m_nodes.size());
    exit(1);
  }
}

void
ArpaNgramReader::read(const char *file)
{
  m_file = fopen(file, "r");
  if (!m_file)
    throw OpenError();
  read(m_file);
  fclose(m_file);
  m_file = NULL;
}

