#include <stdlib.h>
#include <assert.h>

#include "TreeGramArpaReader.hh"
#include "tools.hh"

TreeGramArpaReader::TreeGramArpaReader()
  : m_lineno(0)
{
}

void
TreeGramArpaReader::read_error()
{
  fprintf(stderr, "TreeGramArpaReader::read(): error on line %d\n", m_lineno);
  exit(1);
}

void
TreeGramArpaReader::read(FILE *file, TreeGram *tree_gram)
{
  std::string line;
  std::vector<std::string> vec;

  // Just for efficiency
  line.reserve(128); 
  vec.reserve(16);

  bool ok = true;

  m_lineno = 0;

  // Find header
  while (1) {
    ok = read_line(&line, file);
    chomp(&line);
    m_lineno++;

    if (!ok) {
      fprintf(stderr, "TreeGramArpaReader::read(): "
	      "error on line %d while waiting \\data\\", m_lineno);
      exit(1);
    }

    if (line == "\\data\\")
      break;
  }

  // Read header
  int order = 1;
  int number_of_nodes = 0;
  while (1) {
    ok = read_line(&line, file);
    chomp(&line);
    m_lineno++;

    if (!ok) {
      fprintf(stderr, "TreeGramArpaReader::read(): "
	      "error on line %d while reading counts", m_lineno);
      exit(1);
    }

    // Header ends in a \-command
    if (line[0] == '\\')
      break;

    // Skip empty lines
    if (line.find_first_not_of(" \t\n") == line.npos)
      continue;

    // All non-empty header lines must be ngram counts
    if (line.substr(0, 6) != "ngram ")
      read_error();

    split(line.substr(6), "=", false, &vec);
    if (vec.size() != 2)
      read_error();

    int count = atoi(vec[1].c_str());
    number_of_nodes += count;
    m_counts.push_back(count);
    if (atoi(vec[0].c_str()) != order || m_counts.back() <= 0)
      read_error();
    order++;
  }

  tree_gram->reserve_nodes(number_of_nodes);

  // Read ngrams order by order
  for (order = 1; order <= m_counts.size(); order++) {

    // We must always have the correct header line at this point
    if (line[0] != '\\') {
      fprintf(stderr, "TreeGramArpaReader::read(): "
	      "\\%d-grams expected on line %d\n", order, m_lineno);
      exit(1);
    }
    clean(&line, " \t");
    split(line, "-", false, &vec);

    if (atoi(vec[0].substr(1).c_str()) != order || vec[1] != "grams:") {
      fprintf(stderr, "TreeGramArpaReader::read(): "
	      "unexpected command on line %d: %s\n", m_lineno, line.c_str());
      exit(1);
    }

    // Read the grams of each order
    TreeGram::Gram gram;
    gram.resize(order);
    for (int w = 0; w < m_counts[order-1]; w++) {

      // Read and split the line
      if (!read_line(&line, file))
	read_error();
      clean(&line, " \t\n");
      m_lineno++;

      // Ignore empty lines
      if (line.find_first_not_of(" \t\n") == line.npos) {
	w--;
	continue;
      }

      split(line, " \t", true, &vec);

      // Check the number of columns on the line
      if (vec.size() < order + 1 || vec.size() > order + 2) {
	fprintf(stderr, "TreeGramArpaReader::read(): "
		"%d columns on line %d\n", vec.size(), m_lineno);
	exit(1);
      }
      if (order == m_counts.size() && vec.size() != order + 1)
	fprintf(stderr, "WARNING: %d columns on line %d\n", vec.size(), 
		m_lineno);

      // FIXME: should we deny new words in higher order ngrams?

      // Parse log-probability, back-off weight and word indices
      // FIXME: check the conversion of floats
      float log_prob = strtod(vec[0].c_str(), NULL);
      float back_off = 0;
      if (vec.size() == order + 2)
	back_off = strtod(vec[order + 1].c_str(), NULL);

      for (int i = 0; i < order; i++)
	gram[i] = tree_gram->add_word(vec[i + 1]);

      tree_gram->add_gram(gram, log_prob, back_off);
    }

    // Skip empty lines
    while (1) {
      if (!read_line(&line, file)) {
	if (ferror(file))
	  read_error();
	if (feof(file))
	  break;
      }
      chomp(&line);
      m_lineno++;

      if (line.find_first_not_of(" \t\n") != line.npos)
	break;
    }
  }
}
