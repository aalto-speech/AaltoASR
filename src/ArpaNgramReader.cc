#include <string>

#include "ArpaNgramReader.hh"

// FIXME: ugly code, assumes order < 10
void
ArpaNgramReader::read(std::istream &in)
{
  std::string str; 
  str.reserve(256); // Size just for efficiency

  // Skip header
  while (std::getline(in, str)) {
    if (str == "\\data\\")
      break;
  }
  if (!in)
    throw ReadError();

  // Read counts
  std::vector<int> ngrams; // The number of ngrams for each order
  while (std::getline(in, str)) {
    if (str.compare(0, 5, "ngram ", 5) != 0)
      break;
    ngrams.push_back(atoi(&str.c_str()[8]));
  }
  if (!in)
    throw ReadError();

  // Read ngrams
  int order = 0;
  std::string word;
  word.reserve(256);

  while (std::getline(in, str)) {
    if (str.length() == 0)
      continue;

    // Command
    if (str[0] == '\\') {
      if (str == "\\end\\")
	break;

      if (str.compare(3, 6, "grams:", 6) == 0) {
	int new_order = atoi(&str.c_str()[1]);
	if (new_order != order + 1)
	  throw InvalidOrder();
	order = new_order;
      }
    }

    if (order == 0)
      throw InvalidOrder();

    std::istringstream buf(str);
    buf >> log_prob;
    for (int i = 0; i < order; i++) {
      buf >> word;
      int word_index = m_vocabulary.index(word);
      if (m_vocabulary.oov(word_index))
	throw UnknownWord();

    }
  }

  if (!in)
    throw ReadError();
}
