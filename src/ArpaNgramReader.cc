#include <string>

#include "ArpaNgramReader.hh"

inline double 
ArpaNgramReader::str2double(const char *str)
{
  char *endptr;
  int value = strtod(ptr, &endptr);
  if (endptr == ptr)
    throw InvalidFloat();
  if (errno == ERANGE)
    throw RangeError();
  return value;
}

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
  int total_ngrams = 0;
  std::vector<int> ngrams; // The number of ngrams for each order
  while (std::getline(in, str)) {
    if (str.compare(0, 5, "ngram ", 5) != 0)
      break;
    int value = atoi(&str.c_str()[8]);
    total_ngrams += value;
    ngrams.push_back(value);
  }
  if (!in)
    throw ReadError();

  // Read ngrams
  int order = 0;
  std::vector<int> words;
  while (in >> str) {

    // Command
    if (str[0] == '\\') {
      // End
      if (str == "\\end\\")
	break;

      // N-gram
      if (str.compare(3, 6, "grams:", 6) == 0) {
	int new_order = atoi(&str.c_str()[1]);
	if (new_order != order + 1)
	  throw InvalidOrder();
	order = new_order;
      }
    }
    
    // Ngram
    else {
      double log_prob = str2double(str.c_str(), &endptr);
      
      words.clear();
      for (int i = 0; i < order; i++) {
	int word_id = m_vocabulary.index(str);
	words.push_back(word_id);
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
