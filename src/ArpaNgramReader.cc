#include <string>

#include <errno.h>

#include "ArpaNgramReader.hh"

inline void
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


inline
bool
ArpaNgramReader::regexec(const regex_t *preg, const char *string)
{
  int result = ::regexec(preg, string, m_matches.size(), &m_matches[0], 0);
  return result == 0;
}

inline
void
ArpaNgramReader::split(std::string &str, std::vector<int> &points) 
{
  bool spaces = true;

  points.clear();
  for (int i = 0; i < str.length(); i++) {
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

ArpaNgramReader::ArpaNgramReader(const Vocabulary &vocabulary)
  : m_vocabulary(vocabulary)
{
  int cflags = REG_EXTENDED;

  regcomp(&m_r_count, "^ngram ([0-9]+)=([0-9]+)$", cflags);
  regcomp(&m_r_order, "^\\\\([0-9]+)-grams:$", cflags);

#define RFLOAT "-?[0-9]*\\.[0-9]+"
  regcomp(&m_r_ngram, "^(" RFLOAT ")(.*)(" RFLOAT ")$", cflags);
#undef RFLOAT

  m_matches.resize(4);
}

ArpaNgramReader::~ArpaNgramReader()
{
  regfree(&m_r_count);
  regfree(&m_r_order);
  regfree(&m_r_ngram);
}

inline double 
ArpaNgramReader::str2double(const char *str)
{
  char *endptr;
  double value = strtod(str, &endptr);
  if (endptr == str)
    throw InvalidFloat();
  if (errno == ERANGE)
    throw RangeError();
  return value;
}

// FIXME: ugly code
void
ArpaNgramReader::read(std::istream &in)
{
  m_lineno = 0;

  std::string str; 
  str.reserve(256); // Size just for efficiency

  // Skip header
  while (std::getline(in, str)) {
    m_lineno++;
    if (str == "\\data\\")
      break;
  }
  if (!in)
    throw ReadError();

  // Read counts
  int max_order = 0;
  int total_ngrams = 0;
  std::vector<int> ngrams; // The number of ngrams for each order
  while (std::getline(in, str)) {
    m_lineno++;
    if (!regexec(&m_r_count, str.c_str()))
      break;
    max_order = atoi(&str.c_str()[start(1)]);
    int value = atoi(&str.c_str()[start(2)]);
    total_ngrams += value;
    ngrams.push_back(value);
    if (ngrams.size() != max_order)
      throw InvalidOrder();
  }
  if (!in)
    throw ReadError();

  // Read all ngrams
  std::vector<int> ngram(max_order);
  std::vector<int> points(max_order + 1);
  for (int order = 1; order <= max_order; order++) {
    bool header = false;

    // Read ngram of current order
    for (int ngrams_read = 0; ngrams_read < ngrams[order];) {
      if (!std::getline(in, str))
	throw ReadError();
      m_lineno++;

      // Command
      if (str[0] == '\\') {
	if (regexec(&m_r_order, str.c_str())) {
	  int new_order = atoi(&str.c_str()[start(1)]);
	  if (new_order != order)
	    throw InvalidOrder();
	  header = true;
	}

	else
	  throw InvalidCommand();
      }
    
      // Ngram
      else if (header && regexec(&m_r_ngram, str.c_str())) {
	double log_prob = atof(&str[start(1)]);
	double back_off = 0;
	if (start(3) >= 0)
	  back_off = atof(&str[start(3)]);

	split(str, points);

	if (points.size() < order + 1)
	  throw InvalidNgram();

	ngram.clear();
	for (int i = 0; i < order; i++) {
	  int point = points[i + 1];
	  if (point >= end(2))
	    throw InvalidNgram();
	  int word_id = m_vocabulary.index(&str[point]);
	  ngram.push_back(word_id);
	}
	ngrams_read++;
      }
      else 
	throw InvalidNgram();
    }

    if (order == 0)
      throw InvalidOrder();
  }
  if (!in)
    throw ReadError();
}
