#ifndef ARPANGRAMREADER_HH
#define ARPANGRAMREADER_HH

#include <iostream> // FIXME: istream

#include <regex.h>

#include "Ngram.hh"
#include "Vocabulary.hh"

class Ngram;

class ArpaNgramReader {
public:
  ArpaNgramReader();
  ~ArpaNgramReader();
  void read(std::istream &in);
  void read(const char *file);
  Ngram &ngram() { return m_ngram; }

  int lineno() const { return m_lineno; }

  struct OpenError : public std::exception {
    virtual const char *what() const throw()
      { return "ArpaNgramReader: open error"; }
  };

  struct ReadError : public std::exception {
    virtual const char *what() const throw()
      { return "ArpaNgramReader: read error"; }
  };

  struct RangeError : public std::exception {
    virtual const char *what() const throw()
      { return "ArpaNgramReader: range error"; }
  };
  
  struct InvalidCommand : public std::exception {
    virtual const char *what() const throw()
      { return "ArpaNgramReader: invalid command"; }
  };

  struct InvalidNgram : public std::exception {
    virtual const char *what() const throw()
      { return "ArpaNgramReader: invalid ngram"; }
  };

  struct InvalidFloat : public std::exception {
    virtual const char *what() const throw()
      { return "ArpaNgramReader: invalid float"; }
  };

  struct InvalidOrder : public std::exception {
    virtual const char *what() const throw()
      { return "ArpaNgramReader: invalid order"; }
  };

  struct RegExpError : public std::exception {
    virtual const char *what() const throw()
      { return "ArpaNgramReader: reg exp error"; }
  };

  struct UnigramOrder : public std::exception {
    virtual const char *what() const throw()
      { return "ArpaNgramReader: unigram order"; }
  };

  struct UnknownPrefix : public std::exception {
    virtual const char *what() const throw()
      { return "ArpaNgramReader: unknown prefix"; }
  };

  struct Duplicate : public std::exception {
    virtual const char *what() const throw()
      { return "ArpaNgramReader: duplicate"; }
  };

private:
  float str2float(const char *str);
  void regcomp(regex_t *preg, const char *regex, int cflags);
  bool regexec(const regex_t *preg, const char *string);
  void split(std::string &str, std::vector<int> &points);

  inline regoff_t start(int index) { return m_matches[index].rm_so; }
  inline regoff_t end(int index) { return m_matches[index].rm_eo; }
  inline regoff_t length(int index) { return end(index) - start(index); }
  
  void reset_stacks(int first = 0);
  void read_header();
  void read_counts();
  void read_ngram(int order);
  void read_ngrams(int order);

  Ngram m_ngram;

  // Regular expressions
  int m_lineno;
  regex_t m_r_count;
  regex_t m_r_order;
  regex_t m_r_ngram;
  std::vector<regmatch_t> m_matches;

  // Temporary variables
  std::istream *m_in;
  std::string m_str;
  std::vector<int> m_counts;
  std::vector<int> m_words;
  std::vector<int> m_word_stack;
  std::vector<int> m_index_stack;
  std::vector<int> m_points;
};

#endif /* ARPANGRAMREADER_HH */



