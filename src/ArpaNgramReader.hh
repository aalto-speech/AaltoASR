#ifndef ARPANGRAMREADER_HH
#define ARPANGRAMREADER_HH

#include <iostream> // FIXME: istream

#include <regex.h>

#include "Ngram.hh"
#include "Vocabulary.hh"

class Ngram;

class ArpaNgramReader {
public:
  ArpaNgramReader(const Vocabulary &vocabulary);
  ~ArpaNgramReader();
  void read(std::istream &in);
  Ngram &ngram() { return m_ngram; }

  int lineno() const { return m_lineno; }

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

private:
  double str2double(const char *str);
  void regcomp(regex_t *preg, const char *regex, int cflags);
  bool regexec(const regex_t *preg, const char *string);
  void split(std::string &str, std::vector<int> &points);

  inline regoff_t start(int index) { return m_matches[index].rm_so; }
  inline regoff_t end(int index) { return m_matches[index].rm_eo; }
  inline regoff_t length(int index) { return end(index) - start(index); }
  
  Vocabulary m_vocabulary;
  Ngram m_ngram;

  // Regular expressions
  int m_lineno;
  regex_t m_r_count;
  regex_t m_r_order;
  regex_t m_r_ngram;
  std::vector<regmatch_t> m_matches;
};

#endif /* ARPANGRAMREADER_HH */



