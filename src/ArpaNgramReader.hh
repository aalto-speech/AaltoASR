#ifndef ARPANGRAMREADER_HH
#define ARPANGRAMREADER_HH

#include <istream>

#include "Ngram.hh"

class ArpaNgramReader {
public:
  inline ArpaNgramReader(const Vocabulary &vocabulary) 
    : m_vocabulary(vocabulary) { }
  void read(std::istream &in);
  Ngram &ngram() { return m_ngram; }

  struct ReadError : public std::exception {
    virtual const char *what() const throw()
      { return "ArpaNgramReader: read error"; }
  };

private:
  Vocabulary m_vocabulary;
  Ngram m_ngram;
};

#endif /* ARPANGRAMREADER_HH */
