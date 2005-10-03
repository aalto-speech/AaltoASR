#ifndef TPNOWAYLEXREADER_HH
#define TPNOWAYLEXREADER_HH

#include <stdio.h>
#include <map>
#include <string>

#include "TPLexPrefixTree.hh"
#include "Vocabulary.hh"

// A class that reads the lexicon from a file and calls the lexical prefix
// tree class to create the lexical tree.
//
// FILE FORMAT:
//
// ^word(prob) phone1 phone2 ...
// ^word phone1 phone2 ...
//


class TPNowayLexReader {
public:
  TPNowayLexReader(std::map<std::string,int> &hmm_map, 
                   std::vector<Hmm> &hmms,
                   TPLexPrefixTree &lex_tree,
                   Vocabulary &vocab);
  void read(FILE *file, const std::string &word_boundary);

  void skip_while(FILE *file, const char *chars);
  void get_until(FILE *file, std::string &str, char *delims);

  // Current state for error diagnosis
  inline const std::string &word() const { return m_word; }
  inline const std::string &phone() const { return m_phone; }

  struct ReadError : public std::exception {
    virtual const char *what() const throw()
      { return "TPNowayLexReader: read error"; }
  };

  struct InvalidProbability : public std::exception {
    virtual const char *what() const throw() 
      { return "TPNowayLexReader: invalid probability"; }
  };

  struct UnknownHmm : public std::exception {
    virtual const char *what() const throw()
      { return "TPNowayLexReader: unknown hmm"; }
  };

protected:
  std::map<std::string,int> &m_hmm_map;
  std::vector<Hmm> &m_hmms;
  TPLexPrefixTree &m_lexicon;
  Vocabulary &m_vocabulary;
  int m_line_no;

  std::string m_word;

  // Temporary variables
  std::string m_phone;
};

#endif /* TPNOWAYLEXREADER_HH */
