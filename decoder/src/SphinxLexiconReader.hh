#ifndef SPHINXLEXICONREADER_HH
#define SPHINXLEXICONREADER_HH

#include <iostream>
#include <map>
#include <string>

#include "Lexicon.hh"
#include "Vocabulary.hh"

// A class that creates a lexicon from a dictionary file
//
// NOTES:
//
// Ignores currently phonemes, because these lexicons are used only
// with the WordGraph.
//
// Might assume that words are sorted.
//
// Lexicon may contain different words with identical phoneme
// sequences.  If a word ends in a node which has already a word end,
// we create a new node for the new word.  The new node will be
// inserted further in the branch vector, so all words with the same
// prefix will end up in the old branch.
//
// FILE FORMAT:
//
// ^word(variation) phone1 phone2 ...
// ^word phone1 phone2 ...
//
// LIMITS:
//
// Maximum line length is limited, but at least 4096.

class SphinxLexiconReader {
public:
  SphinxLexiconReader();
  void read(std::istream &in);

  // Data
  inline Lexicon &lexicon() { return m_lexicon; }
  inline const Vocabulary &vocabulary() { return m_vocabulary; }

  // Current state for error diagnosis
  inline const std::string &word() const { return m_word; }
  inline const std::string &phone() const { return m_phone; }

  struct ReadError : public std::exception {
    virtual const char *what() const throw()
      { return "SphinxLexiconReader: read error"; }
  };

protected:
  Vocabulary m_vocabulary;
  int m_line_no;
  std::string m_word;
};

#endif /* SPHINXLEXICONREADER_HH */
