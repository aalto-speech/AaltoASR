#ifndef NOWAYLEXICONREADER_HH
#define NOWAYLEXICONREADER_HH

#include <stdio.h>
#include <map>
#include <string>

#include "Lexicon.hh"
#include "Vocabulary.hh"
#include "Hmm.hh"

// A class that creates a lexicon from a dictionary file
//
// NOTES:
//
// Might assume that words are sorted (adding prob to nodes) FIXME!!
// Even after exception in reading, the read can be continued.
//
// Lexicon may contain different words with identical phoneme
// sequences.  If a word ends in a node which has already a word end,
// we create a new node for the new word.  The new node will be
// inserted further in the branch vector, so all words with the same
// prefix will end up in the old branch.
//
// FILE FORMAT:
//
// ^word(prob) phone1 phone2 ...
// ^word phone1 phone2 ...
//
// LIMITS:
//
// Maximum line length is limited, but at least 4096.

class NowayLexiconReader {
public:
  NowayLexiconReader(const std::map<std::string,int> &hmm_map, 
		     const std::vector<Hmm> &hmms);
  void read(FILE *file);

  // Data
  inline Lexicon &lexicon() { return m_lexicon; }
  inline const Vocabulary &vocabulary() { return m_vocabulary; }

  // Current state for error diagnosis
  inline const std::string &word() const { return m_word; }
  inline const std::string &phone() const { return m_phone; }

  struct ReadError : public std::exception {
    virtual const char *what() const throw()
      { return "NowayLexiconReader: read error"; }
  };

  struct InvalidProbability : public std::exception {
    virtual const char *what() const throw() 
      { return "NowayLexiconReader: invalid probability"; }
  };

  struct UnknownHmm : public std::exception {
    virtual const char *what() const throw()
      { return "NowayLexiconReader: unknown hmm"; }
  };

protected:
  const std::map<std::string,int> &m_hmm_map;
  const std::vector<Hmm> &m_hmms;
  Lexicon m_lexicon;
  Vocabulary m_vocabulary;
  int m_line_no;

  std::string m_word;

  // Temporary variables
  std::string m_phone;
};

#endif /* NOWAYLEXICONREADER_HH */
