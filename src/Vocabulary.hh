#ifndef VOCABULARY_HH
#define VOCABULARY_HH

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <exception>

class Vocabulary {
public:
  // If size() index is used for out-of-vocabulary words.  The
  // vocabulary size is also affected.  The string presentation of the
  // unknown word is "OOV".  OOV is not read/written in/from the
  // vocabulary stream.

  // Return the string of the index'th word
  inline const std::string &word(unsigned int index) const;

  // Returns true if index is OOV
  inline bool oov(int index) const { return index == size(); }

  // Return the index of the string 'word'
  inline int index(const std::string &word) const;

  // Add a word to vocabulary, and return the index of the word.
  // Duplicates are detected and not inserted again.
  int add(const std::string &word);

  // Return the number of words in the vocabulary.  May include OOV.
  inline int size() const { return m_indices.size(); }

  // Read vocabulary from a stream: one word per line.  # comments are
  // removed.  Spaces are removed.
  void read(std::istream &in);
  
  // Write vocabulary to a stream.
  void write(std::ostream &out) const;

  // Exception: supplied index was out of range
  struct OutOfRange : public std::exception {
    virtual const char *what() const throw()
      { return "Vocabulary: out of range"; }
  };

protected:
  std::map<std::string,int> m_indices;
  std::vector<std::string> m_words;
};

const std::string&
Vocabulary::word(unsigned int index) const
{
  if (index < 0 || index > m_words.size())
    throw OutOfRange();

  return m_words[index];
}

int
Vocabulary::index(const std::string &word) const
{
  std::map<std::string,int>::const_iterator i = m_indices.find(word);
  if (i == m_indices.end())
    return m_indices.size();
  return (*i).second;
}

#endif /* VOCABULARY_HH */
