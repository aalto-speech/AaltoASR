#ifndef VOCABULARY_HH
#define VOCABULARY_HH

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <exception>

class Vocabulary {
public:
  Vocabulary();

  // Return the string of the index'th word
  inline const std::string &word(int index) const;

  // Returns true if index is OOV
  inline bool oov(int index) const { return index == 0; }

  // Return the index of the string 'word'
  inline int index(const std::string &word) const;

  // Add a word to vocabulary, and return the index of the word.
  // Duplicates are detected and not inserted again.
  int add(const std::string &word);

  // Return the number of words in the vocabulary.  Does not include OOV.
  inline int size() const { return m_indices.size() - 1; }

  // Set the string for OOV word.  Warning: clears the vocabulary.
  void set_oov(const std::string &word);

  // Read vocabulary from a stream: one word per line.  # comments are
  // removed.  Spaces are removed.
  void read(std::istream &in);
  void read(const char *file);
  
  // Write vocabulary to a stream.
  void write(std::ostream &out) const;

  struct OpenError : public std::exception {
    virtual const char *what() const throw()
      { return "Vocabulary: open error"; }
  };

  struct OutOfRange : public std::exception {
    virtual const char *what() const throw()
      { return "Vocabulary: out of range"; }
  };

protected:
  std::map<std::string,int> m_indices;
  std::vector<std::string> m_words;
};

const std::string&
Vocabulary::word(int index) const
{
  if (index < 0 || index >= m_words.size())
    throw OutOfRange();

  return m_words[index];
}

int
Vocabulary::index(const std::string &word) const
{
  std::map<std::string,int>::const_iterator i = m_indices.find(word);
  if (i == m_indices.end())
    return 0;
  return (*i).second;
}

#endif /* VOCABULARY_HH */
