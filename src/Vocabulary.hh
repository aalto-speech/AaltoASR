#ifndef VOCABULARY_HH
#define VOCABULARY_HH

#include <string>
#include <vector>
#include <map>
#include <exception>
#include <stdio.h>

class Vocabulary {
public:
  Vocabulary();

  // Return the string of the index'th word
  inline const std::string &word(int index) const;

  // Returns true if index is OOV
  inline bool is_oov(int index) const { return index == 0; }

  // Return the index of the string 'word'
  inline int word_index(const std::string &word) const;

  // Add a word to vocabulary, and return the index of the word.
  // Duplicates are detected and not inserted again.
  int add_word(const std::string &word);

  // Return the number of words in the vocabulary.  Includes OOV.
  inline int num_words() const { return m_indices.size(); }

  // Set the string for OOV word.  Warning: clears the vocabulary.
  void set_oov(const std::string &word);

  // Read vocabulary from a stream: one word per line.  # comments are
  // removed.  Spaces are removed.
  void read(FILE *file);
  void read(const char *filename);
  
  // Write vocabulary to a stream.
  void write(FILE *file) const;

protected:
  // Clears the vocabulary without adding the oov.
  void clear_words();

  std::map<std::string,int> m_indices;
  std::vector<std::string> m_words;
};

const std::string&
Vocabulary::word(int index) const
{
  if ((unsigned int)index >= (unsigned int)m_words.size()) {
    fprintf(stderr, "Vocabulary::word(): index %d out of range\n", index);
    exit(1);
  }

  return m_words[index];
}

int
Vocabulary::word_index(const std::string &word) const
{
  std::map<std::string,int>::const_iterator i = m_indices.find(word);
  if (i == m_indices.end())
    return 0;
  return (*i).second;
}

#endif /* VOCABULARY_HH */
