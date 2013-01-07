// Conversion of words to word indices and vice versa.
#ifndef VOCABULARY_HH
#define VOCABULARY_HH

#include <cstdlib>
#include <string>
#include <vector>
#include <exception>
#include <stdio.h>
#include <map>
typedef std::map<std::string,int> vocabmap;

class Vocabulary {
  friend class VCluster;
public:
  Vocabulary();

  /// \brief Resets the vocabulary to the default state, where there exists only
  /// the OOV word "<UNK>".
  ///
  void reset();

  /// \brief Returns the string of the index'th word.
  ///
  inline const std::string &word(int index) const;

  /// \brief Returns true if index is OOV.
  ///
  inline bool is_oov(int index) const { return index == 0; }

  /// \brief Returns the index of \a word, or 0 if not found.
  ///
  inline int word_index(const std::string &word) const;

  /// \brief Adds a word to vocabulary, and return the index of the word.
  ///
  /// Duplicates are detected and not inserted again.
  ///
  int add_word(const std::string &word);

  /// \brief Returns the number of words in the vocabulary.  Includes OOV.
  ///
  inline int num_words() const { return m_indices.size(); }

  /// \brief Set the string for OOV word.  Warning: clears the vocabulary.
  ///
  void set_oov(const std::string &word);

  /// \brief Read vocabulary from a stream: one word per line.  # comments are
  /// removed.  Spaces are removed.
  ///
  void read(FILE *file);
  void read(const std::string &filename);
  
  /// \brief Writes vocabulary to a stream.
  ///
  void write(FILE *file) const;

  /// \brief Clears the vocabulary without adding the oov.
  ///
  void clear_words();

  /// \brief Copies vocabulary to Voc.
  ///
  /// Ugly implementation.
  ///
  inline void copy_vocab_to(Vocabulary &Voc){Voc.copy_helper(m_indices, m_words);}
  inline void copy_helper(vocabmap &ind, std::vector<std::string> &w) {m_indices=ind,m_words=w;}

protected:
  vocabmap m_indices;
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
  vocabmap::const_iterator i = m_indices.find(word);
  if (i == m_indices.end())
    return 0;
  return (*i).second;
}

#endif /* VOCABULARY_HH */
