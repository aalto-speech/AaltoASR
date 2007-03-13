// Conversion of words to word indices and vice versa.
#include <errno.h>
#include "io.hh"
#include "Vocabulary.hh"
#include "str.hh"

Vocabulary::Vocabulary()
{
  m_words.push_back("<UNK>");
  m_indices["<UNK>"] = 0;
}

int
Vocabulary::add_word(const std::string &word)
{
  const vocabmap::iterator i = m_indices.find(word);
  if (i != m_indices.end())
    return (*i).second;

  int index = m_words.size();
  m_indices[word] = index;
  m_words.push_back(word);
  return index;
}

void
Vocabulary::set_oov(const std::string &word)
{
  clear_words();
  m_words.push_back(word);
  m_indices[word] = 0;
}

void
Vocabulary::read(FILE *file)
{
  std::string word;

  while (str::read_line(&word, file, true)) {

    // Remove comments
    int comment = word.find('#');
    if (comment >= 0)
      word = word.substr(0, comment);

    // Remove leading and trailing spaces.  Skip if word is just
    // spaces.
    int start = word.find_first_not_of("\t\n\r ");
    if (start < 0)
      continue;
    int end = word.find_last_not_of("\t\n\r ");
    
    
    // Check if " " or "(" is present and truncate the word to
    // that symbol. You can use dictionary files in CMUdict
    // and Decoder formats this way.
    int end_2 = word.find_first_of(" (")-1;
    if ((end_2>-1) and (end_2<end))
      end=end_2;
    word = word.substr(start, end - start + 1);
    
    if (word.compare("_")==0 or word.compare("__")==0)
      continue;

    // Insert word
    add_word(word);
  }
}

void
Vocabulary::read(const std::string &filename)
{
  io::Stream file(filename, "r");
  read(file.file);
}

void
Vocabulary::write(FILE *file) const
{
  for (unsigned int i = 1; i < m_words.size(); i++)
    fprintf(file, "%s\n", m_words[i].c_str());
}

void
Vocabulary::clear_words()
{
  m_indices.clear();
  m_words.clear();
}
