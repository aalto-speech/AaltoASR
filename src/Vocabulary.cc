#include <fstream>

#include "Vocabulary.hh"

Vocabulary::Vocabulary()
{
  m_words.push_back("<UNK>");
  m_indices["<UNK>"] = 0;
}

int
Vocabulary::add(const std::string &word)
{
  std::map<std::string,int>::iterator i = m_indices.find(word);
  if (i != m_indices.end())
    return (*i).second;

  int index = m_words.size();
  m_indices[word] = index;
  m_words.push_back(word);
  return index;
}

void
Vocabulary::read(std::istream &in)
{
  std::string word;

  while (getline(in, word)) {
    
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
    word = word.substr(start, end - start + 1);

    // Insert word
    add(word);
  }
}

void
Vocabulary::read(const char *file)
{
  std::ifstream in(file);
  if (!in)
    throw OpenError();
  read(in);
}

void
Vocabulary::write(std::ostream &out) const
{
  for (unsigned int i = 1; i < m_words.size(); i++)
    out << m_words[i] << std::endl;
}
