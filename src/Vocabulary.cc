#include <errno.h>

#include "Vocabulary.hh"
#include "tools.hh"

Vocabulary::Vocabulary()
{
  m_words.push_back("<UNK>");
  m_indices["<UNK>"] = 0;
}

int
Vocabulary::add_word(const std::string &word)
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

  while (read_line(&word, file)) {
    chomp(&word);

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
    add_word(word);
  }
}

void
Vocabulary::read(const char *filename)
{
  FILE *file = fopen(filename, "r");
  if (!file) {
    fprintf(stderr, "Vocabulary::read(): could not open %s: %s\n",
	    filename, strerror(errno));
    exit(1);
  }
  read(file);
  fclose(file);
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
