#include <stdio.h>

#include <assert.h>
#include <math.h>

#include "TPNowayLexReader.hh"

TPNowayLexReader::TPNowayLexReader(
  std::map<std::string,int> &hmm_map,
  std::vector<Hmm> &hmms,
  TPLexPrefixTree &lex_tree,
  Vocabulary &vocab)
  : m_hmm_map(hmm_map),
    m_hmms(hmms),
    m_lexicon(lex_tree),
    m_vocabulary(vocab)
{
}

void
TPNowayLexReader::skip_while(FILE *file, const char *chars)
{
  int c;

  WHILE:
  {
    // Read character
    c = fgetc(file);
    if (c == EOF)
      return;

    // Check if in chars
    for (int i = 0; chars[i] != 0; i++)
      if (c == chars[i])
	goto WHILE;
  }

  ungetc(c, file);
}

// Reads chars to 'str' until one of the 'delims' is reached.  The
// delim is left unread.
void
TPNowayLexReader::get_until(FILE *file, std::string &str, char *delims)
{
  str.erase();

  while (1) {
    int c;
    c = fgetc(file);
    if (c == EOF)
      return;

    // Check delimiters
    for (int i = 0; delims[i] != 0; i++) {
      if (c == delims[i]) {
	ungetc(c, file);
	return;
      }
    }
    
    // Allocate space if necessary and append char to string
    if (str.length() == str.capacity())
      str.reserve(str.capacity() > 0 ? str.capacity() * 2 : 1);
    str += c;
  }
}

void
TPNowayLexReader::read(FILE *file)
{
  std::vector<Hmm*> hmm_list;
  m_word.reserve(128); // The size is not necessary, just for efficiency

  m_lexicon.initialize_lex_tree();
  
  while (1) {
    // Read first word
    skip_while(file, " \t\n");
    get_until(file, m_word, " \t\n");

    if (ferror(file))
      throw ReadError();

    if (feof(file) && m_word.length() == 0)
      break;

    // Parse possible probability
    int left = m_word.find('(');
    int right = m_word.rfind(')');
    float prob = 1;
    if (left != -1 || right != -1) {
      if (left == -1 || right == -1)
	throw InvalidProbability();
      std::string tmp = m_word.substr(left + 1, right - left - 1);
      char *end_ptr;
      prob = strtod(tmp.c_str(), &end_ptr);
      if (*end_ptr != '\0')
	throw InvalidProbability();
      m_word.resize(left);
    }

    // Read phones and find the corresponding HMMs
    hmm_list.clear();
    
    while (1) {
      // Skip whitespace and read phone
      skip_while(file, " \t");
      if (ferror(file))
	throw ReadError();
      if (feof(file))
	break;
      
      int peek = fgetc(file);
      ungetc(peek, file);
      if (peek == '\n')
	break;
      get_until(file, m_phone, " \t\n");

      // Find the index of the hmm
      std::map<std::string,int>::const_iterator it = m_hmm_map.find(m_phone);
      if (it == m_hmm_map.end()) {
	fprintf(stderr, "TPNowayLexReader::read(): unknown hmm %s\n",
		m_phone.c_str());
	exit(1);
      }
      int hmm_id = (*it).second;
      hmm_list.push_back(&m_hmms[hmm_id]);
    }

    // Add word to lexicon

    // FIXME! Deal with duplicate word ends? Pronounciation probabilities?
    int word_id = m_vocabulary.add_word(m_word);

    m_lexicon.add_word(hmm_list, word_id);
  }
  m_lexicon.finish_tree();
}
