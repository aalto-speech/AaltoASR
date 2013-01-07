#include <cstdio>

#include <assert.h>
#include <math.h>

#include "TPNowayLexReader.hh"

using namespace std;

TPNowayLexReader::TPNowayLexReader(
  std::map<std::string,int> &hmm_map,
  std::vector<Hmm> &hmms,
  TPLexPrefixTree &lex_tree,
  Vocabulary &vocab)
  : m_hmm_map(hmm_map),
    m_hmms(hmms),
    m_lexicon(lex_tree),
    m_vocabulary(vocab),
    m_silence_is_word(true)
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
TPNowayLexReader::get_until(FILE *file, std::string &str, const char *delims)
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
TPNowayLexReader::read(FILE *file, const std::string &word_boundary)
{
  int word_id;
  vector<Hmm*> hmm_list;
  m_word.reserve(128); // The size is not necessary, just for efficiency

  m_vocabulary.reset();
  m_lexicon.initialize_lex_tree();

  while (1) {
    // Read first word
    skip_while(file, " \t\n");
    get_until(file, m_word, " \t\n");

    if (ferror(file))
      throw ReadError();

    if (feof(file) && m_word.length() == 0)
      break;

    if (m_word.length() == 0) {
        get_until(file, m_word, "\n");
    	cerr << "Empty word on line: " << m_word << endl;
    	continue;
    }

    // Parse possible probability
    int left = m_word.rfind('(');
    int right = m_word.rfind(')');
    float prob = 1;
    if (left != -1 || right != -1) {
      if (left == -1 || right == -1)
        throw InvalidProbability();
      string tmp = m_word.substr(left + 1, right - left - 1);
      char *end_ptr;
      prob = strtod(tmp.c_str(), &end_ptr);
      if (*end_ptr != '\0')
        throw InvalidProbability();
      m_word.resize(left);
    }

    // Read phones and find the corresponding HMMs
    hmm_list.clear();

    bool unknown_phonemes = false;
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
      map<string,int>::const_iterator it = m_hmm_map.find(m_phone);
      if (it == m_hmm_map.end()) {
        // throw UnknownHmm(m_phone, m_word);
        fprintf(stderr, "TPNowayLexReader::read(): unknown hmm %s in word '%s'\n",
                m_phone.c_str(), m_word.c_str());
        hmm_list.clear();
        unknown_phonemes = true;
        break; //exit(1);
      }
      int hmm_id = (*it).second;
      hmm_list.push_back(&m_hmms[hmm_id]);
    }

    if (unknown_phonemes) // Don't add word if it contains unknown HMMs
      continue;

    // Add word to lexicon

    // FIXME! Deal with duplicate word ends? Pronunciation probabilities?
    if (m_word != "_" && (m_word[0] != '_' || m_silence_is_word))
    {
      word_id = m_vocabulary.add_word(m_word);
      if (m_word == word_boundary)
      {
        m_lexicon.set_word_boundary_id(word_id);
      }
    }
    else
      word_id = 0;

    if (hmm_list.size() > 0)
      m_lexicon.add_word(hmm_list, word_id);
  }
  m_lexicon.finish_tree();
}
