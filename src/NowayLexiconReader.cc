#include <iostream> // DEBUG
#include <sstream>

#include <cassert>
#include <cmath>

#include "NowayLexiconReader.hh"

NowayLexiconReader::NowayLexiconReader(
  const std::map<std::string,int> &hmm_map,
  const std::vector<Hmm> &hmms)
  : m_hmm_map(hmm_map),
    m_hmms(hmms)
{
}

void
skip_while(FILE *file, const char *chars)
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
get_until(FILE *file, std::string &str, char *delims)
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
NowayLexiconReader::read(FILE *file)
{
  m_word.reserve(128); // The size is not necessary, just for efficiency

  while (1) {
    // Read first word
    skip_while(file, " \t\n");
    get_until(file, m_word, " \t\n");

    if (ferror(file))
      throw ReadError();

    if (feof(file) && m_word.length() == 0)
      return;

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

    // Read phones and insert them to lexicon
    Lexicon::Node *node = m_lexicon.root();
    Lexicon::Node *prev_node; // Used for duplicating identical word ends
    bool insert_rest = false;

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
      if (it == m_hmm_map.end())
	throw UnknownHmm();
      int hmm_id = (*it).second;
      
      prev_node = node;

      // Find out if the current node has the corresponding hmm already.
      if (!insert_rest) {
	int next_id = 0;
	while (1) {
	  if (next_id == node->next.size()) {
	    insert_rest = true;
	    break;
	  }
	    
	  if (node->next[next_id]->hmm_id == hmm_id) {
	    node = node->next[next_id];
	    break;
	  }

	  next_id++;
	}
      }

      // When we get a branch that is not already in the lexicon,
      // insert the rest phones.
      if (insert_rest) {
	Lexicon::Node *new_node = 
	  new Lexicon::Node(m_hmms[hmm_id].states.size());
	new_node->hmm_id = hmm_id;
	node->next.push_back(new_node);
	node = new_node;
      }
    }

    assert(node != m_lexicon.root()); // FIXME: can pronounciation be empty?

    // Add word to lexicon

    // We duplicate word ends if two words have identical phoneme
    // sequences.  See the header file for more information.
    if (node->word_id >= 0) {
      node = new Lexicon::Node(*node);
      prev_node->next.push_back(node);
    }
    int word_id = m_vocabulary.add_word(m_word);
    node->word_id = word_id;
    node->log_prob = log10(prob);
    m_lexicon.update_words(word_id);
  }
}
