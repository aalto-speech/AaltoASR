#include <iostream> // DEBUG
#include <sstream>

#include <math.h>

#include "NowayLexiconReader.hh"

NowayLexiconReader::NowayLexiconReader(
  const std::map<std::string,int> &hmm_map,
  const std::vector<Hmm> &hmms)
  : m_hmm_map(hmm_map),
    m_hmms(hmms)
{
}

std::istream&
skip_while(std::istream &in, const char *chars)
{
  char c;

  WHILE:
  {
    // Read character
    in.read(&c, 1);
    if (!in)
      return in;

    // Check if in chars
    for (int i = 0; chars[i] != 0; i++)
      if (c == chars[i])
	goto WHILE;
  }

  in.unget();
  return in;
}

// Reads chars to 'str' until one of the 'delims' is reached.  The
// delim is left unread.
std::istream&
get_until(std::istream &in, std::string &str, char *delims)
{
  str.resize(0);

  while (1) {
    char c;
    in.read(&c, 1);

    if (in.bad())
      return in;

    // Clear flags if EOF reached after string.  Leave flags untouched
    // only if nothing read to string so yet.
    if (!in) {
      if (str.size() == 0)
	return in;
      in.clear();
      return in;
    }
    
    // Check delimiters
    for (int i = 0; delims[i] != 0; i++) {
      if (c == delims[i]) {
	in.unget();
	return in;
      }
    }
    
    // Allocate space if necessary and append char to string
    if (str.length() == str.capacity())
      str.reserve(str.capacity() > 0 ? str.capacity() * 2 : 1);
    str += c;
  }
  return in;
}

void
NowayLexiconReader::read(std::istream &in)
{
  m_word.reserve(128); // The size is not necessary, just for efficiency

  while (1) {
    // Read first word
    skip_while(in, " \t\n");
    get_until(in, m_word, " \t\n");
    if (in.bad())
      throw ReadError();
    if (!in)
      return;
    
    // Parse possible probability
    int left = m_word.find('(');
    int right = m_word.rfind(')');
    double prob = 1;
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
      skip_while(in, " \t");
      if (in.bad())
	throw ReadError();
      if (!in)
	break;
      if (in.peek() == '\n')
	break;
      get_until(in, m_phone, " \t\n");

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

    assert(node != m_lexicon.root());

    // Add word to lexicon

    // We duplicate word ends if two words have identical phoneme
    // sequences.  See the header file for more information.
    if (node->word_id >= 0) {
      node = new Lexicon::Node(*node);
      prev_node->next.push_back(node);
    }
    int word_id = m_vocabulary.add(m_word);
    node->word_id = word_id;
    node->log_prob = log(prob);
    m_lexicon.update_words(word_id);
  }
}
