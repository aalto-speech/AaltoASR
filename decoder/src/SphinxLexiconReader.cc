#include <iostream> // DEBUG
#include <sstream>

#include <assert.h>
#include <math.h>

#include "NowayLexiconReader.hh"

NowayLexiconReader::NowayLexiconReader()
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
get_until(std::istream &in, std::string &str, const char *delims)
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
    
    get_until(in, m_phone, "\n");
    int word_id = m_vocabulary.add(m_word);
  }
}
