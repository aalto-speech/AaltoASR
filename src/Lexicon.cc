#include "Lexicon.hh"

int Lexicon::Path::count = 0;

Lexicon::Lexicon()
  : m_words(0), m_root_node(1)
{
}
