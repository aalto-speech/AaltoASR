#include "Toolbox.hh"

Toolbox::Toolbox()
  : m_hmm_reader(),
    m_hmm_map(m_hmm_reader.hmm_map()),
    m_hmms(m_hmm_reader.hmms()),

    m_lexicon_reader(m_hmm_map, m_hmms),
    m_lexicon(m_lexicon_reader.lexicon()),
    m_vocabulary(m_lexicon_reader.vocabulary()),

    m_lna_reader(),

    m_ngram_reader(m_vocabulary),
    m_ngram(m_ngram_reader.ngram()),

    m_expander(m_hmms, m_lexicon, m_lna_reader),
    m_search(m_expander, m_vocabulary, m_ngram)
{
}

void
Toolbox::hmm_read(const char *file)
{
  std::ifstream in(file);
  m_hmm_reader.read(in);
}

void
Toolbox::lex_read(const char *file)
{
  std::ifstream in(file);
  m_lexicon_reader.read(in);
}

void
Toolbox::ngram_read(const char *file)
{
  std::ifstream in(file);
  m_ngram_reader.read(in);
}

void
Toolbox::lna_open(const char *file, int models, int size)
{
  m_lna_reader.open(file, models, size);
}

void
Toolbox::lna_close()
{
  m_lna_reader.close();
}

void
Toolbox::print_hypo(Hypo &hypo)
{
  m_search.debug_print_hypo(hypo);
}
