#include <algorithm>
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
Toolbox::print_words(int words)
{
  std::vector<Expander::Word*> &sorted_words = m_expander.words();
  std::sort(sorted_words.begin(), sorted_words.end(), Expander::WordCompare());

  if (words == 0 && words > sorted_words.size())
    words = sorted_words.size();

  std::cout.setf(std::cout.fixed, std::cout.floatfield);
  std::cout.setf(std::cout.right, std::cout.adjustfield);
  std::cout.precision(2);
  for (int i = 0; i < words; i++) {
    std::cout << sorted_words[i]->frames << "\t"
	      << sorted_words[i]->log_prob << "\t"
	      << sorted_words[i]->avg_log_prob << "\t"
	      << m_vocabulary.word(sorted_words[i]->word_id)
	      << std::endl;
  }
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
