#include <algorithm>

#include <cassert>

#include "Toolbox.hh"

Toolbox::Toolbox()
  : m_hmm_reader(),
    m_hmm_map(m_hmm_reader.hmm_map()),
    m_hmms(m_hmm_reader.hmms()),

    m_lexicon_reader(m_hmm_map, m_hmms),
    m_lexicon(m_lexicon_reader.lexicon()),
    m_vocabulary(m_lexicon_reader.vocabulary()),

    m_lna_reader(),

    m_ngram(),
    m_ngram_reader(),

    m_expander(m_hmms, m_lexicon, m_lna_reader),

    m_search(m_expander, m_vocabulary, m_ngram)
{
}

void
Toolbox::expand(int frame, int frames)
{ 
  m_expander.expand(frame, frames);
  m_expander.sort_words();
}

const std::string&
Toolbox::best_word()
{
  static const std::string noword("*");

  if (m_expander.words().size() > 0)
    return m_vocabulary.word(m_expander.words()[0]->word_id);
  else
    return noword;
}

void
Toolbox::add_history(int word)
{
  while (m_history.size() >= m_ngram.order())
    m_history.pop_front();

  m_history.push_back(word);
}

void
Toolbox::add_history_word(const std::string &word)
{
  add_history(m_vocabulary.index(word));
}

void
Toolbox::add_ngram_probs()
{
  // Not supported at the moment, because structure of Word has changed.
  assert(false);

//   for (int i = 0; i < m_best_words.size(); i++) {
//     m_history.push_back(m_best_words[i]->word_id);
//     m_best_words[i]->log_prob += 
//       m_ngram.log_prob(m_history.begin(), m_history.end()) * 
//       m_best_words[i]->frames;
//     m_best_words[i]->avg_log_prob = m_best_words[i]->log_prob / 
//       m_best_words[i]->frames;
//     m_history.pop_back();
//   }
//   std::sort(m_best_words.begin(), m_best_words.end(), Expander::WordCompare());
}

void
Toolbox::print_words(int words)
{
  m_expander.sort_words(words);
  const std::vector<Expander::Word*> &sorted_words = m_expander.words();

  if (words == 0 || words > sorted_words.size())
    words = sorted_words.size();

  for (int i = 0; i < words; i++) {
    std::cout << m_vocabulary.word(sorted_words[i]->word_id) << " "
	      << sorted_words[i]->best_length << " "
	      << sorted_words[i]->best_log_prob() << " "
	      << sorted_words[i]->best_avg_log_prob << " "
	      << std::endl;
  }
}

int
Toolbox::find_word(const std::string &word)
{
  const std::vector<Expander::Word*> &sorted_words = m_expander.words();
  for (int i = 0; i < sorted_words.size(); i++) {
    if (m_vocabulary.word(sorted_words[i]->word_id) == word)
      return i + 1;
  }
  return -1;
}

void
Toolbox::hmm_read(const char *file)
{
  std::ifstream in(file);
  if (!in)
    throw OpenError();
  m_hmm_reader.read(in);
}

void
Toolbox::lex_read(const char *file)
{
  std::ifstream in(file);
  if (!in)
    throw OpenError();
  m_lexicon_reader.read(in);
}

void
Toolbox::ngram_read(const char *file)
{
  FILE *fp = fopen(file, "r");
  if (!fp)
    throw OpenError();
  m_ngram_reader.read(fp, &m_ngram);
  fclose(fp);
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
  m_search.print_hypo(hypo);
}

bool
Toolbox::runto(int frame)
{
  while (frame > m_search.frame()) {
    bool ok = m_search.run();
    if (!ok)
      return false;
  }

  return true;
}

bool
Toolbox::recognize_segment(int start_frame, int end_frame)
{
  return m_search.recognize_segment(start_frame, end_frame);
}
