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

    m_ngram_reader(),
    m_ngram(m_ngram_reader.ngram()),

    m_expander(m_hmms, m_lexicon, m_lna_reader),
    m_best_words(m_expander.words()),

    m_search(m_expander, m_vocabulary, m_ngram)
{
}

void
Toolbox::expand(int frame, int frames)
{ 
  m_expander.expand(frame, frames);
  std::sort(m_best_words.begin(), m_best_words.end(), Expander::WordCompare());
}

const std::string&
Toolbox::best_word()
{
  static const std::string noword("*");

  if (m_best_words.size() > 0)
    return m_vocabulary.word(m_best_words[0]->word_id);
  else
    return noword;
}

int
Toolbox::best_index()
{
  if (m_best_words.size() > 0)
    return m_best_words[0]->word_id;
  else
    return -1;
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
  for (int i = 0; i < m_best_words.size(); i++) {
    m_history.push_back(m_best_words[i]->word_id);
    m_best_words[i]->log_prob = 0; // REMOVE THIS DEBUG!!
    m_best_words[i]->log_prob += 
      m_ngram.log_prob(m_history.begin(), m_history.end()) * 
      m_best_words[i]->frames;
    m_best_words[i]->avg_log_prob = m_best_words[i]->log_prob / 
      m_best_words[i]->frames;
    m_history.pop_back();
  }
  std::sort(m_best_words.begin(), m_best_words.end(), Expander::WordCompare());
}

void
Toolbox::print_words(int words)
{
  std::vector<Expander::Word*> &sorted_words = m_expander.words();
  std::sort(sorted_words.begin(), sorted_words.end(), Expander::WordCompare());

  if (words == 0 || words > sorted_words.size())
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

int
Toolbox::find_word(const std::string &word)
{
  std::vector<Expander::Word*> &sorted_words = m_expander.words();
  for (int i = 0; i < sorted_words.size(); i++) {
    if (m_vocabulary.word(sorted_words[i]->word_id) == word)
      return i + 1;
  }
  return -1;
}

void
Toolbox::prune(int frame, int top)
{
  this->stack(frame).prune(top);
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
  std::ifstream in(file);
  if (!in)
    throw OpenError();
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
