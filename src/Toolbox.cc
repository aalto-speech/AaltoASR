#include <algorithm>
#include <assert.h>
#include <errno.h>

#include "Toolbox.hh"

Toolbox::Toolbox()
  : m_hmm_reader(),
    m_hmm_map(m_hmm_reader.hmm_map()),
    m_hmms(m_hmm_reader.hmms()),

    m_lexicon_reader(m_hmm_map, m_hmms),
    m_lexicon(m_lexicon_reader.lexicon()),
    m_vocabulary(m_lexicon_reader.vocabulary()),

    m_lna_reader(),

    m_expander(m_hmms, m_lexicon, m_lna_reader),

    m_search(m_expander, m_vocabulary)
{
}

Toolbox::~Toolbox()
{
  while (!m_ngrams.empty()) {
    delete m_ngrams.back();
    m_ngrams.pop_back();
  }
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

void Toolbox::duration_read(const char *dur_file)
{
  std::ifstream dur_in(dur_file);
  if (!dur_in)
    throw OpenError();
  m_hmm_reader.read_durations(dur_in);
  m_expander.set_post_durations(true);
}

void
Toolbox::lex_read(const char *filename)
{
  FILE *file = fopen(filename, "r");
  if (!file)
    throw OpenError();
  m_lexicon_reader.read(file);
  fclose(file);
}

void
Toolbox::ngram_read(const char *file, float weight)
{
  FILE *f = fopen(file, "r");
  if (!f) {
    fprintf(stderr, "ngram_read(): could not open %s: %s\n", 
	    file, strerror(errno));
    exit(1);
  }

  m_ngrams.push_back(new TreeGram());
  m_ngrams.back()->read(f);
  m_search.add_ngram(m_ngrams.back(), weight);

  fclose(f);
}

void
Toolbox::lna_open(const char *file, int size)
{
  m_lna_reader.open(file, size);
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

void
Toolbox::segment(const std::string &str, int start_frame, int end_frame)
{
//   // Create lexicon
//   Lexicon lex;
//   Lexicon::Node *node = lex.root_node();
//   istringstream in(str);
//   std::string label;
//   while (in >> label) {
//     int hmm_id = m_hmm_map[label];
//     Hmm &hmm = m_hmms[hmm_id];
//     Lexicon::Node *next = new Lexicon::Node(hmm.states.size());
//     next->hmm_id = hmm_id;
//     node->next.push_back(next);
//     node = next;
//   }

//   // Expand
//   Expander e(m_hmms, lex, m_acoustics);
//   e.set_forced_end(true);
//   e.expand(start_frame, end_frame - start_frame);
}

void
Toolbox::init(int expand_window)
{
  if (m_lna_reader.num_models() != m_hmm_reader.num_models())
    fprintf(stderr, "WARNING: %d states in LNA, but %d states in HMMs\n",
	    m_lna_reader.num_models(), m_hmm_reader.num_models());

  m_search.init_search(expand_window);
}
