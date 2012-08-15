#include <cstddef>  // NULL
#include <algorithm>
#include <iostream>
#include <assert.h>
#include <errno.h>

#include "Toolbox.hh"
#include "TreeGramArpaReader.hh"
#include "io.hh"

using namespace std;

Toolbox::Toolbox()
  : m_use_stack_decoder(0),

    m_hmm_reader(),
    m_hmm_map(m_hmm_reader.hmm_map()),
    m_hmms(m_hmm_reader.hmms()),

    m_lexicon_reader(m_hmm_map, m_hmms),
    m_lexicon(m_lexicon_reader.lexicon()),
    m_vocabulary(m_lexicon_reader.vocabulary()),
    m_tp_lexicon(m_hmm_map, m_hmms),
    m_tp_lexicon_reader(m_hmm_map, m_hmms, m_tp_lexicon, m_tp_vocabulary),
    m_tp_search(m_tp_lexicon, m_tp_vocabulary, &m_lna_reader),

    m_acoustics(NULL),
    m_lna_reader(),
    m_one_frame_acoustics(),
    m_fsa_lm(NULL),

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
  if (m_use_stack_decoder)
    m_lexicon_reader.read(file);
  else
  {
    m_tp_lexicon_reader.read(file, m_word_boundary);
    m_tp_search.set_word_boundary(m_word_boundary);
  }
  fclose(file);
}

const std::string & Toolbox::lex_word() const
{
	if (m_use_stack_decoder)
		return m_lexicon_reader.word();
	else
		return m_tp_lexicon_reader.word();
}

const std::string & Toolbox::lex_phone() const
{
	if (m_use_stack_decoder)
		return m_lexicon_reader.phone();
	else
		return m_tp_lexicon_reader.phone();
}

int
Toolbox::ngram_read(const char *file, float weight, const bool binary, bool quiet)
{
  io::Stream in(file,"r");
  
  if (!in.file) {
    throw OpenError();
  }

  if (!m_use_stack_decoder && (m_ngrams.size() > 0)) {
    delete m_ngrams[0];
    m_ngrams.clear();
  }

  m_ngrams.push_back(new TreeGram());
  m_ngrams.back()->read(in.file, binary);

  int num_oolm = 0;
  if (m_use_stack_decoder) {
    num_oolm = m_search.add_ngram(m_ngrams.back(), weight);
  }
  else {
    num_oolm = m_tp_search.set_ngram(m_ngrams.back());
  }

  if ((num_oolm > 0) && !quiet) {
    cerr << num_oolm << " words in the vocabulary were not found in the LM." << endl;
  }

  return m_ngrams.back()->order();
}

void
Toolbox::fsa_lm_read(const char *file, bool bin, bool quiet)
{
  io::Stream in(file, "r");
  assert(in.file);
  assert(!m_use_stack_decoder);
  if (m_fsa_lm)
    delete m_fsa_lm;
  m_fsa_lm = new fsalm::LM();
  if (bin)
    m_fsa_lm->read(in.file);
  else {
    m_fsa_lm->read_arpa(in.file, true);
    m_fsa_lm->trim();
  }

  int num_oolm = m_tp_search.set_fsa_lm(m_fsa_lm);

  if ((num_oolm > 0) && !quiet) {
    cerr << num_oolm << " words in the vocabulary were not found in the FSA LM." << endl;
  }
}

void
Toolbox::read_lookahead_ngram(const char *file, const bool binary, bool quiet)
{
  int num_oolm = 0;

  if (strlen(file) == 0)
  {
    if (m_ngrams.size() > 0)
      num_oolm = m_tp_search.set_lookahead_ngram(m_ngrams.back());
  }
  else
  {
    io::Stream in(file,"r");
    if (!in.file) {
        throw OpenError();
    }
    m_lookahead_ngram = new TreeGram();
    m_lookahead_ngram->read(in.file, binary);
    assert(m_lookahead_ngram->get_type()==TreeGram::BACKOFF);
    num_oolm = m_tp_search.set_lookahead_ngram(m_lookahead_ngram);
  }

  if ((num_oolm > 0) && !quiet) {
    cerr << num_oolm << " words in the vocabulary were not found in the lookahead LM." << endl;
  }
}

void
Toolbox::read_word_classes(const char *file, bool quiet)
{
  assert(!m_use_stack_decoder);

  ifstream ifs(file);
  m_word_classes.read(ifs, m_tp_vocabulary);
  m_tp_search.set_word_classes(&m_word_classes);

  int num_oovs = m_word_classes.num_oovs();
  if ((num_oovs > 0) && !quiet) {
    cerr << num_oovs << " words in the class definitions were not found in the vocabulary." << endl;
  }
}

void
Toolbox::lna_open(const char *file, int size)
{
  m_lna_reader.open(file, size);
  m_acoustics = &m_lna_reader;
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
  if (m_lna_reader.num_models() != m_hmm_reader.num_models()) {
    cerr << "WARNING: " << m_lna_reader.num_models() << " states in LNA, but "
    		<< m_hmm_reader.num_models() << " states in HMMs" << endl;
  }

  m_search.init_search(expand_window);
}
