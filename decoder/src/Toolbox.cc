#include <cstddef>  // NULL
#include <algorithm>
#include <iostream>
#include <assert.h>
#include <errno.h>

#include "InterTreeGram.hh"
#include "Toolbox.hh"
#include "TreeGramArpaReader.hh"
#include "io.hh"
#include "misc/str.hh"
#include "HTKLatticeGrammar.hh"

using namespace std;

Toolbox::Toolbox(const char * hmm_path, const char * dur_path)
  : m_hmm_reader(NULL),
    m_hmm_map(NULL),
    m_hmms(NULL),

    m_tp_lexicon(NULL),
    m_tp_lexicon_reader(NULL),
    m_lexicon_read(false),
    m_tp_vocabulary(NULL),
    m_tp_search(NULL),

    m_acoustics(NULL),
    m_lna_reader(NULL),
    m_one_frame_acoustics(),
    m_fsa_lm(NULL),
    m_lookahead_ngram(NULL),

    m_last_guaranteed_history(NULL)
{
    hmm_read(hmm_path);
    if (dur_path != NULL) {
	duration_read(dur_path);
    }
    reinitialize_search();
}

Toolbox::~Toolbox()
{
  while (!m_ngrams.empty()) {
    delete m_ngrams.back();
    m_ngrams.pop_back();
  }
  if (m_lookahead_ngram) {
    delete m_lookahead_ngram;
  }

  if (m_fsa_lm) {
    delete m_fsa_lm;
  }

  if (m_tp_vocabulary) {
    delete m_tp_vocabulary;
  }

  if (m_lna_reader) {
    delete m_lna_reader;
  }

  if (m_tp_lexicon) {
    delete m_tp_lexicon;
  }

  if (m_tp_lexicon_reader) {
    delete m_tp_lexicon_reader;
  }

  if (m_tp_search) {
    delete m_tp_search;
  }

  if (m_hmm_reader) {
    delete m_hmm_reader;
  }
}

void
Toolbox::hmm_read(const char *file)
{
  std::ifstream in(file);
  if (!in)
    throw OpenError();
  if (m_hmm_reader) {
    delete m_hmm_reader;
  }
  m_hmm_reader = new NowayHmmReader();
  m_hmm_map = &(m_hmm_reader->hmm_map());
  m_hmms = &(m_hmm_reader->hmms());
  m_hmm_reader->read(in);
}

void Toolbox::duration_read(const char *dur_file)
{
  std::ifstream dur_in(dur_file);
  if (!dur_in)
    throw OpenError();
  m_hmm_reader->read_durations(dur_in);
}

void
Toolbox::reinitialize_search() {
  m_last_guaranteed_history = NULL;
  m_lexicon_read = false;

  if (m_tp_vocabulary) {
    delete m_tp_vocabulary;
  }

  m_tp_vocabulary = new Vocabulary();
  if (m_lna_reader) {
    delete m_lna_reader;
  }
  m_lna_reader = new LnaReaderCircular;

  if (m_tp_lexicon) {
    delete m_tp_lexicon;
  }
  m_tp_lexicon = new TPLexPrefixTree(*m_hmm_map, *m_hmms);

  if (m_tp_lexicon_reader) {
    delete m_tp_lexicon_reader;
  }
  m_tp_lexicon_reader = new TPNowayLexReader(*m_hmm_map, *m_hmms, *m_tp_lexicon, *m_tp_vocabulary);

  if (m_tp_search) {
    delete m_tp_search;
  }
  m_tp_search = new TokenPassSearch(*m_tp_lexicon, *m_tp_vocabulary, m_lna_reader);
}


void
Toolbox::lex_read(const char *filename)
{
  if (!m_tp_search) {
    reinitialize_search();
  }

  FILE *file = fopen(filename, "r");
  if (!file)
    throw OpenError();
  m_tp_lexicon_reader->read(file, m_word_boundary);
  if (!m_word_boundary.empty()) {
    m_tp_search->set_word_boundary(m_word_boundary);
  }
  fclose(file);
  m_lexicon_read = true;
}


void
Toolbox::interpolated_ngram_read(const std::vector<std::string> lmnames, 
                                 const std::vector<float> weights) {

  // Loading binary models doesn't work yet !
  InterTreeGram *itg = new InterTreeGram(lmnames, weights);
  m_tp_search->set_ngram(itg);
  m_ngrams.push_back(itg);
}


int
Toolbox::ngram_read(const char *file, const bool binary, bool quiet)
{
  io::Stream in(file,"r");
  
  if (!in.file) {
    throw OpenError();
  }

  if (m_ngrams.size() > 0) {
    if (m_ngrams.size() > 1) {
      fprintf(stderr, "Trying to load more than one ngram (%lu). You need to use interploated_ngram_read() instead of ngram_read(). Exit.\n", m_ngrams.size());
      exit(-1);
    }
    fprintf(stderr, "Replacing the current n-gram model\n");
    delete m_ngrams[0];
    m_ngrams.clear();
  }

  m_ngrams.push_back(new TreeGram());
  m_ngrams.back()->read(in.file, binary);

  int num_oolm = 0;
  num_oolm = m_tp_search->set_ngram(m_ngrams.back());

  if ((num_oolm > 0) && !quiet) {
    cerr << num_oolm << " words in the vocabulary were not found in the LM." << endl;
  }

  return m_ngrams.back()->order();
}

void
Toolbox::htk_lattice_grammar_read(const char *file, bool quiet)
{
  io::Stream in(file,"r");
  
  if (!in.file) {
    //throw OpenError(); // FIXME
    fprintf(stderr, "htk_lattice_grammar_read(): could not open %s: %s\n", 
            file, strerror(errno));
    exit(1);
  }
  if (m_ngrams.size() > 0) {
    if (!quiet)
      fprintf(stderr, "Replacing old grammar.\n");
    delete m_ngrams[0];
    m_ngrams.clear();
  }

  m_ngrams.push_back(new HTKLatticeGrammar());
  m_ngrams.back()->read(in.file, false);

  int num_oolm = m_tp_search->set_ngram(m_ngrams.back());
  if ((num_oolm > 0) && !quiet) {
    cerr << num_oolm << " words in the vocabulary were not found in the LM." << endl;
  }
}

void
Toolbox::fsa_lm_read(const char *file, bool bin, bool quiet)
{
  io::Stream in(file, "r");
  assert(in.file);
  if (m_fsa_lm)
    delete m_fsa_lm;
  m_fsa_lm = new fsalm::LM();
  if (bin)
    m_fsa_lm->read(in.file);
  else {
    m_fsa_lm->read_arpa(in.file, true);
    m_fsa_lm->trim();
  }

  int num_oolm = m_tp_search->set_fsa_lm(m_fsa_lm);

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
      num_oolm = m_tp_search->set_lookahead_ngram(m_ngrams.back());
  }
  else
  {
    io::Stream in(file,"r");
    if (!in.file) {
        throw OpenError();
    }
    if (m_lookahead_ngram) {
      delete m_lookahead_ngram;
    }
    m_lookahead_ngram = new TreeGram();
    m_lookahead_ngram->read(in.file, binary);
    assert(m_lookahead_ngram->get_type()==TreeGram::BACKOFF);
    num_oolm = m_tp_search->set_lookahead_ngram(m_lookahead_ngram);
  }

  if ((num_oolm > 0) && !quiet) {
    cerr << num_oolm << " words in the vocabulary were not found in the lookahead LM." << endl;
  }
}

void Toolbox::interpolated_lookahead_ngram_read(const std::vector<std::string> lmnames, const std::vector<float> weights) {
  if (m_lookahead_ngram) {
    delete m_lookahead_ngram;
  }
  //FIXME: Not checking that the type is BACKOFF
  m_lookahead_ngram = new InterTreeGram(lmnames, weights);
  m_tp_search->set_lookahead_ngram(m_lookahead_ngram);;
}

void
Toolbox::read_word_classes(const char *file)
{
#ifdef ENABLE_WORDCLASS_SUPPORT
  ifstream ifs(file);
  m_word_classes.read(ifs, *m_tp_vocabulary);
  m_tp_search->set_word_classes(&m_word_classes);
#endif
}

void
Toolbox::lna_open(const char *file, int size)
{
  m_lna_reader->open_file(file, size);
  m_acoustics = m_lna_reader;
}

void
Toolbox::lna_open_fd(const int fd, int size)
{
  m_lna_reader->open_fd(fd, size);
  m_acoustics = m_lna_reader;
}

void
Toolbox::lna_close()
{
  m_lna_reader->close();
}

const std::vector<timed_token_type> &
Toolbox::best_timed_hypo_string(bool print_all)
{
  if (m_tp_search->get_print_text_result()) {
    fprintf(stderr, "Toolbox::best_timed_hypo_string() should not be used with "
            "print_text_results set true\n");
    throw logic_error("Toolbox::best_timed_hypo_string");
  }

  HistoryVector hist_vec;
  if (print_all)
    m_tp_search->get_best_final_token().get_lm_history(hist_vec, NULL);
  else
    m_tp_search->get_first_token().get_lm_history(hist_vec, m_last_guaranteed_history);

  static std::vector<timed_token_type> retval;
  retval.clear();
  bool all_guaranteed = true;

  for (auto hist : hist_vec) {
    std::string newstring("");
    assert(hist->reference_count > 0);
    if (hist->previous->reference_count == 1) {
      if (all_guaranteed)
        m_last_guaranteed_history = hist;
    }
    else {
      if (all_guaranteed)
        newstring = "* ";
      all_guaranteed = false;
    }

    newstring += word(hist->last().word_id());
    retval.push_back(timed_token_type(newstring,pair<double,double>(hist->word_start_frame,hist->word_first_silence_frame)));
  }
  return retval;
}
const bytestype& Toolbox::best_hypo_string(bool print_all, bool output_time)
{
  if (m_tp_search->get_print_text_result()) {
    fprintf(stderr, "Toolbox::best_hypo_string() should not be used with "
            "print_text_results set true\n");
    throw logic_error("Toolbox::best_hypo_string");
  }

  HistoryVector hist_vec;
  if (print_all)
    m_tp_search->get_best_final_token().get_lm_history(hist_vec, NULL);
  else
    m_tp_search->get_first_token().get_lm_history(hist_vec, m_last_guaranteed_history);

  static std::string retval;
  retval.clear();
  bool all_guaranteed = true;

  for (auto hist : hist_vec) {
    assert(hist->reference_count > 0);
    if (hist->previous->reference_count == 1) {
      if (all_guaranteed)
        m_last_guaranteed_history = hist;
    }
    else {
      if (all_guaranteed)
        retval += "* ";
      all_guaranteed = false;
    }

    if (! output_time)
      retval += word(hist->last().word_id()) + " ";
    else
      retval += (str::fmt(256, "<time=%d> ", hist->word_start_frame) +
                 word(hist->last().word_id()) + " ");
  }
  // Last of the timing information
  //if (output_time) 
  //  retval = retval + str::fmt(256, "%d", frame());
  return retval;
}

void Toolbox::set_word_boundary(const std::string & word)
{
  // set_word_boundary() has no effect after the language model has been read.
  // Calling them in wrong order will result in confusing errors later.
  if ((m_ngrams.size() > 0) || m_lexicon_read) {
    cerr << "Warning, set_word_boundary() has to be called before reading language model or lexicon." << endl;
  }
  m_word_boundary = word;
}
