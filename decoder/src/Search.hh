#ifndef SEARCH_HH
#define SEARCH_HH

#include <cstddef>  // NULL
#include <vector>
#include <deque>

#include <float.h>

#include "TreeGram.hh"
#include "Expander.hh"
#include "Vocabulary.hh"

class HypoPath {
public:
  inline HypoPath(int word_id, int frame, HypoPath *prev);
  inline ~HypoPath();
  inline void link() { m_reference_count++; }
  inline static void unlink(HypoPath *path);
  inline int count() const { return m_reference_count; }
  inline bool guard() { return prev == NULL; }
  int word_id;
  int frame;
  HypoPath *prev;
  float lm_log_prob;
  float ac_log_prob;
  static int g_count;
private:
  int m_reference_count;
};

HypoPath::HypoPath(int word_id, int frame, HypoPath *prev)
  : word_id(word_id), frame(frame), prev(prev),
    lm_log_prob(0), ac_log_prob(0), m_reference_count(0)
{
  if (prev)
    prev->link();
  g_count++;
}

HypoPath::~HypoPath()
{
  g_count--;
}

void
HypoPath::unlink(HypoPath *path)
{
  if (path == NULL)
    return;
  while (path->m_reference_count == 1) {
    HypoPath *prev = path->prev;
    delete path;
    path = prev;
    if (!path)
      return;
  }
  path->m_reference_count--;
}

class Hypo {
public:
  inline Hypo();
  inline Hypo(int frame, float log_prob, HypoPath *path);
  inline Hypo(const Hypo &t);
  inline Hypo &operator=(const Hypo &hypo);
  inline ~Hypo();
  inline void add_path(int word_id, int frame);
  bool operator<(const Hypo &h) const { return log_prob > h.log_prob; }

  int frame;
  float log_prob;
  HypoPath *path;
};


Hypo::Hypo()
  : frame(0), log_prob(0), path(NULL)
{
}

Hypo::Hypo(int frame, float log_prob, HypoPath *path)
 : frame(frame), log_prob(log_prob), path(path)
{
  if (path)
    path->link();
}

Hypo::Hypo(const Hypo &h)
  : frame(h.frame),
    log_prob(h.log_prob),
    path(h.path)
{
  if (path)
    path->link();
}

Hypo&
Hypo::operator=(const Hypo &h)
{
  if (this == &h)
    return *this;

  HypoPath *old_path = path;

  frame = h.frame;
  log_prob = h.log_prob;
  path = h.path;

  if (path)
    path->link();

  if (old_path)
    HypoPath::unlink(old_path);

  return *this;
}

Hypo::~Hypo()
{
  if (path)
    HypoPath::unlink(path);
}

void
Hypo::add_path(int word_id, int frame)
{
  HypoPath *old_path = path;
  path = new HypoPath(word_id, frame, path);
  path->link();
  if (old_path)
    HypoPath::unlink(old_path);
}

class HypoStack : private std::vector<Hypo> {
public:
  inline HypoStack() : m_best_log_prob(-1e10) { }

  // Inherited from vector
  Hypo &operator[](int index) { return std::vector<Hypo>::operator[](index); }
  Hypo &at(int index) { return std::vector<Hypo>::operator[](index); }
  Hypo &front() { return std::vector<Hypo>::front(); }
  Hypo &back() { return std::vector<Hypo>::back(); }
  void clear() { std::vector<Hypo>::clear(); }
  void reserve(int size) { std::vector<Hypo>::reserve(size); }
  size_type size() const { return std::vector<Hypo>::size(); }
  bool empty() const { return std::vector<Hypo>::empty(); }
  void pop_back() { std::vector<Hypo>::pop_back(); }
  void remove(int index) { std::vector<Hypo>::erase(begin() + index); }

  // New functions

  /// \brief Returns the first (and should be the only) hypothesis that has at
  /// least the first \a words words in common with \a hypo.
  ///
  int find_similar(const Hypo &hypo, int words);

  void sorted_insert(const Hypo &hypo);

private:
  float m_best_log_prob;
  int m_best_index;
};

class Search {
public:

  Search(Expander &expander, const Vocabulary &vocabulary);

  // Debug and print
  void print_prunings();
  void print_path(HypoPath *path);
  void print_hypo(const Hypo &hypo);
  void print_sure();

  // Operate
  int add_ngram(NGram *ngram, float weight);
  void reset_search(int start_frame);
  void init_search(int expand_window);
  bool expand_stack(int frame);
  void expand_words(int frame, const std::string &words);
  void go(int frame);
  bool run();
  bool recognize_segment(int start_frame, int end_frame);

  // Info
  int frame() const { return m_frame; }
  int first_frame() const { return m_first_frame; }
  int last_frame() const { return m_last_frame; }
  HypoStack &stack(int frame);
  const HypoStack &stack(int frame) const;

  // Options
  void set_end_frame(int end_frame) { m_end_frame = end_frame; }

  void set_hypo_limit(int hypo_limit) { m_hypo_limit = hypo_limit; }
  void set_word_limit(int word_limit) { m_word_limit = word_limit; }
  void set_word_beam(float word_beam) { m_word_beam = word_beam; }
  void set_lm_scale(float lm_scale) { m_lm_scale = lm_scale; }
  void set_lm_offset(float lm_offset) { m_lm_offset = lm_offset; }
  void set_unk_offset(float unk_offset) { m_unk_offset = unk_offset; }

  /// \brief Sets how many words in the word histories of two hypotheses have to
  /// match for the hypotheses to be considered similar (and only the better to
  /// be saved).
  ///
  void set_prune_similar(int prune_similar) 
  { m_prune_similar = prune_similar; }

  void set_multiple_endings(int multiple_endings) 
  { m_multiple_endings = multiple_endings; }
  void set_hypo_beam(float hypo_beam) { m_hypo_beam = hypo_beam; }
  void set_global_beam(float beam) { m_global_beam = beam; }
  void set_verbose(int verbose) { m_verbose = verbose; }
  void set_print_probs(bool print_probs) { m_print_probs = print_probs; }
  void set_print_indices(bool print_indices)
  {
    m_print_indices = print_indices;
  }
  void set_print_frames(bool print_frames) { m_print_frames = print_frames; }
  void set_word_boundary(const std::string &word);
  void set_dummy_word_boundaries(bool value) { m_dummy_word_boundaries = value; }

  // Exceptions
  struct InvalidFrame : public std::exception {
    virtual const char *what() const throw()
      { return "Search: invalid frame"; }
  };

private:
  void check_stacks();
  void ensure_stack(int frame);
  float compute_lm_log_prob(const Hypo &hypo);
  void update_global_pruning(int frame, float log_prob);

  /// Won't insert the hypothesis, if there already is a better hypothesis that
  /// has at least \a m_prune_similar words in common.
  ///
  void insert_hypo(int target_frame, const Hypo &hypo);

  void expand_hypo_with_word(const Hypo &hypo, int word, int target_frame, 
			     float ac_log_prob);
  void expand_hypo(const Hypo &hypo);
  void find_best_words(int frame);
  void initial_prunings(int frame, HypoStack &stack);
  int frame2stack(int frame) const;

  Expander &m_expander;
  const Vocabulary &m_vocabulary; // Words in lexicon (not the words in lm)

  struct LanguageModel {
    LanguageModel() : ngram(NULL), weight(0) {}
    NGram *ngram;
    //TreeGram *ngram;
    float weight;
    std::vector<int> lex2lm;
  };
  std::vector<LanguageModel> m_ngrams;
  int m_max_lm_order;
  
  // State
  int m_frame;

  // Stacks
  int m_first_frame;	
  int m_last_frame;
  int m_first_stack;
  std::vector<HypoStack> m_stacks;
  int m_last_hypo_frame;

  // Options

  /**
   * Size of the window (in frames) used in Viterbi expansion.
   **/
  int m_expand_window;	
  int m_end_frame;
  float m_lm_scale;
  float m_lm_offset;
  float m_unk_offset;
  int m_verbose;
  bool m_print_probs;
  int m_multiple_endings;
  bool m_print_indices;
  bool m_print_frames;
  HypoPath *m_last_printed_path;

  /** Index of the word boundary in LM context (negative if not used)
   *
   * Currently, when a hypothesis is expanded with word W, it is also
   * expanded with a word bounadry and W. (see m_dummy_word_boundaries)
   *
   * Also subsequent word boundary words are combined into a single
   * word boundary word, because language model does not have doubles.
   **/
  int m_word_boundary;

  /**
   * Currently, when a hypothesis is expanded with word W, it is also
   * expanded with a word bounadry and W. (on by default)
   **/
  int m_dummy_word_boundaries;

  // Pruning options
  int m_word_limit;	// How many best words are expanded
  float m_word_beam;   // Do not expand words outside this beam
  int m_prune_similar;  // Prune similar N-word endings 
  int m_hypo_limit;	// How many best hypos in a stack are expanded
  float m_hypo_beam;

  // Global pruning
  float m_global_best;
  float m_global_beam;
  int m_global_frame;

  // Pruning statistics
  int m_stack_expansions;
  int m_hypo_insertions;
  int m_limit_prunings;
  int m_beam_prunings;
  int m_similar_prunings;

  // Temporary variables
  NGram::Gram m_history_lex;
  NGram::Gram m_history_lm;
};

#endif /* SEARCH_HH */
