#ifndef SEARCH_HH
#define SEARCH_HH

#include <vector>
#include <deque>

#include "Ngram.hh"
#include "Expander.hh"
#include "Vocabulary.hh"

class HypoPath {
public:
  inline HypoPath(int word_id, int frame, HypoPath *prev);
  inline ~HypoPath();
  inline void link() { m_reference_count++; }
  inline static void unlink(HypoPath *path);
  int word_id;
  int frame;
  HypoPath *prev;
  static int count;
private:
  int m_reference_count;
};

class Hypo {
public:
  inline Hypo();
  inline Hypo(int frame, double log_prob, HypoPath *path);
  inline Hypo(const Hypo &t);
  inline Hypo &operator=(const Hypo &hypo);
  inline ~Hypo();
  inline void add_path(int word_id, int frame);
  bool operator<(const Hypo &h) const { return log_prob > h.log_prob; }
  int frame;
  double log_prob;
  HypoPath *path;
};

typedef std::vector<Hypo> HypoStack;

class Search {
public:

  Search(Expander &expander, const Vocabulary &vocabulary, 
	 const Ngram &ngram);

  // Debug
  void debug_print_hypo(Hypo &hypo);
  void debug_print_history(Hypo &hypo);

  // Operate
  void init_search(int frames, int hypos);
  bool expand_stack(int frame);
  void run();

  // Info
  inline int earliest_frame() const { return m_earliest_frame; }
  inline int last_frame() const { return m_earliest_frame + m_frames; }
  int frame2stack(int frame) const;
  inline HypoStack &stack(int frame) { return m_stacks[frame2stack(frame)]; }
  inline const HypoStack &stack(int frame) const
    { return m_stacks[frame2stack(frame)]; }

  // Options
  void set_hypo_limit(int hypo_limit) { m_hypo_limit = hypo_limit; }
  void set_word_limit(int word_limit) { m_word_limit = word_limit; }
  void set_lm_scale(double lm_scale) { m_lm_scale = lm_scale; }

  // Exceptions
  struct ForgottenFrame : public std::exception {
    virtual const char *what() const throw()
      { return "Search: forgotten frame"; }
  };

  struct FutureFrame : public std::exception {
    virtual const char *what() const throw()
      { return "Search: future frame"; }
  };

private:
  Expander &m_expander;
  const Vocabulary &m_vocabulary;
  const Ngram &m_ngram;

  // State
  int m_earliest_frame;	// Earliest frame
  int m_earliest_stack;	// Earliest stack in the circular buffer
  std::vector<HypoStack> m_stacks;

  // Options
  int m_frames;		// Size of circular frame buffer
  double m_lm_scale;
  int m_word_limit;	// How many best words are expanded
  int m_hypo_limit;	// How many best hypos in a stack are expanded

  // Temporary variables
  std::deque<int> m_history;
};

HypoPath::HypoPath(int word_id, int frame, HypoPath *prev)
  : word_id(word_id), frame(frame), prev(prev), m_reference_count(0)
{
  if (prev)
    prev->link();
  count++;
}

HypoPath::~HypoPath()
{
  count--;
}

void
HypoPath::unlink(HypoPath *path)
{
  while (path->m_reference_count == 1) {
    HypoPath *prev = path->prev;
    delete path;
    path = prev;
    if (!path)
      return;
  }
  path->m_reference_count--;
}

Hypo::Hypo()
  : frame(0), log_prob(0), path(NULL)
{
}

Hypo::Hypo(int frame, double log_prob, HypoPath *path)
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

#endif /* SEARCH_HH */
