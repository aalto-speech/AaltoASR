#ifndef SEARCH_HH
#define SEARCH_HH

#include <vector>

#include "Ngram.hh"
#include "Expander.hh"
#include "Vocabulary.hh"

class Search {
public:

  class Path {
  public:
    inline Path(int word_id, int frame, Path *prev);
    inline ~Path();
    inline void link() { m_reference_count++; }
    inline static void unlink(Path *path);
    int word_id;
    int frame;
    Path *prev;
    static int count;
  private:
    int m_reference_count;
  };

  class Hypo {
  public:
    inline Hypo();
    inline Hypo(int frame, double log_prob, Path *path);
    inline Hypo(const Hypo &t);
    inline Hypo &operator=(const Hypo &hypo);
    inline ~Hypo();
    inline void add_path(int word_id, int frame);
    bool operator<(const Hypo &h) const { return log_prob > h.log_prob; }
    int frame;
    double log_prob;
    Path *path;
  };

  typedef std::vector<Hypo> HypoStack;

  Search(Expander &expander, const Vocabulary &vocabulary, 
	 const Ngram &ngram, int frames);
  void run();
  void debug_print_hypo(Hypo &hypo);
  void debug_print_history(Hypo &hypo);

  void set_lm_scale(double lm_scale) { m_lm_scale = lm_scale; }

private:
  Expander &m_expander;
  const Vocabulary &m_vocabulary;
  const Ngram &m_ngram;
  int m_frames;
  
  int m_first_stack;
  std::vector<HypoStack> m_stacks;

  // Options
  double m_lm_scale;
};

Search::Path::Path(int word_id, int frame, Path *prev)
  : word_id(word_id), frame(frame), prev(prev), m_reference_count(0)
{
  if (prev)
    prev->link();
  count++;
}

Search::Path::~Path()
{
  count--;
}

void
Search::Path::unlink(Path *path)
{
  while (path->m_reference_count == 1) {
    Path *prev = path->prev;
    delete path;
    path = prev;
    if (!path)
      return;
  }
  path->m_reference_count--;
}

Search::Hypo::Hypo()
  : frame(0), log_prob(0), path(NULL)
{
}

Search::Hypo::Hypo(int frame, double log_prob, Path *path)
 : frame(frame), log_prob(log_prob), path(path)
{
  if (path)
    path->link();
}

Search::Hypo::Hypo(const Hypo &h)
  : frame(h.frame),
    log_prob(h.log_prob),
    path(h.path)
{
  if (path)
    path->link();
}

Search::Hypo&
Search::Hypo::operator=(const Hypo &h)
{
  if (this == &h)
    return *this;

  Path *old_path = path;

  frame = h.frame;
  log_prob = h.log_prob;
  path = h.path;

  if (path)
    path->link();

//    if (old_path)
//      Path::unlink(old_path);

  return *this;
}

Search::Hypo::~Hypo()
{
  if (path)
    Path::unlink(path);
}

void
Search::Hypo::add_path(int word_id, int frame)
{
  Path *old_path = path;
  path = new Path(word_id, frame, path);
  path->link();
  if (old_path)
    Path::unlink(old_path);
}

#endif /* SEARCH_HH */
