#ifndef SEARCH_HH
#define SEARCH_HH

#include <vector>

#include "Expander.hh"
#include "Vocabulary.hh"

class Search {
public:

  class Path {
  public:
    Path(int word_id, int frame, Path *prev) : 
      word_id(word_id), frame(frame), prev(prev) { }
    int word_id;
    int frame;
    Path *prev;
  };

  class Hypo {
  public:
    Hypo() : frame(0), log_prob(0), path(NULL) { }
    Hypo(int frame, double log_prob, Path *path) :
      frame(frame), log_prob(log_prob), path(path) { }
    bool operator<(const Hypo &h) { return log_prob > h.log_prob; }
    int frame;
    double log_prob;
    Path *path;
  };

  typedef std::vector<Hypo> HypoStack;

  Search(Expander &expander, Vocabulary &vocabulary, int frames);
  void run();
  void debug_print_hypo(Hypo &hypo);
  void debug_print_history(Hypo &hypo);

private:
  Expander &m_expander;
  Vocabulary &m_vocabulary;
  int m_frames;
  
  int m_first_stack;
  std::vector<HypoStack> m_stacks;
};

#endif /* SEARCH_HH */
