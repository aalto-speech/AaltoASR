#include <iomanip>

#include <math.h>
#include "Search.hh"

Search::Search(Expander &expander, Vocabulary &vocabulary, int frames)
  : m_expander(expander),
    m_vocabulary(vocabulary),
    m_frames(frames),
    m_first_stack(0),
    m_stacks(frames)
{
}

void
Search::debug_print_hypo(Hypo &hypo)
{
  static std::vector<Search::Path*> debug_paths(128);

  Path *path = hypo.path;
  std::cout.setf(std::cout.fixed, std::cout.floatfield);
  std::cout.setf(std::cout.right, std::cout.adjustfield);
  std::cout.precision(2);
  std::cout << std::setw(5) << hypo.frame;
  std::cout << std::setw(10) << hypo.log_prob;

  debug_paths.clear();
  while (path != NULL) {
    debug_paths.push_back(path);
    path = path->prev;
  }
  
  for (int i = debug_paths.size() - 1; i >= 0; i--) {
    std::cout << " " << m_vocabulary.word(debug_paths[i]->word_id);
  }

  std::cout << std::endl;
}

void
Search::debug_print_history(Hypo &hypo)
{
  static std::vector<Search::Path*> paths(128);

  Path *path = hypo.path;

  paths.clear();
  while (path != NULL) {
    paths.push_back(path);
    path = path->prev;
  }
  
  for (int i = paths.size() - 1; i > 0; i--) {
    std::cout << paths[i]->frame*128 << " " << paths[i-1]->frame*128
	      << " " << m_vocabulary.word(paths[i]->word_id) << std::endl;
  }
}

void
Search::run()
{
  int frame = 0;
  int word_limit = 10;
  int hypo_limit = 10;

  Hypo hypo;
  m_stacks[0].push_back(hypo);

  double best_score = -1e10;

  while (1) {
//    std::cout << "--- " << frame << " ---" << std::endl;

    // Prune stack
    HypoStack &stack = m_stacks[m_first_stack];
    if (stack.size() > hypo_limit) {
	std::partial_sort(stack.begin(), stack.begin() + hypo_limit, 
			  stack.end());
	stack.resize(word_limit);
    }

    if (!stack.empty()) {

      // Expand
      m_expander.expand(frame, m_frames - 2); // FIXME: magic number
      std::vector<Expander::Word*> words = m_expander.words();
      if (words.size() > word_limit) {
	std::partial_sort(words.begin(), words.begin() + word_limit, 
			  words.end(),
			  Expander::WordCompare());
	words.resize(word_limit);
      }

      // Expand hypotheses in stack
      for (int s = 0; s < stack.size(); s++) {
	Hypo &hypo = stack[s];
//	debug_print_hypo(hypo);
	for (int w = 0; w < words.size(); w++) {
	  Expander::Word *word = words[w];
	  int target_stack = (m_first_stack + word->frames) % m_frames;
	  Hypo new_hypo = Hypo(frame + word->frames,
			       word->log_prob + hypo.log_prob,
			       new Path(word->word_id, hypo.frame, hypo.path));
	  m_stacks[target_stack].push_back(new_hypo);

	  double score = (new_hypo.log_prob - 100) / new_hypo.frame;
	  if (score > best_score) {
	    best_score = score;
	    std::cout << "---" << std::endl;
	    debug_print_history(new_hypo);
	  }

//	  std::cout << "  ";
//	  debug_print_hypo(new_hypo);
	}
      }
    }
    stack.clear();
    m_first_stack = (m_first_stack + 1) % m_frames;
    frame++;
  }
}
