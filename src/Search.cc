#include <algorithm>
#include <iomanip>
#include <deque>

#include <math.h>
#include "Search.hh"

int Search::Path::count = 0;

Search::Search(Expander &expander, const Vocabulary &vocabulary, 
	       const Ngram &ngram, int frames)
  : m_expander(expander),
    m_vocabulary(vocabulary),
    m_ngram(ngram),
    m_frames(frames),
    m_first_stack(0),
    m_stacks(frames),
    m_lm_scale(1)
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
  std::deque<int> history;

  while (1) {

//      for (int j = 0; j < m_stacks.size(); j++) {
//        HypoStack &stack = m_stacks[j];
//        for (int i = 0; i < stack.size(); i++) {
//  	debug_print_hypo(stack[i]);
//        }
//      }

//    std::cout << "--- " << frame << " ---" << std::endl;

    // Prune stack
    HypoStack &stack = m_stacks[m_first_stack];
    if (stack.size() > hypo_limit) {
	std::partial_sort(stack.begin(), stack.begin() + hypo_limit, 
			  stack.end());
	stack.resize(hypo_limit);
    }

    if (frame == m_expander.eof_frame())
      break;
      
    if (!stack.empty()) {

      // FIXME!
      std::cout << Path::count << "\t";
      debug_print_hypo(stack[0]);

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
	  double log_prob = hypo.log_prob + word->log_prob;
	  
	  // Language model
	  // FIX THE HISTORY!!
	  if (m_ngram.order() > 0) {
	    history.clear();
	    history.push_front(word->word_id);
	    Path *path = hypo.path;
	    for (int i = 0; i < m_ngram.order(); i++) {
	      if (!path)
		break;
	      history.push_front(path->word_id);
	      path = path->prev;
	    }
	    log_prob += m_lm_scale * 
	      m_ngram.log_prob(history.begin(), history.end());
	  }

	  // Insert hypo to correct stack
	  Hypo new_hypo(frame + word->frames, log_prob, hypo.path);
	  new_hypo.add_path(word->word_id, hypo.frame);
	  m_stacks[target_stack].push_back(new_hypo);

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
