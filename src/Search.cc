#include <algorithm>
#include <iomanip>

#include <math.h>
#include "Search.hh"

int HypoPath::count = 0;

Search::Search(Expander &expander, const Vocabulary &vocabulary, 
	       const Ngram &ngram)
  : m_expander(expander),
    m_vocabulary(vocabulary),
    m_ngram(ngram),

    // State
    m_earliest_frame(0),
    m_earliest_stack(0),
    m_stacks(),

    // Options
    m_frames(0),
    m_lm_scale(1),
    m_lm_offset(1),
    m_word_limit(0),
    m_hypo_limit(0),
    m_beam(0),

    // Temp
    m_history(0)
{
}

void
Search::debug_print_hypo(Hypo &hypo)
{
  static std::vector<HypoPath*> debug_paths(128);

  HypoPath *path = hypo.path;
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
    std::cout << " " 
	      << "<" << debug_paths[i]->frame << "> "
	      << m_vocabulary.word(debug_paths[i]->word_id);
  }

  std::cout << " <" << hypo.frame << ">" << std::endl;
}

void
Search::debug_print_history(Hypo &hypo)
{
  static std::vector<HypoPath*> paths(128);

  HypoPath *path = hypo.path;

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
Search::init_search(int frames, int hypos)
{
  m_frame = 0;
  m_frames = frames;
  m_stacks.resize(frames);
  m_earliest_stack = 0;
  m_earliest_frame = 0;

  // Clear stacks and reserve some space.  Stacks grow dynamically,
  // but if we need space anyway, it is better to have some already.
  for (int i = 0; i < m_stacks.size(); i++) {
    m_stacks[i].clear();
    m_stacks[i].reserve(hypos);
  }
  
  // Create initial empty hypothesis.
  Hypo hypo;
  m_stacks[0].push_back(hypo);
}

int
Search::frame2stack(int frame) const
{
  // Calculate the position of the corresponding stack
  if (frame < m_earliest_frame)
    throw ForgottenFrame();
  if (frame - m_earliest_frame >= m_frames)
    throw FutureFrame();
  int stack_index = frame - m_earliest_frame;
  stack_index = (m_earliest_stack + stack_index) % m_frames;

  return stack_index;
}

void
Search::sort_stack(int frame, int top)
{
  int stack_index = frame2stack(frame);
  HypoStack &stack = m_stacks[stack_index];
  if (top > 0) {
    if (top > stack.size())
      top = stack.size();
    std::partial_sort(stack.begin(), stack.begin() + top, stack.end());
  }
  else
    std::sort(stack.begin(), stack.end());
}

bool
Search::expand_stack(int frame)
{
  int stack_index = frame2stack(frame);
  HypoStack &stack = m_stacks[stack_index];

  // Prune stack
  if (m_hypo_limit > 0 && stack.size() > m_hypo_limit) {
    std::partial_sort(stack.begin(), stack.begin() + m_hypo_limit, 
		      stack.end());
    stack.resize(m_hypo_limit);
  }
  else {
    std::sort(stack.begin(), stack.end());
  }

//  if (frame % 25 == 0)
//    std::cerr << frame << std::endl;

  if (!stack.empty()) {
    debug_print_hypo(stack[0]);
  }

  // End of input
  if (frame == m_expander.eof_frame())
    return false;
      
  if (!stack.empty()) {

    // Fit word lexicon to acoustic data
    m_expander.expand(frame, m_frames - 2); // FIXME: magic number

    // Get only the best words
    // FIXME: perhaps we want to add LM probs here
    std::vector<Expander::Word*> words = m_expander.words();
    if (m_word_limit > 0 && words.size() > m_word_limit) {
      std::partial_sort(words.begin(), words.begin() + m_word_limit, 
			words.end(),
			Expander::WordCompare());
      words.resize(m_word_limit);
    }

    // Expand all hypotheses in the stack...
    for (int s = 0; s < stack.size(); s++) {
      Hypo &hypo = stack[s];

      // Only if inside beam
      if (hypo.log_prob < stack[0].log_prob - m_beam)
	continue;

      // ... Using the best words
      for (int w = 0; w < words.size(); w++) {
	Expander::Word *word = words[w];
	int target_stack = (stack_index + word->frames) % m_frames;
	double log_prob = hypo.log_prob + word->log_prob;
	  
	// Calculate language model probabilities
	if (m_ngram.order() > 0) {
	  m_history.clear();
	  m_history.push_front(word->word_id);
	  HypoPath *path = hypo.path;
	  for (int i = 0; i < m_ngram.order(); i++) {
	    if (!path)
	      break;
	    m_history.push_front(path->word_id);
	    path = path->prev;
	  }
	  log_prob += m_lm_offset + m_lm_scale * word->frames *
	    m_ngram.log_prob(m_history.begin(), m_history.end());
	}

	// Insert hypo to target stack
	Hypo new_hypo(frame + word->frames, log_prob, hypo.path);
	new_hypo.add_path(word->word_id, hypo.frame);
	m_stacks[target_stack].push_back(new_hypo);
      }
    }

    if (words.size() > 0)
      stack.clear();
  }

  return true;
}

void
Search::go_to(int frame)
{
  while (m_frame < frame) {
    HypoStack &stack = m_stacks[m_earliest_stack];
    stack.clear();
    m_frame++;
    m_earliest_frame = m_frame;
    m_earliest_stack++;
    if (m_earliest_stack >= m_frames)
      m_earliest_stack = 0;
  }
}

bool
Search::run()
{
  HypoStack &stack = m_stacks[m_earliest_stack];

  if (!expand_stack(m_frame))
    return false;

  m_frame++;
  m_earliest_frame = m_frame;
  m_earliest_stack++;
  if (m_earliest_stack >= m_frames)
    m_earliest_stack = 0;

  return true;
}
