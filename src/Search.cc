#include <stack>
#include <algorithm>
#include <iomanip>

#include <math.h>
#include "Search.hh"

int HypoPath::g_count = 0;

// Assumes sorted
void
HypoStack::prune_similar(int length)
{
  for (int h1 = 0; h1 + 1 < size(); h1++) {
    for (int h2 = h1 + 1; h2 < size(); h2++) {
      bool match = true;
      HypoPath *p1 = at(h1).path;
      HypoPath *p2 = at(h2).path;
      for (int i = 0; i < length; i++) {
	if (!p1 || !p2) {
	  match = false;
	  break;
	}

	if (p1->word_id != p2->word_id) {
	  match = false;
	  break;
	}
	p1 = p1->prev;
	p2 = p2->prev;
      }

      if (match) {
	erase(begin() + h2);
	h2--;
      }
    }
  }
}

Search::Search(Expander &expander, const Vocabulary &vocabulary, 
	       const Ngram &ngram)
  : m_expander(expander),
    m_vocabulary(vocabulary),
    m_ngram(ngram),

    // Stacks states
    m_first_frame(0),
    m_last_frame(0),
    m_first_stack(0),
    m_last_hypo_frame(0),

    // Options
    m_end_frame(-1),
    m_lm_scale(1),
    m_lm_offset(0),
    m_unk_offset(0),
    m_verbose(0),
    m_print_probs(false),
    m_last_printed_path(NULL),

    // Pruning options
    m_word_limit(0),
    m_word_beam(1e10),
    m_hypo_limit(0),
    m_prune_similar(0),
    m_beam(1e10),
    m_global_beam(1e10),

    // Temp
    m_history(0)
{
}

//  void
//  Search::print_paths()
//  {
//    std::cout << std::endl << "last hypo frame: " << m_last_hypo_frame << std::endl;

//    for (int f = first_frame(); f < last_frame(); f++) {
//      HypoStack &stack = this->stack(f);
//      if (stack.size() > 0)
//        std::cout << f << std::endl;
//      for (int h = 0; h < stack.size(); h++) {
//        Hypo &hypo = stack.at(h);
//        HypoPath *path = hypo.path;
//        while (path != NULL) {
//  	if (path->printed)
//  	  std::cout << "*";
//  	std::cout << m_vocabulary.word(path->word_id) << path->count() << " ";
//  	path = path->prev;
//        }
//        std::cout << std::endl;
//      }
//    }
//  }

void
Search::print_prunings()
{
  std::cout << "m_stack_expansions: " << m_stack_expansions << std::endl;
  std::cout << "m_hypo_insertions: " << m_hypo_insertions << std::endl;
  std::cout << "m_limit_prunings: " << m_limit_prunings << std::endl;
  std::cout << "m_beam_prunings: " << m_beam_prunings << std::endl;
  std::cout << "m_similar_prunings: " << m_similar_prunings << std::endl;
}

void
Search::debug_print_hypo(Hypo &hypo)
{
  std::stack<HypoPath*> stack;

  HypoPath *path = hypo.path;
//    std::cout.setf(std::cout.fixed, std::cout.floatfield);
//    std::cout.setf(std::cout.right, std::cout.adjustfield);
//    std::cout.precision(2);
//    std::cout << std::setw(5) << hypo.frame;
//    std::cout << std::setw(10) << hypo.log_prob;

  while (path != NULL) {
    stack.push(path);
    if (path == m_last_printed_path)
      break;
    path = path->prev;
  }

  while (!stack.empty()) {
    path = stack.top();
    stack.pop();
    if (path->word_id > 0 && path != m_last_printed_path) {
      std::cout << m_vocabulary.word(path->word_id) << " ";
      if (m_print_probs)
	std::cout << path->ac_log_prob << " "
		  << path->lm_log_prob << " ";
//	      << "<" << path->frame << "> "
//	      << "(" << path->word_id << ")"
    }
  }
  std::cout << hypo.frame << std::endl;
}

void
Search::print_sure()
{
  std::stack<HypoPath*> stack;
  HypoPath *path = this->stack(m_last_hypo_frame).at(0).path;
  
  while (path != NULL) {
    stack.push(path);
    if (path == m_last_printed_path)
      break;
    path = path->prev;
  }

  while (!stack.empty()) {
    path = stack.top();
    stack.pop();
    if (path->count() != 1)
      break;
    if (path->word_id > 0 && path != m_last_printed_path) {
      m_last_printed_path = path;
//      path->printed = true;
      std::cout << m_vocabulary.word(path->word_id) << " ";
//		<< m_lex2lm[path->word_id] << " "
      if (m_print_probs)
	std::cout << path->ac_log_prob << " "
		  << path->lm_log_prob << " ";
    }
  }
  std::cout.flush();
}

void
Search::reset_search(int start_frame)
{
  m_last_printed_path = NULL;

  // FIXME!  Are all beams reset properly here.  Test reinitializing
  // the search!
  m_global_best = 1e10;
  m_global_frame = -1;

  // Clear stacks
  for (int i = 0; i < m_stacks.size(); i++)
    m_stacks[i].clear();
  m_end_frame = -1;
  m_frame = start_frame;
  m_first_frame = start_frame;
  m_last_frame = m_first_frame + m_stacks.size();
  m_first_stack = frame2stack(start_frame);
  m_last_hypo_frame = start_frame;

  // Create initial empty hypothesis.
  Hypo hypo(0, 0, new HypoPath(0, 0, NULL));
  m_stacks[m_first_stack].add(hypo);

  // Reset pruning statistics
  m_stack_expansions = 0;
  m_hypo_insertions = 0;
  m_limit_prunings = 0;
  m_beam_prunings = 0;
  m_similar_prunings = 0;

  m_history.clear();
}

void
Search::init_search(int expand_window, int stacks, int reserved_hypos)
{
  m_expand_window = expand_window;

  // Initialize stacks and reserve some space beforehand
  m_stacks.resize(stacks);
  for (int i = 0; i < m_stacks.size(); i++) {
    m_stacks[i].clear();
    m_stacks[i].reserve(reserved_hypos);
  }

  reset_search(0);

  // Create mapping between words in the lexicon and the language model
  if (m_ngram.order() > 0) {
    int count = 0;
    m_lex2lm.clear();
    m_lex2lm.resize(m_vocabulary.size());
    for (int i = 0; i < m_vocabulary.size(); i++) {
      m_lex2lm[i] = m_ngram.index(m_vocabulary.word(i));
      if (m_lex2lm[i] == 0) {
//	std::cerr << m_vocabulary.word(i) << " not in LM" << std::endl;
	count++;
      }
    }
    if (count > 0)
      std::cerr << "there were " << count << " out-of-LM words" << std::endl;
  }
}

int
Search::frame2stack(int frame) const
{
  // Check that we have the frame in buffer
  if (frame < m_first_frame)
    throw ForgottenFrame();
  if (frame >= m_last_frame) {
    std::cerr << std::endl 
	      << "m_last_frame = " << m_last_frame << " but " << frame 
	      << " requested" << std::endl;
    std::cerr << "eof_frame = " << m_expander.eof_frame() << std::endl;
    std::cerr << "m_last_hypo_frame = " << m_last_hypo_frame << std::endl;
    throw FutureFrame();
  }

  // Find the stack corresponding to the given frame
  int index = frame - m_first_frame;
  index = (m_first_stack + index) % m_stacks.size();

  return index;
}

void
Search::sort_stack(int frame, int top)
{
  int stack_index = frame2stack(frame);
  HypoStack &stack = m_stacks[stack_index];
  stack.partial_sort(top);
}

void
Search::circulate(int &stack)
{
  stack++;
  if (stack >= m_stacks.size())
    stack = 0;
}

void
Search::move_buffer(int frame)
{
  while (m_last_frame <= frame) {
    m_stacks[m_first_stack].clear();
    circulate(m_first_stack);
    m_first_frame++;
    m_last_frame++;
  }
}

void
Search::prune_similar(int frame, int length)
{
  int stack_index = frame2stack(frame);
  HypoStack &stack = m_stacks[stack_index];
  stack.prune_similar(length);
}

bool
Search::expand(int frame)
{
  // End of input?  
  if (frame > m_last_hypo_frame) {
    assert(this->stack(m_last_hypo_frame).size() > 0);
    debug_print_hypo(this->stack(m_last_hypo_frame).at(0));
    return false;
  }
  
  // Sort the current stack
  int stack_index = frame2stack(frame);
  HypoStack &stack = m_stacks[stack_index];
  stack.sort();

  // Prune similar endings
  if (m_prune_similar > 0) {
    int before = stack.size();
    stack.prune_similar(m_prune_similar);
    m_similar_prunings += (before - stack.size());
  }

  // Keep only the best N hypos
  if (m_hypo_limit > 0) {
    int before = stack.size();
    stack.prune(m_hypo_limit);
    m_limit_prunings += (before - stack.size());
  }

  // Beam and global prunings
  double angle = m_global_best / m_global_frame;
  double ref = m_global_best + angle * (frame - m_global_frame);
  if (stack.best_log_prob() > ref)
    ref = stack.best_log_prob();
  for (int i = 0; i < stack.size(); i++) {
    if (stack[i].log_prob + m_beam < ref) {
      m_beam_prunings += stack.size() - i;
      stack.prune(i);
      break;
    }
  }

  // Reset global pruning if current stack is best
  if (m_global_frame == frame) {
    m_global_best = 1e10;
    m_global_frame = -1;
  }

  // Debug print
  if (m_verbose == 1 && !stack.empty()) {
    static int step = 100;
    static int next = 0;
    if (frame > next) {
      print_sure();
      while (frame > next)
	next += step;
    }
  }
  if (m_verbose == 2 && !stack.empty())
    debug_print_hypo(stack.at(0));

      
  // Expand the stack
  if (!stack.empty()) {
    // Fit word lexicon to acoustic data
    if (m_end_frame > 0 && (frame + m_expand_window > m_end_frame))
      m_expander.expand(frame, m_end_frame - frame);
    else
      m_expander.expand(frame, m_expand_window);

    m_stack_expansions++;

    // Get only the best words
    // FIXME: perhaps we want to add LM probs here, perhaps not
    std::vector<Expander::Word*> words = m_expander.words();
    if (m_word_limit > 0 && words.size() > m_word_limit) {
      std::partial_sort(words.begin(), words.begin() + m_word_limit, 
			words.end(),
			Expander::WordCompare());
      words.resize(m_word_limit);
    }

    // Expand all hypotheses in the stack...
    for (int h = 0; h < stack.size(); h++) {

      Hypo &hypo = stack[h];

      // ... Using the best words
      for (int w = 0; w < words.size(); w++) {
	Expander::Word *word = words[w];

	// Prune words much worse than the best words on average
	if (word->avg_log_prob < words[0]->avg_log_prob * m_word_beam)
	  continue; // FIXME: could we break here if words are sorted?

	double log_prob = hypo.log_prob + word->log_prob;
	double lm_log_prob = 0;

	// Calculate language model probabilities
	if (m_ngram.order() > 0 && m_lm_scale > 0) {
	  int lm_word_id = m_lex2lm[word->word_id];
	  m_history.clear();
	  m_history.push_front(lm_word_id);
	  HypoPath *path = hypo.path;
	  for (int i = 0; i < m_ngram.order()-1; i++) {
	    if (!path)
	      break;
	    m_history.push_front(m_lex2lm[path->word_id]);
	    path = path->prev;
	  }

	  double tmp = m_ngram.log_prob(m_history.begin(), m_history.end());
	  lm_log_prob = m_lm_offset + m_lm_scale * 
//	    word->frames * // Do we need this really?!
	    (tmp + 
	     (lm_word_id == 0 ? m_unk_offset : 0));

	  log_prob += lm_log_prob;
	}

	// Ensure stack space
	move_buffer(frame + word->frames);
	assert(frame >= m_first_frame);
	int index = frame2stack(frame + word->frames);
	HypoStack &target_stack = m_stacks[index];

	// Insert hypo to target stack, if inside the beam
	if (log_prob > target_stack.best_log_prob() - m_beam) {
	  int target_frame = frame + word->frames;
	  Hypo new_hypo(target_frame, log_prob, hypo.path);
	  new_hypo.add_path(word->word_id, hypo.frame);
	  
	  new_hypo.path->lm_log_prob = lm_log_prob;

	  new_hypo.path->ac_log_prob = word->log_prob;

	  target_stack.add(new_hypo);
	  if (target_frame > m_last_hypo_frame)
	    m_last_hypo_frame = target_frame;

	  // Update global pruning
	  double avg_log_prob = log_prob / target_frame;
	  if (avg_log_prob > m_global_best / m_global_frame) {
	    m_global_best = log_prob;
	    m_global_frame = target_frame;
	  }

	  m_hypo_insertions++;
	}
	else
	  m_beam_prunings++;
      }
    }
  }

  // FIXME REALLY: is this good idea?!  
  if (m_expander.words().size() > 0)
    stack.clear();

  return true;
}

void
Search::go(int frame)
{
  while (m_frame < frame) {
    int index = frame2stack(m_frame);
    HypoStack &stack = m_stacks[index];
    stack.clear();
    m_frame++;
  }
}

bool
Search::run()
{
  if (!expand(m_frame)) {
    return false;
  }
  else {
    m_frame++;
    return true;
  }
}

bool
Search::recognize_segment(int start_frame, int end_frame)
{
  reset_search(start_frame);
  set_end_frame(end_frame);
  while (m_frame <= end_frame) {
    if (!expand(m_frame))
      return false;
    m_frame++;
  }
  return true;
}
