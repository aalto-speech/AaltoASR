#include <stack>
#include <algorithm>
#include <iomanip>

#include <math.h>
#include "Search.hh"

int HypoPath::g_count = 0;

// FIXME: remove frame2stack

// FIXME: check that every log_prob has the same base!  Now they
// should be log10 everywhere.

// Prunes hypotheses with the same last N words in LM sense.  Only the
// best hypothesis is retained.
//
// FIXME: currently path->word_id is an index in lexicon, not in LM!
// Search class has lex2lm mapping, which maps lexicon words to LM
// words.  Currently, this is relevant only for UNK words.  So now the
// implementation below does not prune different UNK words. 
// Tue Nov 26 11:50:16 EET 2002
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
    m_word_boundary(-1),
    m_word_limit(0),
    m_word_beam(1e10),
    m_hypo_limit(0),
    m_prune_similar(0),
    m_hypo_beam(1e10),
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
//        const Hypo &hypo = stack.at(h);
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
Search::set_word_boundary(const std::string &word)
{
  // FIXME: currently lexicon must be loaded before calling this.
  m_word_boundary = m_vocabulary.index(word);
}

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
Search::print_path(HypoPath *path)
{
  std::cout << m_vocabulary.word(path->word_id);
  if (m_print_indices)
    std::cout << "(" << path->word_id << ")";
  std::cout << " ";

  if (m_print_probs)
    std::cout << path->ac_log_prob << " "
	      << path->lm_log_prob << " ";
//	      << "<" << path->frame << "> "
//	      << "(" << path->word_id << ")"
}

void
Search::print_hypo(const Hypo &hypo)
{
  std::stack<HypoPath*> stack;

  HypoPath *path = hypo.path;
//    std::cout.setf(std::cout.fixed, std::cout.floatfield);
//    std::cout.setf(std::cout.right, std::cout.adjustfield);
//    std::cout.precision(2);
//    std::cout << std::setw(5) << hypo.frame;
//    std::cout << std::setw(10) << hypo.log_prob;

  while (!path->guard()) {
    stack.push(path);
    if (path == m_last_printed_path)
      break;
    path = path->prev;
  }

  while (!stack.empty()) {
    path = stack.top();
    stack.pop();
    if (path != m_last_printed_path) {
      assert(!path->guard());
      print_path(path);
    }
  }
  std::cout << ": " << hypo.frame << " " << hypo.log_prob << std::endl;
}

void
Search::print_sure()
{
  std::stack<HypoPath*> stack;
  HypoPath *path = this->stack(m_last_hypo_frame).at(0).path;
  
  while (1) {
    stack.push(path);
    if (path == m_last_printed_path)
      break;
    path = path->prev;
  }

  while (!stack.empty()) {
    path = stack.top();
    stack.pop();
    if (path->count() > 1)
      break;
    if (path != m_last_printed_path) {
      m_last_printed_path = path;
      print_path(path);
    }
  }
  std::cout.flush();
}

void
Search::reset_search(int start_frame)
{
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
  m_last_hypo_frame = start_frame;

  // Create initial empty hypothesis.
  Hypo hypo(0, 0, new HypoPath(-1, -1, NULL));
  m_last_printed_path = hypo.path;
  stack(m_first_frame).add(hypo);

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
	std::cerr << m_vocabulary.word(i) << " not in LM" << std::endl;
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
    std::cerr << "m_last_hypo_frame = " << m_last_hypo_frame << std::endl;
    throw FutureFrame();
  }

  return frame % m_stacks.size();
}

void
Search::sort_stack(int frame, int top)
{
  HypoStack &stack = this->stack(frame);
  stack.partial_sort(top);
}

void
Search::ensure_stack(int frame)
{
  while (m_last_frame <= frame) {
    stack(m_first_frame).clear();
    m_first_frame++;
    m_last_frame++;
  }
}

float
Search::compute_lm_log_prob(const Hypo &hypo)
{
  float lm_log_prob = 0;

  if (m_ngram.order() > 0 && m_lm_scale > 0) {
    m_history.clear();
    HypoPath *path = hypo.path;
    int last_word = path->word_id;
    for (int i = 0; i < m_ngram.order(); i++) {
      if (path->guard())
	break;
      m_history.push_front(m_lex2lm[path->word_id]);
      path = path->prev;
    }

    float ngram_log_prob = m_ngram.log_prob(m_history.begin(), m_history.end());

    // With m_unk_offset it is possible to distribute the
    // log-probability of the single UNK word of the LM over several
    // imaginary UNK words.
    float unk_offset = (last_word == 0) ? m_unk_offset : 0;

    lm_log_prob = m_lm_offset + m_lm_scale * (ngram_log_prob + unk_offset);
    // FIXME: we could also use the length to normalize the LM probability.
  }

  return lm_log_prob;
}

void
Search::update_global_pruning(int frame, float log_prob)
{
  float avg_log_prob = log_prob / frame;
  if (avg_log_prob > m_global_best / m_global_frame) {
    m_global_best = log_prob;
    m_global_frame = frame;
  }
}

void
Search::insert_hypo(int target_frame, const Hypo &hypo)
{
  assert(hypo.frame == target_frame);

  ensure_stack(target_frame);

  // Check hypo beam
  // FIXME: we could check global beam here too!
  HypoStack &target_stack = stack(target_frame);
  if (hypo.log_prob < target_stack.best_log_prob() - m_hypo_beam) {
    m_beam_prunings++;
    return;
  }
    
  // Insert hypothesis in the stack
  target_stack.add(hypo);
  if (target_frame > m_last_hypo_frame) {
    m_last_hypo_frame = target_frame;
    if (m_verbose == 1)
      print_sure();
  }
  m_hypo_insertions++;

  update_global_pruning(target_frame, hypo.log_prob);
}

void
Search::expand_hypo_with_word(const Hypo &hypo, int word, int target_frame, 
			      float ac_log_prob)
{
  // Add the expanded hypothesis
  Hypo new_hypo(target_frame, 0, hypo.path);
  new_hypo.add_path(word, target_frame);
  new_hypo.path->ac_log_prob = ac_log_prob;
  new_hypo.path->lm_log_prob = compute_lm_log_prob(new_hypo);
  new_hypo.log_prob = hypo.log_prob + ac_log_prob + new_hypo.path->lm_log_prob;
  insert_hypo(target_frame, new_hypo);

  // Add also the hypothesis with word boundary
  // FIXME: it is useless to do this if no LM is used!
  if (m_word_boundary > 0) {
    new_hypo.add_path(m_word_boundary, target_frame);
    new_hypo.path->ac_log_prob = 0;
    new_hypo.path->lm_log_prob = compute_lm_log_prob(new_hypo);
    new_hypo.log_prob += new_hypo.path->lm_log_prob;
    insert_hypo(target_frame, new_hypo);
  }
}

void
Search::expand_hypo(const Hypo &hypo)
{
  std::vector<Expander::Word*> &words = m_expander.words();

  for (int w = 0; w < words.size(); w++) {
    if (w >= m_word_limit)
      break;

    // FIXME: We could break here if words were always sorted.  Are they?
    if (words[w]->avg_log_prob < words[0]->avg_log_prob * m_word_beam)
      continue; 

    expand_hypo_with_word(hypo, words[w]->word_id,
			  hypo.frame + words[w]->frames, 
			  words[w]->log_prob);
  }
}

void
Search::find_best_words(int frame)
{
  // Do the Viterbi search.
  if (m_end_frame > 0 && (frame + m_expand_window > m_end_frame))
    m_expander.expand(frame, m_end_frame - frame);
  else
    m_expander.expand(frame, m_expand_window);

  // Sort the words according to the average log-probability.
  std::vector<Expander::Word*> &words = m_expander.words();
  if (m_word_limit > 0 && words.size() > m_word_limit) {
    std::partial_sort(words.begin(), words.begin() + m_word_limit, 
		      words.end(),
		      Expander::WordCompare());
    words.resize(m_word_limit);
  }
}

void
Search::initial_prunings(int frame, HypoStack &stack)
{
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
  float angle = m_global_best / m_global_frame;
  float ref = m_global_best + angle * (frame - m_global_frame);
  if (stack.best_log_prob() > ref)
    ref = stack.best_log_prob();
  for (int i = 0; i < stack.size(); i++) {
    if (stack[i].log_prob + m_hypo_beam < ref) {
      m_beam_prunings += stack.size() - i;
      stack.prune(i);
      break;
    }
  }

  // Reset global pruning if current stack was the best
  if (m_global_frame == frame) {
    m_global_best = 1e10;
    m_global_frame = -1;
  }
}

void
Search::check_stacks()
{
  for (int f = m_first_frame; f < m_last_frame; f++) {
    HypoStack &stack = this->stack(f);
    if (stack.empty())
      continue;
    for (int h = 0; h < stack.size(); h++) {
      assert(stack[h].frame == f);
    }
  }
}

bool
Search::expand_stack(int frame)
{
  HypoStack &stack = this->stack(frame);
  stack.sort();

  // Check if the end of speech has been reached.
  if (frame > m_last_hypo_frame) {
    std::cerr << "no more hypos after frame " 
	      << m_last_hypo_frame << std::endl;
    print_sure();
    return false;
  }

  initial_prunings(frame, stack);

  // Find the acoustically best words and expand all hypos in the stack.
  if (!stack.empty()) {
    if (m_verbose == 2) {
      std::cout << frame << "\t";
      print_hypo(stack.at(0));
    }

    m_stack_expansions++;
    find_best_words(frame);

    for (int h = 0; h < stack.size(); h++) {
      assert(stack[h].frame == frame);
      expand_hypo(stack[h]);
    }
    stack.clear();
  }

  return true;
}

void
Search::go(int frame)
{
  while (m_frame < frame) {
    HypoStack &stack = this->stack(frame);
    stack.clear();
    m_frame++;
  }
}

bool
Search::run()
{
  if (!expand_stack(m_frame)) {
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
    if (!expand_stack(m_frame))
      return false;
    m_frame++;
  }
  return true;
}
