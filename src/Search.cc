#include <sstream>
#include <stack>
#include <algorithm>

#include <assert.h>
#include <math.h>
#include <stdio.h>

#include "Search.hh"

int HypoPath::g_count = 0;

// FIXME: check that every log_prob has the same base!  Now they
// should be log10 everywhere.

// Returns the first (and should be the only) hypothesis with a
// similar word history of 'words' words.
//
// FIXME: currently path->word_id is an index in lexicon, not in LM!
// Search class has lex2lm mapping, which maps lexicon words to LM
// words.  Currently, this is relevant only for UNK words.  So now the
// implementation below does not prune different UNK words. 
// Tue Nov 26 11:50:16 EET 2002
int
HypoStack::find_similar(const Hypo &hypo, int words)
{
  for (int h = 0; h < size(); h++) {
    HypoPath *path1 = hypo.path;
    HypoPath *path2 = at(h).path;

    for (int i = 0;; i++) {
      // Enough words checked, it is a match
      if (i == words)
	return h;

      // Both hypotheses are short, no match
      if (!path1 && !path2)
	break;

      // Only the one of the hypotheses is short, no match
      if (!path1 || !path2)
	break;

      // Words differ, no match 
      // FIXME: use perhaps m_lex2lm mapping
      if (path1->word_id != path2->word_id)
	break;

      path1 = path1->prev;
      path2 = path2->prev;
    }
  }

  return -1;
}

void
HypoStack::sorted_insert(const Hypo &hypo)
{
  HypoStack::iterator it = lower_bound(begin(), end(), hypo);
  insert(it, hypo);
}

Search::Search(Expander &expander, const Vocabulary &vocabulary)
  : m_expander(expander),
    m_vocabulary(vocabulary),

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
    m_multiple_endings(0),
    m_print_indices(false),
    m_print_frames(false),
    m_last_printed_path(NULL),

    // Pruning options
    m_word_boundary(-1),
    m_dummy_word_boundaries(true),
    m_word_limit(0),
    m_word_beam(1e10),
    m_prune_similar(0),
    m_hypo_limit(0),
    m_hypo_beam(1e10),
    m_global_beam(1e10)
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
  m_word_boundary = m_vocabulary.word_index(word);
  if (m_word_boundary == 0) {
    fprintf(stderr, "Search::set_word_boundary(): word boundary not in vocabulary\n");
    exit(1);
  }
}

void
Search::print_prunings()
{
  printf(
    "stack expansions: %d\n"
    "hypo insertions:  %d\n"
    "limit prunings:   %d\n"
    "similar prunings: %d\n",
    m_stack_expansions, m_hypo_insertions, m_limit_prunings, m_similar_prunings);
}

void
Search::print_path(HypoPath *path)
{
  printf("%s", m_vocabulary.word(path->word_id).c_str());

  if (m_print_frames)
    printf("[%d]", path->frame);

  if (m_print_indices)
    printf("(%d)", path->word_id);

  putchar(' ');

  if (m_print_probs)
    printf("%.2f %.2f ", path->ac_log_prob, path->lm_log_prob);
}

void
Search::print_hypo(const Hypo &hypo)
{
  std::stack<HypoPath*> stack;
  HypoPath *path = hypo.path;

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

//  printf(": %d %.2f\n", hypo.frame, hypo.log_prob);
  fflush(stdout);
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
  fflush(stdout);
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
  Hypo hypo(start_frame, 0, new HypoPath(-1, start_frame, NULL));
  m_last_printed_path = hypo.path;
  stack(m_first_frame).sorted_insert(hypo);

  // Reset pruning statistics
  m_stack_expansions = 0;
  m_hypo_insertions = 0;
  m_limit_prunings = 0;
  m_beam_prunings = 0;
  m_similar_prunings = 0;

  m_history_lex.clear();
}

void
Search::add_ngram(Ngram *ngram, float weight)
{
  int count = 0;

  m_ngrams.push_back(LanguageModel());
  m_ngrams.back().ngram = ngram;
  m_ngrams.back().weight = weight;
  m_ngrams.back().lex2lm.clear();
  m_ngrams.back().lex2lm.resize(m_vocabulary.num_words());

  // Create the mapping between lexicon and the model.
  for (int i = 0; i < m_vocabulary.num_words(); i++) {
    m_ngrams.back().lex2lm[i] = ngram->word_index(m_vocabulary.word(i));

    // Warn about words not in lm.
    if (m_ngrams.back().lex2lm[i] == 0 && i != 0) {
      fprintf(stderr, "%s not in LM %d\n", m_vocabulary.word(i).c_str(), 
	      m_ngrams.size());
      count++;
    }
  }

  if (count > 0)
    fprintf(stderr, "there were %d out-of-LM words in total in LM %d\n", 
	    count, m_ngrams.size());
}

void
Search::init_search(int expand_window)
{
  m_expand_window = expand_window;

  // Initialize stacks and reserve some space beforehand
  m_stacks.resize(expand_window + 1); // FIXME: not sure if one additional is needed
  for (int i = 0; i < m_stacks.size(); i++) {
    m_stacks[i].clear();
    m_stacks[i].reserve(128); // FIXME: magic number
  }

  reset_search(0);
}

int
Search::frame2stack(int frame) const
{
  if (frame < m_first_frame || frame >= m_last_frame)
    throw InvalidFrame();
  return frame % m_stacks.size();
}

HypoStack&
Search::stack(int frame)
{
  return m_stacks[frame2stack(frame)];
}

const HypoStack&
Search::stack(int frame) const
{
  return m_stacks[frame2stack(frame)];
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

  if (m_max_lm_order > 0 && m_lm_scale > 0) {

    // Create history using lexicon indices
    m_history_lex.clear();
    HypoPath *path = hypo.path;
    int last_word = path->word_id;
    for (int i = 0; i < m_max_lm_order; i++) {
      if (path->guard())
	break;
      m_history_lex.push_front(path->word_id);
      path = path->prev;
    }

    // FOR each interpolated LM
    float ngram_log_prob = 0;
    for (int lm = 0; lm < m_ngrams.size(); lm++) {

      // Convert lexicon history to the currente LM indices.
      m_history_lm.clear();
      for (int j = m_history_lex.size(); j > 0; j--) {
	if (m_history_lm.size() == m_ngrams[lm].ngram->order())
	  break;
	m_history_lm.push_front(m_ngrams[lm].lex2lm[m_history_lex[j - 1]]);
      }

      // Add the current LM probability with weight
      ngram_log_prob += m_ngrams[lm].weight *
	m_ngrams[lm].ngram->log_prob(m_history_lm.begin(), m_history_lm.end());
    }

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
  assert(hypo.path->frame == target_frame);
  assert(m_end_frame < 0 || hypo.frame < m_end_frame);

  ensure_stack(target_frame);

  HypoStack &target_stack = stack(target_frame);

  // Check hypo beam
  // FIXME: we could check global beam here too!
//   if (hypo.log_prob < target_stack.best_log_prob() - m_hypo_beam) {
//     m_beam_prunings++;
//     return;
//   }

  // Check if the hypo would fall outside the stack
  if (target_stack.size() >= m_hypo_limit) {
    m_limit_prunings++;
    if (hypo.log_prob < target_stack.back().log_prob)
      return;
    target_stack.pop_back();
  }

  // Find the possible similar hypothesis and store only the better
  int index = target_stack.find_similar(hypo, m_prune_similar);
  if (index >= 0) {
    m_similar_prunings++;
    if (target_stack.at(index).log_prob > hypo.log_prob)
      return;
    target_stack.remove(index);
  }

  // Insert hypothesis in the stack
  target_stack.sorted_insert(hypo);
  if (target_frame > m_last_hypo_frame) {
    m_last_hypo_frame = target_frame;
    if (m_verbose == 1)
      print_sure();
  }
  m_hypo_insertions++;

  assert(target_stack.size() <= m_hypo_limit);

  // update_global_pruning(target_frame, hypo.log_prob);
}

void
Search::expand_hypo_with_word(const Hypo &hypo, int word, int target_frame, 
			      float ac_log_prob)
{
  // Should not be possible!
  assert(ac_log_prob < 0);

  // Add the expanded hypothesis
  Hypo new_hypo(target_frame, hypo.log_prob, hypo.path);
  new_hypo.add_path(word, target_frame);

  // Merge subsequent silences.  Note, that we have to create a new
  // path, and remove the previous silence.  We should not modify the
  // previous silence, because it is still used by other hypotheses.
  // FIXME: ensure that everything goes fine, there was a nasty bug once
  if (word == m_word_boundary && hypo.path->word_id == m_word_boundary) 
  {
    HypoPath *prev = new_hypo.path->prev;
    new_hypo.path->prev = prev->prev;
    prev->prev->link();
    new_hypo.path->frame = target_frame;
    new_hypo.path->ac_log_prob = prev->ac_log_prob + ac_log_prob;
    new_hypo.path->lm_log_prob = prev->lm_log_prob;
    new_hypo.log_prob += ac_log_prob;
    HypoPath::unlink(prev);
  }
  // else add the new word to the path
  else {
    new_hypo.path->ac_log_prob = ac_log_prob;
    new_hypo.path->lm_log_prob = compute_lm_log_prob(new_hypo);
    new_hypo.log_prob += ac_log_prob + new_hypo.path->lm_log_prob;
  }
  insert_hypo(target_frame, new_hypo);

  // Add also the hypothesis with word boundary
  // FIXME: it is useless to do this if no LM is used!
  if (m_dummy_word_boundaries && m_word_boundary > 0 && word != m_word_boundary) {
    new_hypo.add_path(m_word_boundary, target_frame);
    new_hypo.path->ac_log_prob = 0;
    // FIXME: we do not need this here, because of the previous
    // merge_silences() right?
    //
    // merge_silences(new_hypo.path);
    new_hypo.path->lm_log_prob = compute_lm_log_prob(new_hypo);
    new_hypo.log_prob += new_hypo.path->lm_log_prob;
    insert_hypo(target_frame, new_hypo);
  }
}

void
Search::expand_hypo(const Hypo &hypo)
{
  const std::vector<Expander::Word*> &words = m_expander.words();

  for (int w = 0; w < words.size(); w++) {
    if (w >= m_word_limit)
      break;

    // FIXME: We could break here if words were always sorted.  Are they?
    if (words[w]->best_avg_log_prob < words[0]->best_avg_log_prob * m_word_beam)
      continue; 

    // In addition to the best ending time, expand also to
    // neighbouring frames.  The change in log_prob is approximated.
    if (m_multiple_endings > 0) {
      for (int i = -m_multiple_endings; i <= m_multiple_endings; i++) {
	int df = i;

	int length = words[w]->best_length + df;
	if (length < words[w]->first_length || length >= words[w]->last_length)
	  continue;
	
	// Ending in the current frame may be pruned
	if (!words[w]->active_length(length))
	  continue;

	// Expander should not go to m_end_frame or further!
	assert(m_end_frame < 0 || hypo.frame + length < m_end_frame);

	expand_hypo_with_word(hypo, words[w]->word_id,
			      hypo.frame + length, 
			      words[w]->log_probs[length]);
      }
    }

    // Expand only to the best ending time.
    else {
      expand_hypo_with_word(hypo, words[w]->word_id,
			    hypo.frame + words[w]->best_length, 
			    words[w]->best_log_prob());
    }
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
  m_expander.sort_words(m_word_limit);
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
  // Check if the end of speech has been reached.
  if (frame > m_last_hypo_frame) {
    print_hypo(this->stack(m_last_hypo_frame)[0]);
    fprintf(stderr, "no more hypos after frame %d\n", m_last_hypo_frame);
    return false;
  }

  HypoStack &stack = this->stack(frame);

  // Find the acoustically best words and expand all hypos in the stack.
  if (stack.empty())
    return true;

  if (m_verbose == 2) {
    printf("%d\t", frame);
    print_hypo(stack.at(0));
  }

  m_stack_expansions++;
  find_best_words(frame);

  for (int h = 0; h < stack.size(); h++) {
    assert(stack[h].frame == frame);
    expand_hypo(stack[h]);
  }

  // FIXME: ugly, but works for now
  if (m_expander.words().size() > 0)
    stack.clear();

  return true;
}

void
Search::expand_words(int frame, const std::string &words)
{
  std::istringstream in(words);
  std::string str;
  
  if (stack(frame).empty()) {
    fprintf(stderr, "stack empty at frame %d\n", frame);
    return;
  }

  while (in >> str) {
    find_best_words(frame);
    int index = m_vocabulary.word_index(str);
    if (index == 0) {
      fprintf(stderr, "word '%s' not in lexicon\n", str.c_str());
      return;
    }

    Expander::Word *word = m_expander.word(m_vocabulary.word_index(str));
    if (!word->active) {
      fprintf(stderr, "word '%s' did not survive\n", str.c_str());
      return;
    }

    expand_hypo_with_word(stack(frame)[0], 
			  word->word_id, frame + word->best_length,
			  word->best_log_prob());
    frame += word->best_length;
    fprintf(stderr, "%d\t%s\n", frame, str.c_str());
  }
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
  print_hypo(stack(m_last_hypo_frame)[0]);
  return true;
}
