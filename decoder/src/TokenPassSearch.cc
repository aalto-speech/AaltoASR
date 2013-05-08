#include <algorithm>
#include <cstddef>  // NULL
#include <sstream>
#include <stdexcept>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <string>
#include <cctype>

#include "TokenPassSearch.hh"

#define NUM_HISTOGRAM_BINS 100
#define TOKEN_RESERVE_BLOCK 1024

#define DEFAULT_MAX_LOOKAHEAD_SCORE_LIST_SIZE 512
//1031
#define DEFAULT_MAX_NODE_LOOKAHEAD_BUFFER_SIZE 512

#define DEFAULT_MAX_LM_CACHE_SIZE 15000

#define MAX_TREE_DEPTH 60

#define MAX_STATE_DURATION 80

//#define PRUNING_EXTENSIONS
//#define FAN_IN_PRUNING
//#define EQ_WC_PRUNING
//#define EQ_DEPTH_PRUNING
//#define STATE_PRUNING
//#define FAN_OUT_PRUNING

//#define COUNT_LM_LA_CACHE_MISS

using namespace std;

TokenPassSearch::TokenPassSearch(TPLexPrefixTree &lex, Vocabulary &vocab,
                                 Acoustics *acoustics) :
  m_lexicon(lex),
  m_vocabulary(vocab),
#ifdef ENABLE_WORDCLASS_SUPPORT
  m_word_classes(NULL),
#endif
  m_acoustics(acoustics),
  m_end_frame(-1),
  m_frame(0),
  m_best_log_prob(0),
  m_worst_log_prob(0),
  m_best_we_log_prob(0),
  m_best_final_token(NULL),
  m_ngram(NULL),
  m_fsa_lm(NULL),
  m_lookahead_ngram(NULL),
  m_print_probs(0),
  m_print_text_result(0),
  m_print_state_segmentation(false),
  m_keep_state_segmentation(false),
  m_global_beam(1e10),
  m_word_end_beam(1e10),
  m_similar_lm_hist_span(0),
  m_lm_scale(1),
  m_duration_scale(0),
  m_transition_scale(1),
  m_max_num_tokens(0),
  m_verbose(0),
  m_word_boundary_id(0),
  m_lm_lookahead(0),
  m_max_lookahead_score_list_size(DEFAULT_MAX_LOOKAHEAD_SCORE_LIST_SIZE),
  m_max_node_lookahead_buffer_size(DEFAULT_MAX_NODE_LOOKAHEAD_BUFFER_SIZE),
  m_insertion_penalty(0),
  m_sentence_start_id(-1),
  m_sentence_end_id(-1),
  m_use_sentence_boundary(false),
  m_generate_word_graph(false),
  m_require_sentence_end(false),
  m_remove_pronunciation_id(false),
  m_use_word_pair_approximation(false),
  m_use_lm_cache(true),
  m_current_glob_beam(0),
  m_current_we_beam(0),
  m_eq_depth_beam(1e10),
  m_eq_wc_beam(1e10),
  m_fan_in_beam(1e10),
  m_fan_out_beam(1e10),
  m_state_beam(1e10),
  filecount(0),
  m_min_word_count(0),
  m_fan_in_log_prob(0),
  m_fan_out_log_prob(0),
  m_fan_out_last_log_prob(0),
  m_lm_lookahead_initialized(false)
{
  m_active_token_list = new std::vector<TPLexPrefixTree::Token*>;
  m_new_token_list = new std::vector<TPLexPrefixTree::Token*>;
  m_word_end_token_list = new std::vector<TPLexPrefixTree::Token*>;
#ifdef ENABLE_MULTIWORD_SUPPORT
  m_split_multiwords = false;
#endif
}

TokenPassSearch::~TokenPassSearch() {
  delete m_active_token_list;
  delete m_new_token_list;
  delete m_word_end_token_list;
  for (std::vector<LMHistory *>::iterator it=m_lmhist_dealloc_table.begin();
       it!=m_lmhist_dealloc_table.end();++it) {
    delete[] *it;
  }
  for (std::vector<TPLexPrefixTree::Token *>::iterator it=m_token_dealloc_table.begin();
       it!=m_token_dealloc_table.end();++it) {
    delete[] *it;
  }
}

void TokenPassSearch::set_word_boundary(const std::string &word)
{
  assert(!m_ngram);
  assert(!m_fsa_lm);

  if (word.empty()) {
    m_word_boundary_id = 0;
  }
  else {
    m_word_boundary_id = m_vocabulary.word_index(word);
    if (m_word_boundary_id <= 0) {
      // word is not in vocabulary.
      throw invalid_argument("TokenPassSearch::set_word_boundary");
    }
  }
}

void TokenPassSearch::set_sentence_boundary(const std::string &start,
                                            const std::string &end)
{
  assert(!m_ngram);
  assert(!m_fsa_lm);

  m_sentence_start_id = m_vocabulary.word_index(start);
  if (m_sentence_start_id == 0) {
    // start is not in vocabulary.
    throw invalid_argument("TokenPassSearch::set_sentence_boundary");
  }
  m_sentence_end_id = m_vocabulary.word_index(end);
  if (m_sentence_end_id == 0) {
    // end is not in vocabulary.
    throw invalid_argument("TokenPassSearch::set_sentence_boundary");
  }
  m_use_sentence_boundary = true;
  m_lexicon.set_sentence_boundary(m_sentence_start_id, m_sentence_end_id);
}

void TokenPassSearch::clear_hesitation_words()
{
  m_hesitation_ids.clear();
}

void TokenPassSearch::add_hesitation_word(const std::string & word)
{
  int id = m_vocabulary.word_index(word);
  if (id != 0) {
    m_hesitation_ids.push_back(id);
  }
}

void TokenPassSearch::reset_search(int start_frame)
{
  TPLexPrefixTree::Token *t;
  m_frame = start_frame;
  m_end_frame = -1;
  m_best_final_token = NULL;

  if (m_verbose > 0) {
    cerr << m_hesitation_ids.size() << " hesitation words." << endl;
  }

  // Clear existing tokens and create a new token to the root
  for (int i = 0; i < m_active_token_list->size(); i++) {
    if ((*m_active_token_list)[i] != NULL)
      release_token((*m_active_token_list)[i]);
  }
  m_active_token_list->clear();

  m_lexicon.clear_node_token_lists();

  t = acquire_token();
  t->node = m_lexicon.start_node();
  t->next_node_token = NULL;
  t->am_log_prob = 0;
  t->lm_log_prob = 0;
  t->cur_am_log_prob = 0;
  t->cur_lm_log_prob = 0;
  t->lm_history = acquire_lmhist(&m_null_word, NULL);
  hist::link(t->lm_history);

  if (m_generate_word_graph) {
    t->word_history = new TPLexPrefixTree::WordHistory(-1, -1, NULL);
    t->word_history->lex_node_id = t->node->node_id;
    hist::link(t->word_history);

    word_graph.reset();
    int node_index = word_graph.add_node(-1, -1, t->node->node_id, 0);
    word_graph.link(node_index);
    t->recent_word_graph_node = node_index;
    m_recent_word_graph_info.clear();
    m_recent_word_graph_info.resize(m_word_repository.size());
  }

  t->lm_hist_code = 0;
  t->dur = 0;
  t->word_start_frame = -1;

  t->fsa_lm_node = -1;
  if (m_fsa_lm)
    t->fsa_lm_node = m_fsa_lm->initial_node_id();

  if (m_use_sentence_boundary) {
    LMHistory * sentence_start = acquire_lmhist(
      &m_word_repository[m_sentence_start_id], t->lm_history);
    hist::unlink(t->lm_history, &m_lmh_pool);
    t->lm_history = sentence_start;
    hist::link(t->lm_history);
  }

#ifdef PRUNING_MEASUREMENT
  for (int i = 0; i < 6; i++)
    t->meas[i] = 0;
#endif

#if (defined PRUNING_EXTENSIONS || defined PRUNING_MEASUREMENT || defined EQ_WC_PRUNING)
  for (int i = 0; i < MAX_WC_COUNT; i++)
    m_wc_llh[i] = 0;
  m_min_word_count = 0;
#endif

#ifdef COUNT_LM_LA_CACHE_MISS
  for (int i = 0; i < MAX_LEX_TREE_DEPTH; i++)
  {
    lm_la_cache_count[i] = 0;
    lm_la_cache_miss[i] = 0;
  }
  lm_la_word_cache_count = 0;
  lm_la_word_cache_miss = 0;
#endif

  t->depth = 0;
  //t->token_path = new TPLexPrefixTree::PathHistory(0,0,0,NULL);
  //t->token_path->link();
  t->word_count = 0;

  if (m_keep_state_segmentation) {
    t->state_history = new TPLexPrefixTree::StateHistory(0, 0, NULL);
    hist::link(t->state_history);
  }
  else {
    t->state_history = NULL;
  }

  m_active_token_list->push_back(t);

  if (lm_lookahead_score_list.get_num_items() > 0) {
    // Delete the LM lookahead cache
    LMLookaheadScoreList *score_list;
    while (lm_lookahead_score_list.remove_last_item(&score_list))
      delete score_list;
  }

  if (!m_lm_lookahead_initialized && (m_lm_lookahead > 0)) {
    lm_lookahead_score_list.set_max_items(m_max_lookahead_score_list_size);
    m_lexicon.set_lm_lookahead_cache_sizes(m_max_node_lookahead_buffer_size);
    m_lm_lookahead_initialized = true;
  }

  if (m_lm_lookahead > 0) {
    assert( m_lookahead_ngram != NULL);
  }

  if (m_lm_score_cache.get_num_items() > 0) {
    LMScoreInfo *info;
    while (m_lm_score_cache.remove_last_item(&info))
      delete info;
  }
  m_lm_score_cache.set_max_items(DEFAULT_MAX_LM_CACHE_SIZE);

  m_current_glob_beam = m_global_beam;
  m_current_we_beam = m_word_end_beam;
}


bool
TokenPassSearch::run(void)
{
  if (m_generate_word_graph) {
    if (m_similar_lm_hist_span < 2) {
      throw CannotGenerateWordGraph(
        "Similar word history span should be at least 2 if word graph requested.");
    }

    if (!m_use_sentence_boundary) {
      throw CannotGenerateWordGraph(
        "Word graph can be generated only if sentence boundary is used.");
    }
  }

  if (m_verbose > 1)
    printf("run() in frame %d\n", m_frame);
  if ((m_end_frame != -1 && m_frame >= m_end_frame) ||
      !m_acoustics->go_to(m_frame))
  {
    if (m_generate_word_graph || m_require_sentence_end)
      update_final_tokens();

    if (m_print_text_result)
      print_lm_history(stdout, true);

    if (m_print_state_segmentation)
      print_state_history();

    return false;
  }

  propagate_tokens();
  prune_tokens();
#ifdef PRUNING_MEASUREMENT
  analyze_tokens();
#endif
  /*if ((m_frame%5) == 0)
    save_token_statistics(filecount++);*/
  if (m_print_text_result)
    print_lm_history(stdout, false);
  m_frame++;
  return true;
}

#ifdef PRUNING_MEASUREMENT
void
TokenPassSearch::analyze_tokens(void)
{
  /*double avg_log_prob, log_prob_std, ac_log_prob_dev;
    double ac_avg_log_prob, ac_log_prob_std;
    float count;
    float meas1, meas2, meas3, meas4;*/
  int i, j;
  double temp;
  int d;

  for (i = 0; i < m_active_token_list->size(); i++)
  {
    if ((*m_active_token_list)[i] != NULL)
    {
      // Equal depth pruning
      if (!((*m_active_token_list)[i]->node->flags&
            (NODE_FAN_IN|NODE_FAN_OUT|NODE_AFTER_WORD_ID)))
      {
        temp = (*m_active_token_list)[i]->total_log_prob -
          m_depth_llh[(*m_active_token_list)[i]->depth/2];
        if (temp < (*m_active_token_list)[i]->meas[0])
          (*m_active_token_list)[i]->meas[0] = temp;
      }

      if (!((*m_active_token_list)[i]->node->flags&(NODE_FAN_IN|NODE_FAN_OUT)))
      {
        // Equal word count
        temp = (*m_active_token_list)[i]->total_log_prob -
          m_wc_llh[(*m_active_token_list)[i]->word_count-m_min_word_count];
        if (temp < (*m_active_token_list)[i]->meas[1])
          (*m_active_token_list)[i]->meas[1] = temp;
      }

      // Fan-in
      if ((*m_active_token_list)[i]->node->flags&NODE_FAN_IN)
      {
        temp = (*m_active_token_list)[i]->total_log_prob -
          m_fan_in_log_prob;
        if (temp < (*m_active_token_list)[i]->meas[2])
          (*m_active_token_list)[i]->meas[2] = temp;
      }

      if ((*m_active_token_list)[i]->node->flags&NODE_FAN_OUT)
      {
        // Fan-out
        temp = (*m_active_token_list)[i]->total_log_prob -
          m_fan_out_log_prob;
        if (temp < (*m_active_token_list)[i]->meas[3])
          (*m_active_token_list)[i]->meas[3] = temp;
      }

      // Normal state beam
      /*if (!((*m_active_token_list)[i]->node->flags&(NODE_FAN_IN|NODE_FAN_OUT)))
        {
        float log_prob = (*m_active_token_list)[i]->total_log_prob;
        TPLexPrefixTree::Token *cur_token = (*m_active_token_list)[i]->node->token_list;
        temp = 0;
        while (cur_token != NULL)
        {
        if (log_prob - cur_token->total_log_prob < temp)
        temp = log_prob - cur_token->total_log_prob;
        cur_token = cur_token->next_node_token;
        }
        if (temp < (*m_active_token_list)[i]->meas[3])
        (*m_active_token_list)[i]->meas[3] = temp;
        }

        // Fan-out state beam
        if ((*m_active_token_list)[i]->node->flags&NODE_FAN_OUT)
        {
        float log_prob = (*m_active_token_list)[i]->total_log_prob;
        TPLexPrefixTree::Token *cur_token = (*m_active_token_list)[i]->node->token_list;
        temp = 0;
        while (cur_token != NULL)
        {
        if (log_prob - cur_token->total_log_prob < temp)
        temp = log_prob - cur_token->total_log_prob;
        cur_token = cur_token->next_node_token;
        }
        if (temp < (*m_active_token_list)[i]->meas[4])
        (*m_active_token_list)[i]->meas[4] = temp;
        }

        // Fan-in state beam
        if ((*m_active_token_list)[i]->node->flags&NODE_FAN_IN)
        {
        float log_prob = (*m_active_token_list)[i]->total_log_prob;
        TPLexPrefixTree::Token *cur_token = (*m_active_token_list)[i]->node->token_list;
        temp = 0;
        while (cur_token != NULL)
        {
        if (log_prob - cur_token->total_log_prob < temp)
        temp = log_prob - cur_token->total_log_prob;
        cur_token = cur_token->next_node_token;
        }
        if (temp < (*m_active_token_list)[i]->meas[5])
        (*m_active_token_list)[i]->meas[5] = temp;
        }*/

      if ((*m_active_token_list)[i]->node->flags&NODE_USE_WORD_END_BEAM)
      {
        // Word end
        temp = (*m_active_token_list)[i]->total_log_prob - m_best_we_log_prob;
        if (temp < (*m_active_token_list)[i]->meas[5])
          (*m_active_token_list)[i]->meas[5] = temp;
      }
    }
  }
}
#endif

void TokenPassSearch::get_path(HistoryVector &vec, bool use_best_token,
                               LMHistory *limit)
{
  if (m_print_text_result) {
    fprintf(stderr, "TokenPassSearch::get_path() should not be used with "
            "m_print_text_results set true\n");
    abort();
  }

  const TPLexPrefixTree::Token & token =
    use_best_token ? get_best_final_token() : get_first_token();

  vec.clear();
  LMHistory *hist = token.lm_history;
  while (hist->last().word_id() >= 0) {
    if (hist == limit)
      break;
    vec.push_back(hist);
    hist = hist->previous;
  }
  assert(limit == NULL || hist->last().word_id() >= 0);
}

void TokenPassSearch::write_word_history(FILE *file, bool get_best_path)
{
  if (!m_generate_word_graph) {
    throw WordGraphNotGenerated();
  }

  // Use globally best token if no best final token.
  const TPLexPrefixTree::Token & token =
    get_best_path ? get_best_final_token() : get_first_token();

  // Fetch the best path if requested, otherwise the common path to
  // all tokens and not printed yet

  std::vector<TPLexPrefixTree::WordHistory*> stack;

  TPLexPrefixTree::WordHistory *word_history = token.word_history;
  bool collect = get_best_path;
  while (word_history != NULL) {

    // If not getting the whole best path, we want to fetch only the
    // last continuous sequence of history nodes with reference_count
    // one.  There might be nodes with reference_count greater than 1
    // even if we are collecting.
    if (!get_best_path && collect && word_history->reference_count > 1) {
      stack.clear();
      collect = false;
    }

    if (word_history->printed)
      break;
    if (word_history->previous != NULL &&
        word_history->previous->reference_count == 1)
      collect = true;
    if (collect && word_history->word_id >= 0)
      stack.push_back(word_history);
    word_history = word_history->previous;
  }

  // Print path
  for (int i = stack.size() - 1; i >= 0; i--) {
    stack[i]->printed = true;
    string word(m_vocabulary.word(stack[i]->word_id));
    fprintf(file, "%s ", word.c_str());

    int spaces = 16 - word.length();
    if (spaces < 1)
      spaces = 1;
    for (int j = 0; j < spaces; j++)
      fputc(' ', file);
    fprintf(file, "%d\t%d\t%d\t%.3f\t%.3f\t%.3f\n", stack[i]->end_frame,
            stack[i]->lex_node_id, stack[i]->graph_node_id,
            stack[i]->am_log_prob, stack[i]->lm_log_prob,
            get_token_log_prob(stack[i]->cum_am_log_prob,
                               stack[i]->cum_lm_log_prob));
  }
  if (get_best_path)
    fprintf(file, "\n");

  fflush(file);
}

void TokenPassSearch::print_lm_history(FILE *file, bool get_best_path)
{
  const TPLexPrefixTree::Token & token =
    get_best_path ? get_best_final_token() : get_first_token();

  std::vector<LMHistory *> stack;

  LMHistory * lm_history = token.lm_history;
  bool collect = get_best_path;
  while (lm_history != NULL) {

    // If not getting the whole best path, we want to fetch only the
    // last continuous sequence of history nodes with reference_count
    // one.  There might be nodes with reference_count greater than 1
    // even if we are collecting.
    if (!get_best_path && collect && lm_history->reference_count > 1) {
      stack.clear();
      collect = false;
    }

    if (lm_history->printed)
      break;
    if (lm_history->previous != NULL &&
        lm_history->previous->reference_count == 1)
      collect = true;
    if (collect && lm_history->last().word_id() >= 0)
      stack.push_back(lm_history);
    lm_history = lm_history->previous;
  }

  // Print path
  for (int i = stack.size() - 1; i >= 0; i--) {
    stack[i]->printed = true;
    fprintf(file, "%s ",
            m_vocabulary.word(stack[i]->last().word_id()).c_str());
  }
  if (get_best_path)
    fprintf(file, "\n");

#ifdef PRUNING_MEASUREMENT
  for (i = 0; i < 6; i++)
    fprintf(out, "meas%i: %.3g\n", i, (*m_active_token_list)[best_token]->meas[i]);
#endif

#ifdef COUNT_LM_LA_CACHE_MISS
  fprintf(out, "Count: ");
  for (i = 0; i < MAX_LEX_TREE_DEPTH; i++)
    fprintf(out, "%i ", lm_la_cache_count[i]);
  fprintf(out, "\n");
  fprintf(out, "Miss: ");
  for (i = 0; i < MAX_LEX_TREE_DEPTH; i++)
    fprintf(out, "%i ", lm_la_cache_miss[i]);
  fprintf(out, "\n");
  fprintf(out, "WordCount: %d\n", lm_la_word_cache_count);
  fprintf(out, "WordMiss: %d\n", lm_la_word_cache_miss);
#endif

  fflush(file);
}

float TokenPassSearch::get_am_log_prob(bool get_best_path) const
{
  const TPLexPrefixTree::Token & token =
    get_best_path ? get_best_final_token() : get_first_token();
  return token.am_log_prob;
}

float TokenPassSearch::get_lm_log_prob(bool get_best_path) const
{
  const TPLexPrefixTree::Token & token =
    get_best_path ? get_best_final_token() : get_first_token();
  return token.lm_log_prob;
}

float TokenPassSearch::get_total_log_prob(bool get_best_path) const
{
  const TPLexPrefixTree::Token & token =
    get_best_path ? get_best_final_token() : get_first_token();
  return token.total_log_prob;
}

const std::vector<LMHistory::Word> & TokenPassSearch::get_word_repository() const
{
  return m_word_repository;
}

const WordClasses * TokenPassSearch::get_word_classes() const
{
#ifdef ENABLE_WORDCLASS_SUPPORT
  return m_word_classes;
#else
  return NULL;
#endif
}

const Vocabulary & TokenPassSearch::get_vocabulary() const
{
  return m_vocabulary;
}

const NGram * TokenPassSearch::get_ngram() const
{
  return m_ngram;
}

const TPLexPrefixTree::Token &
TokenPassSearch::get_best_final_token() const
{
  if (m_best_final_token != NULL)
    return *m_best_final_token;

  const TPLexPrefixTree::Token * best_final_token = NULL;
  const TPLexPrefixTree::Token * best_nonfinal_token = NULL;

  token_list_type::const_iterator iter = m_active_token_list->begin();
  for (; iter != m_active_token_list->end(); ++iter) {
    TPLexPrefixTree::Token * token = *iter;

    if (token == NULL) {
      continue;
    }

    if (token->node->flags & NODE_FINAL) {
      if ((best_final_token == NULL)
          || (token->total_log_prob > best_final_token->total_log_prob)) {
        best_final_token = token;
      }
    }
    else {
      if ((best_nonfinal_token == NULL)
          || (token->total_log_prob
              > best_nonfinal_token->total_log_prob)) {
        best_nonfinal_token = token;
      }
    }
  }

  if (best_final_token != NULL) {
    return *best_final_token;
  }
  else if (best_nonfinal_token != NULL) {
    fprintf(stderr,
            "WARNING: No tokens in final nodes. The result will be incomplete. Try increasing beam.\n");
    return *best_nonfinal_token;
  }
  else {
    assert(false);
  }
}

const TPLexPrefixTree::Token &
TokenPassSearch::get_first_token() const
{
  for (int i = 0; i < m_active_token_list->size(); i++) {
    const TPLexPrefixTree::Token * token = (*m_active_token_list)[i];

    if (token != NULL)
      return *token;
  }

  assert(false);
}

void TokenPassSearch::print_state_history(FILE *file)
{
  std::vector<TPLexPrefixTree::StateHistory*> stack;
  get_state_history(stack);

  for (int i = stack.size() - 1; i >= 0; i--) {
    int end_time = i == 0 ? m_frame : stack[i - 1]->start_time;
    fprintf(file, "%i %i %i\n", stack[i]->start_time, end_time,
            stack[i]->hmm_model);
  }
  //fprintf(file, "DEBUG: %s\n", state_history_string().c_str());
}

std::string TokenPassSearch::state_history_string()
{
  std::string str;
  std::vector<TPLexPrefixTree::StateHistory*> stack;
  get_state_history(stack);

  std::ostringstream buf;
  for (int i = stack.size() - 1; i >= 0; i--)
    buf << stack[i]->start_time << " " << stack[i]->hmm_model << " ";
  buf << m_frame;

  return buf.str();
}

void TokenPassSearch::get_state_history(
  std::vector<TPLexPrefixTree::StateHistory*> &stack)
{
  const TPLexPrefixTree::Token & token = get_best_final_token();

  // Determine the state sequence
  stack.clear();
  TPLexPrefixTree::StateHistory *state = token.state_history;
  while (state != NULL && state->previous != NULL) {
    stack.push_back(state);
    state = state->previous;
  }
}

void TokenPassSearch::propagate_tokens(void)
{
  int i;

#if (defined PRUNING_EXTENSIONS || defined PRUNING_MEASUREMENT || defined FAN_IN_PRUNING || defined EQ_WC_PRUNING || defined EQ_DEPTH_PRUNING)
  for (i = 0; i < MAX_LEX_TREE_DEPTH/2; i++)
  {
    m_depth_llh[i] = -1e20;
  }
  int j;
  i = 0;
  while (i < MAX_WC_COUNT && m_wc_llh[i++] < -9e19)
    m_min_word_count++;
  for (i = 0; i < MAX_WC_COUNT; i++)
    m_wc_llh[i] = -1e20;

  m_fan_in_log_prob = -1e20;

#endif

  m_fan_out_log_prob = -1e20;

  m_best_log_prob = -1e20;
  m_best_we_log_prob = -1e20;
  m_worst_log_prob = 0;
  //m_lexicon.clear_node_token_lists();
  clear_active_node_token_lists();

  for (i = 0; i < m_active_token_list->size(); i++) {
    TPLexPrefixTree::Token *token = (*m_active_token_list)[i];
    if (token) {
      propagate_token(token);
    }
  }
}

void TokenPassSearch::propagate_token(TPLexPrefixTree::Token *token)
{
  TPLexPrefixTree::Node *source_node = token->node;
  int i;

  // Iterate all the arcs leaving the token's node.
  for (i = 0; i < source_node->arcs.size(); i++) {
    move_token_to_node(token, source_node->arcs[i].next,
                       source_node->arcs[i].log_prob);
  }

  //XXX
  std::vector<int>::const_iterator iter = m_hesitation_ids.begin();
  for (; iter != m_hesitation_ids.end(); ++iter) {
    if (token->lm_history->last().word_id() == *iter) {
      assert(!m_fsa_lm);
      assert(!m_generate_word_graph);

      // Add sentence_end and sentence_start and propagate the token with new word history
      LMHistory * temp_lm_history = token->lm_history;

      token->lm_history = acquire_lmhist(
        &m_word_repository[m_sentence_end_id], token->lm_history);
      hist::link(token->lm_history);
      token->lm_history->word_start_frame = m_frame;
      token->lm_history = acquire_lmhist(
        &m_word_repository[m_sentence_start_id], token->lm_history);
      hist::link(token->lm_history);
      token->lm_history->word_start_frame = m_frame;
      token->lm_hist_code = compute_lm_hist_hash_code(token->lm_history);

      // Iterate all the arcs leaving the token's node.
      for (i = 0; i < source_node->arcs.size(); i++) {
        if (source_node->arcs[i].next != source_node) // Skip self transitions
          move_token_to_node(token, source_node->arcs[i].next,
                             source_node->arcs[i].log_prob);
      }

      hist::unlink(token->lm_history->previous, &m_lmh_pool);
      hist::unlink(token->lm_history, &m_lmh_pool);
      token->lm_history = temp_lm_history;
    }
  }
  //XXX

  if ((source_node->flags & NODE_INSERT_WORD_BOUNDARY) != 0
      && m_generate_word_graph) {
    throw InvalidSetup("Nodes should not have NODE_INSERT_WORD_BOUNRDARY "
                       "when word graphs are used.");
  }

  if ((source_node->flags & NODE_INSERT_WORD_BOUNDARY)
      && (m_word_boundary_id > 0)) {
    if (token->lm_history->last().word_id() != m_word_boundary_id) {
      assert(!m_fsa_lm);
      assert(!m_generate_word_graph);

      // Add word_boundary and propagate the token with new word history
      LMHistory * temp_lm_history = token->lm_history;

      token->word_start_frame = -1; // FIXME? If m_frame, causes an assert
      append_to_word_history(*token,
                             m_word_repository[m_word_boundary_id]);
      token->cur_lm_log_prob = token->lm_log_prob;

      // Iterate all the arcs leaving the token's node.
      for (i = 0; i < source_node->arcs.size(); i++) {
        if (source_node->arcs[i].next != source_node) // Skip self transitions
          move_token_to_node(token, source_node->arcs[i].next,
                             source_node->arcs[i].log_prob);
      }

      hist::unlink(token->lm_history, &m_lmh_pool);
      token->lm_history = temp_lm_history;
    }
  }
}

void TokenPassSearch::append_to_word_history(TPLexPrefixTree::Token & token,
                                             const LMHistory::Word & word)
{
  token.lm_history = acquire_lmhist(&word, token.lm_history);
  hist::link(token.lm_history);
  token.lm_history->word_start_frame = m_frame;
  update_lm_log_prob(token);
}

void
TokenPassSearch::move_token_to_node(TPLexPrefixTree::Token *token,
                                    TPLexPrefixTree::Node *node,
                                    float transition_score)
{
  // FIXME: remove debug
  //   printf("src: %d\t%d\t(%03.3f, %03.3f, %03.3f, %03.3f)\t",
  //     m_frame, node->node_id,
  //     token->am_log_prob, token->lm_log_prob,
  //     token->word_history->am_log_prob,
  //     token->word_history->lm_log_prob);
  //   debug_print_token_lm_history(0, *token);

  TPLexPrefixTree::Token updated_token;
  updated_token.node = node;
  updated_token.depth = token->depth;
  updated_token.am_log_prob = token->am_log_prob
    + m_transition_scale * transition_score;
  updated_token.lm_log_prob = token->lm_log_prob;
  updated_token.word_count = token->word_count;
  updated_token.fsa_lm_node = token->fsa_lm_node;
  updated_token.lm_hist_code = token->lm_hist_code;
  updated_token.lm_history = token->lm_history;
  updated_token.word_history = token->word_history;
  updated_token.state_history = token->state_history;
  updated_token.word_start_frame = token->word_start_frame;

  // Whenever new history structures are created, they are adopted by
  // the automatic structures below.  This ensures that the history
  // structures are destroyed in the end of the scope of the automatic
  // structures unless other links have been created too.  Now we do
  // not have to manually take care of whether the histories have been
  // linked or not.
  hist::Auto<LMHistory> auto_lm_history;
  hist::Auto<TPLexPrefixTree::WordHistory> auto_word_history;
  hist::Auto<TPLexPrefixTree::StateHistory> auto_state_history;

  if (m_generate_word_graph && token->word_history->lex_node_id !=
      word_graph.nodes[token->recent_word_graph_node].lex_node_id)
  {
    fprintf(stderr, 
            "frame %d: word_history->lex_node_id (%d) != recent (%d)\n",
            m_frame, token->word_history->lex_node_id,
            word_graph.nodes[token->recent_word_graph_node].lex_node_id);
    debug_print_token_lm_history(stderr, *token);
  }

  if (updated_token.node != token->node) {
    // Store old word id for possible word history generation
    int old_lm_history_word_id = token->lm_history->last().word_id();

    if (updated_token.node->flags & NODE_FIRST_STATE_OF_WORD) {
      assert(updated_token.word_start_frame < 0);
      updated_token.word_start_frame = m_frame;
    }

    // Moving to another node
    if (!(updated_token.node->flags & NODE_AFTER_WORD_ID)) {

      // If the node has a word identity, add the word into the word history of
      // the token.
      int word_id = updated_token.node->word_id;
      if (word_id != -1) {
        const LMHistory::Word & word = m_word_repository[word_id];

        // Prune words that don't exist (or whose all multiword
        // components don't exist) in the language model.
#ifdef ENABLE_MULTIWORD_SUPPORT
        if (!m_split_multiwords) {
          if (word.lm_id() < 0) {
            return;
          }
        }
        else {
          for (int i = 0; i < word.num_components(); ++i) {
            if (word.component(i).lm_id < 0)
              return;
          }
        }
#else
        if (word.lm_id() < 0) {
          return;
        }
#endif

        // Prune two subsequent word boundaries
        if ((word_id == m_word_boundary_id)
            && (token->lm_history->last().word_id()
                == m_word_boundary_id)) {
          return;
        }

        // Add the word to lm_history.
        assert(updated_token.word_start_frame >= 0);
        updated_token.lm_history = acquire_lmhist(&word,
                                                 token->lm_history);
        updated_token.lm_history->word_start_frame =
          updated_token.word_start_frame;
        updated_token.word_start_frame = -1;
        auto_lm_history.adopt(updated_token.lm_history, &m_lmh_pool);

        update_lm_log_prob(updated_token);

        updated_token.cur_lm_log_prob = updated_token.lm_log_prob;
        updated_token.word_count++;
        if (m_use_sentence_boundary && word_id == m_sentence_end_id) {
          // Sentence boundaries not allowed in the middle of the
          // recognition segment if we are generating a word graph.
          // That is mainly because srilm-toolkit can not rescore such
          // lattices.  Other reason is that silences in lattices will
          // be filled with <s> <w> </s> branches.
          if (m_generate_word_graph)
            return;

          // Add sentence start and word boundary to the LM history
          // after sentence_end_id
          updated_token.lm_history = acquire_lmhist(
            &m_word_repository[m_sentence_start_id],
            updated_token.lm_history);
          updated_token.lm_history->word_start_frame = m_frame;
          if (m_word_boundary_id > 0) {
            updated_token.lm_history = acquire_lmhist(
              &m_word_repository[m_word_boundary_id],
              updated_token.lm_history);
            updated_token.lm_history->word_start_frame = m_frame;
          }
          auto_lm_history.adopt(updated_token.lm_history, &m_lmh_pool);

          if (m_fsa_lm) {
            updated_token.fsa_lm_node = m_fsa_lm->initial_node_id();
            if (m_word_boundary_id > 0) {
              assert(m_word_repository[m_word_boundary_id].lm_id() >= 0);
              updated_token.fsa_lm_node =
                m_fsa_lm->walk(updated_token.fsa_lm_node,
                               m_word_repository[m_word_boundary_id].lm_id(),
                               NULL);
            }
          }
          else
            updated_token.lm_hist_code =
              compute_lm_hist_hash_code(updated_token.lm_history);
        }
      }
      else {
        // LM probability not updated yet. Use either previous LM
        // probability or language model lookahead.
        if ((updated_token.node->possible_word_id_list.size() > 0)
            && (m_lm_lookahead > 0)) {
          updated_token.cur_lm_log_prob = updated_token.lm_log_prob
            + get_lm_lookahead_score(token->lm_history,
                                     updated_token.node, updated_token.depth);
        }
        else {
          updated_token.cur_lm_log_prob = token->cur_lm_log_prob;
        }
      }
    }
    else
      updated_token.cur_lm_log_prob = updated_token.lm_log_prob;

    if (m_keep_state_segmentation && updated_token.node->state != NULL) {
      updated_token.state_history =
        new TPLexPrefixTree::StateHistory(updated_token.node->state->model,
                                          m_frame, token->state_history);
      auto_state_history.adopt(updated_token.state_history);
    }

    // Update duration probability
    updated_token.dur = 0;
    updated_token.depth = token->depth + 1;
    float duration_log_prob = 0;
    if (token->node->state != NULL) {
      // Add duration probability
      int temp_dur = token->dur + 1;
      duration_log_prob = m_duration_scale
        * token->node->state->duration.get_log_prob(temp_dur);
      updated_token.am_log_prob += duration_log_prob;
    }

    // When the token moves to the first state of the next word, update word_history
    // structure (for word graph) from lm_history.
    if (m_generate_word_graph
        && (updated_token.node->flags & NODE_FIRST_STATE_OF_WORD)) {
      // Add symbol from the LMHistory
      updated_token.word_history = 
        new TPLexPrefixTree::WordHistory(old_lm_history_word_id, m_frame,
                                         updated_token.word_history);
      updated_token.word_history->lex_node_id =
        updated_token.node->node_id;
      auto_word_history.adopt(updated_token.word_history);
      updated_token.word_history->cum_am_log_prob = token->am_log_prob
        + m_transition_scale * transition_score + duration_log_prob;
      updated_token.word_history->cum_lm_log_prob = token->lm_log_prob;
      updated_token.word_history->am_log_prob =
        updated_token.word_history->cum_am_log_prob
        - updated_token.word_history->previous->cum_am_log_prob;
      updated_token.word_history->lm_log_prob =
        updated_token.word_history->cum_lm_log_prob
        - updated_token.word_history->previous->cum_lm_log_prob;
      updated_token.word_history->end_frame = m_frame;
    }

    updated_token.cur_am_log_prob = updated_token.am_log_prob;
  }
  else {
    // Self transition
    updated_token.dur = token->dur + 1;
    if (updated_token.dur > MAX_STATE_DURATION && token->node->state != NULL
        && token->node->state->duration.is_valid_duration_model())
      return; // Maximum state duration exceeded, discard token
    updated_token.depth = token->depth;
    updated_token.cur_am_log_prob = token->cur_am_log_prob
      + m_transition_scale * transition_score;
    updated_token.cur_lm_log_prob = token->cur_lm_log_prob;
  }

  if ((updated_token.node->flags & NODE_FAN_IN_FIRST)
      || updated_token.node == m_lexicon.root()
      || (updated_token.node->flags & NODE_SILENCE_FIRST)) {
    updated_token.depth = 0;
  }

  if (updated_token.node->flags & NODE_SILENCE_FIRST
      && updated_token.lm_history->word_first_silence_frame == -1) {
    updated_token.lm_history->word_first_silence_frame = m_frame;
  }

  if (updated_token.node->state == NULL) {

    // Moving to a node without HMM state, pass through immediately.

    // Try beam pruning
    updated_token.total_log_prob =
      get_token_log_prob(updated_token.cur_am_log_prob,
                         updated_token.cur_lm_log_prob);
    if (((updated_token.node->flags & NODE_USE_WORD_END_BEAM)
         && updated_token.total_log_prob
         < m_best_we_log_prob - m_current_we_beam)
        || updated_token.total_log_prob
        < m_best_log_prob - m_current_glob_beam) {
      return;
    }

    // Create temporary token for propagation.
    TPLexPrefixTree::Token temp_token;
    temp_token.node = updated_token.node;
    temp_token.next_node_token = NULL;
    temp_token.am_log_prob = updated_token.am_log_prob;
    temp_token.lm_log_prob = updated_token.lm_log_prob;
    temp_token.cur_am_log_prob = updated_token.cur_am_log_prob;
    temp_token.cur_lm_log_prob = updated_token.cur_lm_log_prob;
    temp_token.total_log_prob = updated_token.total_log_prob;
    temp_token.lm_history = updated_token.lm_history;
    temp_token.lm_hist_code = updated_token.lm_hist_code;
    temp_token.fsa_lm_node = updated_token.fsa_lm_node;
    temp_token.dur = 0;
    temp_token.word_count = updated_token.word_count;
    temp_token.state_history = updated_token.state_history;
    temp_token.word_history = updated_token.word_history;
    temp_token.word_start_frame = updated_token.word_start_frame;
    if (m_generate_word_graph) {
      copy_word_graph_info(token, &temp_token);
      if (updated_token.node->flags & NODE_FIRST_STATE_OF_WORD)
        build_word_graph(&temp_token);
    }

#ifdef PRUNING_MEASUREMENT
    for (int i = 0; i < 6; i++)
      temp_token.meas[i] = token->meas[i];
#endif

    temp_token.depth = updated_token.depth;
    //temp_token.token_path = token->token_path;
    //temp_token.token_path->link();

    propagate_token(&temp_token);
    //TPLexPrefixTree::PathHistory::unlink(temp_token.token_path);
  }
  else
  {
    // Normal propagation
    TPLexPrefixTree::Token *new_token;
    TPLexPrefixTree::Token *similar_lm_hist;
    float ac_log_prob = m_acoustics->log_prob(
      updated_token.node->state->model);

    updated_token.am_log_prob += ac_log_prob;
    updated_token.cur_am_log_prob += ac_log_prob;
    updated_token.total_log_prob = 
      get_token_log_prob(updated_token.cur_am_log_prob,
                         updated_token.cur_lm_log_prob);
    
    // Apply beam pruning
    if (updated_token.node->flags & NODE_USE_WORD_END_BEAM) {
      if (updated_token.total_log_prob
          < m_best_we_log_prob - m_current_we_beam) {
        return;
      }
    }
    if (updated_token.total_log_prob
        < m_best_log_prob
        - m_current_glob_beam
#ifdef PRUNING_EXTENSIONS
        || ((updated_token.node->flags&NODE_FAN_IN)?
            (updated_token.total_log_prob < m_fan_in_log_prob - m_fan_in_beam) :
            ((!(updated_token.node->flags&(NODE_FAN_IN|NODE_FAN_OUT)) &&
              (updated_token.total_log_prob<m_wc_llh[updated_token.word_count-m_min_word_count]-
               m_eq_wc_beam ||
               (!(updated_token.node->flags&(NODE_AFTER_WORD_ID)) &&
                updated_token.total_log_prob<m_depth_llh[updated_token.depth/2]-m_eq_depth_beam)))))
#endif
#ifdef FAN_IN_PRUNING
        || ((updated_token.node->flags&NODE_FAN_IN) &&
            updated_token.total_log_prob < m_fan_in_log_prob - m_fan_in_beam)
#endif
#ifdef EQ_WC_PRUNING
        || (!(updated_token.node->flags&(NODE_FAN_IN|NODE_FAN_OUT)) &&
            (updated_token.total_log_prob<m_wc_llh[updated_token.word_count-m_min_word_count]-
             m_eq_wc_beam))
#endif
#ifdef EQ_DEPTH_PRUNING
        || ((!(updated_token.node->flags&(NODE_FAN_IN|NODE_FAN_OUT|NODE_AFTER_WORD_ID)) &&
             updated_token.total_log_prob<m_depth_llh[updated_token.depth/2]-m_eq_depth_beam))
#endif
#ifdef FAN_OUT_PRUNING
        || ((updated_token.node->flags&NODE_FAN_OUT) &&
            updated_token.total_log_prob < m_fan_out_log_prob - m_fan_out_beam)
#endif
      ) {
      return;
    }

#ifdef STATE_PRUNING
    if (updated_token.node->flags&(NODE_FAN_OUT|NODE_FAN_IN))
    {
      TPLexPrefixTree::Token *cur_token = updated_token.node->token_list;
      while (cur_token != NULL)
      {
        if (updated_token.total_log_prob <
            cur_token->total_log_prob - m_state_beam)
        {
          return;
        }
        cur_token = cur_token->next_node_token;
      }
    }
#endif

    if (updated_token.node->token_list == NULL) {
      // No tokens in the node,  create new token
      m_active_node_list.push_back(updated_token.node); // Mark the node active
      new_token = acquire_token();
      new_token->node = updated_token.node;
      new_token->next_node_token = updated_token.node->token_list;
      updated_token.node->token_list = new_token;
      // Add to the list of propagated tokens
      if (updated_token.node->flags & NODE_USE_WORD_END_BEAM)
        m_word_end_token_list->push_back(new_token);
      else
        m_new_token_list->push_back(new_token);
    }
    else {
      // Recombination of search paths that are identical up to
      // m_similar_lm_hist_span words.
      if (m_fsa_lm) {
        similar_lm_hist = find_similar_fsa_token(
          updated_token.fsa_lm_node,
          updated_token.node->token_list);
      }
      else {
        similar_lm_hist = find_similar_lm_history(
          updated_token.lm_history, updated_token.lm_hist_code,
          updated_token.node->token_list);
      }

      if (similar_lm_hist == NULL)
      {
        // New word history for this node, create new token
        new_token = acquire_token();
        new_token->node = updated_token.node;
        new_token->next_node_token = updated_token.node->token_list;
        updated_token.node->token_list = new_token;
        // Add to the list of propagated tokens
        if (updated_token.node->flags & NODE_USE_WORD_END_BEAM)
          m_word_end_token_list->push_back(new_token);
        else
          m_new_token_list->push_back(new_token);
      }
      else
      {
        // Found the same word history, pick the best token.
        if (updated_token.total_log_prob
            > similar_lm_hist->total_log_prob) {
          // Replace the previous token
          new_token = similar_lm_hist;
          hist::unlink(new_token->lm_history, &m_lmh_pool);
          hist::unlink(new_token->word_history);
          hist::unlink(new_token->state_history);

          //TPLexPrefixTree::PathHistory::unlink(new_token->token_path);
        }
        else
        {
          // Discard this token
          return;
        }
      }
    }
    if (updated_token.node->flags & NODE_USE_WORD_END_BEAM) {
      if (updated_token.total_log_prob > m_best_we_log_prob)
        m_best_we_log_prob = updated_token.total_log_prob;
    }
    if (updated_token.total_log_prob > m_best_log_prob)
      m_best_log_prob = updated_token.total_log_prob;

#if (defined PRUNING_EXTENSIONS || defined PRUNING_MEASUREMENT)
    if (updated_token.node->flags&NODE_FAN_IN)
    {
      if (updated_token.total_log_prob > m_fan_in_log_prob)
        m_fan_in_log_prob = updated_token.total_log_prob;
      if (m_wc_llh[updated_token.word_count-m_min_word_count] < -1e19)
        m_wc_llh[updated_token.word_count-m_min_word_count] = -1e18;
    }
    else if (!(updated_token.node->flags&(NODE_FAN_IN|NODE_FAN_OUT)))
    {
      if (!(updated_token.node->flags&NODE_AFTER_WORD_ID) &&
          updated_token.total_log_prob > m_depth_llh[updated_token.depth/2])
        m_depth_llh[updated_token.depth/2] = updated_token.total_log_prob;
      if (updated_token.total_log_prob > m_wc_llh[updated_token.word_count-m_min_word_count])
        m_wc_llh[updated_token.word_count-m_min_word_count] = updated_token.total_log_prob;
    }
    else if (m_wc_llh[updated_token.word_count-m_min_word_count] < -1e19)
      m_wc_llh[updated_token.word_count-m_min_word_count] = -1e18;
#endif
#ifdef FAN_IN_PRUNING
    if (updated_token.node->flags&NODE_FAN_IN)
    {
      if (updated_token.total_log_prob > m_fan_in_log_prob)
        m_fan_in_log_prob = updated_token.total_log_prob;
    }
#endif
#ifdef EQ_WC_PRUNING
    if (!(updated_token.node->flags&(NODE_FAN_IN|NODE_FAN_OUT)))
    {
      if (updated_token.total_log_prob > m_wc_llh[updated_token.word_count-m_min_word_count])
        m_wc_llh[updated_token.word_count-m_min_word_count] = updated_token.total_log_prob;
    }
    else if (m_wc_llh[updated_token.word_count-m_min_word_count] < -1e19)
      m_wc_llh[updated_token.word_count-m_min_word_count] = -1e18;
#endif
#ifdef EQ_DEPTH_PRUNING
    if (!(updated_token.node->flags&(NODE_FAN_IN|NODE_FAN_OUT|NODE_AFTER_WORD_ID)))
    {
      if (updated_token.total_log_prob > m_depth_llh[updated_token.depth/2])
        m_depth_llh[updated_token.depth/2] = updated_token.total_log_prob;
    }
#endif

#if (defined FAN_OUT_PRUNING || defined PRUNING_MEASUREMENT)
    if (updated_token.node->flags&NODE_FAN_OUT)
    {
      if (updated_token.total_log_prob > m_fan_out_log_prob)
        m_fan_out_log_prob = updated_token.total_log_prob;
    }
#endif

    if (updated_token.total_log_prob < m_worst_log_prob)
      m_worst_log_prob = updated_token.total_log_prob;

    new_token->lm_history = updated_token.lm_history;
    if (new_token->lm_history != NULL)
      hist::link(new_token->lm_history);
    new_token->lm_hist_code = updated_token.lm_hist_code;
    new_token->fsa_lm_node = updated_token.fsa_lm_node;
    new_token->am_log_prob = updated_token.am_log_prob;
    new_token->cur_am_log_prob = updated_token.cur_am_log_prob;
    new_token->lm_log_prob = updated_token.lm_log_prob;
    new_token->cur_lm_log_prob = updated_token.cur_lm_log_prob;
    new_token->total_log_prob = updated_token.total_log_prob;
    new_token->dur = updated_token.dur;
    new_token->word_count = updated_token.word_count;
    new_token->state_history = updated_token.state_history;
    new_token->word_history = updated_token.word_history;
    new_token->word_start_frame = updated_token.word_start_frame;
    if (updated_token.word_history != NULL)
      hist::link(new_token->word_history);
    if (updated_token.state_history != NULL)
      hist::link(new_token->state_history);

    if (m_generate_word_graph) {
      copy_word_graph_info(token, new_token);
      if ((updated_token.node->flags & NODE_FIRST_STATE_OF_WORD)
          && updated_token.node != token->node)
        build_word_graph(new_token);
    }

#ifdef PRUNING_MEASUREMENT
    for (int i = 0; i < 6; i++)
      new_token->meas[i] = token->meas[i];
#endif

    new_token->depth = updated_token.depth;
    //assert(token->token_path != NULL);
    /*new_token->token_path = new TPLexPrefixTree::PathHistory(
      updated_token.total_log_prob,
      token->token_path->dll + ac_log_prob, updated_token.depth,
      token->token_path);
      new_token->token_path->link();*/
    /*new_token->token_path = token->token_path;
      new_token->token_path->link();*/
  }
}

TPLexPrefixTree::Token*
TokenPassSearch::find_similar_fsa_token(int fsa_lm_node,
                                        TPLexPrefixTree::Token *token_list)
{
  assert(m_fsa_lm);

  TPLexPrefixTree::Token *token = token_list;
  while (token != NULL) {
    if (fsa_lm_node == token->fsa_lm_node)
      break;
    token = token->next_node_token;
  }
  return token;
}

TPLexPrefixTree::Token*
TokenPassSearch::find_similar_lm_history(LMHistory *wh, int lm_hist_code,
                                         TPLexPrefixTree::Token *token_list)
{
  assert(!m_fsa_lm);

  TPLexPrefixTree::Token *cur_token = token_list;
  for (; cur_token != NULL; cur_token = cur_token->next_node_token) {
    if ((lm_hist_code == cur_token->lm_hist_code)
        && (is_similar_lm_history(wh, cur_token->lm_history))) {
      return cur_token;
    }
  }
  return cur_token;  // NULL
}

inline bool TokenPassSearch::is_similar_lm_history(LMHistory *wh1,
                                                   LMHistory *wh2)
{
  LMHistory::ConstReverseIterator iter1 = wh1->rbegin();
  LMHistory::ConstReverseIterator iter2 = wh2->rbegin();

  for (int i = 0; i < m_similar_lm_hist_span; ++i) {
    if ((iter1->word_id == -1) || (iter1->word_id == m_sentence_end_id)) {
      return (iter2->word_id == -1)
        || (iter2->word_id == m_sentence_end_id);
    }

    // Use LM ID instead of word ID so that when using class-based language
    // models, we will use the class information for recombining similar
    // paths. SE 24.9.2012
    if (iter1->lm_id != iter2->lm_id) {
      // Different histories.
      return false;
    }

    ++iter1;
    ++iter2;
  }

  /*    for (int i = 0; i < m_similar_lm_hist_span; i++) {
        if (wh1->last().word_id() == -1
        || wh1->last().word_id() == m_sentence_end_id) {
        if (wh2->last().word_id() == -1
        || wh2->last().word_id() == m_sentence_end_id) {
        // Encountered a context reset in both histories.
        return true;
        }
        else {
        // Different histories.
        return false;
        }
        }

        // Use LM ID instead of word ID so that when using class-based language
        // models, we will use the class information for recombining similar
        // paths. SE 24.9.2012
        if (wh1->last().lm_id() != wh2->last().lm_id()) {
        // Different histories.
        return false;
        }

        wh1 = wh1->previous;
        wh2 = wh2->previous;
        }*/

  // Similar histories up to m_similar_lm_hist_span.
  return true;
}

int TokenPassSearch::compute_lm_hist_hash_code(LMHistory *wh) const
{
  unsigned int code = 0;

  LMHistory::ConstReverseIterator iter = wh->rbegin();
  for (int i = 0; i < m_similar_lm_hist_span; ++i) {
    if (iter->word_id == -1)
      break;

    // Use LM ID instead of word ID so that when using class-based language
    // models, we will use the class information for recombining similar
    // paths.
    code += iter->lm_id;
    code += (code << 10);
    code ^= (code >> 6);

    if (iter->word_id == m_sentence_start_id)
      break;

    ++iter;
  }

  code += (code << 3);
  code ^= (code >> 11);
  code += (code << 15);
  return code & 0x7fffffff;
}

void TokenPassSearch::prune_tokens()
{
  int i;
  int num_active_tokens;
  float beam_limit = m_best_log_prob - m_current_glob_beam; //m_global_beam;
  float we_beam_limit = m_best_we_log_prob - m_current_we_beam;

  if (m_verbose > 1)
    printf("%zd new tokens\n",
           m_new_token_list->size() + m_word_end_token_list->size());

  // At first, remove inactive tokens.
  for (i = 0; i < m_active_token_list->size(); i++) {
    if ((*m_active_token_list)[i] != NULL)
      release_token((*m_active_token_list)[i]);
  }
  m_active_token_list->clear();
  std::vector<TPLexPrefixTree::Token *> * temp = m_active_token_list;
  m_active_token_list = m_new_token_list;
  m_new_token_list = temp;

  // Prune the word end tokens and add them to m_active_token_list
  for (i = 0; i < m_word_end_token_list->size(); i++) {
    if ((*m_word_end_token_list)[i]->total_log_prob < we_beam_limit)
      release_token((*m_word_end_token_list)[i]);
    else
      m_active_token_list->push_back((*m_word_end_token_list)[i]);
  }
  m_word_end_token_list->clear();

  // Fill the token path information
  /*  TPLexPrefixTree::PathHistory *prev_path;
      for (int i = 0; i < m_active_token_list->size(); i++)
      {
      //if ((*m_active_token_list)[i] != NULL)
      {
      prev_path = (*m_active_token_list)[i]->token_path;
      assert( prev_path !=  NULL);
      (*m_active_token_list)[i]->token_path = new TPLexPrefixTree::PathHistory(
      (*m_active_token_list)[i]->total_log_prob,
      m_best_log_prob,
      (*m_active_token_list)[i]->depth,
      prev_path);
      TPLexPrefixTree::PathHistory::unlink(prev_path);
      (*m_active_token_list)[i]->token_path->link();
      }
      }*/

  // Then beam prune the active tokens and find the worst accepted log prob
  // for histogram pruning.
  // Note! After this, the token lists in the nodes are no longer valid.
  num_active_tokens = 0;
  if (m_active_token_list->size() > m_max_num_tokens &&
      m_max_num_tokens > 0)
  {
    // Do also histogram pruning
    // Approximate the worst log prob after beam pruning has been applied.
    if (m_worst_log_prob < beam_limit)
      m_worst_log_prob = beam_limit;
    int bins[NUM_HISTOGRAM_BINS];
    float bin_adv = (m_best_log_prob - m_worst_log_prob)
      / (NUM_HISTOGRAM_BINS - 1);
    float new_min_log_prob;
    memset(bins, 0, NUM_HISTOGRAM_BINS * sizeof(int));

    for (i = 0; i < m_active_token_list->size(); i++)
    {
      float total_log_prob = (*m_active_token_list)[i]->total_log_prob;
      if (total_log_prob
          < beam_limit
#ifdef PRUNING_EXTENSIONS
          || ((flags&NODE_FAN_IN)?
              (total_log_prob < m_fan_in_log_prob - m_fan_in_beam) :
              ((!(flags&(NODE_FAN_IN|NODE_FAN_OUT)) &&
                (total_log_prob < m_wc_llh[(*m_active_token_list)[i]->word_count-m_min_word_count] - m_eq_wc_beam ||
                 (!(flags&(NODE_AFTER_WORD_ID)) &&
                  total_log_prob < m_depth_llh[(*m_active_token_list)[i]->depth/2]-m_eq_depth_beam)))))
#endif
#ifdef FAN_IN_PRUNING
          || ((flags&NODE_FAN_IN) &&
              total_log_prob < m_fan_in_log_prob - m_fan_in_beam)
#endif
#ifdef EQ_WC_PRUNING
          || (!(flags&(NODE_FAN_IN|NODE_FAN_OUT)) &&
              (total_log_prob < m_wc_llh[(*m_active_token_list)[i]->word_count-m_min_word_count] - m_eq_wc_beam))
#endif
#ifdef EQ_DEPTH_PRUNING
          || (!(flags&(NODE_FAN_IN|NODE_FAN_OUT|NODE_AFTER_WORD_ID)) &&
              total_log_prob < m_depth_llh[(*m_active_token_list)[i]->depth/2]-m_eq_depth_beam)
#endif
#ifdef FAN_OUT_PRUNING
          || ((flags&NODE_FAN_OUT) &&
              total_log_prob < m_fan_out_log_prob - m_fan_out_beam)
#endif
        )
      {
        release_token((*m_active_token_list)[i]);
      }
      else
      {
        bins[(int) floorf((total_log_prob - m_worst_log_prob) / bin_adv)]++;
        m_new_token_list->push_back((*m_active_token_list)[i]);
      }
    }
    m_active_token_list->clear();
    temp = m_active_token_list;
    m_active_token_list = m_new_token_list;
    m_new_token_list = temp;
    if (m_verbose > 1)
      printf("%zd tokens after beam pruning\n",
             m_active_token_list->size());
    num_active_tokens = m_active_token_list->size();
    if (num_active_tokens > m_max_num_tokens)
    {
      for (i = 0; i < NUM_HISTOGRAM_BINS - 1; i++) {
        num_active_tokens -= bins[i];
        if (num_active_tokens < m_max_num_tokens)
          break;
      }
      int deleted = 0;
      new_min_log_prob = m_worst_log_prob + (i + 1) * bin_adv;
      for (i = 0; i < m_active_token_list->size(); i++) {
        if ((*m_active_token_list)[i]->total_log_prob
            < new_min_log_prob) {
          release_token((*m_active_token_list)[i]);
          (*m_active_token_list)[i] = NULL;
          deleted++;
        }
      }
      if (m_verbose > 1)
        printf("%zd tokens after histogram pruning\n",
               m_active_token_list->size() - deleted);

      // Pass the new beam limit to next token propagation
      m_current_glob_beam = std::min((m_best_log_prob - new_min_log_prob),
                                     m_global_beam);
      m_current_we_beam = m_current_glob_beam / m_global_beam
        * m_word_end_beam;
    }
  }
  else
  {
    // Only do the beam pruning
    for (i = 0; i < m_active_token_list->size(); i++)
    {
      if ((*m_active_token_list)[i]->total_log_prob < beam_limit
#ifdef PRUNING_EXTENSIONS
          || ((flags&NODE_FAN_IN)?
              (total_log_prob < m_fan_in_log_prob - m_fan_in_beam) :
              ((!(flags&(NODE_FAN_IN|NODE_FAN_OUT)) &&
                (total_log_prob < m_wc_llh[(*m_active_token_list)[i]->word_count-m_min_word_count] - m_eq_wc_beam ||
                 (!(flags&(NODE_AFTER_WORD_ID)) &&
                  total_log_prob < m_depth_llh[(*m_active_token_list)[i]->depth/2]-m_eq_depth_beam)))))
#endif
#ifdef FAN_IN_PRUNING
          || ((flags&NODE_FAN_IN) &&
              total_log_prob < m_fan_in_log_prob - m_fan_in_beam)
#endif
#ifdef EQ_WC_PRUNING
          || (!(flags&(NODE_FAN_IN|NODE_FAN_OUT)) &&
              (total_log_prob < m_wc_llh[(*m_active_token_list)[i]->word_count-m_min_word_count] - m_eq_wc_beam))
#endif
#ifdef EQ_DEPTH_PRUNING
          || (!(flags&(NODE_FAN_IN|NODE_FAN_OUT|NODE_AFTER_WORD_ID)) &&
              total_log_prob < m_depth_llh[(*m_active_token_list)[i]->depth/2]-m_eq_depth_beam)
#endif
#ifdef FAN_OUT_PRUNING
          || ((flags&NODE_FAN_OUT) &&
              total_log_prob < m_fan_out_log_prob - m_fan_out_beam)
#endif
        )
      {
        release_token((*m_active_token_list)[i]);
      }
      else
        m_new_token_list->push_back((*m_active_token_list)[i]);
    }
    m_active_token_list->clear();
    temp = m_active_token_list;
    m_active_token_list = m_new_token_list;
    m_new_token_list = temp;
    if (m_verbose > 1)
      printf("%zd tokens after beam pruning\n",
             m_active_token_list->size());
    if (m_current_glob_beam < m_global_beam)
    {
      // Determine new beam
      m_current_glob_beam = m_current_glob_beam * 1.1;
      m_current_glob_beam = std::min(m_global_beam, m_current_glob_beam);
      m_current_we_beam = m_current_glob_beam / m_global_beam
        * m_word_end_beam;
    }
  }
  if (m_verbose > 1)
    printf("Current beam: %.1f   Word end beam: %.1f\n",
           m_current_glob_beam, m_current_we_beam);
}

void TokenPassSearch::clear_active_node_token_lists(void)
{
  for (int i = 0; i < m_active_node_list.size(); i++)
    m_active_node_list[i]->token_list = NULL;
  m_active_node_list.clear();
}

void TokenPassSearch::set_word_classes(const WordClasses * x)
{
  assert(!m_fsa_lm);
  assert(!m_ngram);

#ifdef ENABLE_WORDCLASS_SUPPORT
  m_word_classes = x;
#endif
}

int TokenPassSearch::set_ngram(NGram *ngram)
{
  assert(!m_fsa_lm);
  m_ngram = ngram;
  // Initialize LM lookahead caches again.
  m_lm_lookahead_initialized = false;
  return create_word_repository();
}

int TokenPassSearch::set_fsa_lm(fsalm::LM *lm)
{
  assert(!m_ngram);
  m_fsa_lm = lm;
  return create_word_repository();
}

int TokenPassSearch::set_lookahead_ngram(NGram *ngram)
{
  assert( m_ngram != NULL || m_fsa_lm != NULL);
  m_lookahead_ngram = ngram;
  return create_word_repository();
}

int TokenPassSearch::create_word_repository()
{
  m_word_repository.clear();
  m_word_repository.resize(m_vocabulary.num_words());

  int num_not_found = 0;

  for (int i = 0; i < m_vocabulary.num_words(); ++i) {
    string word = m_vocabulary.word(i);

    if (m_remove_pronunciation_id) {
      // Remove :[0-9]+ from the tail (pronunciation ID).
      string::const_reverse_iterator iter = word.rbegin();
      if (isdigit(*iter)) {
        ++iter;
        while (isdigit(*iter)) ++iter;
        if (*iter == ':') {
          ++iter;
          int tail_size = iter - word.rbegin();
          word.resize(word.size() - tail_size);
        }
      }
    }

    if (word.size() == 0) {
      if (m_verbose > 0) {
        cerr
          << "TokenPassSearch::create_word_repository: Ignoring empty word in vocabulary."
          << endl;
      }
      continue;
    }

    int lm_id;
    float cm_log_prob;
    find_word_from_lm(i, word, lm_id, cm_log_prob);
    int lookahead_lm_id = find_word_from_lookahead_lm(i, word);

    bool not_found = (lm_id < 0) && (i != 0);
    if ((m_lookahead_ngram != NULL)
        && ((lookahead_lm_id == 0) && (i != 0))) {
      not_found = true;
    }

    m_word_repository.at(i).set_ids(i, lm_id, lookahead_lm_id);

#ifdef ENABLE_MULTIWORD_SUPPORT
    if (word[0] == '_') {
      // Don't treat silences as multiwords.
      m_word_repository[i].add_component(i, lm_id, lookahead_lm_id);
    }
    else {
      cm_log_prob = 0;
      string::iterator component_first = word.begin();
      while (true) {
        string::iterator component_last = find(component_first,
                                               word.end(), '_');
        string component(component_first, component_last);
        if (component.size() == 0) {
          if (m_verbose > 0) {
            cerr
              << "TokenPassSearch::create_word_repository: Ignoring empty multiword component in '"
              << word << "'." << endl;
          }
        }
        else {
          // In theory it's possible that multiword components are not found
          // from the vocabulary as individual words. It won't prevent using
          // them as long as they exist in the language model. Just make sure
          // we have a word ID for every component.
          int word_id = m_vocabulary.add_word(component);
          m_word_repository.resize(m_vocabulary.num_words());

          float component_cm_log_prob;
          find_word_from_lm(word_id, component, lm_id,
                            component_cm_log_prob);
          lookahead_lm_id = find_word_from_lookahead_lm(word_id,
                                                        component);
          m_word_repository[i].add_component(word_id, lm_id,
                                             lookahead_lm_id);
          if ((lm_id < 0) && (i != 0)) {
            not_found = true;
          }
          if ((m_lookahead_ngram != NULL)
              && ((lookahead_lm_id == 0) && (i != 0))) {
            not_found = true;
          }
          cm_log_prob += component_cm_log_prob;
        }

        if (component_last == word.end())
          break;
        component_first = component_last;
        ++component_first;  // Skip the underscore we found last time.
      }
    }
#endif

    m_word_repository[i].set_cm_log_prob(cm_log_prob);

    if (not_found) {
      ++num_not_found;
    }
  }

  // We may have added words to the vocabulary along the way but I think the
  // new words should have been added to the word repository in the end.
  assert(m_vocabulary.num_words() == m_word_repository.size());

  return num_not_found;
}

void TokenPassSearch::find_word_from_lm(int word_id, std::string word,
                                        int & lm_id, float & cm_log_prob) const
{
#ifdef ENABLE_WORDCLASS_SUPPORT
  if (m_word_classes != NULL) {
    try {
      const WordClasses::Membership & class_membership =
        m_word_classes->get_membership(word_id);
      cm_log_prob = class_membership.log_prob;
      word = m_word_classes->get_class_name(class_membership.class_id);
    }
    catch (out_of_range &) {
      // The word does not exist in the class definitions. See if it
      // exists in the language model as it is.
      cm_log_prob = 0;
    }
  }
#else
  cm_log_prob = 0;
#endif

  if (m_fsa_lm) {
    lm_id = m_fsa_lm->symbol_map().index_nothrow(word);
  }
  else {
    if (m_ngram == NULL)
      lm_id = -1;
    else
    {
      lm_id = m_ngram->word_index(word);
      if (lm_id == 0) {
        // Vocabulary::word_index() returns 0 for unknown words.
        lm_id = -1;
      }
    }
  }
}

int TokenPassSearch::find_word_from_lookahead_lm(int word_id,
                                                 std::string word) const
{
  if (m_lookahead_ngram == NULL)
    return 0;

#ifdef ENABLE_WORDCLASS_SUPPORT
  if (m_word_classes != NULL) {
    try {
      const WordClasses::Membership & class_membership =
        m_word_classes->get_membership(word_id);
      word = m_word_classes->get_class_name(class_membership.class_id);
    }
    catch (out_of_range &) {
      // The word does not exist in the class definitions. See if it
      // exists in the language model as it is.
    }
  }
#endif

  return m_lookahead_ngram->word_index(word);
}

#ifdef ENABLE_MULTIWORD_SUPPORT
float TokenPassSearch::split_and_compute_ngram_score(LMHistory * history)
{
  float result = 0;

  for (int skip = 0; skip < history->last().num_components(); ++skip) {
    // Create an n-gram, where n is the model order, from the LM history,
    // starting from each of the components of the last multiword.
    m_history_ngram.clear();
    LMHistory::ConstReverseIterator iter = history->rbegin();
    for (int i = 0; i < skip; ++i) {
      ++iter;
    }
    for (int words_needed = m_ngram->order(); words_needed > 0;
         --words_needed) {
      if (iter->word_id == -1)
        break;  // Reached the beginning of the history.
      m_history_ngram.push_front(iter->lm_id);
      if (iter->word_id == m_sentence_start_id)
        break;  // Reached the beginning of the sentence.
      ++iter;
    }
    result += m_ngram->log_prob(m_history_ngram);
  }

  return result;
}
#endif

void TokenPassSearch::create_history_ngram(LMHistory * history,
                                           int words_needed)
{
  m_history_ngram.clear();

  while (words_needed > 0) {
    if (history->last().word_id() == -1)
      return;  // Reached the beginning of the history.

    m_history_ngram.push_front(history->last().lm_id());
    --words_needed;

    if (history->last().word_id() == m_sentence_start_id)
      return;

    history = history->previous;
  }
}

float TokenPassSearch::compute_ngram_score(LMHistory * history)
{
  if (m_ngram->order() <= 0) {
    return 0;
  }

#ifdef ENABLE_MULTIWORD_SUPPORT
  if (m_split_multiwords) {
    return split_and_compute_ngram_score(history);
  }
  else {
    create_history_ngram(history, m_ngram->order());
    return m_ngram->log_prob(m_history_ngram);
  }
#else
  create_history_ngram(history, m_ngram->order());
  return m_ngram->log_prob(m_history_ngram);
#endif
}

float TokenPassSearch::get_ngram_score(LMHistory *lm_hist, int lm_hist_code)
{
  if (!m_use_lm_cache)
    return compute_ngram_score(lm_hist);

  float score;
  LMScoreInfo *info, *old = NULL;
  bool collision = false;
  int i;

  if (m_lm_score_cache.find(lm_hist_code, &info)) {
    // Check this is correct word history
    LMHistory *wh = lm_hist;
    for (int i = 0; i < info->lm_hist.size(); i++) {
      if (wh->last().word_id() != info->lm_hist[i]) // Also handles 'word_id==-1' case
      {
        collision = true;
        goto get_ngram_score_no_cached;
      }
      wh = wh->previous;
    }
    if (info->lm_hist.size() <= m_ngram->order()) {
      if (wh->last().word_id() != -1
          && wh->last().word_id() != m_sentence_end_id) {
        collision = true;
        goto get_ngram_score_no_cached;
      }
    }
    return info->lm_score;
  }
get_ngram_score_no_cached: if (collision) {
    // In case of collision remove the old item
    if (!m_lm_score_cache.remove_item(lm_hist_code, &old))
      assert( 0);
    delete old;
  }
  score = compute_ngram_score(lm_hist);

  info = new LMScoreInfo;
  info->lm_score = score;
  LMHistory *wh = lm_hist;
  for (i = 0; i <= m_ngram->order() && wh->last().word_id() != -1; i++) {
    info->lm_hist.push_back(wh->last().word_id());
    if (wh->last().word_id() == m_sentence_start_id)
      break;

    wh = wh->previous;
  }
  if (m_lm_score_cache.insert(lm_hist_code, info, &old))
    delete old;

  return score;
}

void TokenPassSearch::advance_fsa_lm(TPLexPrefixTree::Token & token)
{
  const LMHistory::Word & word = token.lm_history->last();

  if (word.lm_id() < 0) {
    cerr << "TokenPassSearch::advance_fsa_lm: Word is not found from the LM: "
         << m_vocabulary.word(word.word_id()) << endl;
    exit(1);
  }
#ifdef ENABLE_MULTIWORD_SUPPORT
  if (m_split_multiwords) {
    for (int i = 0; i < word.num_components(); ++i) {
      token.fsa_lm_node = m_fsa_lm->walk(token.fsa_lm_node, word.lm_id(),
                                         &token.lm_log_prob);
    }
  }
  else {
    token.fsa_lm_node = m_fsa_lm->walk(token.fsa_lm_node, word.lm_id(),
                                       &token.lm_log_prob);
  }
#else
  token.fsa_lm_node = m_fsa_lm->walk(token.fsa_lm_node,
                                     word.lm_id(), &token.lm_log_prob);
#endif
}

void TokenPassSearch::update_lm_log_prob(TPLexPrefixTree::Token & token)
{
  const LMHistory::Word & word = token.lm_history->last();

  if (m_fsa_lm) {
    if (word.word_id() != m_sentence_start_id) {
      advance_fsa_lm(token);
      token.lm_log_prob += word.cm_log_prob();
      token.lm_log_prob += m_insertion_penalty;
    }
  }
  else {  // n-gram language model
    token.lm_hist_code = compute_lm_hist_hash_code(token.lm_history);

    if (word.word_id() != m_sentence_start_id) {
      float lm_score = get_ngram_score(token.lm_history,
                                       token.lm_hist_code);
      token.lm_log_prob += lm_score;
      token.lm_log_prob += word.cm_log_prob();
      token.lm_log_prob += m_insertion_penalty;
#ifdef PRUNING_MEASUREMENT
      if (token.meas[4] > lm_score)
        token.meas[4] = lm_score;
#endif
    }
  }
}

float TokenPassSearch::get_lm_lookahead_score(LMHistory *lm_hist,
                                              TPLexPrefixTree::Node *node, int depth)
{
  // The last word or its last component.
  LMHistory::ConstReverseIterator iter = lm_hist->rbegin();
  int w2 = iter->word_id;
  if (w2 == -1 || w2 == m_sentence_end_id) {
    // This is the beginning of the history or the end of a sentence.
    return 0;
  }

  if (m_lm_lookahead == 1) {
    return get_lm_bigram_lookahead(w2, node, depth);
  }

  // The component before the last component of the last word, or the word
  // before the last word.
  ++iter;
  if (iter == lm_hist->rend()) {
    return 0;
  }
  int w1 = iter->word_id;
  if (w1 == -1 || w1 == m_sentence_end_id) {
    // This is the beginning of the history or the end of a sentence.
    return 0;
  }
  return get_lm_trigram_lookahead(w1, w2, node, depth);
}

float TokenPassSearch::get_lm_bigram_lookahead(int prev_word_id,
                                               TPLexPrefixTree::Node *node, int depth)
{
#ifdef COUNT_LM_LA_CACHE_MISS
  lm_la_cache_count[depth]++;
#endif

  float score;
  if (node->lm_lookahead_buffer.find(prev_word_id, &score))
    return score;

#ifdef COUNT_LM_LA_CACHE_MISS
  lm_la_cache_miss[depth]++;
  lm_la_word_cache_count++;
#endif

  // Not found from cache. Compute the LM bigram lookahead score for every
  // word pair starting with prev_word_id (unless the LM scores have been
  // computed already.
  LMLookaheadScoreList * score_list = NULL;
  if (!lm_lookahead_score_list.find(prev_word_id, &score_list)) {
#ifdef COUNT_LM_LA_CACHE_MISS
    lm_la_word_cache_miss++;
#endif
    // FIXME! Is it necessary to compute the scores for all the words?
    if (m_verbose > 2)
      printf("Compute lm lookahead scores for \'%s'\n",
             m_vocabulary.word(prev_word_id).c_str());
    score_list = new LMLookaheadScoreList;
    LMLookaheadScoreList * old_score_list = NULL;
    if (lm_lookahead_score_list.insert(prev_word_id, score_list,
                                       &old_score_list))
      delete old_score_list; // Old list was removed
    score_list->index = prev_word_id;
    score_list->lm_scores.insert(score_list->lm_scores.end(),
                                 m_word_repository.size(), 0);

    vector<float> extensions;
    m_lookahead_ngram->fetch_bigram_list(
      m_word_repository[prev_word_id].lookahead_lm_id(), extensions);

    // Map lookahead LM IDs to word IDs.
    for (int i = 0; i < m_word_repository.size(); ++i) {
      score_list->lm_scores.at(i) =
        extensions.at(m_word_repository[i].lookahead_lm_id());
    }
  }

  // Compute the lookahead score by selecting the maximum LM score of possible
  // word ends.
  score = -1e10;
  for (int i = 0; i < node->possible_word_id_list.size(); i++) {
    if (score_list->lm_scores[node->possible_word_id_list[i]] > score)
      score = score_list->lm_scores[node->possible_word_id_list[i]];
  }

  // Add the score to the node's buffer
  node->lm_lookahead_buffer.insert(prev_word_id, score, NULL);

  return score;
}

float TokenPassSearch::get_lm_trigram_lookahead(int w1, int w2,
                                                TPLexPrefixTree::Node *node, int depth)
{
#ifdef COUNT_LM_LA_CACHE_MISS
  lm_la_cache_count[depth]++;
#endif

  int index = w1 * m_word_repository.size() + w2;
  float score;
  if (node->lm_lookahead_buffer.find(index, &score))
    return score;

#ifdef COUNT_LM_LA_CACHE_MISS
  lm_la_cache_miss[depth]++;
  lm_la_word_cache_count++;
#endif

  // Not found from cache. Compute the LM trigram lookahead score for every
  // word triplet starting with w1 w2 (unless the LM scores have been computed
  // already).
  LMLookaheadScoreList * score_list = NULL;
  if (!lm_lookahead_score_list.find(index, &score_list)) {
#ifdef COUNT_LM_LA_CACHE_MISS
    lm_la_word_cache_miss++;
#endif
    // FIXME! Is it necessary to compute the scores for all the words?
    if (m_verbose > 2)
      printf("Compute lm lookahead scores for (%s,%s)\n",
             m_vocabulary.word(w1).c_str(),
             m_vocabulary.word(w2).c_str());
    score_list = new LMLookaheadScoreList;
    LMLookaheadScoreList * old_score_list = NULL;
    if (lm_lookahead_score_list.insert(index, score_list, &old_score_list))
      delete old_score_list; // Old list was removed
    score_list->index = index;
    score_list->lm_scores.insert(score_list->lm_scores.end(),
                                 m_word_repository.size(), 0);

    vector<float> extensions;
    m_lookahead_ngram->fetch_trigram_list(
      m_word_repository[w1].lookahead_lm_id(),
      m_word_repository[w2].lookahead_lm_id(), extensions);

    // Map lookahead LM IDs to word IDs.
    for (int i = 0; i < m_word_repository.size(); ++i)
      score_list->lm_scores.at(i) =
        extensions.at(m_word_repository[i].lookahead_lm_id());
  }

  // Compute the lookahead score by selecting the maximum LM score of
  // possible word ends.
  score = -1e10;
  for (int i = 0; i < node->possible_word_id_list.size(); i++) {
    if (score_list->lm_scores[node->possible_word_id_list[i]] > score)
      score = score_list->lm_scores[node->possible_word_id_list[i]];
  }

  // Add the score to the node's buffer
  node->lm_lookahead_buffer.insert(index, score, NULL);

  return score;
}

TPLexPrefixTree::Token*
TokenPassSearch::acquire_token(void)
{
  if (m_token_pool.size() == 0) {
    TPLexPrefixTree::Token *tt =
      new TPLexPrefixTree::Token[TOKEN_RESERVE_BLOCK];
    m_token_dealloc_table.push_back(tt);
    for (int i = 0; i < TOKEN_RESERVE_BLOCK; i++)
      m_token_pool.push_back(&tt[i]);
  }
  TPLexPrefixTree::Token *t = m_token_pool.back();
  m_token_pool.pop_back();
  t->recent_word_graph_node = -1;
  t->word_history = NULL;
  return t;
}

LMHistory *
TokenPassSearch::acquire_lmhist(const LMHistory::Word * last_word, LMHistory * previous) {
  if (m_lmh_pool.size() == 0) {
    LMHistory * lmh_block = new LMHistory[TOKEN_RESERVE_BLOCK];
    m_lmhist_dealloc_table.push_back(lmh_block);
    for (int i = 0; i < TOKEN_RESERVE_BLOCK; i++)
      m_lmh_pool.push_back(&lmh_block[i]);
  }
  LMHistory *lmh = m_lmh_pool.back();
  m_lmh_pool.pop_back();
  lmh->last_word = last_word;
  lmh->previous = previous;
  if (previous) hist::link(lmh->previous);
  return lmh;
}

void TokenPassSearch::release_token(TPLexPrefixTree::Token *token)
{
  if (token->recent_word_graph_node >= 0)
    word_graph.unlink(token->recent_word_graph_node);
  token->recent_word_graph_node = -1;
  hist::unlink(token->lm_history, &m_lmh_pool);
  hist::unlink(token->word_history);
  hist::unlink(token->state_history);
  //TPLexPrefixTree::PathHistory::unlink(token->token_path);
  m_token_pool.push_back(token);
}

void TokenPassSearch::release_lmhist(LMHistory *lmhist) {
  lmhist->last_word=NULL;
  lmhist->previous=NULL;
  lmhist->reference_count = 0;
  lmhist->printed = false;
  lmhist->word_start_frame = 0;
  lmhist->word_first_silence_frame=-1;
  m_lmh_pool.push_back(lmhist);
}


void TokenPassSearch::save_token_statistics(int count)
{
  int *buf = new int[MAX_TREE_DEPTH];
  int i, j;
  char fname[30];
  FILE *fp;
  int val;

  for (i = 0; i < MAX_TREE_DEPTH; i++) {
    buf[i] = 0;
  }
  for (i = 0; i < m_active_token_list->size(); i++) {
    if ((*m_active_token_list)[i] != NULL) {
      if ((*m_active_token_list)[i]->depth < MAX_TREE_DEPTH) {
        val = (int) (MAX_TREE_DEPTH - 1
                     - ((m_best_log_prob
                         - (*m_active_token_list)[i]->total_log_prob)
                        / m_global_beam * (MAX_TREE_DEPTH - 2) + 0.5));
        buf[val]++;
      }
    }
  }

  sprintf(fname, "llh_%d", count);
  fp = fopen(fname, "w");
  for (j = 0; j < MAX_TREE_DEPTH; j++) {
    fprintf(fp, "%d ", buf[j]);
  }
  fprintf(fp, "\n");
  fclose(fp);
  delete buf;
}

void TokenPassSearch::debug_ensure_all_paths_contain_history(LMHistory *limit)
{
  fprintf(stderr, "DEBUG: ensure_all_paths_contain_history\n");
  for (int t = 0; t < m_active_token_list->size(); t++) {
    TPLexPrefixTree::Token *token = (*m_active_token_list)[t];
    if (token == NULL)
      continue;
    LMHistory *hist = token->lm_history;
    while (hist != NULL) {
      if (hist == limit)
        break;
      hist = hist->previous;
    }

    if (hist == NULL) {
      fprintf(stderr, "ERROR: path does not contain history %p\n", limit);
      fprintf(stderr, "token = %p\n", token);
      fprintf(stderr, "node = %d\n", token->node->node_id);
      fprintf(stderr, "history:");
      hist = token->lm_history;
      while (hist->last().word_id() >= 0) {
        fprintf(stderr, " %d %s 0x%p (ref %d)", hist->word_start_frame,
                m_vocabulary.word(hist->last().word_id()).c_str(), hist,
                hist->reference_count);
        hist = hist->previous;
      }
      fprintf(stderr, "\n");
      exit(1);
    }
  }
}

void TokenPassSearch::update_final_tokens()
{
  assert(m_generate_word_graph || m_require_sentence_end);

  // Add the sentence end symbol to the tokens in the final nodes.

  m_best_final_token = NULL;
  for (int i = 0; i < m_active_token_list->size(); i++) {
    TPLexPrefixTree::Token *token = m_active_token_list->at(i);
    if (token == NULL)
      continue;

    // For tokens in a final node, update the last word from LMHistory to
    // WordHistory.
    //
    // FIXME: this is quite a hack, because WordHistory is normally updated
    // using LMHistory, when token moves to the first state of the next
    // word. Thus, tokens that are in a final node do not have the current
    // word in their word histories.
    if (m_generate_word_graph && token->node->flags & NODE_FINAL) {
      token->word_history = new TPLexPrefixTree::WordHistory(
        token->lm_history->last().word_id(), m_frame,
        token->word_history);
      token->word_history->lex_node_id = token->node->node_id;
      token->word_history->cum_am_log_prob = token->am_log_prob;
      token->word_history->cum_lm_log_prob = token->lm_log_prob;
      token->word_history->am_log_prob =
        token->word_history->cum_am_log_prob
        - token->word_history->previous->cum_am_log_prob;
      token->word_history->lm_log_prob =
        token->word_history->cum_lm_log_prob
        - token->word_history->previous->cum_lm_log_prob;
      token->word_history->end_frame = m_frame;
      build_word_graph(token);
    }

    // Add sentence end in LMHistory.
    token->lm_history = acquire_lmhist(&m_word_repository[m_sentence_end_id],
                                      token->lm_history);
    hist::link(token->lm_history);
    token->lm_history->word_start_frame = m_frame;
    update_lm_log_prob(*token);
    if (m_fsa_lm) {
      token->fsa_lm_node = m_fsa_lm->initial_node_id();
    }
    token->word_count++;
    token->total_log_prob = get_token_log_prob(token->am_log_prob,
                                               token->lm_log_prob);

    // For tokens in a final node, add sentence end also in WordHistory.
    if (m_generate_word_graph && token->node->flags & NODE_FINAL) {
      if (m_best_final_token == NULL
          || token->total_log_prob
          > m_best_final_token->total_log_prob)
        m_best_final_token = token;

      token->word_history = new TPLexPrefixTree::WordHistory(
        token->lm_history->last().word_id(), m_frame,
        token->word_history);
      token->word_history->lex_node_id = token->node->node_id;
      token->word_history->cum_am_log_prob = token->am_log_prob;
      token->word_history->cum_lm_log_prob = token->lm_log_prob;
      token->word_history->am_log_prob =
        token->word_history->cum_am_log_prob
        - token->word_history->previous->cum_am_log_prob;
      token->word_history->lm_log_prob =
        token->word_history->cum_lm_log_prob
        - token->word_history->previous->cum_lm_log_prob;
      token->word_history->end_frame = m_frame;

      build_word_graph(token);
    }
  }
}

/*void
  TokenPassSearch::print_token_path(TPLexPrefixTree::PathHistory *hist)
  {
  std::vector<TPLexPrefixTree::PathHistory*> path_list;
  TPLexPrefixTree::PathHistory *cur_item;
  int i;
  char fname[30];
  FILE *fp;

  sprintf(fname, "best_path_%i", filecount++);
  fp = fopen(fname, "w");

  cur_item = hist;
  while (cur_item != NULL)
  {
  path_list.push_back(cur_item);
  cur_item = cur_item->prev;
  }
  // Print the best path
  for (i = path_list.size()-1; i >= 0; i--)
  {
  fprintf(fp, "%d\t%.1f\t%.1f\n", path_list[i]->depth,
  path_list[i]->ll, path_list[i]->dll);
  }
  fclose(fp);
  }
*/

void TokenPassSearch::copy_word_graph_info(TPLexPrefixTree::Token *src_token,
                                           TPLexPrefixTree::Token *tgt_token)
{
  tgt_token->recent_word_graph_node = src_token->recent_word_graph_node;
  word_graph.link(tgt_token->recent_word_graph_node);
}

void TokenPassSearch::build_word_graph_aux(TPLexPrefixTree::Token *new_token,
                                           TPLexPrefixTree::WordHistory *word_history)
{
  int word_id = word_history->word_id;
  if (word_id < 0)
    return;

  // Check if we have already created a node for this word at this frame with
  // the same lex_node_id. If so, use the old node. Otherwise, create a new
  // word graph node.
  int target_node = 0;
  bool create_new_node = true;
  WordGraphInfo &info = m_recent_word_graph_info[word_id];
  if (info.frame != m_frame) {
    // Frame number from previous node insertion for this word differs, so
    // we create a new node and start collecting a new list of node IDs that
    // exist at this frame.
    info.items.clear();
    info.frame = m_frame;
  }
  for (int i = 0; i < info.items.size(); i++) {
    if (info.items[i].lex_node_id == word_history->lex_node_id) {
      // This node matches all (word_id, frame, lex_node_id).
      target_node = info.items[i].graph_node_id;
      create_new_node = false;
      break;
    }
  }
  if (create_new_node) {
    target_node = word_graph.add_node(m_frame, word_id,
                                      word_history->lex_node_id);
    WordGraphInfo::Item item;
    item.graph_node_id = target_node;
    item.lex_node_id = word_history->lex_node_id;
    info.items.push_back(item);
  }

  // The new arc starts from the previous end point of this path.
  int source_node = new_token->recent_word_graph_node;
  float am = word_history->am_log_prob;
  float lm = word_history->lm_log_prob;

  // Multiply LM weights with LM scale so that it's easier to prune arcs in
  // WordGraph::add_arc(). The weights will be divided by LM scale before
  // writing the final lattice.
  word_graph.add_arc(source_node, target_node, am, lm * m_lm_scale,
                     m_use_word_pair_approximation);

  word_graph.unlink(new_token->recent_word_graph_node);
  new_token->recent_word_graph_node = target_node;
  word_graph.link(target_node);

  new_token->word_history->graph_node_id = target_node;
}

void TokenPassSearch::build_word_graph(TPLexPrefixTree::Token *new_token)
{
  if (new_token->word_history->word_id < 0)
    return;

  // assert(new_token->word_history->word_id == new_token->lm_history->word_id);
  // WordGraph::Node &node = word_graph.nodes[new_token->recent_word_graph_node];
  // assert(new_token->word_history->previous == NULL ||
  // node.symbol == new_token->word_history->previous->word_id);
  build_word_graph_aux(new_token, new_token->word_history);
}

void TokenPassSearch::write_word_graph(const std::string &file_name)
{
  if (!m_generate_word_graph) {
    throw WordGraphNotGenerated();
  }

  FILE *file = fopen(file_name.c_str(), "w");
  if (!file) {
    throw IOError("Could not open word graph file for writing.");
  }
  write_word_graph(file);
  fclose(file);
}

void TokenPassSearch::write_word_graph(FILE *file)
{
  const TPLexPrefixTree::Token & best_token = get_best_final_token();

  if (1) {
    word_graph.reset_reachability();
    word_graph.mark_reachable_nodes(best_token.recent_word_graph_node);
  }
  else
    word_graph.reset_reachability(true);

  // Count reachable nodes and arcs
  int nodes = 0;
  int arcs = 0;
  for (int n = 0; n < word_graph.nodes.size(); n++) {

    // Print reachable nodes
    WordGraph::Node &node = word_graph.nodes[n];
    if (!node.reachable)
      continue;

    nodes++;
    int a = node.first_arc;
    while (a >= 0) {
      arcs++;
      WordGraph::Arc &arc = word_graph.arcs[a];
      a = arc.sibling_arc;
    }
  }

  fprintf(file, "VERSION=1.1\n"
          "base=10\n"
          "dir=f\n"
          "lmscale=%f wdpenalty=%f\n"
          "N=%d\tL=%d\n"
          "start=0 end=%d\n", m_lm_scale, m_insertion_penalty, nodes, arcs,
          best_token.recent_word_graph_node);

  for (int n = 0; n < word_graph.nodes.size(); n++) {

    // Print reachable nodes
    WordGraph::Node &node = word_graph.nodes[n];
    if (!node.reachable)
      continue;

    //    fprintf(file, "I=%d\tt=%.2f,%d,%d\n", n, node.path_weight, node.frame,
    //      node.lex_node_id);
    fprintf(file, "I=%d\tt=%d\n", n, node.frame);
  }

  int arc_count = 0;
  for (int n = 0; n < word_graph.nodes.size(); n++) {

    // Consider only reachable nodes
    WordGraph::Node &node = word_graph.nodes[n];
    if (!node.reachable)
      continue;

    // Print arcs
    int a = node.first_arc;
    std::string word;
    while (a >= 0) {
      WordGraph::Arc &arc = word_graph.arcs[a];
      float am_log_prob = arc.am_weight;
      float lm_log_prob = arc.lm_weight / m_lm_scale
        - m_insertion_penalty;

      word = m_vocabulary.word(node.symbol);
      if (word == "<s>" || word == "</s>")
        word = "!NULL";

      fprintf(file, "J=%d\tS=%d\tE=%d\tW=%s\tv=0\ta=%e\tl=%e\n",
              arc_count++, arc.source_node_id, n, word.c_str(),
              am_log_prob, lm_log_prob);
      a = arc.sibling_arc;
    }
  }
}

// void
// TokenPassSearch::write_word_graph(FILE *file)
// {
//   word_graph.reset_reachability();
//   word_graph.mark_reachable_nodes(m_final_word_graph_node); 

//   int aux_node = word_graph.nodes.size();
//   for (int n = 0; n < word_graph.nodes.size(); n++) {

//     // Print reachable nodes
//     WordGraph::Node &node = word_graph.nodes[n];
//     if (!node.reachable)
//       continue;

//     // Print arcs
//     int a = node.first_arc;
//     while (a >= 0) {
//       WordGraph::Arc &arc = word_graph.arcs[a];
//       WordGraph::Node &target_node = word_graph.nodes[arc.target_node];

//       float am_log_prob = arc.aux_weight;
//       float lm_log_prob = arc.weight - am_log_prob;

//       fprintf(file, "%d %d %s-%d %f\n", arc.target_node, aux_node,
//            m_vocabulary.word(arc.symbol).c_str(), node.frame, -am_log_prob, 
//            target_node.frame, node.frame);
//       fprintf(file, "%d %d %s %f\n", aux_node, n, "<lm>", -lm_log_prob);

//       aux_node++;
//       a = arc.sibling_arc;
//     }
//   }
//   fprintf(file, "%d\n", m_final_word_graph_node);
// }

void TokenPassSearch::debug_print_best_lm_history()
{
  std::vector<int> word_hist;
  LMHistory *cur_word;
  float max_log_prob = -1e20;
  int i;
  int best_token = -1;

  // Find the best token
  for (i = 0; i < m_active_token_list->size(); i++) {
    if ((*m_active_token_list)[i] != NULL) {
      if ((*m_active_token_list)[i]->total_log_prob > max_log_prob) {
        best_token = i;
        max_log_prob = (*m_active_token_list)[i]->total_log_prob;
      }
    }
  }
  assert(best_token >= 0);
  // Determine the word sequence
  cur_word = (*m_active_token_list)[best_token]->lm_history;
  while (cur_word != NULL) {
    word_hist.push_back(cur_word->last().word_id());
    cur_word = cur_word->previous;
  }
  // Print the best path
  for (i = word_hist.size() - 1; i >= 0; i--) {
    if (word_hist[i] < 0)
      printf("* ");
    else
      printf("%s ", m_vocabulary.word(word_hist[i]).c_str());
  }
  printf("\n");
  //print_token_path((*m_active_token_list)[best_token]->token_path);
#ifdef PRUNING_MEASUREMENT
  for (i = 0; i < 6; i++)
    printf("meas%i: %.3g\n", i, (*m_active_token_list)[best_token]->meas[i]);
#endif

#ifdef COUNT_LM_LA_CACHE_MISS
  printf("Count: ");
  for (i = 0; i < MAX_LEX_TREE_DEPTH; i++)
    printf("%i ", lm_la_cache_count[i]);
  printf("\n");
  printf("Miss: ");
  for (i = 0; i < MAX_LEX_TREE_DEPTH; i++)
    printf("%i ", lm_la_cache_miss[i]);
  printf("\n");
  printf("WordCount: %d\n", lm_la_word_cache_count);
  printf("WordMiss: %d\n", lm_la_word_cache_miss);
#endif
}

void TokenPassSearch::debug_print_token_lm_history(FILE * file,
                                                   const TPLexPrefixTree::Token & token)
{
  if (file == NULL)
    file = stdout;

  // Determine the word sequence
  std::vector<LMHistory*> stack;
  LMHistory * lm_history = token.lm_history;
  while (lm_history != NULL) {
    stack.push_back(lm_history);
    lm_history = lm_history->previous;
  }

  // Print the words
  while (!stack.empty()) {
    lm_history = stack.back();
    stack.pop_back();

    if (lm_history->last().word_id() < 0)
      fprintf(file, "* ");
    else
      fprintf(file, "%s ",
              m_vocabulary.word(lm_history->last().word_id()).c_str());
  }
  fprintf(file, "%.2f %d\n", token.total_log_prob,
          token.recent_word_graph_node);
}

void TokenPassSearch::debug_print_token_word_history(FILE * file,
                                                     const TPLexPrefixTree::Token & token)
{
  if (!m_generate_word_graph) {
    throw WordGraphNotGenerated();
  }

  if (file == NULL)
    file = stdout;

  // Determine the word sequence
  std::vector<TPLexPrefixTree::WordHistory*> stack;
  TPLexPrefixTree::WordHistory * word_history = token.word_history;
  while (word_history != NULL) {
    stack.push_back(word_history);
    word_history = word_history->previous;
  }

  // Print the path
  for (int i = stack.size() - 1; i >= 0; i--) {
    int word_id = stack[i]->word_id;
    if (word_id > 0) {
      std::string word(m_vocabulary.word(stack[i]->word_id));
      fprintf(file, "%s ", word.c_str());
    }

    fprintf(file, "%d\t%d\t%d\t%.3f\t%.3f\t%.3f\n", stack[i]->end_frame,
            stack[i]->lex_node_id, stack[i]->graph_node_id,
            stack[i]->am_log_prob, stack[i]->lm_log_prob,
            get_token_log_prob(stack[i]->cum_am_log_prob,
                               stack[i]->cum_lm_log_prob));
  }
  fprintf(file, "\n");

  fflush(file);
}
