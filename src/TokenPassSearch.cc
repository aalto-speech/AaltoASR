#include <stdio.h>
#include <math.h>
#include "TokenPassSearch.hh"

#define NUM_HISTOGRAM_BINS 100
#define TOKEN_RESERVE_BLOCK 1024

#define DEFAULT_MAX_LOOKAHEAD_SCORE_LIST_SIZE 512
#define DEFAULT_MAX_NODE_LOOKAHEAD_BUFFER_SIZE 512

#define MAX_TREE_DEPTH 60

#define MAX_STATE_DURATION 80

#define EQ_DEPTH_PRUNING
#define EQ_WC_PRUNING


TokenPassSearch::TokenPassSearch(TPLexPrefixTree &lex, Vocabulary &vocab,
                                 Acoustics &acoustics) :
  m_lexicon(lex),
  m_vocabulary(vocab),
  m_acoustics(acoustics)
{
  m_root = lex.root();
  m_end_frame  = -1;
  m_global_beam = 1e10;
  m_word_end_beam = 1e10;
  m_eq_depth_beam = 1e10;
  m_eq_wc_beam = 1e10;

  m_similar_word_hist_span = 0;
  m_lm_scale = 1;
  m_max_num_tokens = 0;
  m_active_token_list = new std::vector<TPLexPrefixTree::Token*>;
  m_new_token_list = new std::vector<TPLexPrefixTree::Token*>;
  m_word_end_token_list = new std::vector<TPLexPrefixTree::Token*>;
  m_ngram = NULL;
  m_word_boundary_id = 0;
  m_duration_scale = 0;
  m_transition_scale = 1;
  m_lm_bigram_lookahead = false;
  m_max_lookahead_score_list_size = DEFAULT_MAX_LOOKAHEAD_SCORE_LIST_SIZE;
  m_max_node_lookahead_buffer_size = DEFAULT_MAX_NODE_LOOKAHEAD_BUFFER_SIZE;
  m_insertion_penalty = 0;
  m_avg_ac_log_prob_alpha = 0.2;
  filecount = 0;
}


void
TokenPassSearch::set_word_boundary(const std::string &word)
{
  m_word_boundary_id = m_vocabulary.word_index(word);
  if (m_word_boundary_id == 0) {
    fprintf(stderr, "Search::set_word_boundary(): word boundary not in vocabulary\n");
    exit(1);
  }
}


void
TokenPassSearch::reset_search(int start_frame)
{
  TPLexPrefixTree::Token *t;
  m_frame = start_frame;
  m_end_frame = -1;

  m_word_boundary_lm_id = m_lex2lm[m_word_boundary_id];

  // Clear existing tokens and create a new token to the root
  for (int i = 0; i < m_active_token_list->size(); i++)
  {
    if ((*m_active_token_list)[i] != NULL)
      release_token((*m_active_token_list)[i]);
  }
  m_active_token_list->clear();

  m_lexicon.clear_node_token_lists();

  t = acquire_token();
  t->node = m_root;
  t->next_node_token = NULL;
  t->am_log_prob = 0;
  t->lm_log_prob = 0;
  t->cur_am_log_prob = 0;
  t->cur_lm_log_prob = 0;
  t->prev_word = new TPLexPrefixTree::WordHistory(-1, -1, NULL);
  t->prev_word->link();
  t->dur = 0;
  t->avg_ac_log_prob = 0;

#ifdef PRUNING_MEASUREMENT
  for (int i = 0; i < 6; i++)
    t->meas[i] = 0;
#endif

#ifdef EQ_WC_PRUNING
  for (int i = 0; i < MAX_WC_COUNT; i++)
    m_wc_llh[i] = 0;
  m_min_word_count = 0;
#endif
  
  t->depth = 0;
  //t->token_path = new TPLexPrefixTree::PathHistory(0,0,0,NULL);
  //t->token_path->link();
  t->word_count = 0;

  m_active_token_list->push_back(t);

  if (lm_lookahead_score_list.get_num_items() > 0)
  {
    // Delete the LM lookahead cache
    LM2GLookaheadScoreList *score_list;
    while (lm_lookahead_score_list.remove_last_item(&score_list))
      delete score_list;
  }
  lm_lookahead_score_list.set_max_items(DEFAULT_MAX_LOOKAHEAD_SCORE_LIST_SIZE);

  m_current_glob_beam = m_global_beam;
  m_current_we_beam = m_word_end_beam;
}


bool
TokenPassSearch::run(void)
{
  if (m_verbose > 1)
    printf("run() in frame %d\n", m_frame);
  if (m_frame >= m_end_frame)
  {
    print_best_path(true);
    return false;
  }
  if (!m_acoustics.go_to(m_frame))
    return false;

  propagate_tokens();
  prune_tokens();
#ifdef PRUNING_MEASUREMENT
  analyze_tokens();
#endif
  /*if ((m_frame%5) == 0)
    save_token_statistics(filecount++);*/
  m_frame++;
  print_guaranteed_path();
  return true;
}

#ifdef PRUNING_MEASUREMEMT
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
      temp = (*m_active_token_list)[i]->total_log_prob -
        m_depth_llh[(*m_active_token_list)[i]->depth/2];
      if (temp < (*m_active_token_list)[i]->meas[0])
        (*m_active_token_list)[i]->meas[0] = temp;
    }
  }
}
#endif


void
TokenPassSearch::print_guaranteed_path(void)
{
  // Start from some token and proceed to find those word history records
  // that belong to every token and has not yet been printed.
  TPLexPrefixTree::WordHistory *word_hist;
  int i;
  std::vector<TPLexPrefixTree::WordHistory*> word_list;
  bool collecting = false;

  for (i = 0; i < m_active_token_list->size(); i++)
    if ((*m_active_token_list)[i] != NULL)
      break;
  assert( i < m_active_token_list->size() );
  word_hist = (*m_active_token_list)[i]->prev_word;
  while (word_hist->prev_word != NULL &&
         word_hist->printed == 0)
  {
    if (word_hist->prev_word->get_num_references() == 1)
      collecting = true;
    else if (collecting)
    {
      word_list.clear();
      collecting = false;
    }
    if (collecting)
    {
      word_list.push_back(word_hist);
    }
    word_hist = word_hist->prev_word;
  }
  if (word_list.size())
  {
    // Print the guaranteed path and mark the words printed
    for (i = word_list.size()-1; i >= 0; i--)
    {
      fprintf(stdout, "%s ",m_vocabulary.word(word_list[i]->word_id).c_str());
      word_list[i]->printed = 1;
    }
    fflush(stdout);
  }
}


void
TokenPassSearch::print_best_path(bool only_not_printed)
{
  std::vector<int> word_hist;
  TPLexPrefixTree::WordHistory *cur_word;
  float max_log_prob = -1e20;
  int i, best_token;
  
  // Find the best token
  for (i = 0; i < m_active_token_list->size(); i++)
  {
    if ((*m_active_token_list)[i] != NULL)
    {
      if ((*m_active_token_list)[i]->total_log_prob > max_log_prob)
      {
        best_token = i;
        max_log_prob = (*m_active_token_list)[i]->total_log_prob;
      }
    }
  }
  // Determine the word sequence
  cur_word = (*m_active_token_list)[best_token]->prev_word;
  while (cur_word != NULL)
  {
    if (only_not_printed && cur_word->printed)
      break;
    word_hist.push_back(cur_word->word_id);
    cur_word = cur_word->prev_word;
  }
  // Print the best path
  for (i = word_hist.size()-1; i >= 0; i--)
  {
    printf("%s ",m_vocabulary.word(word_hist[i]).c_str());
  }
  printf("\n");
  //print_token_path((*m_active_token_list)[best_token]->token_path);
#ifdef PRUNING_MEASUREMENT
  for (i = 0; i < 6; i++)
    printf("meas%i: %.3g\n", i, (*m_active_token_list)[best_token]->meas[i]);
#endif
}


void
TokenPassSearch::propagate_tokens(void)
{
  int i;

#ifdef EQ_DEPTH_PRUNING
  for (i = 0; i < MAX_LEX_TREE_DEPTH/2; i++)
  {
    m_depth_llh[i] = -1e20;
  }
#endif
#ifdef EQ_WC_PRUNING
  int j;
  i = 0;
  while (i < MAX_WC_COUNT && m_wc_llh[i++] < -9e19)
    m_min_word_count++;
  for (i = 0; i < MAX_WC_COUNT; i++)
    m_wc_llh[i] = -1e20;
#endif
  
  m_best_log_prob = -1e20;
  m_best_we_log_prob = -1e20;
  m_worst_log_prob = 0;
  //m_lexicon.clear_node_token_lists();
  clear_active_node_token_lists();
  for (i = 0; i < m_active_token_list->size(); i++)
  {
    if ((*m_active_token_list)[i] != NULL)
    {
      propagate_token((*m_active_token_list)[i]);
    }
  }
}


void
TokenPassSearch::propagate_token(TPLexPrefixTree::Token *token)
{
  TPLexPrefixTree::Node *source_node = token->node;
  int i;
  bool we_flag = false;

  if (token->node == m_root) // New word start, use tighter beam
    we_flag = true;

  // Iterate all the arcs leaving the token's node
  if (source_node->word_id != -1)
  {
    // Word end (unless looping into itself).
    // Iterate all the arcs leaving the token's node.
    for (i = 0; i < source_node->arcs.size(); i++)
    {
      if (source_node->arcs[i].next != source_node)
      {
        // Moving away from a word end, add the word history record and
        // the language model score.
        // NOTE! It seems like we should not add any special code for handling
        // word breaks with silence or prevent two sequental word boundaries.
        // However, this branch seems to be quite insignificant to the
        // efficiency (about 0.3% of running time).
        float lm_score_add;
        float total_log_prob;
        TPLexPrefixTree::WordHistory *new_prev_word, *word_boundary;
        new_prev_word = new TPLexPrefixTree::WordHistory(
          source_node->word_id, m_lex2lm[source_node->word_id],
          token->prev_word);
        lm_score_add = compute_lm_log_prob(new_prev_word) +
          m_insertion_penalty;
        total_log_prob = get_token_log_prob(token->am_log_prob+
                                            source_node->arcs[i].log_prob,
                                            token->lm_log_prob + lm_score_add);
        // Prune as early as possible
        if (total_log_prob < m_best_we_log_prob - m_current_we_beam)
        {
          new_prev_word->link();
          TPLexPrefixTree::WordHistory::unlink(new_prev_word);
          continue;
        }
        word_boundary = new TPLexPrefixTree::WordHistory(m_word_boundary_id,
                                                         m_word_boundary_lm_id,
                                                         new_prev_word);
        word_boundary->link();
        token->word_count++;
        move_token_to_node(token, source_node->arcs[i].next,
                           source_node->arcs[i].log_prob,
                           lm_score_add, new_prev_word,
                           true);
        if (source_node->word_id != m_word_boundary_id)
        {
          lm_score_add += compute_lm_log_prob(word_boundary);
          move_token_to_node(token, source_node->arcs[i].next,
                             source_node->arcs[i].log_prob,
                             lm_score_add, word_boundary,
                             true);
        }
        token->word_count--;
        TPLexPrefixTree::WordHistory::unlink(word_boundary);
      }
      else
      {
        // Self loop, no  word end.
        move_token_to_node(token, source_node->arcs[i].next,
                           source_node->arcs[i].log_prob,
                           0, token->prev_word,
                           we_flag);
      }
    }
  }
  else
  {
    // Iterate all the arcs leaving the token's node.
    for (i = 0; i < source_node->arcs.size(); i++)
    {
      // Moving without a word end.
      move_token_to_node(token, source_node->arcs[i].next,
                         source_node->arcs[i].log_prob,
                         0, token->prev_word,
                         we_flag);
    }
  }
}


void
TokenPassSearch::move_token_to_node(TPLexPrefixTree::Token *token,
                                    TPLexPrefixTree::Node *node,
                                    float transition_score,
                                    float lm_score,
                                    TPLexPrefixTree::WordHistory *word_hist,
                                    bool word_end_flag)
{
  int new_dur;
  int depth;
  float new_cur_am_log_prob;
  float new_cur_lm_log_prob;
  float real_am_log_prob = token->am_log_prob + transition_score;
  float real_lm_log_prob = token->lm_log_prob + lm_score;
  float total_token_log_prob;

  int i;

  if (node != token->node)
  {
    // Changing node
    new_dur = 0;
    depth = token->depth + 1;
    if (token->node->state != NULL)
    {
      // Add duration probability
      int temp_dur = token->dur+1;
      real_am_log_prob += m_duration_scale*
        token->node->state->duration.get_log_prob(temp_dur);
    }
    new_cur_am_log_prob = real_am_log_prob;
    new_cur_lm_log_prob = token->cur_lm_log_prob;

    if (word_end_flag)
      new_cur_lm_log_prob = real_lm_log_prob;
    if (node->possible_word_id_list.size() > 0 && m_lm_bigram_lookahead)
    {
      // Add language model lookahead
      new_cur_lm_log_prob += get_lm_bigram_lookahead(word_hist, node);
    }
  }
  else
  {
    // Self transition
    new_dur = token->dur + 1;
    if (new_dur > MAX_STATE_DURATION && token->node->state != NULL &&
        token->node->state->duration.is_valid_duration_model())
        return; // Maximum state duration exceeded, discard token
    depth = token->depth;
    new_cur_am_log_prob = token->cur_am_log_prob +
      m_transition_scale*transition_score;
    new_cur_lm_log_prob = token->cur_lm_log_prob;
  }

  if (node == m_root)
  {
    depth = 0;
  }
  if (node->state == NULL)
  {
    // Moving to a node without HMM state, pass through immediately.

    // Try beam pruning
    total_token_log_prob =
      get_token_log_prob(new_cur_am_log_prob, new_cur_lm_log_prob);
    if (total_token_log_prob+token->avg_ac_log_prob <
        m_best_log_prob - m_current_glob_beam)
        return;
    
    // Create temporary token for propagation.
    TPLexPrefixTree::Token temp_token;
    temp_token.node = node;
    temp_token.next_node_token = NULL;
    temp_token.am_log_prob = real_am_log_prob;
    temp_token.lm_log_prob = real_lm_log_prob;
    temp_token.cur_am_log_prob = new_cur_am_log_prob;
    temp_token.cur_lm_log_prob = new_cur_lm_log_prob;
    temp_token.total_log_prob = token->total_log_prob;
    temp_token.prev_word = word_hist;
    temp_token.dur = 0;
    temp_token.word_count = token->word_count;

#ifdef PRUNING_MEASUREMENT
    for (i = 0; i < 6; i++)
      temp_token.meas[i] = token->meas[i];
#endif

    temp_token.avg_ac_log_prob = token->avg_ac_log_prob;

    temp_token.depth = depth;
    //temp_token.token_path = token->token_path;
    //temp_token.token_path->link();
    
    propagate_token(&temp_token);
    //TPLexPrefixTree::PathHistory::unlink(temp_token.token_path);
  }
  else
  {
    // Normal propagation
    TPLexPrefixTree::Token *new_token;
    TPLexPrefixTree::Token *similar_word_hist;
    float ac_log_prob = m_acoustics.log_prob(node->state->model);
    float new_avg_ac_log_prob;

    real_am_log_prob += ac_log_prob;
    new_cur_am_log_prob += ac_log_prob;
    total_token_log_prob =
      get_token_log_prob(new_cur_am_log_prob, new_cur_lm_log_prob);
    
    // Apply beam pruning once more
    if (word_end_flag)
    {
      if (total_token_log_prob < m_best_we_log_prob - m_current_we_beam)
        return;
    }
    if (total_token_log_prob < m_best_log_prob - m_current_glob_beam)
      return;
#ifdef EQ_DEPTH_PRUNING
    if (total_token_log_prob < m_depth_llh[depth/2] - m_eq_depth_beam)
      return;
#endif
#ifdef EQ_WC_PRUNING
    if (total_token_log_prob < m_wc_llh[token->word_count-m_min_word_count] -
        m_eq_wc_beam)
      return;
#endif

    // Apply "acoustic lookahead"
    new_avg_ac_log_prob = m_avg_ac_log_prob_alpha * ac_log_prob +
      (1-m_avg_ac_log_prob_alpha) * token->avg_ac_log_prob;

    if (node->token_list == NULL)
    {
      // No tokens in the node,  create new token
      m_active_node_list.push_back(node); // Mark the node active
      new_token = acquire_token();
      new_token->node = node;
      new_token->next_node_token = node->token_list;
      node->token_list = new_token;
      // Add to the list of propagated tokens
      if (word_end_flag)
        m_word_end_token_list->push_back(new_token);
      else
        m_new_token_list->push_back(new_token);
    }
    else
    {
      similar_word_hist = find_similar_word_history(word_hist,
                                                    node->token_list);
      if (similar_word_hist == NULL)
      {
        // New word history for this node, create new token
        new_token = acquire_token();
        new_token->node = node;
        new_token->next_node_token = node->token_list;
        node->token_list = new_token;
        // Add to the list of propagated tokens
        if (word_end_flag)
          m_word_end_token_list->push_back(new_token);
        else
          m_new_token_list->push_back(new_token);
      }
      else
      {
        // Found the same word history, pick the best token.
        if (total_token_log_prob > similar_word_hist->total_log_prob)
        {
          // Replace the previous token
          new_token = similar_word_hist;
          TPLexPrefixTree::WordHistory::unlink(new_token->prev_word);

          //TPLexPrefixTree::PathHistory::unlink(new_token->token_path);
        }
        else
        {
          // Discard this token
          return;
        }
      }
    }
    if (word_end_flag)
    {
      if (total_token_log_prob > m_best_we_log_prob)
      {
        m_best_we_log_prob = total_token_log_prob;
      }
    }
    if (total_token_log_prob > m_best_log_prob)
    {
      m_best_log_prob = total_token_log_prob;
    }

#ifdef EQ_DEPTH_PRUNING
    if (total_token_log_prob > m_depth_llh[depth/2])
      m_depth_llh[depth/2] = total_token_log_prob;
#endif

#ifdef EQ_WC_PRUNING
    if (total_token_log_prob > m_wc_llh[token->word_count-m_min_word_count])
      m_wc_llh[token->word_count-m_min_word_count] = total_token_log_prob;
#endif
        
    if (total_token_log_prob < m_worst_log_prob)
      m_worst_log_prob = total_token_log_prob;
    new_token->prev_word = word_hist;
    if (word_hist != NULL)
      word_hist->link();
    new_token->am_log_prob = real_am_log_prob;
    new_token->cur_am_log_prob = new_cur_am_log_prob;
    new_token->lm_log_prob = real_lm_log_prob;
    new_token->cur_lm_log_prob = new_cur_lm_log_prob;
    new_token->total_log_prob = total_token_log_prob;
    new_token->dur = new_dur;
    new_token->word_count = token->word_count;

#ifdef PRUNING_MEASUREMENT
    for (i = 0; i < 6; i++)
      new_token->meas[i] = token->meas[i];
#endif

    new_token->depth = depth;
    new_token->avg_ac_log_prob = new_avg_ac_log_prob;
    //assert(token->token_path != NULL);
    /*new_token->token_path = new TPLexPrefixTree::PathHistory(
      total_token_log_prob,
      token->token_path->dll + ac_log_prob, depth,
      token->token_path);
      new_token->token_path->link();*/
    /*new_token->token_path = token->token_path;
      new_token->token_path->link();*/
  }
}


TPLexPrefixTree::Token*
TokenPassSearch::find_similar_word_history(TPLexPrefixTree::WordHistory *wh,
                                           TPLexPrefixTree::Token *token_list)
{
  TPLexPrefixTree::Token *cur_token = token_list;
  int i;
  while (cur_token != NULL)
  {
    if (is_similar_word_history(wh, cur_token->prev_word))
      break;
    cur_token = cur_token->next_node_token;
  }
  return cur_token;
}

bool
TokenPassSearch::is_similar_word_history(TPLexPrefixTree::WordHistory *wh1,
                                         TPLexPrefixTree::WordHistory *wh2)
{
  for (int i = 0; i < m_similar_word_hist_span; i++)
  {
    if (wh1->word_id == -1)
    {
      if (wh2->word_id == -1)
        return true;
      return false;
    }
    if (wh1->word_id != wh2->word_id)
      return false;
    wh1 = wh1->prev_word;
    wh2 = wh2->prev_word;
  }
  return true; // Similar word histories up to m_similar_word_hist_span words
}

void
TokenPassSearch::prune_tokens(void)
{
  int i;
  std::vector<TPLexPrefixTree::Token*> *temp;
  int num_active_tokens;
  float beam_limit = m_best_log_prob - m_current_glob_beam; //m_global_beam;
  float we_beam_limit = m_best_we_log_prob - m_current_we_beam;

  if (m_verbose > 1)
    printf("%d new tokens\n",
           m_new_token_list->size() + m_word_end_token_list->size());

  // At first, remove inactive tokens.
  for (i = 0; i < m_active_token_list->size(); i++)
  {
    if ((*m_active_token_list)[i] != NULL)
      release_token((*m_active_token_list)[i]);
  }
  m_active_token_list->clear();
  temp = m_active_token_list;
  m_active_token_list = m_new_token_list;
  m_new_token_list = temp;

  // Prune the word end tokens and add them to m_active_token_list
  for (i = 0; i < m_word_end_token_list->size(); i++)
  {
    if ((*m_word_end_token_list)[i]->total_log_prob < we_beam_limit)
      release_token((*m_word_end_token_list)[i]);
    else
    {
      m_active_token_list->push_back((*m_word_end_token_list)[i]);
    }
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
    float bin_adv = (m_best_log_prob-m_worst_log_prob)/(NUM_HISTOGRAM_BINS-1);
    float new_min_log_prob;
    memset(bins, 0, NUM_HISTOGRAM_BINS*sizeof(int));
    
    for (i = 0; i < m_active_token_list->size(); i++)
    {
      if ((*m_active_token_list)[i]->total_log_prob < beam_limit
#ifdef EQ_DEPTH_PRUNING
          || (*m_active_token_list)[i]->total_log_prob <
          m_depth_llh[(*m_active_token_list)[i]->depth/2] -
          m_eq_depth_beam
#endif
#ifdef EQ_WC_PRUNING
          || (*m_active_token_list)[i]->total_log_prob <
          m_wc_llh[(*m_active_token_list)[i]->word_count-m_min_word_count] -
          m_eq_wc_beam
#endif
        )
      {
        release_token((*m_active_token_list)[i]);
      }
      else
      {
        bins[(int)floorf(((*m_active_token_list)[i]->total_log_prob-
                          m_worst_log_prob)/bin_adv)]++;
        m_new_token_list->push_back((*m_active_token_list)[i]);
      }
    }
    m_active_token_list->clear();
    temp = m_active_token_list;
    m_active_token_list = m_new_token_list;
    m_new_token_list = temp;
    if (m_verbose > 1)
      printf("%d tokens after beam pruning\n", m_active_token_list->size());
    num_active_tokens = m_active_token_list->size();
    if (num_active_tokens > m_max_num_tokens)
    {
      for (i = 0; i < NUM_HISTOGRAM_BINS-1; i++)
      {
        num_active_tokens -= bins[i];
        if (num_active_tokens < m_max_num_tokens)
          break;
      }
      new_min_log_prob = m_worst_log_prob + i*bin_adv;
      for (i = 0; i < m_active_token_list->size(); i++)
      {
        if ((*m_active_token_list)[i]->total_log_prob < new_min_log_prob)
        {
          release_token((*m_active_token_list)[i]);
          (*m_active_token_list)[i] = NULL;
        }
      }
      if (m_verbose > 1)
        printf("%d tokens after histogram pruning\n", num_active_tokens);

      // Pass the new beam limit to next token propagation
      m_current_glob_beam = std::min((m_best_log_prob-new_min_log_prob),
                                     m_global_beam);
      m_current_we_beam = m_current_glob_beam/m_global_beam * m_word_end_beam;
    }
  }
  else
  {
    // Only do the beam pruning
    for (i = 0; i < m_active_token_list->size(); i++)
    {
      if ((*m_active_token_list)[i]->total_log_prob < beam_limit
#ifdef EQ_DEPTH_PRUNING
          || (*m_active_token_list)[i]->total_log_prob <
          m_depth_llh[(*m_active_token_list)[i]->depth/2] -
          m_eq_depth_beam
#endif
#ifdef EQ_WC_PRUNING
          || (*m_active_token_list)[i]->total_log_prob <
          m_wc_llh[(*m_active_token_list)[i]->word_count-m_min_word_count] -
          m_eq_wc_beam
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
      printf("%d tokens after beam pruning\n", m_active_token_list->size());
    if (m_current_glob_beam < m_global_beam)
    {
      // Determine new beam
      m_current_glob_beam = m_current_glob_beam*1.1;
      m_current_glob_beam = std::min(m_global_beam, m_current_glob_beam);
      m_current_we_beam = m_current_glob_beam/m_global_beam * m_word_end_beam;
    }
  }
  if (m_verbose > 1)
    printf("Current beam: %.1f   Word end beam: %.1f\n",
           m_current_glob_beam, m_current_we_beam);
}


void
TokenPassSearch::clear_active_node_token_lists(void)
{
  for (int i = 0; i < m_active_node_list.size(); i++)
    m_active_node_list[i]->token_list = NULL;
  m_active_node_list.clear();
}



void
TokenPassSearch::set_ngram(TreeGram *ngram)
{
  int count = 0;

  m_ngram = ngram;
  m_lex2lm.clear();
  m_lex2lm.resize(m_vocabulary.num_words());

  // Create the mapping between lexicon and the model.
  for (int i = 0; i < m_vocabulary.num_words(); i++) {
    m_lex2lm[i] = m_ngram->word_index(m_vocabulary.word(i));

    // Warn about words not in lm.
    if (m_lex2lm[i] == 0 && i != 0) {
      fprintf(stderr, "%s not in LM\n", m_vocabulary.word(i).c_str());
      count++;
    }
  }

  if (count > 0)
    fprintf(stderr, "there were %d out-of-LM words in total in LM\n", 
	    count);
}

float
TokenPassSearch::compute_lm_log_prob(TPLexPrefixTree::WordHistory *word_hist)
{
  float lm_log_prob = 0;

  if (m_ngram->order() > 0)
  {
    // Create history
    m_history_lm.clear();
    int last_word = word_hist->word_id;
    TPLexPrefixTree::WordHistory *word = word_hist;
    for (int i = 0; i < m_ngram->order(); i++) {
      if (word->word_id == -1)
	break;
      m_history_lm.push_front(word->lm_id);
      word = word->prev_word;
    }

    lm_log_prob = m_ngram->log_prob(m_history_lm);
  }

  return lm_log_prob;
}


float
TokenPassSearch::get_lm_bigram_lookahead(TPLexPrefixTree::WordHistory *prev_word,
                                         TPLexPrefixTree::Node *node)
{
  int i;
  int prev_word_id = prev_word->word_id;
  int real_word_id = prev_word_id;
  LM2GLookaheadScoreList *score_list, *old_score_list;
  float score;

  if (prev_word_id == -1)
    prev_word_id = m_lexicon.words();
  
  if (node->lm_lookahead_buffer.get_num_items() == 0)
  {
    node->lm_lookahead_buffer.set_max_items(DEFAULT_MAX_NODE_LOOKAHEAD_BUFFER_SIZE);
  }
  else
  {
    if (node->lm_lookahead_buffer.find(prev_word_id, &score))
      return score;
  }

  // Not found, compute the LM bigram lookahead scores.
  // At first, determine if the LM scores have been computed already.

  if (!lm_lookahead_score_list.find(prev_word_id, &score_list))
  {
    // Not found, compute the scores.
    // FIXME! Is it necessary to compute the scores for all the words?
    if (m_verbose > 2)
      printf("Compute lm lookahead scores for \'%s'\n",
             real_word_id==-1?"NULL":m_vocabulary.word(prev_word_id).c_str());
    score_list = new LM2GLookaheadScoreList;
    if (lm_lookahead_score_list.insert(prev_word_id, score_list,
                                       &old_score_list))
      delete old_score_list; // Old list was removed
    score_list->prev_word_id = prev_word_id;
    score_list->lm_scores.insert(score_list->lm_scores.end(),
                                 m_lexicon.words(), 0);    
    m_ngram->fetch_bigram_list(prev_word->lm_id, m_lex2lm,
                               score_list->lm_scores);
  }
  
  // Compute the lookahead score by selecting the maximum LM score of
  // possible word ends.
  score = -1e10;
  for (i = 0; i < node->possible_word_id_list.size(); i++)
  {
    if (score_list->lm_scores[node->possible_word_id_list[i]] > score)
      score = score_list->lm_scores[node->possible_word_id_list[i]];
  }
  // Add the score to the node's buffer
  node->lm_lookahead_buffer.insert(prev_word_id, score, NULL);
  return score;
}


TPLexPrefixTree::Token*
TokenPassSearch::acquire_token(void)
{
  TPLexPrefixTree::Token *t;
  if (m_token_pool.size() == 0)
  {
    TPLexPrefixTree::Token *tt =
      new TPLexPrefixTree::Token[TOKEN_RESERVE_BLOCK];
    for (int i = 0; i < TOKEN_RESERVE_BLOCK; i++)
      m_token_pool.push_back(&tt[i]);
  }
  t = m_token_pool.back();
  m_token_pool.pop_back();
  return t;
}


void
TokenPassSearch::release_token(TPLexPrefixTree::Token *token)
{
  TPLexPrefixTree::WordHistory::unlink(token->prev_word);
  //TPLexPrefixTree::PathHistory::unlink(token->token_path);
  m_token_pool.push_back(token);
}



void
TokenPassSearch::save_token_statistics(int count)
{
  int *buf = new int[MAX_TREE_DEPTH];
  int i, j;
  char fname[30];
  FILE *fp;
  int x, y;
  int val;
  
  for (i = 0; i < MAX_TREE_DEPTH; i++)
  {
    buf[i] = 0;
  }
  for (i = 0; i < m_active_token_list->size(); i++)
  {
    if ((*m_active_token_list)[i] != NULL)
    {
      if ((*m_active_token_list)[i]->depth < MAX_TREE_DEPTH)
      {
        /*val = 63 - ((m_best_log_prob -
                     (*m_active_token_list)[i]->total_log_prob)
                     /m_global_beam*50+0.5);*/
        val = MAX_TREE_DEPTH-1 - ((m_best_log_prob -
                     (*m_active_token_list)[i]->total_log_prob)
                    /m_global_beam*(MAX_TREE_DEPTH-2)+0.5);
        x = (*m_active_token_list)[i]->depth;
        buf[val]++;
      }
    }
  }

  sprintf(fname, "llh_%d", count);
  fp = fopen(fname, "w");
  for (j = 0; j < MAX_TREE_DEPTH; j++)
  {
    fprintf(fp, "%d ", buf[j]);
  }
  fprintf(fp, "\n");
  fclose(fp);
  delete buf;
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
