#include <iomanip>
#include <iostream>
#include <algorithm>

#include <assert.h>
#include <float.h>
#include <stdio.h>

#include "Expander.hh"

struct TokenCompare {
  inline bool operator()(Lexicon::Token *a, Lexicon::Token *b) {
    return b->log_prob < a->log_prob;
  }
};


Expander::Expander(const std::vector<Hmm> &hmms, Lexicon &lexicon,
		   Acoustics &acoustics)
  : m_hmms(hmms),
    m_lexicon(lexicon),
    m_acoustics(acoustics),

    m_forced_end(false),
    m_token_limit(0),
    m_beam(1e10),
    m_max_state_duration(0x7fff), // FIXME
    m_duration_scale(1),
    m_transition_scale(1),
    m_post_durations(false),
    m_rabiner_post_mode(0),
    m_words(),
    m_active_words()
{
}

Expander::~Expander()
{
  for (int i = 0; i < m_token_pool.size(); i++)
    delete m_token_pool[i];
  m_token_pool.clear();
}

void
Expander::sort_best_tokens(int tokens)
{
  if (tokens > m_tokens.size())
    tokens = m_tokens.size();

  std::partial_sort(m_tokens.begin(), 
		    m_tokens.begin() + tokens, 
		    m_tokens.end(),
		    TokenCompare());
}

// PRECONDITIONS:
// - Tokens are in incoming_token slots.
// POSTCONDITIONS:
// - Removes tokens that are not among best 'tokens' tokens.
// - Does not necessarily sort the remaining tokens.
void
Expander::keep_best_tokens(int tokens)
{
  if (tokens >= m_tokens.size())
    return;

  // Ensure that 'tokens' best tokens are in the beginning of the
  // list.  The order of the rest is undefined.
  std::partial_sort(m_tokens.begin(), 
		    m_tokens.begin() + tokens, 
		    m_tokens.end(),
		    TokenCompare());

  // Remove worst tokens.
  while (m_tokens.size() > tokens) {
    Lexicon::Token *token = m_tokens.back();
    Lexicon::Node *node = token->node;
    Lexicon::State &state = node->states[token->state];
    assert(state.incoming_token == token);
    assert(state.outgoing_token == NULL);
    state.incoming_token = NULL;
    //delete token;
    release_token(token);
    m_tokens.pop_back();
  }
}

Lexicon::Token*
Expander::token_to_state(const Lexicon::Token *source_token, 
			 Lexicon::State &source_state,
			 Lexicon::State &target_state,
			 float new_log_prob, float new_dur_log_prob,
                         float aco_log_prob,
			 bool update_best, bool same_state, bool silence,
                         const HmmState &target_hmm_state)
{
  Lexicon::Token *new_token;
  int next_state_duration;

  // Just update the log probs
  new_log_prob += aco_log_prob;
  new_dur_log_prob += aco_log_prob;
  
  // Increase duration
  if (same_state && !silence)
  {
    next_state_duration = source_token->state_duration + 1;
  }
  else
  {
    next_state_duration = 0;
  }
  
  // Target state has already an incoming token.
  if (target_state.incoming_token != NULL) {

    // New token is worse?
    if (target_state.incoming_token->log_prob > new_log_prob)
      return NULL;
    // Old token is worse.  We replace the contents of the old token
    // with the new token
    else {
      new_token = target_state.incoming_token;

      assert(new_token != source_token); // FIXME! Can this happen?!

      *new_token = *source_token;
    }
  }
	  
  // Target state is empty.
  else {
    //new_token = new Lexicon::Token(*source_token);
    new_token = acquire_token(source_token);
    m_tokens.push_back(new_token);
    target_state.incoming_token = new_token;
  }

  new_token->state_duration = next_state_duration;

  new_token->log_prob = new_log_prob;
  new_token->dur_log_prob = new_dur_log_prob;
  
  // Update beam threshold, but only if we are not in sink state!
  //if (update_best && new_log_prob > m_beam_best_tmp)
  //  m_beam_best_tmp = new_log_prob;
  if (update_best && new_token->log_prob > m_beam_best_tmp)
    m_beam_best_tmp = new_token->log_prob;


  return new_token;
}

// Only for sanity debug checks
void
Expander::check_words()
{
  for (int i = 0; i < m_words.size(); i++) {
    
    if (std::find(m_active_words.begin(), m_active_words.end(), &m_words[i]) ==
		  m_active_words.end())
    {
      assert(!m_words[i].active);
      assert(m_words[i].first_length == -1);
      assert(m_words[i].last_length == -1);
    }
    else {
      assert(m_words[i].active);
      assert(m_words[i].first_length >= 0);
      assert(m_words[i].last_length > m_words[i].first_length);
    }
  }
}

// Only for sanity debugging
void
Expander::check_best(int info, bool tmp)
{
  float best_found = -1e10;
  float best = tmp ? m_beam_best_tmp : m_beam_best;

  for (int i = 0; i < m_tokens.size(); i++) {
    const Lexicon::Token *token = m_tokens[i];
    if (token->log_prob > best_found)
      best_found = token->log_prob;
  }
  
  if (best > -1e10 && best_found != best) {
    fprintf(stderr, "%d best found %f, should be %f\n", info, best_found, 
	    best);
    abort();
  }
}

// PRECONDITIONS:
//
// - Tokens are in incoming_token slots.
// - Acoustics has been updated to current time.
// - No tokens in sink or source states. FIXME: check this with asserts!
//
// POSTCONDITIONS:
//
// - The order and number of tokens is undefined.
// - Tokens are in incoming_token slots.
// - No tokens in sink or source states. FIXME: check this with asserts!
//
// IMPLEMENTATION NOTES:
//
// - It is quite tricky to handle the token vector.  We have the
// following situations: 
//
// 1) Remove outgoing_token, while we do not know where the token is
// located in the vector. 
//
// Because the remove happens when a new token is about to replace the
// outgoing_token, we can just replace the outgoing_token with the new
// token.  (Was: Mark the token, and remove it later)
//
// 2) Remove the source_token.  We know the location, but we do not
// simply want to remove in the middle of vector. 
//
// Replace the token with the last token in the vector.  Note that we
// may have to process the replacing token also.
//
// 3) Add new tokens in the end of the vector so that they will be
// processed in the same round.
//
// BUG WARNING!!!
//
// - When keeping track of the best token, ignore tokens in dummy sink
//   states!
//
// IMPORTANT ASSUMPTIONS:
// 
// - Lexicon tree may contain the same word id in several nodes.
// - Lexicon may skip states
// - Is it allowed to have big loops?

void
Expander::move_all_tokens()
{
  // FIXME: remove stupid asserts
  for (int i = 0; i < m_tokens.size(); i++) {
    const Lexicon::Token *token = m_tokens[i];
    Lexicon::Node *node = token->node;
    Lexicon::State &state = node->states[token->state];
    assert(state.outgoing_token == NULL);
    assert(state.incoming_token == token);
    state.outgoing_token = state.incoming_token;
    state.incoming_token = NULL;

    // Tokens in sink states not allowed at all!
    const Hmm &hmm = m_hmms[node->hmm_id];
    assert(!hmm.is_sink(token->state));
  }
  
  // ITERATE ALL TOKENS
  for (int t = 0; t < m_tokens.size(); t++) {
    Lexicon::Token *source_token = m_tokens[t];
    Lexicon::Node *source_node = source_token->node;
    Lexicon::State &source_state = source_node->states[source_token->state];
    const Hmm &hmm = m_hmms[source_node->hmm_id];
    const HmmState &hmm_state = hmm.states[source_token->state];
    bool silence = false;

    if (hmm.label[0]=='_')
    {
      silence = true;
    }

    if (source_token == source_state.incoming_token)
      continue;

    assert(source_token == source_state.outgoing_token);
    source_state.outgoing_token = NULL;

    // Beam pruning using the beam calculated in the last frame
    if (source_token->log_prob < m_beam_best - m_beam) {
      // FIXME: this should never happen anymore, because we have
      // applied the beam pruning already before calling this
      // function! (24.6.2003 thirsima)
      // 
      // Actually this may happen.  Probably because tokens ending in
      // dummy states are moved again during the same frame, and may
      // fall outside the beam! (24.6.2003 thirsima)
      //
      // But why are we using the beam calculated in the last frame?
      // Reason: Even if it is not correct, the previous best is a
      // good estimate before we have found the best token in this
      // frame. (28.10.2003 thirsima)
      goto token_finished;
    }

    // ITERATE TRANSITIONS
    for (int r = 0; r < hmm_state.transitions.size(); r++) {
      const HmmTransition &transition = hmm_state.transitions[r];
      int target_state_id = transition.target;
      const HmmState &target_hmm_state = hmm.states[target_state_id];
      float log_prob = source_token->log_prob +
                       m_transition_scale*transition.log_prob;
      float dur_log_prob = source_token->dur_log_prob;

      if (m_post_durations)
      {
        // NOTE: It seems like the best result is obtained if the staying
        // in silence states is not penalized at all.
        if (target_state_id != source_token->state && !silence) {
          dur_log_prob +=
            m_duration_scale *
            hmm_state.duration.get_log_prob(source_token->state_duration + 1);
        }

        if (m_rabiner_post_mode)
        {
          dur_log_prob += m_transition_scale*transition.log_prob;
        }
      }

      // Target state is a sink state.  Clone the token to the next
      // nodes in the lexicon.
      if (hmm.is_sink(target_state_id)) {

	// If token is at word end, update the best words.
	int word_id = source_node->word_id;
	if (word_id >= 0) {
	  Word *word = &m_words[word_id];

	  assert(word->first_length <= m_frame);
	  assert(word->last_length <= m_frame + 1);

	  // Note that we use (m_frame) instead of (m_frame + 1),
	  // because the current frame is actually already the start
	  // of the next word.  This also assumes that there can not
	  // be an empty word in lexicon.
	  if (!m_forced_end || m_frame == m_frames - 1) {
	    float node_log_prob = (m_post_durations ? dur_log_prob : log_prob);

	    // FIXME: is this correct?  Do we really want to add the
	    // lexicon node probabilities only at word ends?
	    node_log_prob += source_node->log_prob;
	    assert(node_log_prob < 0);

	    // Update the best
	    float avg_log_prob = node_log_prob / m_frame;
	    if (!word->active || avg_log_prob > word->best_avg_log_prob) {
	      word->best_avg_log_prob = avg_log_prob;
	      word->best_length = m_frame;
	    }

	    // Store word end probabilities for each frame.  Note that
	    // there may be several transitions leading to this state.
	    // Store only the best.
            if (word->last_length < m_frame + 1 ||
                word->log_probs[m_frame] < node_log_prob)
            {
              word->log_probs[m_frame] = node_log_prob;
              word->last_length = m_frame + 1;
            }
            
            // Word not in active list yet, add it.
	    if (!word->active) {
	      word->first_length = m_frame;
	      m_active_words.push_back(word);
	      word->active = true;
	    }
	    assert(word->first_length < word->last_length);
	  }
	}

	// ITERATE NEXT NODES
	for (int n = 0; n < source_node->next.size(); n++) {
	  Lexicon::Node *target_node = source_node->next[n];
	  Lexicon::State &target_state = target_node->states[0];
	  Lexicon::Token *new_token;

	  // Our target state is a source state and may already
	  // contain an outgoing token.  Check whether we have better
	  // token here.
	  if (target_state.outgoing_token != NULL) {
	    if (target_state.outgoing_token->log_prob < log_prob) {
	      new_token = target_state.outgoing_token;
	      (*new_token) = (*source_token);
	    }
	    else
	      new_token = NULL;
	  }

	  // Our target source state is empty.
	  else {
	    new_token = token_to_state(source_token, source_state,
                                       target_state, log_prob, dur_log_prob,
                                       0, false, false, false,
                                       target_hmm_state);

	    // The new token is in source state now.  We want to move
	    // it again during current token loop.
	    target_state.incoming_token = NULL;
	    target_state.outgoing_token = new_token;
	  }

	  if (new_token != NULL) {
	    new_token->state = 0;
	    new_token->node = target_node;
	    new_token->state_duration = 0;
//	    new_token->add_path(target_node->hmm_id, new_token->frame + 1,
//				log_prob);
	  }
	}
      }

      // Target state is not a sink state.  Just clone the token.
      else {

        float aco_prob;

        aco_prob = m_acoustics.log_prob(hmm.states[target_state_id].model);
        
	// Beam pruning using already the temporary beam_best, which
	// is under calculation for the next frame.
	// FIXME: this might be quite unnecessary
  	// if (source_token->log_prob < m_beam_best_tmp - m_beam)
	//   continue;

	Lexicon::State &target_state = source_node->states[target_state_id];
	Lexicon::Token *new_token = 
	  token_to_state(source_token, source_state, target_state, log_prob,
			 dur_log_prob,
                         aco_prob,
                         true, target_state_id == source_token->state,
                         silence,
                         target_hmm_state);
	if (new_token != NULL) {
	  new_token->state = target_state_id;
	  new_token->frame++;
	}
      }
    }

  token_finished:
    // The current token has been cloned (or pruned) according to all
    // transitions.  Replace the token by the last token in the
    // vector.
    //delete source_token;
    release_token(source_token);
    if (m_tokens.size() > t + 1)
      m_tokens[t] = m_tokens[m_tokens.size() - 1];
    m_tokens.pop_back();
    t--;
  }

//   // FIXME: REMOVE debug
//   for (int i = 0; i < m_tokens.size(); i++) {
//     const Lexicon::Token *token = m_tokens[i];
//     Lexicon::Node *node = token->node;
//     Lexicon::State &state = node->states[token->state];
//     assert(state.outgoing_token == NULL);
//     assert(state.incoming_token == token);
//   }
}

void
Expander::clear_tokens()
{
  for (int t = 0; t < m_tokens.size(); t++) {
    const Lexicon::Token *token = m_tokens[t];
    Lexicon::Node *node = token->node;
    Lexicon::State &state = node->states[token->state];
    state.incoming_token = NULL;
    //delete m_tokens[t];
    release_token(m_tokens[t]);
  }
  m_tokens.clear();
}

void
Expander::create_initial_tokens(int start_frame)
{
  Lexicon::Node *node = m_lexicon.root();
  Lexicon::Token *token;
  
  for (int next_id = 0; next_id < node->next.size(); next_id++) {
    Lexicon::State &state = node->next[next_id]->states[0];
    //token = new Lexicon::Token();
    token = acquire_token();

    state.incoming_token = token;
    token->frame = start_frame - 1;
    token->state = 0;
    token->node = node->next[next_id];
    token->log_prob = 0;
    token->dur_log_prob = 0;
//    token->add_path(node->next[next_id]->hmm_id, 0, start_frame);
    m_tokens.push_back(token);
  }
}

void
Expander::debug_print_history(Lexicon::Token *token)
{
  std::vector<Lexicon::Path*> paths;
  for (Lexicon::Path *path = token->path; path != NULL; path = path->prev)
    paths.push_back(path);
  std::cout << std::setw(8) << token->log_prob << " ";
  for (int i = paths.size() - 1; i >= 0; i--) {
    Lexicon::Path *path = paths[i];
    std::cout << m_hmms[path->hmm_id].label;
  }
}

void
Expander::debug_print_timit(Lexicon::Token *token)
{
  std::vector<Lexicon::Path*> paths;
  for (Lexicon::Path *path = token->path; path != NULL; path = path->prev)
    paths.push_back(path);
  for (int i = paths.size() - 1; i >= 0; i--) {
    Lexicon::Path *path = paths[i];
    std::cout << path->frame << " ";
    if (i > 0)
      std::cout << paths[i-1]->frame;
    else
      std::cout << token->frame;
    std::cout << " " << m_hmms[path->hmm_id].label << std::endl;
  }
}

void
Expander::debug_print_tokens()
{
  int tokens = 20;
  
  if (tokens > m_tokens.size())
    tokens = m_tokens.size();
  std::partial_sort(m_tokens.begin(), 
		    m_tokens.begin() + tokens,
		    m_tokens.end(), 
		    TokenCompare());
  for (int t = 0; t < tokens; t++) {
    Lexicon::Token *token = m_tokens[t];
    Lexicon::Node *node = token->node;

    std::vector<Lexicon::Path*> paths;
    for (Lexicon::Path *path = token->path; path != NULL; path = path->prev)
      paths.push_back(path);
    std::cout << t << "\t" 
	      << m_hmms[node->hmm_id].label << (int)token->state
	      << "(" << (int)token->state_duration << ")\t" 
	      << std::setprecision(4)
	      << token->log_prob - m_tokens[0]->log_prob << "\t";
    float old_log_prob = 0;
    for (int i = paths.size() - 1; i >= 0; i--) {
      std::cout << m_hmms[paths[i]->hmm_id].label;
      std::cout	<< paths[i]->frame 
		<< "(" << paths[i]->log_prob - old_log_prob << ") ";
      old_log_prob = paths[i]->log_prob;
    }
    std::cout << token->frame << std::endl;
  }
}

void
Expander::expand(int start_frame, int frames)
{
  if (m_words.size() != m_lexicon.words()) {
    // Changing lexicon size is not supported at the moment
    assert(m_words.size() == 0);
    m_active_words.clear();
    m_active_words.reserve(m_lexicon.words());
    m_words.resize(m_lexicon.words());
    for (int i = 0; i < m_words.size(); i++)
      m_words[i].word_id = i;
  }

  // Resize and clear log-prob vectors, if number of frames grows.
  if (m_words[0].log_probs.size() < frames) {
    for (int w = 0; w < m_words.size(); w++) {
      m_words[w].log_probs.resize(frames);
      for (int l = 0; l < frames; l++)
	m_words[w].clear_length(l);
    }
  }

  // FIXME: it is enough to iterate over m_active_words only, right?!
  // Clear best words list
  for (int w = 0; w < m_active_words.size(); w++) {
    Word *word = m_active_words[w];
    word->active = false;

    // Clear also the log-probability storage area for each modified frame
    for (int l = word->first_length; l < word->last_length; l++)
      word->clear_length(l);
    word->first_length = -1;
    word->last_length = -1;
  }
  m_active_words.clear();    

  create_initial_tokens(start_frame);

  m_frames = frames;
  m_beam_best_tmp = -1e10;
  for (m_frame = 0; m_frames < 0 || m_frame < m_frames; m_frame++) {
    m_beam_best = m_beam_best_tmp;
    m_beam_best_tmp = -1e10;

    // FIXME: REMOVE debug
    // fprintf(stderr, "%d\t%.2f\t%d\n", 
    //    m_frame, m_beam_best, m_tokens.size());

    // FIXED 24.6.2003: we should apply beam pruning before
    // keep_best_tokens, because after pruning, the sorting may be
    // useless!
    
    // Beam pruning using the beam calculated in the last frame
    for (int t = 0; t < m_tokens.size(); t++) {
      Lexicon::Token *token = m_tokens[t];
      Lexicon::Node *node = token->node;
      Lexicon::State &state = node->states[token->state];
      assert(state.incoming_token == token);
      assert(state.outgoing_token == NULL);
      assert(token->frame == start_frame + m_frame -1);
      
      if (token->log_prob < m_beam_best - m_beam) {
	// Delete the token
	//delete m_tokens[t];
        release_token(m_tokens[t]);
	state.incoming_token = NULL;

	// Replace the token with the last token in the vector, in
	// order to avoid unnecessary copying.
	if (m_tokens.size() > t + 1) {
	  m_tokens[t] = m_tokens[m_tokens.size() - 1];
	}
	m_tokens.pop_back();
	t--;
      }
    }

    // Limit pruning
    if (m_token_limit > 0)
      keep_best_tokens(m_token_limit);

    assert(m_tokens.size() > 0);

    if (!m_acoustics.go_to(start_frame + m_frame))
      break;
    move_all_tokens();

    // It should be impossible to lose all tokens!
    assert(!m_tokens.empty());

//      std::cout << m_frame << ": ";
//      debug_print_history(m_tokens[0]);
//      std::cout << std::endl;

//     std::cout << "--- " << m_frame << " ---" << std::endl;
//     debug_print_tokens();

//      if (m_frame % 125 == 0)
//        std::cout << std::setw(8) << m_frame << ": " 
//  		<< m_tokens.size() << std::endl;

  }

  clear_tokens();
}

void
Expander::sort_words(int top)
{
  if (top > 0 && top < m_active_words.size())
    std::partial_sort(m_active_words.begin(), m_active_words.begin() + top, 
		      m_active_words.end(), Expander::WordCompare());
  else 
    std::sort(m_active_words.begin(), m_active_words.end(), 
	      Expander::WordCompare());
}


Lexicon::Token* Expander::acquire_token(void)
{
  Lexicon::Token *t;
  if (m_token_pool.size() == 0)
    t = new Lexicon::Token();
  else
  {
    t = m_token_pool.back();
    m_token_pool.pop_back();
  }
  return t;
}

Lexicon::Token* Expander::acquire_token(const Lexicon::Token *source_token)
{
  Lexicon::Token *t;
  if (m_token_pool.size() == 0)
    t = new Lexicon::Token(*source_token);
  else
  {
    t = m_token_pool.back();
    *t = *source_token;
    m_token_pool.pop_back();
  }
  return t;
}

void Expander::release_token(Lexicon::Token *token)
{
  if (token->path)
    Lexicon::Path::unlink(token->path);
  m_token_pool.push_back(token);
}
