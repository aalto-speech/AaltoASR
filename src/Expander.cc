#include <iomanip>
#include <iostream>
#include <algorithm>

#include <float.h>

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
    m_words(),
    m_sorted_words()
{
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
    state.incoming_token = NULL;
    delete token;
    m_tokens.pop_back();
  }
}

Lexicon::Token*
Expander::token_to_state(const Lexicon::Token *source_token, 
			 Lexicon::State &source_state,
			 Lexicon::State &target_state,
			 float new_log_prob)
{
  Lexicon::Token *new_token;

  // Target state has already an incoming token.
  if (target_state.incoming_token != NULL) {

    // New token is worse.  
    if (target_state.incoming_token->log_prob > new_log_prob)
      return NULL;

    // Old token is worse.  We replace the contents of the old token
    // with the new token.
    else {
      new_token = target_state.incoming_token;

      assert(new_token != source_token); // FIXME! Can this happen?!

      *new_token = *source_token;
    }
  }
	  
  // Target state is empty.
  else {
    new_token = new Lexicon::Token(*source_token);
    m_tokens.push_back(new_token);
    target_state.incoming_token = new_token;
  }

  new_token->log_prob = new_log_prob;

  // Update beam threshold
  if (new_log_prob > m_beam_best_tmp)
    m_beam_best_tmp = new_log_prob;

  return new_token;
}

// PRECONDITIONS:
//
// - Tokens are in incoming_token slots.
// - Acoustics has been updated to current time.
//
// POSTCONDITIONS:
//
// - The order and number of tokens is undefined.
// - Tokens are in incoming_token slots.
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

void
Expander::move_all_tokens()
{
  for (int i = 0; i < m_tokens.size(); i++) {
    const Lexicon::Token *token = m_tokens[i];
    Lexicon::Node *node = token->node;
    Lexicon::State &state = node->states[token->state];
    assert(state.outgoing_token == NULL);
    assert(state.incoming_token == token);
    state.outgoing_token = state.incoming_token;
    state.incoming_token = NULL;
  }

  // ITERATE ALL TOKENS
  for (int t = 0; t < m_tokens.size(); t++) {
    Lexicon::Token *source_token = m_tokens[t];
    Lexicon::Node *source_node = source_token->node;
    Lexicon::State &source_state = source_node->states[source_token->state];
    const Hmm &hmm = m_hmms[source_node->hmm_id];
    const HmmState &hmm_state = hmm.states[source_token->state];

    if (source_token == source_state.incoming_token)
      continue;

    assert(source_token == source_state.outgoing_token);
    source_state.outgoing_token = NULL;

    // Beam pruning using the beam calculated in the last frame
    if (source_token->log_prob < m_beam_best - m_beam)
      goto token_finished;

    // ITERATE TRANSITIONS
    for (int r = 0; r < hmm_state.transitions.size(); r++) {
      const HmmTransition &transition = hmm_state.transitions[r];
      int target_state_id = transition.target;
      float log_prob = source_token->log_prob + transition.log_prob;

      // Target state is a sink state.  Clone the token to the next
      // nodes in the lexicon.
      if (hmm.is_sink(target_state_id)) {

	// If token is at word end, update the best words.
	int word_id = source_node->word_id;
	if (word_id >= 0) {
	  Word *word = &m_words[word_id];

	  // Note that we use (m_frame) instead of (m_frame + 1),
	  // because the current frame is actually already the start
	  // of the next word.  This also assumes that there can not
	  // be an empty word in lexicon.
	  if (!m_forced_end || m_frame == m_frames - 1) {
	    float log_prob = source_token->log_prob;
	    float avg_log_prob = log_prob / m_frame;

	    // FIXME: is this correct?  Do we really want to add this
	    // only in word ends?
	    log_prob += source_node->log_prob;

	    if (!word->active || avg_log_prob > word->avg_log_prob) {
	      if (!word->active)
		m_sorted_words.push_back(word);
	      word->avg_log_prob = avg_log_prob;
	      word->log_prob = log_prob;
	      word->frames = m_frame;
	      word->active = true;
	    }
	  }
	}

	// ITERATE NEXT NODES
	for (int n = 0; n < source_node->next.size(); n++) {
	  Lexicon::Node *target_node = source_node->next[n];

	  // FIXME: why we do not want to do this here?  Why only in
	  // word ends?  
	  // log_prob += target_node->log_prob;

	  Lexicon::State &target_state = target_node->states[0];
	  Lexicon::Token *new_token = 
	    token_to_state(source_token, source_state, target_state, log_prob);
	  if (new_token != NULL) {
	    new_token->state = 0;
	    new_token->node = target_node;
	    new_token->state_duration = 0;
//	    new_token->add_path(target_node->hmm_id, new_token->frame + 1,
//				log_prob);

	    // The new token is in source state now.  We want to move
	    // it again during current token loop.
	    target_state.incoming_token = NULL;
	    target_state.outgoing_token = new_token;
	  }
	}
      }

      // Target state is not a sink state.  Just clone the token.
      else {

	// FIXME: Maximum state duration test
	if (target_state_id == source_token->state 
	    && hmm.label[0] != '_') {
	  source_token->state_duration++;
	  if (source_token->state_duration >= m_max_state_duration)
	    continue;
	}
	else {
	  source_token->state_duration = 0;
	}

	log_prob += m_acoustics.log_prob(hmm.states[target_state_id].model); 

	// Beam pruning using already the temporary beam_best, which
	// is under calculation for the next frame.
	// FIXME: this might be quite unnecessary
  	if (source_token->log_prob < m_beam_best_tmp - m_beam)
  	  continue;

	Lexicon::State &target_state = source_node->states[target_state_id];
	Lexicon::Token *new_token = 
	  token_to_state(source_token, source_state, target_state, log_prob);
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
    delete source_token;
    if (m_tokens.size() > t + 1)
      m_tokens[t] = m_tokens[m_tokens.size() - 1];
    m_tokens.pop_back();
    t--;
  }
}

void
Expander::clear_tokens()
{
  for (int t = 0; t < m_tokens.size(); t++) {
    const Lexicon::Token *token = m_tokens[t];
    Lexicon::Node *node = token->node;
    Lexicon::State &state = node->states[token->state];
    state.incoming_token = NULL;
    delete m_tokens[t];
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
    token = new Lexicon::Token();

    state.incoming_token = token;
    token->frame = start_frame - 1;
    token->state = 0;
    token->node = node->next[next_id];
    token->log_prob = 0;
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
    m_sorted_words.clear();
    m_sorted_words.reserve(m_lexicon.words());
    m_words.resize(m_lexicon.words());
    for (int i = 0; i < m_words.size(); i++)
      m_words[i].word_id = i;
  }

  // Clear best words list
  for (int i = 0; i < m_words.size(); i++)
    m_words[i].active = false;
  m_sorted_words.clear();    

  create_initial_tokens(start_frame);

  m_frames = frames;
  m_beam_best_tmp = -1e10;
  for (m_frame = 0; m_frames < 0 || m_frame < m_frames; m_frame++) {
    m_beam_best = m_beam_best_tmp;
    m_beam_best_tmp = -1e10;

    if (m_token_limit > 0)
      keep_best_tokens(m_token_limit);
    if (!m_acoustics.go_to(start_frame + m_frame))
      break;
    move_all_tokens();

    if (m_tokens.empty())
      break;

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
