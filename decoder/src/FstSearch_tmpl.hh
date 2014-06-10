#include "FstSearch.hh"
#include "OneFrameAcoustics.hh"

#include <algorithm>
#include <set>

inline std::string FstToken::str() const {
  std::ostringstream os;
  os << "Token " << node_idx << " " << logprob << " dur " << state_dur << " '";
  for (const auto s: unemitted_words) {
    os << " " << s;
  }
  os << " '";
  return os.str();
}

// Constructor, if acoustics is created by the caller
template <typename T>
FstSearch_base<T>::FstSearch_base(const char * search_fst_fname, FstAcoustics *fst_acu):
  verbose(0), m_fst_acoustics(fst_acu), m_delete_acoustics(false), 
  m_duration_scale(3.0f), m_beam(2600.0f), m_token_limit(5000), m_transition_scale(1.0f), 
  m_one_token_per_node(false)
{
  m_fst.read(search_fst_fname);
  if (m_one_token_per_node) m_node_best_token.resize(m_fst.nodes.size());
}

// Constructor, if acoustics created here
template <typename T>
FstSearch_base<T>::FstSearch_base(const char * search_fst_fname, const char * hmm_path, const char * dur_path):
  m_fst_acoustics(new FstAcoustics(hmm_path, dur_path)), m_delete_acoustics(true), 
  m_duration_scale(3.0f), m_beam(2600.0f), m_token_limit(5000), m_transition_scale(1.0f), 
  m_one_token_per_node(false)
{
  m_fst.read(search_fst_fname);
  if (m_one_token_per_node) m_node_best_token.resize(m_fst.nodes.size());
}

template <typename T>
FstSearch_base<T>::~FstSearch_base() {
  if (m_delete_acoustics && m_fst_acoustics) delete m_fst_acoustics;
}

template <typename T>
void FstSearch_base<T>::init_search() {
  //if (verbose) fprintf(stderr, "Init search\n");
  m_new_tokens.resize(1);
  T &t=m_new_tokens[0];
  t.node_idx = m_fst.initial_node_idx;
  if (m_one_token_per_node) std::fill(m_node_best_token.begin(), m_node_best_token.end(), -1);
}

template <typename T>
void FstSearch_base<T>::propagate_tokens() {
  m_active_tokens.swap(m_new_tokens); 
  //fprintf(stderr, "Working on frame %d, %ld active_tokens\n", m_frame, m_active_tokens.size());

  // Clean up the buffers that will hold the new values
  m_new_tokens.clear();

  float best_logprob=-999999999.0f;
  for (auto t: m_active_tokens) {
    float blp = propagate_token(t, best_logprob-m_beam);
    if (best_logprob<blp) {
      best_logprob = blp;
    }
  }
  
  // sort and prune
  std::sort(m_new_tokens.begin(), m_new_tokens.end(),
            [](T const & a, T const &b){return a.logprob > b.logprob;});

  if (m_one_token_per_node) {
    std::fill(m_node_best_token.begin(), m_node_best_token.end(), -1);
    /*fprintf(stderr, "Sorted\n");
      int c=0;
      for (auto t: m_new_tokens) {
      fprintf(stderr, "  %d(%d): %s\n", ++c, m_frame, t.str().c_str());
      }*/
    if (m_new_tokens.size() > m_token_limit) {
      m_new_tokens.resize(m_token_limit);
    }
    //fprintf(stderr, "size after token limit %ld\n", m_new_tokens.size());
  } else {
    // For each active node, keep only one hypo with the same tokens
    int num_accepted_tokens=0;
    auto orig_tokens(std::move(m_new_tokens));
    m_new_tokens.clear();
    //std::map<int, std::unordered_set<std::vector<std::string > > > histmap; // no hash function for type
    std::map<int, std::set<std::vector<std::string > > > histmap;
    for (auto t: orig_tokens) {
      //fprintf(stderr, "Is there already?");
      auto &valset = histmap[t.node_idx];
      if (valset.find(t.unemitted_words) !=valset.end() ) {
        //fprintf(stderr, " Yes!\n");
        continue;
      }
      //fprintf(stderr, " Nope!\n");
      valset.insert(t.unemitted_words);
      m_new_tokens.push_back(std::move(t));
      if (num_accepted_tokens>= m_token_limit) break;
      num_accepted_tokens++;
    }
  }
  
  int beam_prune_idx = 1;
  best_logprob = m_new_tokens[0].logprob;
  while (beam_prune_idx < m_new_tokens.size() && m_new_tokens[beam_prune_idx].logprob > best_logprob-m_beam) {
    beam_prune_idx++;
  }
  m_new_tokens.resize(beam_prune_idx);
  //fprintf(stderr, "size after beam %ld\n", m_new_tokens.size());
}

template <typename T>
void FstSearch_base<T>::run() {
  while (m_fst_acoustics->next_frame()) {
    propagate_tokens();
  }
  //fprintf(stderr, "%s\n", tokens_at_final_states().c_str());
  //fprintf(stderr, "%s\n", best_tokens().c_str());
  
}


template <typename T>
bytestype FstSearch_base<T>::tokens_at_final_states() {
  std::ostringstream os;
  os << "Tokens at final nodes:" << std::endl;
  for (const auto t: m_new_tokens) {
    if (m_fst.nodes[t.node_idx].end_node) {
      os << "  " << t.str() << std::endl;
    }
  }
  return os.str();
}

template <typename T>
bytestype FstSearch_base<T>::best_tokens(int n) {
  std::ostringstream os;
  os << "Best tokens:" << std::endl;
  int c=0;
  for (const auto t: m_new_tokens) {
    os << "  " << t.str() << std::endl;
    if (c++>n) break;
  }
  return os.str();
}

template <typename T>
bytestype FstSearch_base<T>::get_result_and_logprob(float &logprob) {
  for (const auto t: m_new_tokens) {
    if (!m_fst.nodes[t.node_idx].end_node) {
      continue;
    }
    logprob = t.logprob;
    std::ostringstream os;
    for (const auto w: t.unemitted_words) {
      os << w << " ";
    }
    std::string retval(os.str()); // The best hypo at a final node
    return retval.substr(0, retval.size()-1); // Remove the trailing space
  }
  // FIXME: We should throw an exception if we end up here !!!!
  logprob=-1.0f;
  return "";
}

template <typename T>
float FstSearch_base<T>::get_best_final_token_logprob() {
  for (const auto t: m_new_tokens) {
    if (!m_fst.nodes[t.node_idx].end_node) continue;
    return t.logprob;
  }
  return -9999999.9f;
}

template <typename T>
float FstSearch_base<T>::propagate_token( T &t, float beam_prune_threshold) {
  float best_logprob=-999999999.0f;
  Fst::Node n = m_fst.nodes[t.node_idx];
  //fprintf(stderr, "Propagate token at node %d\n", t.node_idx);
  //fprintf(stderr, " num arcs %ld\n", n.arcidxs.size());
  for (const auto arcidx: n.arcidxs) {
    auto arc = m_fst.arcs[arcidx];
    auto node = m_fst.nodes[arc.target];
    //fprintf(stderr, "%s\n", arc.str().c_str());
    //fprintf(stderr, "%s\n", node.str().c_str());

    // Is this necessary, do objects have a default copy constructor?
    T updated_token(t);
    
    //Token updated_token;
    //updated_token.logprob = t.logprob;
    //updated_token.state_dur = t.state_dur;
    //updated_token.unemitted_words = t.unemitted_words;

    updated_token.node_idx = arc.target;

    int source_emission_pdf_idx = m_fst.nodes[arc.source].emission_pdf_idx;
    //fprintf(stderr, "Add trans logprob %.5f\n", arc.transition_logprob);
    updated_token.logprob += m_transition_scale * arc.transition_logprob;
    if (node.emission_pdf_idx >= 0) {
      //fprintf(stderr, "Emit logprob %.5f\n", m_acoustics->log_prob(node.emission_pdf_idx));
      updated_token.logprob += m_fst_acoustics->log_prob(node.emission_pdf_idx);  
    }
    if (arc.target != arc.source) {
      if (source_emission_pdf_idx >=0) {
        //fprintf(stderr, "Adding dur logprob %d -> %d (%d)\n", arc.source, arc.target, updated_token.state_dur);
        // Add the duration from prev state at state change boundary
        updated_token.logprob += m_duration_scale * 
          m_fst_acoustics->duration_logprob(source_emission_pdf_idx, updated_token.state_dur);
        updated_token.state_dur = 1;
      } //else fprintf(stderr, "Skip duration model.\n");
    } else {
      //fprintf(stderr, "Increasing state dur %d\n", arc.source);
      updated_token.state_dur +=1;
    }
    if (arc.emit_symbol.size()) {
      updated_token.unemitted_words.push_back(arc.emit_symbol);
    }
    //fprintf(stderr, "m_nbt size %ld, idx %d\n", m_node_best_token.size(), updated_token.node_idx);
    //fprintf(stderr, "%d\n", m_node_best_token[updated_token.node_idx]);
    int best_token_idx = m_one_token_per_node? m_node_best_token[updated_token.node_idx] : -1;
    T *best_token = best_token_idx == -1 ? nullptr : &(m_new_tokens[best_token_idx]);
    if (updated_token.logprob > beam_prune_threshold && // Do approximate beam pruning here, exact later
        ( best_token_idx ==-1 || updated_token.logprob > best_token->logprob)) {
      if (m_one_token_per_node) {
        m_node_best_token[updated_token.node_idx] = m_new_tokens.size();
      }
      //fprintf(stderr, "Accepted token %s\n", updated_token.str().c_str());
      if (best_logprob < updated_token.logprob) {
        best_logprob = updated_token.logprob;
      }
      m_new_tokens.push_back(std::move(updated_token));
    }
  }
  return best_logprob;
}


