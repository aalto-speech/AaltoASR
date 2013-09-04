#include "FstSearch.hh"

#include <algorithm>
#include <math.h>

FstSearch::FstSearch(const char * search_fst_fname, const char * hmm_path, const char * dur_path):
  duration_scale(3.0f), beam(2600.0f), token_limit(5000), m_frame(0), m_hmm_reader(NULL), 
  m_hmms(NULL), m_acoustics(NULL), m_lna_reader(NULL)
{
  m_fst.read(search_fst_fname);
  m_node_best_token.resize(m_fst.nodes.size());

  hmm_read(hmm_path);
  if (dur_path != NULL) {
    duration_read(dur_path);
  }

  m_lna_reader = new LnaReaderCircular;

}

FstSearch::~FstSearch() {
  if (m_hmm_reader) delete m_hmm_reader;
  if (m_lna_reader) delete m_lna_reader;
}


// This is a direct copy from Toolbox.cc. FIXME: Code duplication
void
FstSearch::hmm_read(const char *file)
{
  std::ifstream in(file);
  if (!in)
    throw OpenError();
  if (m_hmm_reader) {
    delete m_hmm_reader;
  }
  m_hmm_reader = new NowayHmmReader();
  //m_hmm_map = &(m_hmm_reader->hmm_map());
  m_hmms = &(m_hmm_reader->hmms());
  m_hmm_reader->read(in);
}

// This is a direct copy from Toolbox.cc. FIXME: Code duplication
void FstSearch::lna_open(const char *file, int size)
{
  m_lna_reader->open_file(file, size);
  m_acoustics = m_lna_reader;
}

// This is a direct copy from Toolbox.cc. FIXME: Code duplication
void FstSearch::lna_open_fd(const int fd, int size)
{
  m_lna_reader->open_fd(fd, size);
  m_acoustics = m_lna_reader;
}

// This is a direct copy from Toolbox.cc. FIXME: Code duplication
void FstSearch::lna_close()
{
  m_lna_reader->close();
}

std::string FstSearch::run() {
  // FIXME: Set a token to the initial state !!!
  m_new_tokens.resize(1);
  Token &t=m_new_tokens[0];
  t.node_idx = m_fst.initial_node_idx;
  std::fill(m_node_best_token.begin(), m_node_best_token.end(), -1);

  while (m_acoustics->go_to(m_frame)) {
    m_active_tokens.swap(m_new_tokens); 
    //fprintf(stderr, "Working on frame %d, %ld active_tokens\n", m_frame, m_active_tokens.size());

    // Clean up the buffers that will hold the new values
    m_new_tokens.clear();

    float best_logprob=-999999999.0f;
    for (auto t: m_active_tokens) {
      float blp = propagate_token(t, best_logprob-beam);
      if (best_logprob<blp) {
        best_logprob = blp;
      }
    }

    // sort and prune
    std::fill(m_node_best_token.begin(), m_node_best_token.end(), -1);
    std::sort(m_new_tokens.begin(), m_new_tokens.end(),
              [](Token const & a, Token const &b){return a.logprob > b.logprob;});
    
    //fprintf(stderr, "Sorted\n");
    //for (auto t: m_new_tokens) {
      //fprintf(stderr, "  %s\n", t.str().c_str());
    //}
    if (m_new_tokens.size() > token_limit) {
      m_new_tokens.resize(token_limit);
    }
    //fprintf(stderr, "size after token limit %ld\n", m_new_tokens.size());

    int beam_prune_idx = 1;
    best_logprob = m_new_tokens[0].logprob;
    while (beam_prune_idx < m_new_tokens.size() && m_new_tokens[beam_prune_idx].logprob > best_logprob-beam) {
      beam_prune_idx++;
    }
    m_new_tokens.resize(beam_prune_idx);
    //fprintf(stderr, "size after beam %ld\n", m_new_tokens.size());

    m_frame++;
  }

  // Print tokens at final states
  //fprintf(stderr, "Tokens at final nodes:\n");
  for (auto t: m_new_tokens) {
    if (!m_fst.nodes[t.node_idx].end_node) {
      continue;
    }
    std::ostringstream os;
    for (int i=0; i<t.unemitted_words.size()-1; ++i) {
      os << t.unemitted_words[i] << " ";
    }
    os << t.unemitted_words[t.unemitted_words.size()-1];
    return os.str(); // The best hypo at a final node
    //fprintf(stderr, "  %s\n", t.str().c_str());
  }
}

float FstSearch::propagate_token( Token &t, float beam_prune_threshold) {
  float best_logprob=-999999999.0f;
  Fst::Node n = m_fst.nodes[t.node_idx];
  //fprintf(stderr, "Propagate token at node %d\n", t.node_idx);
  //fprintf(stderr, " num arcs %ld\n", n.arcidxs.size());
  for (auto arcidx: n.arcidxs) {
    auto arc = m_fst.arcs[arcidx];
    auto node = m_fst.nodes[arc.target];
    //fprintf(stderr, "%s\n", arc.str().c_str());
    //fprintf(stderr, "%s\n", node.str().c_str());

    Token updated_token(t);
    int source_emission_pdf_idx = m_fst.nodes[arc.source].emission_pdf_idx;
    //fprintf(stderr, "Add trans logprob %.5f\n", arc.transition_logprob);
    updated_token.logprob += arc.transition_logprob;
    if (node.emission_pdf_idx >= 0) {
      //fprintf(stderr, "Emit logprob %.5f\n", m_acoustics->log_prob(node.emission_pdf_idx));
      updated_token.logprob += m_acoustics->log_prob(node.emission_pdf_idx);  
    }
    if (arc.target != arc.source && source_emission_pdf_idx >= 0 ) {
      //fprintf(stderr, "Adding dur logprob %d -> %d (%d)\n", arc.source, arc.target, updated_token.state_dur);
      // Add the duration from prev state at state change boundary
      updated_token.logprob += duration_scale * 
        duration_logprob(source_emission_pdf_idx, updated_token.state_dur);
      updated_token.state_dur = 1;
    } else {
      //fprintf(stderr, "Increasing state dur %d\n", arc.source);
      updated_token.state_dur +=1;
    }
    if (arc.emit_symbol.size()) {
      updated_token.unemitted_words.push_back(arc.emit_symbol);
    }
    updated_token.node_idx = arc.target;
    //fprintf(stderr, "m_nbt size %ld, idx %d\n", m_node_best_token.size(), updated_token.node_idx);
    //fprintf(stderr, "%d\n", m_node_best_token[updated_token.node_idx]);
    int best_token_idx = m_node_best_token[updated_token.node_idx];
    Token *best_token = best_token_idx == -1 ? NULL : &(m_new_tokens[best_token_idx]);
    if (updated_token.logprob > beam_prune_threshold && // Do approximate beam pruning here, exact later
        ( best_token_idx ==-1 || updated_token.logprob > best_token->logprob)) {
      m_node_best_token[updated_token.node_idx] = m_new_tokens.size();
      //fprintf(stderr, "Accepted token %s\n", updated_token.str().c_str());
      m_new_tokens.push_back(updated_token);
      if (best_logprob < updated_token.logprob) {
        best_logprob = updated_token.logprob;
      }
    }
  }
  return best_logprob;
}

float FstSearch::duration_logprob(int emission_pdf_idx, int duration) {
  //fprintf(stderr, "Request dur for %d (%d)\n", emission_pdf_idx, duration);
  float a = m_a_table[emission_pdf_idx];
  if (a<=0) {
    return 0.0f;
  }

  float b = m_b_table[emission_pdf_idx];
  float const_term = -a*logf(b)-lgammaf(a);
  return (a-1)*logf(duration)-duration/b+const_term;
}

void FstSearch::duration_read(const char *fname) {
  std::ifstream dur_in(fname);
  if (!dur_in) 
    throw OpenError();

  int version;
  float a,b;
  dur_in >> version;
  if (version!=4) 
    throw InvalidFormat();
  
  int num_states, state_id;
  dur_in >> num_states;
  //fprintf(stderr, "reading %d duration models\n", num_states);
  m_a_table.resize(num_states);
  m_b_table.resize(num_states);
  
  for (int i=0; i<num_states; i++) {
    dur_in >> state_id;
    if (state_id != i) {
      throw InvalidFormat();
    }
    dur_in >> a >> b;
    m_a_table.push_back(a);
    m_b_table.push_back(b);
  }
}
