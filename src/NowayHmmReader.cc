#include <math.h>
#include <string.h>

#include "NowayHmmReader.hh"

void
NowayHmmReader::read_hmm(std::istream &in, Hmm &hmm)
{
  int hmm_id = 0;
  int num_states = 0;
  std::string label;
  std::vector<HmmState> states;
  
  in >> hmm_id;
  in >> num_states >> label;
  states.resize(num_states);

  // Real model ids for each state
  for (int s = 0; s < num_states; s++)
    in >> states[s].model;

  // Read transitions
  for (int s = 0; s < num_states; s++) {
    HmmState &state = states[s];
    int from;
    int to;
    int num_transitions = 0;
    float prob;

    in >> from >> num_transitions;
    state.transitions.resize(num_transitions);

    for (int t = 0; t < num_transitions; t++) {
      in >> to >> prob;
      state.transitions[t].target = to;
      state.transitions[t].log_prob = log10(prob);
    }
  }

  label.swap(hmm.label);
  states.swap(hmm.states);
}

void
NowayHmmReader::read(std::istream &in)
{
  std::istream::iostate old_state = in.exceptions();
  in.exceptions(in.badbit | in.failbit | in.eofbit);

  try {
    char buf[5];
    in.read(buf, 5);
    if (!in || strncmp(buf, "PHONE", 5))
      throw InvalidFormat();
    
    int num_hmms;
    in >> num_hmms;
    m_hmms.resize(num_hmms);
    for (int h = 0; h < num_hmms; h++) {
      // FIXME: should we check that hmm_ids read from stream match
      // with 'h'?
      read_hmm(in, m_hmms[h]);
      m_hmm_map[m_hmms[h].label] = h;
    }
    
  }
  catch (std::exception &e) {
    in.exceptions(old_state);
    throw InvalidFormat();
  }
  
  in.exceptions(old_state);
}
