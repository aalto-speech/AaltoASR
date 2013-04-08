#include <math.h>
#include <string.h>
#include <cstdlib>

#include "NowayHmmReader.hh"

NowayHmmReader::NowayHmmReader()
  : m_num_models(0)
{

}

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
  for (int s = 0; s < num_states; s++) {
    int model_index;
    in >> model_index;
    if (model_index + 1 > m_num_models)
      m_num_models = model_index + 1;
    states[s].model = model_index;
  }

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
      if (to >= num_states || to < 1) {
	std::cerr << "hmm '" << label << "' has invalid transition" << std::endl;
	throw InvalidFormat();
      }
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
  m_num_models = 0;
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

void NowayHmmReader::read_durations(std::istream &in)
{
  std::istream::iostate old_state = in.exceptions();
  in.exceptions(in.badbit | in.failbit | in.eofbit);
  int version;
  int hmm_id;
  float a,b;
  float a0,a1,b0,b1;

  try {
    if (m_hmms.size() == 0)
    {
      std::cerr << "NowayHmmReader::read_durations(): Error: HMMs must be loaded before duration file!" << std::endl;
      exit(1);
    }
    in >> version;
    if (version != 1 && version != 2 && version != 3 && version != 4)
      throw InvalidFormat();
    if (version == 3 || version == 4)
    {
      std::vector<float> a_table;
      std::vector<float> b_table;
      int num_states, state_id;
      in >> num_states;
      if (version == 3)
        num_states++; // Used to be the index of the last state
      a_table.reserve(num_states);
      b_table.reserve(num_states);
      for (int i = 0; i < num_states; i++)
      {
        in >> state_id;
        if (state_id != i)
          throw InvalidFormat();
        in >> a >> b;
        a_table.push_back(a);
        b_table.push_back(b);
      }
      for (int i = 0; i < m_hmms.size(); i++)
      {
        for (int s = 2; s < m_hmms[i].states.size(); s++)
        {
          HmmState &state = m_hmms[i].states[s];
          if (state.model >= num_states)
            throw StateOutOfRange();
          state.duration.set_parameters(a_table[state.model],
                                        b_table[state.model]);
        }
      }
    }
    else
    {
      for (int i = 0; i < m_hmms.size(); i++)
      {
        in >> hmm_id;
        if (hmm_id != i+1)
          throw InvalidFormat();

        if (version == 2)
        {
          in >> a >> b;
          m_hmms[i].set_pho_dur_stat(a,b);
        }

        for (int s = 2; s < m_hmms[i].states.size(); s++)
        {
          HmmState &state = m_hmms[i].states[s];
          in >> a >> b;
          state.duration.set_parameters(a,b);
        }

        if (version == 2)
        {
          for (int s = 2; s < m_hmms[i].states.size(); s++)
          {
            HmmState &state = m_hmms[i].states[s];
            in >> a0 >> a1 >> b0 >> b1;
            state.duration.set_sr_parameters(a0,a1,b0,b1);
          }
        }
      }
    }
  }
  catch (std::exception &e) {
    in.exceptions(old_state);
    throw InvalidFormat();
  }
  
  in.exceptions(old_state);
}

