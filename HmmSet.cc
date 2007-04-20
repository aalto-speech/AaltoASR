#include <assert.h>
#include <fstream>
#include <math.h>
#include <iostream>
#include <values.h>

#include "HmmSet.hh"
#include "util.hh"

#define MIN_STATE_PROB 1e-30
#define MIN_KERNEL_POS_PROB 1e-30



void
Hmm::resize(int states)
{
  m_states.resize(states);
}

HmmSet::HmmSet()
{
}


HmmSet::HmmSet(const HmmSet &hmm_set)
{
  copy(hmm_set);
}


void
HmmSet::copy(const HmmSet &hmm_set)
{
  m_pool = hmm_set.m_pool;
  m_hmm_map = hmm_set.m_hmm_map;
  m_transitions = hmm_set.m_transitions;
  m_states = hmm_set.m_states;
  m_hmms = hmm_set.m_hmms;
}


void
HmmSet::reset()
{
  m_pool.reset();

  for (int s = 0; s < num_states(); s++)
    m_states[s].emission_pdf.reset();

  for (int t = 0; t < num_transitions(); t++)
    m_transitions[t].prob = 0;
}


void
HmmSet::reserve_states(int states)
{

  m_states.resize(states);
  m_state_likelihoods.resize(states);
  for (int i=0; i<states; i++)
    m_states[i].emission_pdf.set_pool(&m_pool);
}


Hmm&
HmmSet::new_hmm(const std::string &label)
{
  // Check that label does not exist already
  std::map<std::string,int>::iterator it = m_hmm_map.find(label);
  if (it != m_hmm_map.end())
    throw DuplicateHmm();

  // Insert new hmm
  m_hmm_map[label] = m_hmms.size();
  m_hmms.push_back(Hmm());

  Hmm &hmm = m_hmms.back();
  hmm.label = label;

  return hmm;
}


Hmm&
HmmSet::add_hmm(const std::string &label, int num_states)
{
  Hmm &hmm = new_hmm(label);
  hmm.resize(num_states);
  return hmm;
}


HmmTransition&
HmmSet::add_transition(int bind_index, int target, double prob)
{
  std::vector<int> hmm_transitions = m_states[bind_index].m_transitions;
  int index;
  if (bind_index >= 0)
  {
    for (index = 0; index < (int)m_transitions.size(); index++)
    {
      if (m_transitions[index].bind_index == bind_index &&
          m_transitions[index].target == target)
        break;
    }
  }
  else
    index = (int)m_transitions.size();
  if (index == (int)m_transitions.size())
  {
    m_transitions.push_back(HmmTransition(target, prob));
    m_transitions.back().bind_index = bind_index;
  }

  hmm_transitions.push_back(index);
  return m_transitions[index];
}

int
HmmSet::clone_transition(int index)
{
  m_transitions.push_back(m_transitions[index]);
  return m_transitions.size() - 1;
}


void
HmmSet::read_mc(const std::string &filename)
{
  std::ifstream in(filename.c_str());
  if (!in) {
    fprintf(stderr, "HmmSet::read_mc(): could not open %s\n", 
	    filename.c_str());
    throw OpenError();
  }

  int states = 0;

  in >> states;
  reserve_states(states);

  for (int s = 0; s < states; s++) {
    HmmState &state = m_states[s];
    state.emission_pdf.read(in);
  }

  if (!in)
    throw ReadError();
}

void
HmmSet::read_ph(const std::string &filename)
{
  std::ifstream in(filename.c_str());
  if (!in) {
    fprintf(stderr, "HmmSet::read_ph(): could not open %s\n", 
	    filename.c_str());
    throw OpenError();
  }

  std::string buf;
  std::string label;
  int phonemes = 0;

  // Check that the first line is "PHONE"
  // Second line has the total amount of phonemes
  in >> buf >> phonemes;
  if (buf != "PHONE")
    throw ReadError();

  // Reserve one Hmm for each phoneme
  m_hmms.reserve(phonemes);

  for (int h = 0; h < phonemes; h++) {
    // Read phone
    int index = 0;
    int states = 0;
    in >> index >> states >> label; 
    if (!in)
      throw ReadError();

    Hmm &hmm = add_hmm(label, states);

    // Read states
    states -= 2;

    // FIXME: add_hmm does this already(?)
    //    hmm.resize(states);
    int state;
    in >> state >> state;
    for (int s = 0; s < states; s++)
      in >> hmm.state(s);

    // Read transitions
    for (int s = -2; s < states; s++) {
      int transitions = 0;
      int source = 0;

      in >> source >> transitions;
      source -= 2;

      for (int t = 0; t < transitions; t++) {
	int target;
	double prob;
	in >> target >> prob;

        assert(target > 0);
        assert(prob > 0);

	if (target == 1)
	  target = -2;
	else
	  target -= 2;

	if (source >= 0)
	  add_transition(hmm.state(source), target, prob);
      }
    }
  }

  if (!in)
    throw ReadError();
}


void
HmmSet::read_gk(const std::string &filename)
{
  m_pool.read_gk(filename);
}


void
HmmSet::read_all(const std::string &base)
{
  read_mc(base + ".mc");
  read_ph(base + ".ph");
  read_gk(base + ".gk");
}


void
HmmSet::write_mc(const std::string &filename)
{
  std::ofstream out(filename.c_str());

  out << m_states.size() << std::endl;
  
  for (int s = 0; s < (int)m_states.size(); s++) {
    m_states[s].emission_pdf.write(out);
  }
}

void
HmmSet::write_ph(const std::string &filename)
{
  std::ofstream out(filename.c_str());

  out << "PHONE" << std::endl;
  out << m_hmms.size() << std::endl;

  // Write hmms
  for (int h = 0; h < (int)m_hmms.size(); h++) {
    Hmm &hmm = m_hmms[h];

    out << h+1 << " " << hmm.num_states() + 2 << " " << hmm.label << std::endl;

    // Write states
    out << "-1 -2";
    for (int m = 0; m < hmm.num_states(); m++)
      out << " " << hmm.state(m);
    out << std::endl;

    // Write transitions
    out << "0 1 2 1" << std::endl;
    out << "1 0" << std::endl;
    for (int s = 0; s < hmm.num_states(); s++) {

      std::vector<int> &hmm_transitions = m_states[hmm.state(s)].transitions();
      
      int source = s + 2;
      if (source == 1)
	source = 0;

      out << source << " " << hmm_transitions.size();

      for (int t = 0; t < (int)hmm_transitions.size(); t++) {
	HmmTransition &transition = this->transition(hmm_transitions[t]);

	int target = transition.target + 2;
	if (target == 0)
	  target = 1;

	out << " " << target 
	    << " " << transition.prob;
      }
      out << std::endl;
    }
  }
}


void
HmmSet::write_gk(const std::string &filename)
{
  m_pool.write_gk(filename);
}


void
HmmSet::write_all(const std::string &base)
{
  write_mc(base + ".mc");
  write_ph(base + ".ph");
  write_gk(base + ".gk");
}


void
HmmSet::reset_cache()
{
  // Mark all values uncalculated
  while (!m_valid_state_likelihoods.empty()) {
    m_state_likelihoods[m_valid_state_likelihoods.back()]=-1.0;
    m_valid_state_likelihoods.pop_back();
  }
  // Clear also cache for base distributions
  m_pool.reset_cache();
}


double
HmmSet::state_likelihood(const int s, const FeatureVec &feature) 
{
  if (m_state_likelihoods[s] > 0)
    return m_state_likelihoods[s];

  HmmState &state = m_states[s];
  m_state_likelihoods[s] = state.emission_pdf.compute_likelihood(feature);
  if (m_state_likelihoods[s] < MIN_STATE_PROB)
    m_state_likelihoods[s] = MIN_STATE_PROB;
  m_valid_state_likelihoods.push_back(s);

  return m_state_likelihoods[s];
}


void
HmmSet::precompute_likelihoods(const FeatureVec &f)
{
  // Precompute base distribution likelihoods
  m_pool.precompute_likelihoods(f);

  // Precompute state likelihoods
  for (int s=0; s<num_states(); s++) {
    m_state_likelihoods[s] = m_states[s].emission_pdf.compute_likelihood(f);
    if (m_state_likelihoods[s] < MIN_STATE_PROB)
      m_state_likelihoods[s] = MIN_STATE_PROB;
    m_valid_state_likelihoods.push_back(s);
  }
}


void
HmmSet::start_accumulating()
{
  m_transition_accum = m_transitions;
  for (unsigned int i=0; i<m_transition_accum.size(); i++)
    m_transition_accum[i].prob=0;
  for (int s=0; s<num_states(); s++)
    m_states[s].emission_pdf.start_accumulating();
}


void
HmmSet::accumulate_ml(const FeatureVec &f, int state, int transition)
{
  assert(m_transition_accum.size() > 0);
  
  // Accumulate state transition probabilities
  int transition_index = m_states[state].transitions().at(transition);
  m_transition_accum[transition_index].prob++;

  // Accumulate state pdf
  m_states[state].emission_pdf.accumulate_ml(1, f);
}


void
HmmSet::dump_all_statistics(const std::string base) const
{
  assert(m_transition_accum.size() > 0);
}


void
HmmSet::accumulate_from_dumped_statistics(const std::string base)
{
  assert(m_transition_accum.size() > 0);

}

 
void 
HmmSet::stop_accumulating()
{
  assert(m_transition_accum.size() > 0);

  float sum;
  
  /* Update transition probabilities */
  for (int s = 0; s < num_states(); s++) {
    sum = 0.0;
    
    std::vector<int> &state_transitions = m_states[s].transitions();
    for (int t = 0; t < (int)state_transitions.size(); t++)
      sum += m_transition_accum[state_transitions[t]].prob;
    
    // If no data, do nothing
    if (sum != 0.0) {
      for (int t = 0; t < (int)state_transitions.size(); t++) {
	
	m_transition_accum[state_transitions[t]].prob = 
	  m_transition_accum[state_transitions[t]].prob/sum;
	if (m_transition_accum[state_transitions[t]].prob < .001)
	  m_transition_accum[state_transitions[t]].prob = .001;
      }
    }
  }
  
  // Stop accumulation also for state distributions
  for (int s=0; s<num_states(); s++) {
    HmmState &curr_state = state(s);
    curr_state.emission_pdf.stop_accumulating();
  }
 
  // Clear accumulator
  m_transition_accum.clear();
}
