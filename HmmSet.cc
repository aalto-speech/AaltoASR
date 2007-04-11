#include <assert.h>
#include <fstream>
#include <math.h>
#include <iostream>
#include <values.h>

#include "HmmSet.hh"
#include "util.hh"
#include "mtl/lu.h"

#define MIN_STATE_PROB 1e-30
#define MIN_KERNEL_POS_PROB 1e-30



void
Hmm::resize(int states)
{
  m_states.resize(states);
  m_transitions.resize(states+1);
}

HmmSet::HmmSet()
  : m_dim(0)
{
}


HmmSet::HmmSet(const HmmSet &hmm_set)
{
  copy(hmm_set);
}


void
HmmSet::copy(const HmmSet &hmm_set)
{
  obs_log_probs = hmm_set.obs_log_probs;
  obs_kernel_likelihoods = hmm_set.obs_kernel_likelihoods;
  m_dim = hmm_set.m_dim;
  m_cov_type = hmm_set.m_cov_type;
  m_hmm_map = hmm_set.m_hmm_map;
  m_transitions = hmm_set.m_transitions;
  m_states = hmm_set.m_states;
  m_hmms = hmm_set.m_hmms;
  m_state_probs = hmm_set.m_state_probs;
  m_kernel_likelihoods = hmm_set.m_kernel_likelihoods;
  m_valid_stateprobs = hmm_set.m_valid_stateprobs;
  m_valid_kernel_likelihoods = hmm_set.m_valid_kernel_likelihoods;
  m_viterbi_scale_coeff = hmm_set.m_viterbi_scale_coeff;
}


void
HmmSet::reset()
{
  for (int k = 0; k < num_kernels(); k++) {
    std::fill(m_kernels[k].center.begin(), m_kernels[k].center.end(), 0);
    m_kernels[k].cov.reset();
  }

  for (int s = 0; s < num_states(); s++) {
    for (int w = 0; w < (int)m_states[s].weights.size(); w++)
      m_states[s].weights[w].weight = 0;
  }

  for (int t = 0; t < num_transitions(); t++)
    m_transitions[t].prob = 0;
}


void
HmmSet::reserve_states(int states)
{
  m_states.resize(states);
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

void
HmmSet::clone_hmm(const std::string &source, const std::string &target)
{
  Hmm &target_hmm = new_hmm(target);
  target_hmm = hmm(source);
  target_hmm.label = target;
}

void
HmmSet::untie_transitions(const std::string &label)
{
  Hmm &hmm = this->hmm(label);
  for (int s = 0; s < hmm.num_states(); s++) {
    std::vector<int> &transitions = hmm.transitions(s);
    for (int t = 0; t < (int)transitions.size(); t++)
      transitions[t] = clone_transition(transitions[t]); 
  }
}

HmmTransition&
HmmSet::add_transition(int h, int source, int target, float prob, int bind_index)
{
  std::vector<int> &hmm_transitions = m_hmms[h].transitions(source);
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
    int weights = 0;
    in >> weights;

    state.weights.resize(weights);
    for (int w = 0; w < weights; w++)
      in >> state.weights[w].kernel >> state.weights[w].weight;
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

  in >> buf >> phonemes;
  if (buf != "PHONE")
    throw ReadError();

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
    hmm.resize(states);
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

      if (source >= 0)
	hmm.transitions(source).reserve(transitions);
      for (int t = 0; t < transitions; t++) {
	int target;
	float prob;
	in >> target >> prob;

        assert(target > 0);
        assert(prob > 0);

	if (target == 1)
	  target = -2;
	else
	  target -= 2;

	if (source >= 0)
	  add_transition(h, source, target, prob, hmm.state(source));
      }
    }
  }

  if (!in)
    throw ReadError();
}

void
HmmSet::read_all(const std::string &base)
{
  read_gk(base + ".gk");
  read_mc(base + ".mc");
  read_ph(base + ".ph");
}


void
HmmSet::write_mc(const std::string &filename)
{
  std::ofstream out(filename.c_str());

  out << m_states.size() << std::endl;
  
  // Format: NUM_WEIGHTS  KERNEL WEIGHT  KERNEL WEIGHT...
  for (int s = 0; s < (int)m_states.size(); s++) {
    HmmState &state = m_states[s];
    out << state.weights.size();
    for (int w = 0; w < (int)state.weights.size(); w++) {
      out << " " << m_states[s].weights[w].kernel 
	  << " " << m_states[s].weights[w].weight;
    }
    out << std::endl;
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
      std::vector<int> &hmm_transitions = hmm.transitions(s);

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
HmmSet::write_all(const std::string &base)
{
  write_gk(base + ".gk");
  write_mc(base + ".mc");
  write_ph(base + ".ph");
}


void
HmmSet::compute_observation_log_probs(const FeatureVec &feature)
{
  double sum = 0;

  if (pcgmm.basis_dim() > 0)
    pcgmm.compute_all_likelihoods(feature, obs_kernel_likelihoods);
  else if (scgmm.basis_dim() > 0)
    scgmm.compute_all_likelihoods(feature, obs_kernel_likelihoods);
  else {
    obs_kernel_likelihoods.resize(num_kernels());
    for (int k = 0; k < num_kernels(); k++) {
      obs_kernel_likelihoods[k] = compute_kernel_likelihood(k, feature);
    }
  }

  obs_log_probs.resize(num_states());
  for (int s = 0; s < num_states(); s++) {
    HmmState &state = m_states[s];

    obs_log_probs[s] = 0;
    for (int w = 0; w < (int)state.weights.size(); w++) {
      obs_log_probs[s] += (double)state.weights[w].weight * 
	obs_kernel_likelihoods[state.weights[w].kernel];
    }
    sum += obs_log_probs[s];
  }

  if (sum == 0)
    sum = 1;
  
  for (int s = 0; s < (int)obs_log_probs.size(); s++)
    obs_log_probs[s] = util::safe_log(obs_log_probs[s] / sum);
}


void 
HmmSet::reset_state_probs() 
{
  if (m_state_probs.size()==0) { // First call to the func
    m_state_probs.resize(m_states.size(),-1.0);
    m_kernel_likelihoods.resize(m_kernels.size(),-1.0);    
  } 
  // Mark all values uncalculated
  while (!m_valid_stateprobs.empty()) {
    m_state_probs[m_valid_stateprobs.back()]=-1.0;
    m_valid_stateprobs.pop_back();
  }
  while (!m_valid_kernel_likelihoods.empty()) {
    m_kernel_likelihoods[m_valid_kernel_likelihoods.back()]=-1.0;
    m_valid_kernel_likelihoods.pop_back();
  }
}


float 
HmmSet::state_prob(const int s, const FeatureVec &feature) 
{
  int k;
  double temp;
  
  // Is there a valid value ?
  if (m_state_probs[s] >= 0.0) 
    return(m_state_probs[s]);

  // Calculate the valid value
  HmmState &state = m_states[s];
  temp = 0;
  for (int w = 0; w < (int)state.weights.size(); w++) {
    k = state.weights[w].kernel;
    // Is there a valid value?
    if (m_pdf_likelihoods[k] < 0)
    {
      m_pdf_likelihoods[k] = compute_kernel_likelihood(k, feature);
      m_valid_kernel_likelihoods.push_back(k);
    }
    temp += state.weights[w].weight * m_kernel_likelihoods[k];
  }

  if (temp < MIN_STATE_PROB)
    temp = MIN_STATE_PROB;
  m_state_probs[s] = temp;
  m_valid_stateprobs.push_back(s);
  return(m_state_probs[s]);
}
