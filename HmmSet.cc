#include <assert.h>
#include <fstream>
#include <math.h>
#include <iostream>
#include <values.h>

#include "HmmSet.hh"
#include "util.hh"
#include "str.hh"

#define MIN_STATE_PROB 1e-50


void
Hmm::resize(int states)
{
  m_states.resize(states);
}

HmmSet::HmmSet()
{
  m_mode=PDF::ML;
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
  m_emission_pdfs = hmm_set.m_emission_pdfs;
  m_hmms = hmm_set.m_hmms;
}


void
HmmSet::reset()
{
  m_pool.reset();

  for (int i = 0; i < num_emission_pdfs(); i++)
    m_emission_pdfs[i].reset();

  for (int t = 0; t < num_transitions(); t++)
    m_transitions[t].prob = 0;
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


int
HmmSet::add_transition(int source, int target, double prob)
{
  int index = (int)m_transitions.size();
  m_transitions.push_back(HmmTransition(source, target, prob));
  m_states[source].m_transitions.push_back(index);  
  return index;
}


int
HmmSet::add_state(int pdf_index)
{
  int index = (int)m_states.size();
  m_states.push_back(HmmState(pdf_index));
  return index;
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

  int pdfs = 0;

  in >> pdfs;

  m_emission_pdfs.resize(pdfs);
  m_pdf_likelihoods.resize(pdfs);
  m_valid_pdf_likelihoods.clear();
  
  for (int i = 0; i < pdfs; i++) {
    Mixture &pdf = m_emission_pdfs[i];
    pdf.set_pool(&m_pool);
    pdf.read(in);
    m_pdf_likelihoods[i] = -1;
  }

  if (!in)
    throw ReadError();
}

bool
HmmSet::read_ph(const std::string &filename)
{
  std::string buf;
  bool legacy_mode = false;
  std::ifstream in(filename.c_str());
  if (!in) {
    fprintf(stderr, "HmmSet::read_ph(): could not open %s\n", 
	    filename.c_str());
    throw OpenError();
  }

  // Check that the first line is "PHONE"
  in >> buf;
  if (buf == "PHONE")
  {
    legacy_mode = true;
    read_legacy_ph(in);
  }
  else
    throw ReadError();

  if (!in)
    throw ReadError();
  return legacy_mode;
}


void
HmmSet::read_legacy_ph(std::ifstream &in)
{
  std::string label;
  int phonemes = 0;
  std::vector<std::vector<HmmTransition> > state_info;
  
  // Second line has the total amount of phonemes
  in >> phonemes;
  
  // Reserve one Hmm for each phoneme
  m_hmms.reserve(phonemes);

  for (int h = 0; h < phonemes; h++) {
    // Read a phone HMM
    int index = 0;
    int states = 0;
    in >> index >> states >> label; 
    if (!in)
      throw ReadError();

    states -= 2; // Remove the dummy states
    Hmm &hmm = add_hmm(label, states);

    // In legacy ph-files the states are tied according to the emission pdfs.
    // If the state with the same emission pdf already exists, the transition
    // information is ignored.
    int state;
    int pdf;
    std::vector<bool> load_transitions;
    in >> state >> state;
    for (int s = 0; s < states; s++)
    {
      in >> pdf;
      if (pdf >= (int)state_info.size())
        state_info.resize(pdf+1); // Reserve more states

      hmm.state(s) = pdf; // Assign the state (same index as with PDFs)
      
      if ((int)state_info[pdf].size() == 0)
      {
        // New state, load transitions
        hmm.state(s) = pdf;
        load_transitions.push_back(true);
      }
      else
      {
        // The state already exists, do not load the transitions
        load_transitions.push_back(false);
      }
    }

    // Read transitions
    for (int s = -2; s < states; s++) {
      int transitions = 0;
      int source = 0;

      in >> source >> transitions;
      source -= 2;

      if (source >= states)
      {
        throw str::fmt(128, "HmmSet::read_legacy_ph: Invalid source state number %i (only %i states)", source, states);
      }

      for (int t = 0; t < transitions; t++) {
	int target;
	double prob;
	in >> target >> prob;

        assert(target > 0);
        assert(prob > 0);

        if (source >= 0 && load_transitions[source])
        {
          if (target == 1)
            target = states - source; // Sink state
          else
          {
            target -= 2;
            if (target > states)
              throw str::fmt(128, "HmmSet::read_legacy_ph: Invalid target state number %i (only %i states)", source, states);
            target -= source; // Make it relative
          }

          state_info[hmm.state(source)].push_back(
            HmmTransition(hmm.state(source), target, prob));
        }
      }

      if (source >= 0 && !load_transitions[source])
      {
        // Check the transitions are valid
        if (source >= 0)
        {
          for (int i = 0; i < (int)state_info[hmm.state(source)].size(); i++)
          {
            if (source+state_info[hmm.state(source)][i].target_offset > states)
              throw str::fmt(128, "HmmSet::read_legacy_ph: Invalid target state number %i on existing state %i (only %i states)", source, hmm.state(source), states);
          }
        }
      }
    }
  }

  // Fill the HmmSet with states and transitions in correct order.
  // The states and PDFs share the same indices and the transitions are
  // numbered sequentially.
  for (int s = 0; s < (int)state_info.size(); s++)
  {
    int index = add_state(s);
    assert( index == s );

    for (int i = 0; i < (int)state_info[s].size(); i++)
    {
      add_transition(s, state_info[s][i].target_offset, state_info[s][i].prob);
    }
  }
}


int
HmmSet::get_state_with_pdf(int pdf_index)
{
  for (int i = 0; i < (int)m_states.size(); i++)
  {
    if (m_states[i].emission_pdf == pdf_index)
      return i;
  }
  return -1;
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

  out << m_emission_pdfs.size() << std::endl;
  
  for (int i = 0; i < (int)m_states.size(); i++) {
    m_emission_pdfs[i].write(out);
  }
}


void
HmmSet::write_ph(const std::string &filename)
{
  write_legacy_ph(filename);
}

void
HmmSet::write_legacy_ph(const std::string &filename)
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

	int target = transition.target_offset + 2 + s;
	if (target == hmm.num_states()+2)
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
  while (!m_valid_pdf_likelihoods.empty()) {
    m_pdf_likelihoods[m_valid_pdf_likelihoods.back()]=-1.0;
    m_valid_pdf_likelihoods.pop_back();
  }
  // Clear also cache for base distributions
  m_pool.reset_cache();
}


double
HmmSet::pdf_likelihood(const int p, const FeatureVec &feature) 
{
  if (m_pdf_likelihoods[p] > 0)
    return m_pdf_likelihoods[p];

  m_pdf_likelihoods[p] = m_emission_pdfs[p].compute_likelihood(feature);
  if (m_pdf_likelihoods[p] < MIN_STATE_PROB)
    m_pdf_likelihoods[p] = MIN_STATE_PROB;
  m_valid_pdf_likelihoods.push_back(p);

  return m_pdf_likelihoods[p];
}


void
HmmSet::precompute_likelihoods(const FeatureVec &f)
{
  // Clear cache
  reset_cache();
  
  // Precompute base distribution likelihoods
  m_pool.precompute_likelihoods(f);

  m_valid_pdf_likelihoods.clear();
  // Precompute state likelihoods
  for (int i = 0; i < num_emission_pdfs(); i++) {
    m_pdf_likelihoods[i] = m_emission_pdfs[i].compute_likelihood(f);
    if (m_pdf_likelihoods[i] < MIN_STATE_PROB)
      m_pdf_likelihoods[i] = MIN_STATE_PROB;
    m_valid_pdf_likelihoods.push_back(i);
  }
}


void
HmmSet::start_accumulating()
{
  m_transition_accum = m_transitions;
  m_accumulated.resize(m_transition_accum.size());
  for (unsigned int i=0; i<m_transition_accum.size(); i++) {
    m_transition_accum[i].prob=0;
    m_accumulated[i]=false;
  }

  for (int i = 0; i < num_emission_pdfs(); i++)
    m_emission_pdfs[i].start_accumulating();
}


void
HmmSet::accumulate_distribution(const FeatureVec &f, int pdf, double gamma, int pos)
{
  m_emission_pdfs[pdf].accumulate(gamma, f, pos);
}


void
HmmSet::accumulate_transition(int transition_index, double prior)
{
  assert(m_transition_accum.size() > 0);

  // Accumulate state transition probabilities
  m_transition_accum[transition_index].prob += prior;
  m_accumulated[transition_index] = true;
}


void
HmmSet::dump_statistics(const std::string base) const
{
  dump_ph_statistics(base+".phs");
  dump_mc_statistics(base+".mcs");
  dump_gk_statistics(base+".gks");
}


void
HmmSet::dump_ph_statistics(const std::string filename) const
{
  if (m_transition_accum.size() > 0) {
    std::ofstream phs(filename.c_str());
    if (!phs) {
      fprintf(stderr, "HmmSet::dump_ph_statistics(): could not open %s\n", filename.c_str());
      throw OpenError();
    }
  
    // Write out every transition as "state relative_target occ_count"
    phs << m_transition_accum.size() << std::endl;
    for (unsigned int t=0; t<m_transition_accum.size(); t++) {
      if (m_accumulated[t]) {
	phs << m_transition_accum[t].source_index << " ";
	phs << m_transition_accum[t].target_offset << " ";
	phs << m_transition_accum[t].prob << std::endl;
      }
    }
    
    if (!phs)
      throw WriteError();  
    phs.close();
  }
}


void
HmmSet::dump_mc_statistics(const std::string filename) const
{
  std::ofstream mcs(filename.c_str());
  if (!mcs) {
    fprintf(stderr, "HmmSet::dump_mc_statistics(): could not open %s\n", filename.c_str());
    throw OpenError();
  }  
  
  mcs << num_emission_pdfs() << std::endl;

  for (int i = 0; i < num_emission_pdfs(); i++) {
    if (m_emission_pdfs[i].accumulated(0)) {
      mcs << i << " num ";
      m_emission_pdfs[i].dump_statistics(mcs, 0);
      mcs << std::endl;
    }

    if (m_emission_pdfs[i].estimation_mode() == PDF::MMI) {
      if (m_emission_pdfs[i].accumulated(1)) {
	mcs << i << " den ";
	m_emission_pdfs[i].dump_statistics(mcs, 1);
	mcs << std::endl;
      }
    }
  }
  
  if (!mcs)
    throw WriteError();  
  mcs.close();
}


void
HmmSet::dump_gk_statistics(const std::string filename) const
{
  std::ofstream gks(filename.c_str());
  if (!gks) {
    fprintf(stderr, "HmmSet::dump_gk_statistics(): could not open %s\n", filename.c_str());
    throw OpenError();
  }  

  gks << m_pool.size() << " " << m_pool.dim() << std::endl;
  for (int g=0; g<m_pool.size(); g++) {
    if (m_pool.get_pdf(g)->accumulated(0)) {
      gks << g << " num ";
      m_pool.get_pdf(g)->dump_statistics(gks, 0);
      gks << std::endl;
    }

    if (m_pool.get_pdf(g)->estimation_mode() == PDF::MMI) {
      if (m_pool.get_pdf(g)->accumulated(1)) {
	gks << g << " den ";
	m_pool.get_pdf(g)->dump_statistics(gks, 1);
	gks << std::endl;
      }
    }
  }
  
  if (!gks)
    throw WriteError();  
  gks.close();
}


void
HmmSet::accumulate_from_dump(const std::string base)
{
  assert(m_transition_accum.size() > 0);

  accumulate_ph_from_dump(base+".phs");
  accumulate_mc_from_dump(base+".mcs");
  accumulate_gk_from_dump(base+".gks");
}


void
HmmSet::accumulate_ph_from_dump(const std::string filename)
{
  std::ifstream phs(filename.c_str());
  if (!phs) {
    fprintf(stderr, "HmmSet::accumulate_ph_from_dump(): could not open %s\n", filename.c_str());
    throw OpenError();
  }
  
  unsigned int num_transitions;
  phs >> num_transitions;
  if (m_transition_accum.size() != num_transitions)
    throw std::string("HmmSet::accumulate_ph_from_dump: the number of transitions in: %s doesn't match the earlier accumulations\n", filename.c_str());
  
  int source, target, pos; double occ;
  for (unsigned int t=0; t<num_transitions; t++) {
    phs >> source >> target >> occ;
    pos=-1;
    // Find this transition in m_transition_accum
    for (unsigned int tsearch=0; tsearch<m_transition_accum.size(); tsearch++)
    {
      if (m_transition_accum[tsearch].source_index == source &&
          m_transition_accum[tsearch].target_offset == target)
      {
	pos=tsearch;
	break;
      }
    }
    if (pos==-1)
      throw str::fmt(128, "HmmSet::accumulate_ph_from_dump: the transition %i could not be accumulated", t);
    
    // Accumulate the statistics
    m_transition_accum[pos].prob += occ;
  }
  
  phs.close();
}


void
HmmSet::accumulate_mc_from_dump(const std::string filename)
{
  std::ifstream mcs(filename.c_str());
  if (!mcs) {
    fprintf(stderr, "HmmSet::accumulate_mc_from_dump(): could not open %s\n", filename.c_str());
    throw OpenError();
  }

  int n, pdf;
  mcs >> n;

  if (n != num_emission_pdfs())
    throw str::fmt(128, "HmmSet::accumulate_mc_from_dump: the number of PDFs in: %s is wrong\n", filename.c_str());

  while(mcs >> pdf)
    m_emission_pdfs[pdf].accumulate_from_dump(mcs);
  
  mcs.close();
}


void
HmmSet::accumulate_gk_from_dump(const std::string filename)
{
  std::ifstream gks(filename.c_str());
  if (!gks) {
    fprintf(stderr, "HmmSet::accumulate_gk_from_dump(): could not open %s\n", filename.c_str());
    throw OpenError();
  }

  int num_pdfs, dim, pdf;
  gks >> num_pdfs >> dim;
  if (num_pdfs != m_pool.size())
    throw std::string("HmmSet::accumulate_gk_from_dump: the number of mixture base distributions in: %s is wrong\n", filename.c_str());

  if (dim != m_pool.dim())
    throw std::string("HmmSet::accumulate_gk_from_dump: the dimensionality of mixture base distributions in: %s is wrong\n", filename.c_str());

  while(gks >> pdf)
    m_pool.get_pdf(pdf)->accumulate_from_dump(gks);
  
  gks.close();
}
 

void 
HmmSet::stop_accumulating()
{
  assert(m_transition_accum.size() > 0);

  // Stop accumulation also for state distributions
  for (int i=0; i<num_emission_pdfs(); i++)
    m_emission_pdfs[i].stop_accumulating();
 
  // Clear accumulator
  m_transition_accum.clear();
  m_accumulated.clear();
}


void
HmmSet::estimate_parameters()
{
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

    m_emission_pdfs[state(s).emission_pdf].estimate_parameters();
  }
}


void
HmmSet::set_estimation_mode(PDF::EstimationMode mode)
{
  m_mode = mode;
  
  for (int i=0; i<m_pool.size(); i++)
    m_pool.get_pdf(i)->set_estimation_mode(m_mode);
  
  for (int i=0; i<num_emission_pdfs(); i++)
    m_emission_pdfs[i].set_estimation_mode(m_mode);
}
  

PDF::EstimationMode
HmmSet::get_estimation_mode()
{
  return m_mode;
}


void
HmmSet::set_minvar(double minvar)
{
  for (int i=0; i<m_pool.size(); i++) {
    Gaussian *g = (Gaussian*)m_pool.get_pdf(i);
    g->set_minvar(minvar);
  }
}
