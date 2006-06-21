#include <assert.h>
#include <fstream>
#include <math.h>
#include <iostream>
#include <values.h>

#include "HmmSet.hh"
#include "util.hh"
#include "mtl/lu.h"

#define MIN_STATE_PROB 1e-20
#define MIN_KERNEL_POS_PROB 1e-30

void
HmmCovariance::reset(int value)
{
  if (m_cov_type == FULL) {
    mtl::set(full, 0);
    mtl::set(full_inv_cholesky_transpose, 0);
  }
  else
    std::fill(data.begin(), data.end(), value);
}

void
HmmCovariance::resize(int dim, Type type)
{
  m_cov_type = type;
  if (m_cov_type == SINGLE)
    data.resize(1);
  else if (m_cov_type == DIAGONAL)
    data.resize(dim);
  else if (m_cov_type == FULL) {
    full.resize(dim, dim);
    full_inv_cholesky_transpose.resize(dim,dim);
  }
  else
    throw std::string("Unknown covariance type");
  reset();
}


void
HmmKernel::resize(int dim, HmmCovariance::Type cov_type)
{
  center.resize(dim);
  cov.resize(dim, cov_type);
}

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

  m_kernels.resize(hmm_set.m_kernels.size());
  for (unsigned int i=0; i<m_kernels.size(); i++) {
    m_kernels[i].center = hmm_set.m_kernels[i].center;
    m_kernels[i].cov.resize(m_dim, hmm_set.m_kernels[i].cov.type());
    m_kernels[i].cov.data = hmm_set.m_kernels[i].cov.data;
    m_kernels[i].cov.cov_det = hmm_set.m_kernels[i].cov.cov_det;
    if (m_cov_type == HmmCovariance::FULL) {
      m_kernels[i].cov.full.resize(m_dim, m_dim);
      m_kernels[i].cov.full_inv_cholesky_transpose.resize(m_dim, m_dim);
      mtl::copy(hmm_set.m_kernels[i].cov.full, m_kernels[i].cov.full);
      mtl::copy(hmm_set.m_kernels[i].cov.full_inv_cholesky_transpose, m_kernels[i].cov.full_inv_cholesky_transpose);
    }
  }
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
HmmSet::set_covariance_type(HmmCovariance::Type type)
{
  m_cov_type = type;
}

void
HmmSet::set_dim(int dim)
{
  m_dim = dim;
}

void
HmmSet::reserve_states(int states)
{
  m_states.resize(states);
}

void
HmmSet::reserve_kernels(int kernels)
{
  m_kernels.resize(kernels);
}

// ASSUMES:
// - dimension and cov_type are set already
HmmKernel&
HmmSet::add_kernel()
{
  HmmKernel dummy;
  m_kernels.push_back(dummy);
  HmmKernel &kernel = m_kernels[m_kernels.size()-1];
  kernel.resize(m_dim, m_cov_type);
  return kernel;
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
HmmSet::read_gk(const std::string &filename)
{
  std::ifstream in(filename.c_str());
  if (!in) {
    fprintf(stderr, "HmmSet::read_gk(): could not open %s\n", 
	    filename.c_str());
    throw OpenError();
  }
  
  int kernels = 0;
  std::string cov_str;

  in >> kernels >> m_dim >> cov_str;
  if (cov_str == "single_cov")
    m_cov_type = HmmCovariance::SINGLE;
  else if (cov_str == "diagonal_cov")
    m_cov_type = HmmCovariance::DIAGONAL;
  else if (cov_str == "full_cov")
    m_cov_type = HmmCovariance::FULL;
 else
    throw std::string("Unknown covariance type");

  assert(m_cov_type != HmmCovariance::INVALID);

  reserve_kernels(kernels);
  for (int k = 0; k < kernels; k++) {
    HmmKernel &kernel = m_kernels[k];

    // Read center
    kernel.resize(m_dim, m_cov_type);
    for (int d = 0; d < m_dim; d++)
      in >> kernel.center[d];

    // Read covariance
    if (m_cov_type == HmmCovariance::SINGLE) {
      in >> kernel.cov.var();
    } 
    else if (m_cov_type == HmmCovariance::DIAGONAL) {
      for (int d = 0; d < m_dim; d++)
	in >> kernel.cov.diag(d);
    } 
    else if (m_cov_type == HmmCovariance::FULL) {
      for (int i=0; i<m_dim; i++)
	for (int j=0; j<m_dim; j++) {
	  float f = 0;
	  in >> f;
	  kernel.cov.full(i,j) = f;
	  kernel.cov.full(j,i) = f;
	}
    }
    else {
      throw std::string("Unknown covariance type");
    }
  }

  // Covariances have changed, compute the determinants
  compute_covariance_determinants();

  if (!in)
    throw ReadError();
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
HmmSet::write_gk(const std::string &filename)
{
  std::ofstream out(filename.c_str());

  out << m_kernels.size() << " " << m_dim << " ";
  if (m_cov_type == HmmCovariance::SINGLE)
    out << "single_cov";
  else if (m_cov_type == HmmCovariance::DIAGONAL)
    out << "diagonal_cov";
  else if (m_cov_type == HmmCovariance::FULL)
    out << "full_cov";
  else
    throw std::string("Unknown covariance type");
  out << std::endl;

  // Write kernels
  for (int k = 0; k < (int)m_kernels.size(); k++) {
    HmmKernel &kernel = m_kernels[k];

    // Write centers
    for (int d = 0; d < m_dim; d++) {
      if (d > 0)
	out << " ";
      out << kernel.center[d];
    }

    // Write covariance
    if (m_cov_type == HmmCovariance::SINGLE) {
      out << " " << kernel.cov.var() << std::endl;
    } 
    else if (m_cov_type == HmmCovariance::DIAGONAL) {
      for (int d = 0; d < m_dim; d++)
	out << " " << kernel.cov.diag(d);
      out << std::endl;
    }
    else if (m_cov_type == HmmCovariance::FULL) {
      for (int i=0; i<m_dim; i++)
	for (int j=0; j<m_dim; j++)
	  out << " " << kernel.cov.full(i,j);
      out << std::endl;
    }
  }
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
HmmSet::compute_covariance_determinants(void)
{
  double det;
  int d;
  
  for (int k = 0; k < num_kernels(); k++) {
    HmmKernel &kernel = m_kernels[k];

    if (kernel.cov.type() == HmmCovariance::DIAGONAL)
    {
      det = 1;
      for (d = 0; d < m_dim; d++)
        det *= kernel.cov.diag(d);
      kernel.cov.cov_det = det;
    }

    else if (kernel.cov.type() == HmmCovariance::FULL)
      {
      	Matrix t(m_dim, m_dim);
	Matrix precision(m_dim, m_dim);
	Matrix precision_cholesky(m_dim, m_dim);
	mtl::copy(kernel.cov.full, t);
	mtl::dense1D<int> pivots(m_dim, 0);
	mtl::lu_factor(t, pivots);
	mtl::lu_inverse(t, pivots, precision);
	cholesky_factor(precision, precision_cholesky);
	mtl::transpose(precision_cholesky, kernel.cov.full_inv_cholesky_transpose);
	
	det=1;
	for (d=0; d<m_dim; d++)
	  // Sqrt of the determinant of the inverse
	  det *= precision_cholesky(d,d);
	kernel.cov.cov_det = det;
	if (det <= 0) {
	  print_all_matrix(kernel.cov.full);
	  assert(det>0);
	}
	// Clean up, only cholesky factors for the precision are needed
	kernel.cov.full.resize(0,0);
      }
  }
}


void
HmmSet::compute_observation_log_probs(const FeatureVec &feature)
{
  double sum = 0;

  obs_kernel_likelihoods.resize(num_kernels());
  for (int k = 0; k < num_kernels(); k++) {
    obs_kernel_likelihoods[k] = compute_kernel_likelihood(k, feature);
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
    if (m_kernel_likelihoods[k] < 0)
    {
      m_kernel_likelihoods[k] = compute_kernel_likelihood(k, feature);
      m_valid_kernel_likelihoods.push_back(k);
    }
    temp += state.weights[w].weight * m_kernel_likelihoods[k];
  }
  if (temp < MIN_STATE_PROB)
    m_state_probs[s] = MIN_STATE_PROB;
  else
    m_state_probs[s] = temp;
  m_valid_stateprobs.push_back(s);
  return(m_state_probs[s]);
}

float 
HmmSet::compute_kernel_likelihood(const int k, const FeatureVec &feature) 
{
  float result;
  double dist = 0.0;
  double dif;
  Vector s(m_dim);
  Vector t(m_dim);
  double temp;
  
  HmmKernel &kernel = m_kernels[k];
  
  switch (kernel.cov.type())
  {
  case HmmCovariance::SINGLE:
    for (int i = 0; i < m_dim; i++)
    {
      dif = feature[i] - kernel.center[i];
      dist += dif*dif;
    }
    result = expf(-0.5 * dist) / kernel.cov.var();
    break;

  case HmmCovariance::DIAGONAL:
    for (int i = 0; i < m_dim; i++)
    {
      dif = feature[i] - kernel.center[i];
      dist += dif*dif/(double)kernel.cov.diag(i);
    }
    result = exp(-0.5 * dist) / sqrt(kernel.cov.cov_det);
    break;

  case HmmCovariance::FULL:
    for (int i = 0; i < m_dim; i++)
      s[i] = feature[i]-kernel.center[i];

    for (int i=0; i<m_dim; i++)
      for (int j=i; j<m_dim; j++)
	t[i] += kernel.cov.full_inv_cholesky_transpose(i,j)*s[j];

    temp = mtl::dot(t, t);
    result = kernel.cov.cov_det * exp(-0.5 * temp);
    break;  

  default:
    throw std::string("Unknown covariance type");
  }

  if (result < MIN_KERNEL_POS_PROB)
    result = 0;

  return(result);
}



void cholesky_factor(const Matrix &A, Matrix &B)
{
  assert(A.nrows() == A.ncols());
  assert(B.nrows() == B.ncols());
  assert(A.nrows() == B.nrows());
  mtl::copy(A, B);
  
  for (unsigned int j=0; j<B.nrows(); j++) 
    {
      for (unsigned int k=0; k<j; k++)
	for (unsigned int i=j; i<B.nrows(); i++)
	  B(i,j) = B(i,j)-B(i,k)*B(j,k);
      B(j,j) = sqrt(B(j,j));
      for (unsigned int k=j+1; k<B.nrows(); k++)
	B(k,j) = B(k,j)/B(j,j);
    }
  
  for (unsigned int i=0; i<B.nrows(); i++)
    for (unsigned int j=i+1; j<B.ncols(); j++)
      B(i,j) = 0;
}
