#include <cassert>

#include "Distributions.hh"
#include "str.hh"

#include "blaspp.h"
#include "blas1pp.h"
#include "blas2pp.h"
#include "blas3pp.h"



int
PDF::accumulator_position(std::string stats_type)
{
  if (stats_type == "num")
    return 0;
  else if (stats_type == "den")
    return 1;
  else return -1;
}


void
Gaussian::split(Gaussian &g1, Gaussian &g2) const
{
  assert(dim() != 0);

  Vector mean1; get_mean(mean1);
  Vector mean2; get_mean(mean2);
  Matrix cov; get_covariance(cov);

  // Add/Subtract standard deviations
  // FIXME: should we use eigvals/vecs for full covs?
  double sd=0;
  for (int i=0; i<dim(); i++) {
    sd=0.00001*sqrt(cov(i,i));
    mean1(i) -= sd;
    mean2(i) += sd;
  }
  
  g1.set_mean(mean1);
  g2.set_mean(mean2);
  g1.set_covariance(cov);
  g2.set_covariance(cov);
}


// Merge using:
// f1=n1/(n1+n2) ; f2=n2/(n1+n2)
// mu=f1*mu1+f2*mu2
// sigma=f1*sigma1+f2*sigma2+f1*f2*(mu1-mu2)^2
void
Gaussian::merge(double weight1, const Gaussian &m1, 
		double weight2, const Gaussian &m2)
{
  assert(m1.dim() == m2.dim());
  assert(weight1 <1 && weight2 <1 && weight1>0 && weight2>0);
  
  reset(m1.dim());

  double f1=weight1/(weight1+weight2);
  double f2=weight2/(weight1+weight2);

  // Merged mean
  Vector mu; m1.get_mean(mu);
  Vector vtemp; m2.get_mean(vtemp);
  Blas_Scale(weight1, mu);
  Blas_Add_Mult(mu, weight2, vtemp);
  set_mean(mu);

  // Merged covariance
  Matrix covariance; m1.get_covariance(covariance);
  Matrix mtemp; m2.get_covariance(mtemp);
  Blas_Scale(weight1, covariance);
  Blas_Scale(weight2, mtemp);
  // FIXME: is this ok?
  covariance = covariance + mtemp;
  Vector diff; m1.get_mean(diff);
  Blas_Add_Mult(diff, -1, vtemp);
  Blas_R1_Update(covariance, diff, diff, f1*f2);
  set_covariance(covariance);
}


double
Gaussian::kullback_leibler(Gaussian &g) const
{
  assert(dim()==g.dim());
  
  LaVectorLongInt pivots(dim(),1);
  LaGenMatDouble t1; g.get_covariance(t1);
  LaGenMatDouble t2; g.get_covariance(t2);
  LaVectorDouble t3; g.get_mean(t3);
  LaVectorDouble t4; g.get_mean(t4);
  LUFactorizeIP(t1, pivots);
  LaLUInverseIP(t1, pivots);
    
  LaGenMatDouble cov; get_covariance(cov);
  LaVectorDouble mean; get_mean(mean);
  Blas_Mat_Mat_Mult(t1, cov, t2, 1.0, 0.0);
  Blas_Add_Mult(t3, -1, mean); 
  Blas_Mat_Vec_Mult(t1, t3, t4);
  
  double value=LinearAlgebra::determinant(t1)
    /LinearAlgebra::determinant(cov);
  //  FIXME: logarithm ok?
  value = log2(value);
  value += t2.trace();
  value += Blas_Dot_Prod(t3, t4);
  value -= dim();
  value *= 0.5;

  return value;
}


DiagonalGaussian::DiagonalGaussian(int dim)
{
  reset(dim);
}


DiagonalGaussian::DiagonalGaussian(const DiagonalGaussian &g)
{
  reset(g.dim());
  g.get_mean(m_mean);
  g.get_covariance(m_covariance);
  for (int i=0; i<dim(); i++)
    m_precision(i)=1/m_covariance(i);
}


DiagonalGaussian::~DiagonalGaussian()
{
  stop_accumulating();
}


void
DiagonalGaussian::reset(int dim)
{
  m_mean.resize(dim);
  m_covariance.resize(dim);
  m_precision.resize(dim);
  m_mean=0; m_covariance=0; m_precision=0;
  m_dim=dim;
}


double
DiagonalGaussian::compute_likelihood(const FeatureVec &f) const
{
  return exp(compute_log_likelihood(f));
}


double
DiagonalGaussian::compute_log_likelihood(const FeatureVec &f) const
{
  double ll=0;
  // FIXME!
  //  Vector diff(f);
  Vector diff(f.dim());
  for (int i=0; i<dim(); i++)
    diff(i)=f[i];
  Blas_Add_Mult(diff, -1, m_mean);
  for (int i=0; i<dim(); i++)
    ll += diff(i)*diff(i)*m_precision(i);
  ll *= -0.5;
  ll += m_constant;

  return ll;
}


void
DiagonalGaussian::write(std::ostream &os) const
{
  os << "diag ";
  for (int i=0; i<dim(); i++)
    os << m_mean(i) << " ";
  for (int i=0; i<dim()-1; i++)
    os << m_covariance(i) << " ";
  os << m_covariance(dim()-1);
}


void
DiagonalGaussian::read(std::istream &is)
{
  for (int i=0; i<dim(); i++)
    is >> m_mean(i);
  for (int i=0; i<dim(); i++) {
    is >> m_covariance(i);
    m_precision(i) = 1/m_covariance(i);
  }
  m_constant=1;
  for (int i=0; i<dim(); i++)
    m_constant *= m_covariance(i);
  m_constant = 1/sqrt(m_constant);
  m_constant = log(m_constant);
}


void
DiagonalGaussian::start_accumulating()
{
  if (m_mode == ML) {
    m_accums.resize(1);
    m_accums[0] = new DiagonalAccumulator(dim());
  }
  else if (m_mode == MMI) {
    m_accums.resize(2);
    m_accums[0] = new DiagonalAccumulator(dim());
    m_accums[1] = new DiagonalAccumulator(dim());
  }
}


void
DiagonalGaussian::accumulate(double prior,
			     const FeatureVec &f,
			     int accum_pos)
{
  assert((int)m_accums.size() > accum_pos);
  assert(m_accums[accum_pos] != NULL);

  m_accums[accum_pos]->accumulated = true;
  m_accums[accum_pos]->gamma += prior;
  for (int d=0; d<dim(); d++) {
    m_accums[accum_pos]->mean(d) += prior*f[d];
    m_accums[accum_pos]->cov(d) += prior*f[d]*f[d];
  }
}


void 
DiagonalGaussian::dump_statistics(std::ostream &os,
				  int accum_pos) const
{
  assert((int)m_accums.size() > accum_pos);
  assert(m_accums[accum_pos] != NULL);  

  if (m_accums[accum_pos]->accumulated) {
    os << m_accums[accum_pos]->gamma << " ";
    for (int i=0; i<dim(); i++)
      os << m_accums[accum_pos]->mean(i) << " ";
    for (int i=0; i<dim()-1; i++)
      os << m_accums[accum_pos]->cov(i) << " ";
    os << m_accums[accum_pos]->cov(dim()-1);
  }
}

  
void 
DiagonalGaussian::accumulate_from_dump(std::istream &is)
{
  std::string type;
  is >> type;
  int accum_pos = accumulator_position(type);

  assert((int)m_accums.size() > accum_pos);
  assert(m_accums[accum_pos] != NULL);

  double gamma;
  Vector mean(dim());
  Vector cov(dim());

  is >> gamma;
  for (int i=0; i<dim(); i++)
    is >> mean(i);
  for (int i=0; i<dim(); i++)
    is >> cov(i);

  m_accums[accum_pos]->gamma += gamma;
  for (int i=0; i<dim(); i++) {
    m_accums[accum_pos]->mean(i) += gamma*mean(i);
    m_accums[accum_pos]->cov(i) += gamma*cov(i);
  }

  m_accums[accum_pos]->accumulated = true;
}


void
DiagonalGaussian::stop_accumulating()
{
  if (m_accums.size() > 0)
    for (unsigned int i=0; i<m_accums.size(); i++)
      if (m_accums[i] != NULL)
	delete m_accums[i];
}


bool
DiagonalGaussian::accumulated(int accum_pos) const
{
  if ((int)m_accums.size() <= accum_pos)
    return false;
  if (m_accums[accum_pos] == NULL)
    return false;
  return m_accums[accum_pos]->accumulated;
}


void
DiagonalGaussian::estimate_parameters()
{
  if (m_mode == ML) {
    m_mean.copy(m_accums[0]->mean);
    m_covariance.copy(m_accums[0]->cov);
    Blas_Scale(m_accums[0]->gamma, m_mean);
    Blas_Scale(m_accums[0]->gamma, m_covariance);
  }

  // FIXME: do MMI
  if (m_mode == MMI) {
    m_mean.copy(m_accums[0]->mean);
    m_covariance.copy(m_accums[0]->cov);
    Blas_Scale(m_accums[0]->gamma, m_mean);
    Blas_Scale(m_accums[0]->gamma, m_covariance);
  }
}


void
DiagonalGaussian::get_mean(Vector &mean) const
{
  mean.copy(m_mean);
}


void
DiagonalGaussian::get_covariance(Matrix &covariance) const
{
  covariance.resize(dim(),dim());
  covariance=0;
  for (int i=0; i<dim(); i++)
    covariance(i,i)=m_covariance(i);
}


void
DiagonalGaussian::set_mean(const Vector &mean)
{
  assert(mean.size()==dim());
  m_mean.copy(mean);
}


void
DiagonalGaussian::set_covariance(const Matrix &covariance)
{
  assert(covariance.rows()==dim());
  assert(covariance.cols()==dim());
  m_covariance.copy(covariance);
}


void
DiagonalGaussian::get_covariance(Vector &covariance) const
{
  covariance.copy(m_covariance);
}


void
DiagonalGaussian::set_covariance(const Vector &covariance)
{
  assert(covariance.size()==dim());
  m_covariance.copy(covariance);
}


FullCovarianceGaussian::FullCovarianceGaussian(int dim)
{
  reset(dim);
}


FullCovarianceGaussian::FullCovarianceGaussian(const FullCovarianceGaussian &g)
{
  reset(g.dim());
  g.get_mean(m_mean);
  g.get_covariance(m_covariance);
  LinearAlgebra::inverse(m_covariance, m_precision);
}


FullCovarianceGaussian::~FullCovarianceGaussian()
{
  stop_accumulating();
}


void
FullCovarianceGaussian::reset(int dim)
{
  m_mean.resize(dim);
  m_covariance.resize(dim,dim);
  m_precision.resize(dim,dim);
  m_mean=0; m_covariance=0; m_precision=0;
}


double
FullCovarianceGaussian::compute_likelihood(const FeatureVec &f) const
{
  return exp(compute_log_likelihood(f));
}


double
FullCovarianceGaussian::compute_log_likelihood(const FeatureVec &f) const
{
  double ll=0;

  // FIXME
  //  Vector diff(f);
  Vector diff(f.dim());
  Vector t(f.dim());

  Blas_Add_Mult(diff, -1, m_mean);
  Blas_Mat_Vec_Mult(m_precision, diff, t, 1, 0);
  ll = Blas_Dot_Prod(diff, t);
  ll *= -0.5;
  ll += m_constant;

  return ll;
}


void
FullCovarianceGaussian::write(std::ostream &os) const
{
  os << "full ";
  for (int i=0; i<dim(); i++)
    os << m_mean(i) << " ";
  for (int i=0; i<dim(); i++)
    for (int j=0; j<dim(); j++)
      if (!(i==dim()-1 && j==dim()-1))
	os << m_covariance(i,j) << " ";
  os << m_covariance(dim()-1, dim()-1);
}


void
FullCovarianceGaussian::read(std::istream &is)
{
  std::string type;
  is >> type;
  if (type != "full")
    throw std::string("Error reading FullCovarianceGaussian parameters from a stream\n");
  
  std::string line;
  getline(is, line);
  str::chomp(&line);
  
  for (int i=0; i<dim(); i++)
    is >> m_mean(i);
  for (int i=0; i<dim(); i++)
    for (int j=0; j<dim(); j++)
      is >> m_covariance(i,j);
}


void
FullCovarianceGaussian::start_accumulating()
{
  if (m_mode == ML) {
    m_accums.resize(1);
    m_accums[0] = new FullCovarianceAccumulator(dim());
  }
  else if (m_mode == MMI) {
    m_accums.resize(2);
    m_accums[0] = new FullCovarianceAccumulator(dim());
    m_accums[1] = new FullCovarianceAccumulator(dim());
  }
}


void
FullCovarianceGaussian::accumulate(double prior,
				   const FeatureVec &f,
				   int accum_pos)
{
  assert((int)m_accums.size() > accum_pos);
  assert(m_accums[accum_pos] != NULL);

  m_accums[accum_pos]->accumulated = true;  
  m_accums[accum_pos]->gamma += prior;
  for (int i=0; i<dim(); i++) {
    m_accums[accum_pos]->mean(i) += prior*f[i];
    m_accums[accum_pos]->cov(i,i) += prior*f[i]*f[i];
  }
}


bool
FullCovarianceGaussian::accumulated(int accum_pos) const
{
  if ((int)m_accums.size() <= accum_pos)
    return false;
  if (m_accums[accum_pos] == NULL)
    return false;
  return m_accums[accum_pos]->accumulated;
}


void 
FullCovarianceGaussian::dump_statistics(std::ostream &os,
					int accum_pos) const
{
}


void 
FullCovarianceGaussian::accumulate_from_dump(std::istream &is)
{
}


void
FullCovarianceGaussian::stop_accumulating()
{
  if (m_accums.size() > 0)
    for (unsigned int i=0; i<m_accums.size(); i++)
      if (m_accums[i] != NULL)
	delete m_accums[i];
}


void
FullCovarianceGaussian::estimate_parameters()
{
  if (m_mode == ML) {
    m_mean.copy(m_accums[0]->mean);
    m_covariance.copy(m_accums[0]->cov);
    Blas_Scale(m_accums[0]->gamma, m_mean);
    Blas_Scale(m_accums[0]->gamma, m_covariance);
  }

  // FIXME!
  else if (m_mode == MMI) {
    m_mean.copy(m_accums[0]->mean);
    m_covariance.copy(m_accums[0]->cov);
    Blas_Scale(m_accums[0]->gamma, m_mean);
    Blas_Scale(m_accums[0]->gamma, m_covariance);
  }
}


void
FullCovarianceGaussian::get_mean(Vector &mean) const
{
  mean.copy(m_mean);
}


void
FullCovarianceGaussian::get_covariance(Matrix &covariance) const
{
  covariance.copy(m_covariance);
}


void
FullCovarianceGaussian::set_mean(const Vector &mean)
{
  assert(mean.size()==dim());
  m_mean.copy(mean);
}


void
FullCovarianceGaussian::set_covariance(const Matrix &covariance)
{
  assert(covariance.rows()==dim());
  assert(covariance.cols()==dim());
  m_covariance.copy(covariance);
}




Mixture::Mixture()
{
}


Mixture::Mixture(PDFPool *pool)
{
  m_pool = pool;
}


Mixture::~Mixture()
{
}


void
Mixture::reset()
{
  m_weights.resize(0);
  m_pointers.resize(0);
}


void
Mixture::set_pool(PDFPool *pool)
{
  m_pool = pool;
}


void
Mixture::set_components(const std::vector<int> &pointers,
			const std::vector<double> &weights)
{
  assert(pointers.size()==weights.size());

  m_pointers.resize(pointers.size());
  m_weights.resize(weights.size());

  for (unsigned int i=0; i<pointers.size(); i++) {
    m_pointers[i] = pointers[i];
    m_weights[i] = weights[i];
  }
}


PDF&
Mixture::get_basis_pdf(int index)
{
  assert(index < size());
  return m_pool->get_pdf(m_pointers[index]);
}

void
Mixture::get_components(std::vector<int> &pointers,
			std::vector<double> &weights)
{
  pointers.resize(m_pointers.size());
  weights.resize(m_weights.size());
  
  for (unsigned int i=0; i<m_pointers.size(); i++) {
    pointers[i] = m_pointers[i];
    weights[i] = m_weights[i];
  }
}


void
Mixture::add_component(int pool_index,
		       double weight)
{
  assert(m_weights.size() == m_pointers.size());
  m_pointers.push_back(pool_index);
  m_weights.push_back(weight);
}


void
Mixture::normalize_weights()
{
  double sum=0;
  for (unsigned int i=0; i<m_weights.size(); i++)
    sum += m_weights[i];
  for (unsigned int i=0; i<m_weights.size(); i++)
    m_weights[i] /= sum;
}


double
Mixture::compute_likelihood(const FeatureVec &f) const
{
  double l = 0;
  for (unsigned int i=0; i< m_pointers.size(); i++) {
    l += m_weights[i]*m_pool->compute_likelihood(f, m_pointers[i]);
  }
  return l;
}


double
Mixture::compute_log_likelihood(const FeatureVec &f) const
{
  double ll = 0;
  for (unsigned int i=0; i< m_pointers.size(); i++) {
    ll += m_weights[i]*m_pool->compute_likelihood(f, m_pointers[i]);
  }
  return util::safe_log(ll);

}


void
Mixture::start_accumulating()
{
  if (m_mode == ML) {
    m_accums.resize(1);
    m_accums[ML] = new MixtureAccumulator(dim());
  }
  else if (m_mode == MMI) {
    m_accums.resize(2);
    m_accums[ML] = new MixtureAccumulator(dim());
    m_accums[MMI] = new MixtureAccumulator(dim());
  }
  
  for (int i=0; i<size(); i++)
    m_pool->get_pdf(m_pointers[i]).start_accumulating();
}


void
Mixture::accumulate(double prior,
		    const FeatureVec &f,
		    int accum_pos)
{
  double total_likelihood, this_likelihood;

  // Compute the total likelihood for this mixture
  total_likelihood = compute_likelihood(f);
  
  // Accumulate all basis distributions with some priors
  for (int i=0; i<size(); i++) {
    this_likelihood = 
      prior * m_weights[i] * m_pool->compute_likelihood(f, m_pointers[i])/total_likelihood;
    m_accums[accum_pos]->gamma[i] += this_likelihood;
    get_basis_pdf(i).accumulate(this_likelihood, f, accum_pos);
  }
  m_accums[accum_pos]->accumulated = true;
}


bool
Mixture::accumulated(int accum_pos) const
{
  if ((int)m_accums.size() <= accum_pos)
    return false;
  if (m_accums[accum_pos] == NULL)
    return false;
  return m_accums[accum_pos]->accumulated;
}


void 
Mixture::dump_statistics(std::ostream &os,
			 int accum_pos) const
{
  assert((int)m_accums.size() > accum_pos);
  assert(m_accums[accum_pos] != NULL);

  if (m_accums[accum_pos]->accumulated) {
    os << size() << " ";
    for (int i=0; i<size()-1; i++) {
      os << m_pointers[i] << " " << m_accums[accum_pos]->gamma[i] << " ";
      os << m_pointers[size()-1] << " " << m_accums[accum_pos]->gamma[size()-1];
    }
  }
}


void 
Mixture::accumulate_from_dump(std::istream &is)
{
  std::string type;
  is >> type;
  int accum_pos = accumulator_position(type);

  assert((int)m_accums.size() > accum_pos);
  assert(m_accums[accum_pos] != NULL);

  int pointer, sz;
  double acc;
  is >> sz;
  assert(sz==size());

  for (int i=0; i<size(); i++) {
    is >> pointer >> acc;    
    assert(m_pointers[i] == pointer);
    m_accums[accum_pos]->gamma[i] += acc;
  }

  m_accums[accum_pos]->accumulated = true;
}


void
Mixture::stop_accumulating()
{
  if (m_accums.size() > 0)
    for (unsigned int i=0; i<m_accums.size(); i++)
      if (m_accums[i] != NULL)
	delete m_accums[i];
  
  for (int i=0; i<size(); i++)
    get_basis_pdf(i).stop_accumulating();
}


void
Mixture::estimate_parameters()
{
  if (m_mode == ML) {    
    double total_gamma = 0;
    for (int i=0; i<size(); i++)
      total_gamma += m_accums[0]->gamma[i];
    
    for (int i=0; i<size(); i++) {
      m_weights[i] = m_accums[0]->gamma[i]/total_gamma;
      get_basis_pdf(i).estimate_parameters();
    }
  }
  

  // FIXME
  else if (m_mode == MMI) {
    double total_gamma = 0;
    for (int i=0; i<size(); i++)
      total_gamma += m_accums[0]->gamma[i];
    
    for (int i=0; i<size(); i++) {
      m_weights[i] = m_accums[0]->gamma[i]/total_gamma;
      get_basis_pdf(i).estimate_parameters();
    }
  }
}


void
Mixture::write(std::ostream &os) const
{
  os << m_pointers.size();
  for (unsigned int w = 0; w < m_pointers.size(); w++) {
    os << " " << m_pointers[w]
	<< " " << m_weights[w];
  }
  os << std::endl;
}


void
Mixture::read(std::istream &is)
{
  int num_weights = 0;
  is >> num_weights;
  
  int index;
  double weight;
  for (int w = 0; w < num_weights; w++) {
    is >> index >> weight;
    add_component(index, weight);
  }
  normalize_weights();
}


PDFPool::~PDFPool() {
  for (unsigned int i=0; i<m_pool.size(); i++)
    delete m_pool[i];
}


void
PDFPool::reset() {
  m_dim=0;
  m_pool.clear();
  m_likelihoods.clear();
}


PDF&
PDFPool::get_pdf(int index) const
{
  return *m_pool[index];
}


void
PDFPool::set_pdf(int pdfindex, PDF *pdf)
{  
}


void
PDFPool::reset_cache()
{
  while (!m_valid_likelihoods.empty()) {
    m_likelihoods[m_valid_likelihoods.back()]=-1.0;
    m_valid_likelihoods.pop_back();
  }
}


double
PDFPool::compute_likelihood(const FeatureVec &f, int index)
{
  if (m_likelihoods[index] > 0)
    return m_likelihoods[index];
  m_likelihoods[index] = m_pool[index]->compute_likelihood(f);
  m_valid_likelihoods.push_back(index);
  return m_likelihoods[index];
}


void
PDFPool::precompute_likelihoods(const FeatureVec &f)
{
  for (int i=0; i<size(); i++) {
    m_likelihoods[i] = m_pool[i]->compute_likelihood(f);
    m_valid_likelihoods.push_back(i);
  }
}


void
PDFPool::read_gk(const std::string &filename)
{
  std::ifstream in(filename.c_str());
  if (!in)
    throw std::string("PDFPool::read_gk(): could not open %s\n", filename.c_str());
  
  int pdfs = 0;
  std::string type_str;
  in >> pdfs >> m_dim >> type_str;
  m_pool.resize(pdfs);
  m_likelihoods.resize(pdfs);

  // New implementation
  if (type_str == "variable") {
    for (unsigned int i=0; i<m_pool.size(); i++) {
      in >> type_str;

      if (type_str == "diagonal_cov") {
	m_pool[i]=new DiagonalGaussian(m_dim);
      }
      else if (type_str == "full_cov") {
	m_pool[i]=new FullCovarianceGaussian(m_dim);
      }
      /*
      else if (type_str == "pcgmm") {
	m_pool[i]=new PrecisionConstrainedGaussian(m_dim);
      }
      else if (type_str == "scgmm") {
	m_pool[i]=new SubspaceConstrainedGaussian(m_dim);
      }
      */
    }
  }

  // For compliance
  else {
    if (type_str == "diagonal_cov") {
      for (unsigned int i=0; i<m_pool.size(); i++) {
	m_pool[i]=new DiagonalGaussian(m_dim);
	m_pool[i]->read(in);
      }
    }
    else if (type_str == "full_cov") {
      for (unsigned int i=0; i<m_pool.size(); i++) {
	m_pool[i]=new FullCovarianceGaussian(m_dim);
	m_pool[i]->read(in);
      }      
    }
    /*
    else if (type_str == "pcgmm") {
      for (unsigned int i=0; i<m_pool.size(); i++) {
	m_pool[i]=new PrecisionConstrainedGaussian(m_dim);
	m_pool[i]->read(in);
      }      
    }
    else if (type_str == "scgmm") {
      for (unsigned int i=0; i<m_pool.size(); i++) {
	m_pool[i]=new SubspaceConstrainedGaussian(m_dim);
	m_pool[i]->read(in);
      }            
    }
    */
    else
      throw std::string("Unknown model type\n");
  }
  
  if (!in)
    throw std::string("PDFPool::read_gk(): error reading file: %s\n");
}


void
PDFPool::write_gk(const std::string &filename) const
{
  std::ofstream out(filename.c_str());
  if (!out)
    throw std::string("PDFPool::write_gk(): could not open %s\n", filename.c_str());
  
  out << m_pool.size() << " " << m_dim << " variable\n";

  for (unsigned int i=0; i<m_pool.size(); i++)
    m_pool[i]->write(out);

  if (!out)
    throw std::string("PDFPool::write_gk(): error writing file: %s\n", filename.c_str());
}
