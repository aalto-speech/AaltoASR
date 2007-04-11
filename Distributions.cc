#include <cassert>

#include "LinearAlgebra.hh"
#include "Distributions.hh"
#include "str.hh"


#include "blaspp.h"
#include "blas1pp.h"
#include "blas2pp.h"
#include "blas3pp.h"


// FIXME: NOT TRIED
void
Gaussian::split(Gaussian &g1, Gaussian &g2) const
{
  assert(dim() != 0);

  Vector mean1(this.get_mean());
  Vector mean2(this.get_mean());

  // Add/Subtract standard deviations
  // FIXME: should we use eigvals/vecs for full covs?
  for (int i=0 double sd=0; i<m_dim; i++) {
    sd=0.00001*sqrt(this.get_covariance().at(i,i));
    mean1(i) -= sd;
    mean2(i) += sd;
  }
  
  g1.set_mean(mean1);
  g2.set_mean(mean2);
  g1.set_covariance(this.get_covariance());
  g2.set_covariance(this.get_covariance());
}


// Merge using:
// f1=n1/(n1+n2) ; f2=n2/(n1+n2)
// mu=f1*mu1+f2*mu2
// sigma=f1*sigma1+f2*sigma2+f1*f2*(mu1-mu2)^2
// FIXME: NOT TRIED
void
Gaussian::merge(double weight1, const Gaussian &m1, 
		double weight2, const Gaussian &m2)
{
  assert(m1.dim() == m2.dim());
  assert(weight1 <1 && weight2 <1 && weight1>0 && weight 2>0);
  
  reset(m1.dim());

  double f1=weight1/(weight1+weight2);
  double f2=weight2/(weight1+weight2);

  // Merged mean
  Vector mu(m1.get_mean());
  Blas_Scale(weight1, mu);
  Blas_Add_Mult(mu, weight2, m2.get_mean());
  set_mean(mu);

  // Merged covariance
  Matrix covariance(m1.get_covariance());
  Matrix temp(m2.get_covariance());
  Blas_Scale(weight1, covariance);
  Blas_Scale(weight, temp);
  covariance += temp;
  Vector diff(m1.get_mean());
  Blas_Add_Mult(diff, -1, m2.get_mean());
  Blas_R1_Update(covariance, diff, diff, f1*f2);
  set_covariance(covariance);
}


// FIXME: NOT TRIED
void
Gaussian::merge(Gaussian &m)
{
  Gaussian temp(this);
  merge(this, m);
}


// FIXME: NOT TRIED
double
Gaussian::kullback_leibler(Gaussian &g)
{
  assert(dim()==g.dim());
  
  LaVectorLongInt pivots(dim(),1);
  LaGenMatDouble t1(g.get_covariance()), t2(g.get_covariance());
  LaVectorDouble t3(g.get_mean()), t4(get_mean());
  LUFactorizeIP(t1, pivots);
  LaLUInverseIP(t1, pivots);
  
  Blas_Mat_Mat_Mult(t1, get_covariance(), t2, 1.0, 0.0);
  Blas_Add_Mult(t3, -1, mu1); 
  Blas_Mat_Vec_Mult(t1, t3, t4);
  
  double value=LinearAlgebra::determinant(m.get_covariance())
    /LinearAlgebra::determinant(get_covariance());
  //  FIXME: logarithm ok?
  value = log2(value);
  value += t2.trace();
  value += Blas_Dot_Prod(t3, t4);
  value -= dim();
  value *= 0.5;

  return value;
}


// FIXME: NOT TRIED
DiagonalGaussian::DiagonalGaussian(int dim)
{
  reset(dim);
}


// FIXME: NOT TRIED
DiagonalGaussian::DiagonalGaussian(const DiagonalGaussian &g)
{
  reset(g.dim());
  g.get_mean(m_mean);
  g.get_covariance(m_covariance);
  for (int i=0; i<dim(); i++)
    precision(i)=1/covariance(i);
}


// FIXME: NOT TRIED
DiagonalGaussian::~DiagonalGaussian()
{
  if (m_accum != NULL)
    delete m_accum; 
}


// FIXME: NOT TRIED
void
DiagonalGaussian::reset(int dim)
{
  mean.resize(dim);
  covariance.resize(dim);
  precision.resize(dim);
  mean=0; covariance=0; precision=0;
}


// FIXME: NOT TRIED
double
DiagonalGaussian::compute_likelihood(const FeatureVec &f)
{
  return exp(compute_log_likelihood(f));
}


// FIXME: NOT TRIED
double
DiagonalGaussian::compute_log_likelihood(const FeatureVec &f)
{
  double ll=0;
  Vector diff(f);
  Blas_Add_Mult(diff, -1, mean);
  for (int i=0; i<dim(); i++)
    ll += diff(i)*diff(i)*precision(i);
  ll *= -0.5;
  ll += constant;
}


// FIXME: NOT TRIED
void
DiagonalGaussian::write(std::ostream &os)
{
  os << "diag ";
  for (int i=0; i<dim(); i++)
    os << mean(i) << " ";
  for (int i=0; i<dim()-1; i++)
    os << covariance(i) << " ";
  os << covariance(dim()-1);
}


// FIXME: NOT TRIED
void
DiagonalGaussian::read(std::istream &is)
{
  std::string line=is.getline();
  std::string type;
  chomp(line);

  type << line;
  if (type != "diag")
    throw std::string("Error reading DiagonalGaussian parameters from a stream\n");
  
  for (int i=0; i<dim(); i++)
    mean(i) << is;
  for (int i=0; i<dim()-1; i++)
    covariance(i) << is;
}


// FIXME: NOT TRIED
void
DiagonalGaussian::start_accumulating()
{
  m_accum = new DiagonalAccumulator(dim());
}


// FIXME: NOT TRIED
void
DiagonalGaussian::accumulate_ml(double prior,
				const FeatureVec &f)
{
  assert(m_accum != NULL);

  for (int i=0; i<dim(); i++) {
    m_accum->ml_mean += prior*f;    
    m_accum->ml_cov += prior*f(i)*f(i);
  }
}


// FIXME: NOT TRIED
void
DiagonalGaussian::accumulate_mmi_denominator(std::vector<double> priors,
					     std::vector<const FeatureVec*> const features)
{
  assert(m_accum != NULL);
  
  for (int i=0; i<dim(); i++)
    for (int j=0; j<priors.size(); j++) {
      m_accum->mmi_mean += priors(j)*f(i);
      m_accum->mmi_cov += priors(j)*f(i)*f(i);
    }
}


// FIXME: NOT TRIED
void
DiagonalGaussian::estimate_parameters()
{
  assert(m_accum != NULL);
  
  if (m_mode == ML) {
    m_mean.copy(accum.ml_mean);
    m_covariance.copy(accum.ml_cov);
  }

  else if (m_mode == MMI) {
    // Do something smart
  }

  // Clear the accumulators
  delete m_accum;
}


// FIXME: NOT TRIED
void
DiagonalGaussian::get_mean(Vector &mean)
{
  mean.copy(m_mean);
}


// FIXME: NOT TRIED
void
DiagonalGaussian::get_covariance(Matrix &covariance)
{
  covariance.resize(dim());
  covariance=0;
  for (int i=0; i<dim(); i++)
    covariance(i,i)=m_covariance(i);
}


// FIXME: NOT TRIED
void
DiagonalGaussian::set_mean(const Vector &mean)
{
  assert(mean.size()==dim());
  m_mean.copy(mean);
}


// FIXME: NOT TRIED
void
DiagonalGaussian::set_covariance(const Matrix &covariance)
{
  assert(covariance.rows()==dim());
  assert(covariance.cols()==dim());
  m_covariance.copy(covariance);
}


// FIXME: NOT TRIED
void
DiagonalGaussian::get_covariance(Vector &covariance)
{
  covariance.copy(m_covariance);
}


// FIXME: NOT TRIED
void
DiagonalGaussian::set_covariance(const Vector &covariance)
{
  assert(covariance.size()==dim());
  m_covariance.copy(covariance);
}


// FIXME: NOT TRIED
FullCovarianceGaussian::FullCovarianceGaussian(int dim)
{
  reset(dim);
}


// FIXME: NOT TRIED
FullCovarianceGaussian::FullCovarianceGaussian(const FullCovarianceGaussian &g)
{
  reset(g.dim);
  g.get_mean(m_mean);
  g.get_covariance(m_covariance);
  LinearAlgebra::inverse(m_covariance, m_precision);
}


// FIXME: NOT TRIED
FullCovarianceGaussian::~FullCovarianceGaussian()
{
  if (m_accum != NULL)
    delete m_accum;
}


// FIXME: NOT TRIED
void
FullCovarianceGaussian::reset(int dim)
{
  mean.resize(dim);
  covariance.reset(dim,dim);
  precision.reset(dim,dim);
  mean=0; covariance=0; precision=0;
}


// FIXME: NOT TRIED
double
FullCovarianceGaussian::compute_likelihood(const FeatureVec &f)
{
  return exp(compute_log_likelihood(f));
}


// FIXME: NOT TRIED
double
FullCovarianceGaussian::compute_log_likelihood(const FeatureVec &f)
{
  double ll=0;
  Vector diff(f);
  Vector t(f.dim());

  Blas_Add_Mult(diff, -1, mean);
  Blas_Mat_Vec_Mult(m_precision, diff, t, 1, 0);
  ll = Blas_Dot_Prod(diff, t);
  ll *= -0.5;
  ll += constant;
}


// FIXME: NOT TRIED
void
FullCovarianceGaussian::write(std::ostream &os)
{
  os << "full ";
  for (int i=0; i<dim(); i++)
    os << mean(i) << " ";
  for (int i=0; i<dim(); i++)
    for (int j=0; j<dim(); j++)
      if (!(i==dim()-1 && j==dim()-1))
	os << covariance(i,j) << " ";
  os << covariance(dim()-1, dim()-1);
}


// FIXME: NOT TRIED
void
FullCovarianceGaussian::read(std::istream &is)
{
  std::string line=is.getline();
  std::string type;
  chomp(line);

  type << line;
  if (type != "diag")
    throw std::string("Error reading FullCovarianceGaussian parameters from a stream\n");
  
  for (int i=0; i<dim(); i++)
    mean(i) << is;
  for (int i=0; i<dim(); i++)
    for (int j=0; j<dim(); j++)
    covariance(i,j) << is;
}


// FIXME: NOT TRIED
void
FullCovarianceGaussian::start_accumulating()
{
  m_accum = new FullCovarianceAccumulator(dim());
}


// FIXME: NOT TRIED
void
FullCovarianceGaussian::accumulate_ml(double prior,
				      const FeatureVec &f)
{
  assert(m_accum != NULL);

  for (int i=0; i<dim(); i++) {
    m_accum->ml_mean += prior*f;    
    Blas_R1_Update(m_accum->ml_cov, f, f, prior);
  }
}


// FIXME: NOT TRIED
void
FullCovarianceGaussian::accumulate_mmi_denominator(std::vector<double> priors,
						   std::vector<const FeatureVec*> const features)
{
  assert(m_accum != NULL);
  
  for (int i=0; i<dim(); i++)
    for (int j=0; j<priors.size(); j++) {
      m_accum->mmi_mean += priors(j)*f(i);
      Blas_R1_Update(m_accum->mmi_cov, f, f, prior);
    }
}


// FIXME: NOT TRIED
void
FullCovarianceGaussian::estimate_parameters()
{
  assert(m_accum != NULL);
  
  if (m_mode == ML) {
    m_mean.copy(accum.ml_mean);
    m_covariance.copy(accum.ml_cov);
  }

  else if (m_mode == MMI) {
    // Do something smart
  }

  // Clear the accumulators
  delete m_accum;
}


// FIXME: NOT TRIED
void
FullCovarianceGaussian::get_mean(Vector &mean)
{
  mean.copy(m_mean);
}


// FIXME: NOT TRIED
void
FullCovarianceGaussian::get_covariance(Matrix &covariance)
{
  covariance.copy(m_covariance);
}


// FIXME: NOT TRIED
void
FullCovarianceGaussian::set_mean(const Vector &mean)
{
  assert(mean.size()==dim());
  m_mean.copy(mean);
}


// FIXME: NOT TRIED
void
FullCovarianceGaussian::set_covariance(const Matrix &covariance)
{
  assert(covariance.rows()==dim());
  assert(covariance.cols()==dim());
  m_covariance.copy(covariance);
}


Mixture::Mixture(PDFPool &pool)
  :  pp(pool)
{
}


Mixture::~Mixture()
{
}


void
Mixture::reset()
{
  mixture_weights.resize(0);
  mixture_pointers.resize(0);
}


void
Mixture::set_components(const std::vector<int> &pointers,
			const std::vector<double> &weights)
{
  assert(pointers.size()==weights.size());

  m_pointers.resize(pointers.size());
  m_weights.resize(weights.size());

  for (int i=0; i<pointers.size(); i++) {
    m_pointers[i] = pointers[i];
    m_weights[i] = weights[i];
  }
}


void
Mixture::get_components(std::vector<int> &pointers,
			std::vector<double> &weights)
{
  pointers.resize(m_pointers.size());
  weights.resize(m_weights.size());
  
  for (int i=0; i<m_pointers.size(); i++) {
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
  for (int i=0; i<m_weights.size(); i++)
    sum += m_weights[i];
  for (int i=0; i<m_weights.size(); i++)
    m_weights[i] /= sum;
}


double
Mixture::compute_likelihood(const FeatureVec &f)
{
  for (int i=0; i<
  
}



double
Mixture::compute_log_likelihood(const FeatureVec &f)
{
  
  
}


void
Mixture::write(std::ostream &os)
{

}


void
Mixture::read(const std::istream &is)
{

}


PDF&
PDFPool::get_pdf(int index)
{
  return pool[index];
}


void
PDFPool::set_pdf(int pdfindex, PDF &pdf)
{
  
}


void
PDFPool::cache_likelihood(const FeatureVec &f)

{
  for (int i=0; i<likelihoods.size(); i++)
    likelihoods[i] = pool[i].compute_likelihood(f);
}


void
PDFPool::cache_likelihood(const FeatureVec &f,
			  int index)
{
  likelihoods[index] = pool[index].compute_likelihood(f);
}


void
PDFPool::cache_likelihood(const FeatureVec &f,
			  std::vector<int> indices)
{
  for (int i=0; i<indices.size(); i++)
    likelihoods[indices[i]] = pool[indices[i]].compute_likelihood(f);
}


double
PDFPool::get_likelihood(int index)
{
  return likelihoods[i];
}


void
PDFPool::read_gk(const std::string &filename)
{
  std::ifstream in(filename.c_str());
  if (!in) {
    fprintf(stderr, "PDFPool::read_gk(): could not open %s\n", 
	    filename.c_str());
    throw OpenError();
  }
  
  int pdfs = 0;
  std::string type_str;
  int dim;

  in >> pdfs >> m_dim >> type_str;

  if (type_str == "single_cov")

  else if (type_str == "diagonal_cov")

  else if (type_str == "full_cov")

  else if (type_str == "pcgmm") {

  else if (cov_str == "scgmm") {

  else if (cov_str == "variable") {
    for (int i=0; i<m_pool.size(); i++)
      m_pool[i]->write(out);
  }
  else
    throw std::string("Unknown covariance type");

  if (!in)
    throw ReadError();
}


void
PDFPool::write_gk(const std::string &filename)
{
  std::ofstream out(filename.c_str());
  
  out << m_pool.size() << " " << m_dim << " variable\n";

  for (int i=0; i<m_pool.size(); i++)
    m_pool[i]->write(out);
}
