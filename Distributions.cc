#include <cassert>

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
DiagonalGaussian::~DiagonalGaussian() { }


// FIXME: NOT TRIED
DiagonalGaussian::DiagonalGaussian(const DiagonalGaussian &g)
{
  reset(g.dim);
  mean.copy(g.get_mean());
  covariance.copy(g.get_covariance());
  for (int i=0; i<dim; i++)
    precision(i)=1/covariance(i);
}


// FIXME: NOT TRIED
void
DiagonalGaussian::reset(int dim)
{
  mean.resize(dim);
  covariance.reset(dim);
  precision.reset(dim);
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
  mean.resize(dim());
  mean=0;
  for (int i=0; i<dim(); i++)
    mean(i)=m_mean(i);
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


