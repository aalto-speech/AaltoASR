#include <cassert>

#include "Distributions.hh"
#include "str.hh"

#include "blaspp.h"
#include "blas1pp.h"
#include "blas2pp.h"
#include "blas3pp.h"


void
GaussianAccumulator::get_accumulated_mean(Vector &mean) const
{
  mean.copy(m_mean);
}


void
GaussianAccumulator::get_mean_estimate(Vector &mean) const
{
  mean.copy(m_mean);
  Blas_Scale(1/m_gamma, mean);
}


void 
FullStatisticsAccumulator::dump_statistics(std::ostream &os) const
{
  float t;
  if (accumulated()) {
    os.write((char*)&m_feacount, sizeof(int));
    os.write((char*)&m_gamma, sizeof(double));
    for (int i=0; i<dim(); i++) {
      t=m_mean(i);
      os.write((char*)&t, sizeof(float));
    }
    for (int i=0; i<dim(); i++)
      for (int j=0; j<=i; j++) {
        t=m_second_moment(i,j);
        os.write((char*)&t, sizeof(float));
      }
  }
}

  
void 
FullStatisticsAccumulator::accumulate_from_dump(std::istream &is)
{
  int feacount;
  double gamma;
  float t;

  is.read((char*)&feacount, sizeof(int));
  is.read((char*)&gamma, sizeof(double));
  m_feacount += feacount;
  m_gamma += gamma;
  m_accumulated = true;
  
  for (int i=0; i<dim(); i++) {
    is.read((char*)&t, sizeof(float));
    m_mean(i) += t;
  }
  for (int i=0; i<dim(); i++)
    for (int j=0; j<=i; j++) {
      is.read((char*)&t, sizeof(float));
      m_second_moment(i,j) += t;
    }
}


void
FullStatisticsAccumulator::get_accumulated_second_moment(Matrix &second_moment) const
{
  LaGenMatDouble temp = LaGenMatDouble(m_second_moment);
  second_moment.copy(temp);
}


void
FullStatisticsAccumulator::get_covariance_estimate(Matrix &covariance_estimate) const
{
  Vector mean_estimate;
  get_mean_estimate(mean_estimate);
  get_accumulated_second_moment(covariance_estimate);
  Blas_Scale(1/m_gamma, covariance_estimate);
  Blas_R1_Update(covariance_estimate, mean_estimate, mean_estimate, -1);
}


void
FullStatisticsAccumulator::accumulate(int feacount, double gamma, const FeatureVec &f)
{
  m_feacount += feacount;
  m_gamma += gamma;
  m_accumulated = true;
  const Vector *feature = f.get_vector();
  Blas_Add_Mult(m_mean, gamma, *feature);
  Blas_R1_Update(m_second_moment, *feature, gamma, 1.0, true);
}


void 
DiagonalStatisticsAccumulator::dump_statistics(std::ostream &os) const
{
  float t;
  if (accumulated()) {
    os.write((char*)&m_feacount, sizeof(int));
    os.write((char*)&m_gamma, sizeof(double));
    for (int i=0; i<dim(); i++) {
      t=m_mean(i);
      os.write((char*)&t, sizeof(float));
    }
    for (int i=0; i<dim(); i++) {
      t=m_second_moment(i);
      os.write((char*)&t, sizeof(float));
    }
  }
}

  
void 
DiagonalStatisticsAccumulator::accumulate_from_dump(std::istream &is)
{
  int feacount;
  double gamma;
  float t;
  
  is.read((char*)&feacount, sizeof(int));
  is.read((char*)&gamma, sizeof(double));
  m_feacount += feacount;
  m_gamma += gamma;
  m_accumulated = true;
  
  for (int i=0; i<dim(); i++) {
    is.read((char*)&t, sizeof(float));
    m_mean(i) += t;
  }
  for (int i=0; i<dim(); i++) {
    is.read((char*)&t, sizeof(float));
    m_second_moment(i) += t;
  }
}


void
DiagonalStatisticsAccumulator::get_covariance_estimate(Matrix &covariance_estimate) const
{
  Vector mean_estimate;
  get_mean_estimate(mean_estimate);
  covariance_estimate.resize(dim(), dim());
  covariance_estimate=0;
  for (int i=0; i<dim(); i++) {
    covariance_estimate(i,i) = m_second_moment(i) / m_gamma;
    covariance_estimate(i,i) -= mean_estimate(i)*mean_estimate(i);
  }
}


void
DiagonalStatisticsAccumulator::get_accumulated_second_moment(Matrix &second_moment) const
{
  second_moment.resize(dim(), dim());
  second_moment=0;
  for (int i=0; i<dim(); i++)
    second_moment(i,i) = m_second_moment(i);
}


void
DiagonalStatisticsAccumulator::accumulate(int feacount, double gamma, const FeatureVec &f)
{
  m_feacount += feacount;
  m_gamma += gamma;
  m_accumulated = true;
  const Vector *feature = f.get_vector();
  Blas_Add_Mult(m_mean, gamma, *feature);
  for (int i=0; i<dim(); i++)
    m_second_moment(i) += gamma * (*feature)(i) * (*feature)(i);
}


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
Gaussian::accumulate(double gamma,
                     const FeatureVec &f,
                     int accum_pos)
{
  assert((int)m_accums.size() > accum_pos);
  assert(m_accums[accum_pos] != NULL);
  
  m_accums[accum_pos]->accumulate(1, gamma, f);
}


void 
Gaussian::dump_statistics(std::ostream &os,
                          int accum_pos) const
{
  assert((int)m_accums.size() > accum_pos);
  assert(m_accums[accum_pos] != NULL);  

  m_accums[accum_pos]->dump_statistics(os);
}
  
void 
Gaussian::accumulate_from_dump(std::istream &is)
{
  int accum_pos;  
  is.read((char*)&accum_pos, sizeof(int));

  assert((int)m_accums.size() > accum_pos);
  assert(m_accums[accum_pos] != NULL);

  m_accums[accum_pos]->accumulate_from_dump(is);
}


void
Gaussian::stop_accumulating()
{
  if (m_accums.size() > 0)
    for (unsigned int i=0; i<m_accums.size(); i++)
      if (m_accums[i] != NULL)
	delete m_accums[i];
  m_accums.resize(0);
}


bool
Gaussian::accumulated(int accum_pos) const
{
  if ((int)m_accums.size() <= accum_pos)
    return false;
  if (m_accums[accum_pos] == NULL)
    return false;
  return m_accums[accum_pos]->accumulated();
}


void
Gaussian::estimate_parameters(double minvar, double covsmooth,
                              double c1, double c2)
{
  assert(accumulated(0));

  Vector new_mean;
  Matrix new_covariance;
  
  if (m_mode == ML) {
    m_accums[0]->get_mean_estimate(new_mean);
    m_accums[0]->get_covariance_estimate(new_covariance);
  }
  else if (m_mode == MMI) {

    assert( accumulated(1) );

    Vector old_mean;
    Matrix old_covariance;
    get_mean(old_mean);
    get_covariance(old_covariance);

    // c & mu~ & sigma~
    double c = m_accums[0]->gamma() - m_accums[1]->gamma();
    LaVectorDouble mu_tilde;
    LaVectorDouble temp_denominator_mu;
    m_accums[0]->get_accumulated_mean(mu_tilde);
    m_accums[1]->get_accumulated_mean(temp_denominator_mu);
    Blas_Add_Mult(mu_tilde, -1, temp_denominator_mu);

    LaGenMatDouble sigma_tilde;
    LaGenMatDouble temp_denominator_sigma;
    m_accums[0]->get_accumulated_second_moment(sigma_tilde);
    m_accums[1]->get_accumulated_second_moment(temp_denominator_sigma);
    Blas_Add_Mat_Mult(sigma_tilde, -1, temp_denominator_sigma);
    
    // a0
    LaGenMatDouble a0(sigma_tilde);
    Blas_Scale(c, a0);
    Blas_R1_Update(a0, mu_tilde, mu_tilde, -1);
    
    // a1
    LaGenMatDouble a1(old_covariance);
    Blas_R1_Update(a1, old_mean, old_mean, 1);
    Blas_Scale(c, a1);
    Blas_R1_Update(a1, old_mean, mu_tilde, -1);
    Blas_R1_Update(a1, mu_tilde, old_mean, -1);
    Blas_Add_Mat_Mult(a1, 1, sigma_tilde);
    
    // a2
    LaGenMatDouble a2(old_covariance);

    // Solve the quadratic eigenvalue problem
    // A_0 + D A_1 + D^2 A_2
    // by linearizing to a generalized eigenvalue problem
    // See: www.cs.utk.edu/~dongarra/etemplates/node333.html
    // Note: complex eigenvalues, we want only the max real eigenvalue!
    LaGenMatDouble A=LaGenMatDouble::zeros(2*dim());
    LaGenMatDouble B=LaGenMatDouble::zeros(2*dim());
    LaGenMatDouble identity=LaGenMatDouble::eye(dim());

    A(LaIndex(dim(),2*dim()-1),LaIndex(0,dim()-1)).inject(a0);
    A(LaIndex(dim(),2*dim()-1),LaIndex(dim(),2*dim()-1)).inject(a1);
    Blas_Scale(-1,A);
    A(LaIndex(0,dim()-1),LaIndex(dim(),2*dim()-1)).inject(identity);

    B(LaIndex(0,dim()-1),LaIndex(0,dim()-1)).inject(identity);
    B(LaIndex(dim(),2*dim()-1),LaIndex(dim(),2*dim()-1)).inject(a2);
    
    LaVectorComplex eigvals;
    LaGenMatComplex eigvecs;
    LinearAlgebra::generalized_eigenvalues(A,B,eigvals,eigvecs);

    double d=0;
    for (int i=0; i<eigvals.size(); i++)
      if (eigvals(i).i < 0.000000001)
        d=std::max(d, eigvals(i).r);
    
    assert(d>0);
    assert(c2>1);
    d=std::max(c1*m_accums[1]->gamma(), c2*d);
    
    // COMPUTE NEW MEAN
    // new_mean=(obs_num-obs_den+D*old_mean)/(gamma_num-gamma_den+D)
    new_mean.copy(old_mean);
    Blas_Scale(d, new_mean);
    Blas_Add_Mult(new_mean, 1, mu_tilde);
    Blas_Scale(1/(c+d), new_mean);

    // COMPUTE NEW COVARIANCE
    // new_cov=(obs_num-obs_den+D*old_cov+old_mean*old_mean')/(gamma_num-gamma_den+D)-new_mean*new_mean'
    new_covariance.copy(old_covariance);
    Blas_R1_Update(new_covariance, old_mean, old_mean, 1);
    Blas_Scale(d, new_covariance);
    Blas_Add_Mat_Mult(new_covariance, 1, sigma_tilde);
    Blas_Scale(1/(c+d), new_covariance);
    Blas_R1_Update(new_covariance, new_mean, new_mean, -1);
  }
  else
    assert( 0 );

  // Check that covariances are valid
  if (m_accums[0]->feacount() > 1)
  {
    for (int i=0; i<dim(); i++)
      if (new_covariance(i,i) <= 0)
        fprintf(stderr, "Warning: Variance in dimension %i is %g (gamma %g)\n",
                i, new_covariance(i,i), m_accums[0]->gamma());
  }
  
  // Common tweaking
  for (int i=0; i<dim(); i++)
    if (new_covariance(i,i) < minvar)
      new_covariance(i,i) = minvar;
  
  if (covsmooth != 0) {
    for (int i=0; i<dim(); i++)
      for (int j=0; j<dim(); j++)
        if (i != j)
          new_covariance(i,j) *= m_accums[0]->feacount()/(m_accums[0]->feacount()+covsmooth);
  }

  // Set the parameters
  set_mean(new_mean);
  set_covariance(new_covariance);
}


void
Gaussian::split(Gaussian &g1, Gaussian &g2, double perturbation) const
{
  assert(dim() != 0);
  
  Vector mean1; get_mean(mean1);
  Vector mean2; get_mean(mean2);
  Matrix cov; get_covariance(cov);

  Matrix eigvecs(cov);
  Vector eigvals(dim());
  LaEigSolveSymmetricVecIP(eigvecs, eigvals);
  for (int i=0; i<dim(); i++) {
    double t=0;
    for (int j=0; j<dim(); j++)
      t +=  eigvals(j) * eigvecs(i,j);
    // FIXME!? IS THIS CORRECT??
    t = perturbation * sqrt(t);
    mean1(i) -= t;
    mean2(i) += t;
  }
  
  g1.set_mean(mean1);
  g2.set_mean(mean2);
  g1.set_covariance(cov);
  g2.set_covariance(cov);
}


void
Gaussian::split(Gaussian &g2, double perturbation)
{
  split(*this, g2, perturbation);
}


// Merge using:
// f1=n1/(n1+n2) ; f2=n2/(n1+n2)
// mu=f1*mu1+f2*mu2
// sigma=f1*sigma1+f2*sigma2+f1*f2*(mu1-mu2)^2
void
Gaussian::merge(double weight1, const Gaussian &m1, 
		double weight2, const Gaussian &m2,
                bool finish_statistics)
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
  set_covariance(covariance, finish_statistics);
}


void
Gaussian::merge(const std::vector<double> &weights,
                const std::vector<const Gaussian*> &gaussians,
                bool finish_statistics)
{
  assert( weights.size() == gaussians.size() );

  Vector new_mean(m_dim);
  Matrix new_covariance = Matrix::zeros(m_dim, m_dim);
  Matrix eye = Matrix::eye(m_dim, m_dim);
  double weight_sum = 0;
  new_mean = 0;
  
  for (int i = 0; i < (int)weights.size(); i++)
  {
    Matrix cur_covariance;
    Vector cur_mean;
    gaussians[i]->get_covariance(cur_covariance);
    gaussians[i]->get_mean(cur_mean);
    Blas_R1_Update(cur_covariance, cur_mean, cur_mean);
    //Blas_Mat_Mat_Mult(cur_covariance, eye, new_covariance, weights[i], 1.0);
    Blas_Add_Mat_Mult(new_covariance, weights[i], cur_covariance);
    Blas_Add_Mult(new_mean, weights[i], cur_mean);
    weight_sum += weights[i];
  }
  Blas_Scale(1.0/weight_sum, new_mean);
  Blas_Scale(1.0/weight_sum, new_covariance);
  Blas_R1_Update(new_covariance, new_mean, new_mean, -1.0);
  set_mean(new_mean);
  set_covariance(new_covariance, finish_statistics);
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


bool
Gaussian::full_stats_accumulated() const
{
  if (!m_full_stats)
    return false;
  return accumulated();
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
  {
    if (m_covariance(i) > 0)
      m_precision(i)=1/m_covariance(i);
    else
      m_covariance(i) = 0;
  }
  set_constant();
}


DiagonalGaussian::~DiagonalGaussian()
{
  stop_accumulating();
}


void
DiagonalGaussian::reset(int dim)
{
  m_dim=dim;
  m_mode=ML;
  
  m_mean.resize(dim);
  m_covariance.resize(dim);
  m_precision.resize(dim);
  m_mean=0; m_covariance=0; m_precision=0;

  m_constant=0;
  m_full_stats=false;
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

  Vector diff(*(f.get_vector()));

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
    if (m_covariance(i) > 0)
      m_precision(i) = 1/m_covariance(i);
    else
      m_precision(i) = 0;
  }
  set_constant();
}


void
DiagonalGaussian::start_accumulating()
{
  if (m_mode == ML) {
    m_accums.resize(1);
    if (m_full_stats)
      m_accums[0] = new FullStatisticsAccumulator(dim());
    else
      m_accums[0] = new DiagonalStatisticsAccumulator(dim());
  }
  else if (m_mode == MMI) {
    m_accums.resize(2);
    if (m_full_stats) {
      m_accums[0] = new FullStatisticsAccumulator(dim());
      m_accums[1] = new FullStatisticsAccumulator(dim());
    }
    else {
      m_accums[0] = new DiagonalStatisticsAccumulator(dim());
      m_accums[1] = new DiagonalStatisticsAccumulator(dim());
    }
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
DiagonalGaussian::set_covariance(const Matrix &covariance,
                                 bool finish_statistics)
{
  assert(covariance.rows()==dim());
  assert(covariance.cols()==dim());

  for (int i=0; i<dim(); i++) {
    assert(covariance(i,i)>0);
    m_covariance(i) = covariance(i,i);
    if (finish_statistics && m_covariance(i) > 0)
      m_precision(i) = 1/m_covariance(i);
    else
      m_precision(i) = 0;
  }
  if (finish_statistics)
    set_constant();
}


void
DiagonalGaussian::get_covariance(Vector &covariance) const
{
  covariance.copy(m_covariance);
}


void
DiagonalGaussian::set_covariance(const Vector &covariance,
                                 bool finish_statistics)
{
  assert(covariance.size()==dim());
  m_covariance.copy(covariance);

  for (int i=0; i<dim(); i++) {
    m_covariance(i) = covariance(i);
    if (finish_statistics && m_covariance(i) > 0)
      m_precision(i) = 1/m_covariance(i);
    else
      m_precision(i) = 0;
  }
  if (finish_statistics)
    set_constant();
}


void
DiagonalGaussian::set_constant(void)
{
  m_constant=1;
  for (int i=0; i<dim(); i++)
    m_constant *= m_precision(i);
  if (m_constant > 0)
    m_constant = log(sqrt(m_constant));
}


void
DiagonalGaussian::split(Gaussian &g1, Gaussian &g2, double perturbation) const
{
  assert(dim() != 0);
  
  Vector mean1; get_mean(mean1);
  Vector mean2; get_mean(mean2);
  Matrix cov; get_covariance(cov);

  // Add/Subtract standard deviations
  double sd=0;
  for (int i=0; i<dim(); i++) {
    sd=perturbation*sqrt(cov(i,i));
    mean1(i) -= sd;
    mean2(i) += sd;
  }
  
  g1.set_mean(mean1);
  g2.set_mean(mean2);
  g1.set_covariance(cov);
  g2.set_covariance(cov);
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
  m_precision.copy(g.m_precision);
  m_constant = g.m_constant;
}


FullCovarianceGaussian::~FullCovarianceGaussian()
{
  stop_accumulating();
}


void
FullCovarianceGaussian::reset(int dim)
{
  m_dim=dim;
  m_mode=ML;
  
  m_mean.resize(dim);
  m_covariance.resize(dim,dim);
  m_precision.resize(dim,dim);
  m_mean=0; m_covariance=0; m_precision=0;

  m_constant=0;
  m_full_stats=true;
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

  Vector diff(*(f.get_vector()));
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
  for (int i=0; i<dim(); i++)
    is >> m_mean(i);
  for (int i=0; i<dim(); i++)
    for (int j=0; j<dim(); j++)
      is >> m_covariance(i,j);

  if (LinearAlgebra::is_spd(m_covariance))
  {
    LinearAlgebra::inverse(m_covariance, m_precision);
    m_constant = log(sqrt(LinearAlgebra::spd_determinant(m_precision)));
  }
  else
  {
    m_precision = 0;
    m_constant = 0;
  }
}


void
FullCovarianceGaussian::start_accumulating()
{
  if (m_mode == ML) {
    m_accums.resize(1);
    m_accums[0] = new FullStatisticsAccumulator(dim());
  }
  else if (m_mode == MMI) {
    m_accums.resize(2);
    m_accums[0] = new FullStatisticsAccumulator(dim());
    m_accums[1] = new FullStatisticsAccumulator(dim());
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
FullCovarianceGaussian::set_covariance(const Matrix &covariance,
                                       bool finish_statistics)
{
  assert(covariance.rows()==dim());
  assert(covariance.cols()==dim());
  
  m_covariance.copy(covariance);
  if (finish_statistics && LinearAlgebra::is_spd(m_covariance))
  {
    LinearAlgebra::inverse(m_covariance, m_precision);
    m_constant = log(sqrt(LinearAlgebra::spd_determinant(m_precision)));
  }
  else
  {
    m_precision = 0;
    m_constant = 0;
  }
}


PrecisionConstrainedGaussian::PrecisionConstrainedGaussian()
{
}


PrecisionConstrainedGaussian::PrecisionConstrainedGaussian(PrecisionSubspace *space)
{
  m_ps = space;
}


PrecisionConstrainedGaussian::PrecisionConstrainedGaussian(const PrecisionConstrainedGaussian &g)
{
  g.get_mean(m_transformed_mean);
  g.get_precision_coeffs(m_coeffs);
  m_ps = g.get_subspace();
}


PrecisionConstrainedGaussian::~PrecisionConstrainedGaussian()
{
}


void
PrecisionConstrainedGaussian::reset(int feature_dim)
{
  m_dim=feature_dim;
  m_mode=ML;
  
  m_transformed_mean.resize(feature_dim);
  m_transformed_mean=0;
  m_coeffs=0;
  
  m_constant=0;
  m_full_stats=true;
}


double
PrecisionConstrainedGaussian::compute_likelihood(const FeatureVec &f) const
{
  return exp(compute_log_likelihood(f));
}


double
PrecisionConstrainedGaussian::compute_log_likelihood(const FeatureVec &f) const
{
  if (!m_ps->computed())
    m_ps->precompute(f);
  
  double result=m_constant
    +Blas_Dot_Prod(m_transformed_mean, *f.get_vector())
    +m_ps->dotproduct(m_coeffs);
  return result;
}


void
PrecisionConstrainedGaussian::write(std::ostream &os) const
{
  os << "pcgmm ";
  for (int i=0; i<dim(); i++)
    os << m_transformed_mean(i) << " ";
  os << subspace_dim() << " ";
  for (int i=0; i<subspace_dim()-1; i++)
    os << m_coeffs(i) << " ";
  os << m_coeffs(subspace_dim()-1);  
}


void
PrecisionConstrainedGaussian::read(std::istream &is)
{
  int ss_dim;
  
  for (int i=0; i<dim(); i++)
    is >> m_transformed_mean(i);
  is >> ss_dim;
  m_coeffs.resize(ss_dim);
  for (int i=0; i<subspace_dim(); i++)
    is >> m_coeffs(i);

  Matrix precision;
  m_ps->compute_precision(m_coeffs, precision);
  m_constant = LinearAlgebra::determinant(precision);
  m_constant = log(sqrt(m_constant));
}


void
PrecisionConstrainedGaussian::start_accumulating()
{
  if (m_mode == ML) {
    m_accums.resize(1);
    m_accums[0] = new FullStatisticsAccumulator(dim());
  }
  else if (m_mode == MMI) {
    m_accums.resize(2);
    m_accums[0] = new FullStatisticsAccumulator(dim());
    m_accums[1] = new FullStatisticsAccumulator(dim());
  }
}


void
PrecisionConstrainedGaussian::get_mean(Vector &mean) const
{
  Matrix precision;
  Matrix covariance;
  m_ps->compute_precision(m_coeffs, precision);
  LinearAlgebra::inverse(precision, covariance);
  Blas_Mat_Vec_Mult(covariance, m_transformed_mean, mean);
}


void
PrecisionConstrainedGaussian::get_covariance(Matrix &covariance) const
{
  Matrix precision;
  m_ps->compute_precision(m_coeffs, precision);
  LinearAlgebra::inverse(precision, covariance);
}


void
PrecisionConstrainedGaussian::set_mean(const Vector &mean)
{
  Matrix precision;
  m_ps->compute_precision(m_coeffs, precision);
  Blas_Mat_Vec_Mult(precision, mean, m_transformed_mean);
}


void
PrecisionConstrainedGaussian::set_covariance(const Matrix &covariance,
                                             bool finish_statistics)
{
  m_ps->optimize_coefficients(covariance, m_coeffs);
}


SubspaceConstrainedGaussian::SubspaceConstrainedGaussian()
{
}


SubspaceConstrainedGaussian::SubspaceConstrainedGaussian(ExponentialSubspace *space)
{
  m_es = space;
}


SubspaceConstrainedGaussian::SubspaceConstrainedGaussian(const SubspaceConstrainedGaussian &g)
{
  g.get_subspace_coeffs(m_coeffs);
  m_es = g.get_subspace();
}


SubspaceConstrainedGaussian::~SubspaceConstrainedGaussian()
{
}


void
SubspaceConstrainedGaussian::reset(int feature_dim)
{
  m_dim=feature_dim;
  m_mode=ML;
  
  m_coeffs=0;
  
  m_constant=0;
  m_full_stats=true;
}


double
SubspaceConstrainedGaussian::compute_likelihood(const FeatureVec &f) const
{
  return exp(compute_log_likelihood(f));
}


double
SubspaceConstrainedGaussian::compute_log_likelihood(const FeatureVec &f) const
{
  if (!m_es->computed())
    m_es->precompute(f);
  
  double result=m_constant+m_es->dotproduct(m_coeffs);
  return result;
}


void
SubspaceConstrainedGaussian::write(std::ostream &os) const
{
  os << "scgmm ";
  for (int i=0; i<m_coeffs.size()-1; i++)
    os << m_coeffs(i) << " ";
  os << m_coeffs(m_coeffs.size()-1);
}


void
SubspaceConstrainedGaussian::read(std::istream &is)
{
  

  
}


void
SubspaceConstrainedGaussian::start_accumulating()
{
  if (m_mode == ML) {
    m_accums.resize(1);
    m_accums[0] = new FullStatisticsAccumulator(dim());
  }
  else if (m_mode == MMI) {
    m_accums.resize(2);
    m_accums[0] = new FullStatisticsAccumulator(dim());
    m_accums[1] = new FullStatisticsAccumulator(dim());
  }
}


void
SubspaceConstrainedGaussian::get_mean(Vector &mean) const
{
}


void
SubspaceConstrainedGaussian::get_covariance(Matrix &covariance) const
{
}


void
SubspaceConstrainedGaussian::set_mean(const Vector &mean)
{
}


void
SubspaceConstrainedGaussian::set_covariance(const Matrix &covariance,
                                            bool finish_statistics)
{
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


PDF*
Mixture::get_base_pdf(int index)
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
    m_accums[0] = new MixtureAccumulator(size());
  }
  else if (m_mode == MMI) {
    m_accums.resize(2);
    m_accums[0] = new MixtureAccumulator(size());
    m_accums[1] = new MixtureAccumulator(size());
  }
  
  for (int i=0; i<size(); i++)
    get_base_pdf(i)->start_accumulating();
}


void
Mixture::accumulate(double gamma,
		    const FeatureVec &f,
		    int accum_pos)
{
  double total_likelihood, this_likelihood;

  // Compute the total likelihood for this mixture
  total_likelihood = compute_likelihood(f);
  
  // Accumulate all basis distributions with some gamma
  if (total_likelihood > 0) {
    for (int i=0; i<size(); i++) {
      this_likelihood = gamma * m_weights[i] * m_pool->compute_likelihood(f, m_pointers[i]);
      m_accums[accum_pos]->gamma[i] += this_likelihood / total_likelihood;
      get_base_pdf(i)->accumulate(this_likelihood / total_likelihood, f, accum_pos);
    }
    m_accums[accum_pos]->accumulated = true;
  }
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
    os << size();
    for (int i=0; i<size()-1; i++)
      os << " " << m_pointers[i] << " " << m_accums[accum_pos]->gamma[i];
    os << " " << m_pointers[size()-1] << " " << m_accums[accum_pos]->gamma[size()-1];    
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
  m_accums.resize(0);
  
  for (int i=0; i<size(); i++)
    get_base_pdf(i)->stop_accumulating();
}


void
Mixture::estimate_parameters(void)
{
  assert(accumulated(0));
  
  if (m_mode == ML || !accumulated(1)) {    
    double total_gamma = 0;
    for (int i=0; i<size(); i++)
      total_gamma += m_accums[0]->gamma[i];    
    for (int i=0; i<size(); i++)
      m_weights[i] = m_accums[0]->gamma[i]/total_gamma;
  }

  // There are many alternatives for updating mixture coefficients
  // This implementation follows Woodland & Povey, '02
  else if (m_mode == MMI) {

    double diff=1, norm;

    // Iterate while the weights change
    std::vector<double> old_weights = m_weights;
    while (diff > 0.0001) {

      diff=0;
      std::vector<double> previous_weights = m_weights;

      for (int i=0; i<size(); i++) {

        // Set c_new = c_old * gam_num / gam_den
        m_weights[i] = old_weights[i] * m_accums[0]->gamma[i]/m_accums[1]->gamma[i];

        // Renormalize weights
        norm=0;
        for (int j=0; j<size(); j++)
          if (i != j)
            norm += m_weights[j];
        norm = (1 - m_weights[i]) / norm;
        for (int j=0; j<size(); j++) {
          if (i != j)
            m_weights[j] *= norm;
        }
      }

      for (int i=0; i<size(); i++)
        diff += std::abs(m_weights[i]-previous_weights[i]);
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


PDFPool::PDFPool() {
  reset();
}


PDFPool::PDFPool(int dim)
{
  reset();
  m_dim=dim;
}


PDFPool::~PDFPool()
{
  for (unsigned int i=0; i<m_pool.size(); i++)
    delete m_pool[i];
}


void
PDFPool::reset()
{
  m_dim=0;
  m_minvar = 0;
  m_covsmooth = 0;
  m_c1 = 1;
  m_c2 = 2;
  m_pool.clear();
  m_likelihoods.clear();
  m_valid_likelihoods.clear();
}


PDF*
PDFPool::get_pdf(int index) const
{
  return m_pool[index];
}


void
PDFPool::set_pdf(int pdfindex, PDF *pdf)
{
  if ((unsigned int)pdfindex >= m_pool.size())
    m_pool.resize(pdfindex+1);
  m_pool[pdfindex]=pdf;
}


int
PDFPool::add_pdf(PDF *pdf)
{
  int index = (int)m_pool.size();
  m_pool.push_back(pdf);
  return index;
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
  reset_cache();
  for (int i=0; i<size(); i++) {
    m_likelihoods[i] = m_pool[i]->compute_likelihood(f);
    m_valid_likelihoods.push_back(i);
  }
}


void
PDFPool::set_gaussian_parameters(double minvar, double covsmooth,
                                 double c1, double c2)
{
  m_minvar = minvar;
  m_covsmooth = covsmooth;
  m_c1 = c1;
  m_c2 = c2;
}


void
PDFPool::estimate_parameters(void)
{
  for (int i=0; i<size(); i++)
  {
    Gaussian *temp = dynamic_cast< Gaussian* > (m_pool[i]);
    if (temp != NULL)
      temp->estimate_parameters(m_minvar, m_covsmooth, m_c1, m_c2);
    else
      m_pool[i]->estimate_parameters();
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
  m_valid_likelihoods.clear();
  for (int i=0; i<pdfs; i++)
    m_likelihoods[i] = -1;
  
  // New implementation
  if (type_str == "variable") {
    for (int i=0; i<pdfs; i++) {
      in >> type_str;

      if (type_str == "diag") {
        m_pool[i]=new DiagonalGaussian(m_dim);
        m_pool[i]->read(in);
      }
      else if (type_str == "full") {
	m_pool[i]=new FullCovarianceGaussian(m_dim);
        m_pool[i]->read(in);
      }
      else if (type_str == "pcgmm") {
	m_pool[i]=new PrecisionConstrainedGaussian();
        m_pool[i]->read(in);
      }
      else if (type_str == "scgmm") {
	m_pool[i]=new SubspaceConstrainedGaussian();
        m_pool[i]->read(in);
      }
    }
  }

  // For compliance
  else {
    if (type_str == "diagonal_cov") {
      for (int i=0; i<pdfs; i++) {
        m_pool[i]=new DiagonalGaussian(m_dim);
	m_pool[i]->read(in);
      }
    }
    else if (type_str == "full_cov") {
      for (int i=0; i<pdfs; i++) {
	m_pool[i]=new FullCovarianceGaussian(m_dim);
	m_pool[i]->read(in);
      }      
    }
    else if (type_str == "pcgmm") {
      for (unsigned int i=0; i<m_pool.size(); i++) {
	m_pool[i]=new PrecisionConstrainedGaussian();
	m_pool[i]->read(in);
      }      
    }
    else if (type_str == "scgmm") {
      for (unsigned int i=0; i<m_pool.size(); i++) {
	m_pool[i]=new SubspaceConstrainedGaussian();
	m_pool[i]->read(in);
      }            
    }
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

  for (unsigned int i=0; i<m_pool.size(); i++) {
    m_pool[i]->write(out);
    out << std::endl;
  }

  if (!out)
    throw std::string("PDFPool::write_gk(): error writing file: %s\n", filename.c_str());
}


bool
PDFPool::split_gaussian(int index, int &new_index, double minocc, int minfeas)
{
  Gaussian *gaussian = dynamic_cast< Gaussian* > (m_pool[index]);
  if (gaussian == NULL)
    return false;

  Gaussian *new_gaussian = gaussian->copy_gaussian();

  if (gaussian->m_accums[0]->gamma() < minocc)
    return false;
  if (gaussian->m_accums[0]->feacount() < minfeas)
    return false;

  gaussian->split(*new_gaussian, 0.2);
  new_index = add_pdf(new_gaussian);
  return true;
}
