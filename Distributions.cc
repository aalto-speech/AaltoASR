#include <cassert>

#include <iostream>
#include <fstream>
#include <math.h>

#include "Distributions.hh"
#include "str.hh"

#include "blaspp.h"
#include "blas1pp.h"
#include "blas2pp.h"
#include "blas3pp.h"


namespace aku {

void
GaussianAccumulator::get_accumulated_mean(Vector &mean) const
{
  mean.copy(m_mean);
}


void
GaussianAccumulator::set_accumulated_mean(Vector &mean)
{
  m_mean.copy(mean);
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
    os.write((char*)&m_aux_gamma, sizeof(double));
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
  double aux_gamma = 0;
  float t;

  is.read((char*)&feacount, sizeof(int));
  is.read((char*)&gamma, sizeof(double));
  is.read((char*)&aux_gamma, sizeof(double));
  if (is.fail())
    fprintf(stderr, "Error while reading statistics dump\n");

  if (feacount < 0)
    throw std::string("Invalid statistics dump\n");
  
  m_feacount += feacount;
  m_gamma += gamma;
  m_aux_gamma += aux_gamma;
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
  SymmetricMatrix temp = m_second_moment; // Discard const qualifier
  second_moment = LaGenMatDouble(temp);
}

// Diagonal covariance version of get_accumulated_second_moment
void
FullStatisticsAccumulator::get_accumulated_second_moment(Vector &second_moment) const
{
  second_moment.resize(dim());
  for (int i = 0; i < dim(); i++)
    second_moment(i) = m_second_moment(i,i);
}

void
FullStatisticsAccumulator::set_accumulated_second_moment(Matrix &second_moment)
{
  for (int i = 0; i < dim(); i++)
    for (int j = 0; j <= i; j++)
      m_second_moment(i, j) = second_moment(i, j);
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
FullStatisticsAccumulator::accumulate(int feacount, double gamma, const Vector &f)
{
  m_feacount += feacount;
  m_gamma += gamma;
  m_accumulated = true;
  Blas_Add_Mult(m_mean, gamma, f);
  Blas_R1_Update(m_second_moment, f, gamma, 1.0, true);
}


void
FullStatisticsAccumulator::reset()
{
  m_feacount=0;
  m_mean=0;
  m_second_moment=0;
  m_gamma=0;
  m_aux_gamma = 0;
  m_accumulated=false;
}


void 
DiagonalStatisticsAccumulator::dump_statistics(std::ostream &os) const
{
  float t;
  if (accumulated()) {
    os.write((char*)&m_feacount, sizeof(int));
    os.write((char*)&m_gamma, sizeof(double));
    os.write((char*)&m_aux_gamma, sizeof(double));
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
  double aux_gamma = 0;
  float t;
  
  is.read((char*)&feacount, sizeof(int));
  is.read((char*)&gamma, sizeof(double));
  is.read((char*)&aux_gamma, sizeof(double));
  if (is.fail())
    fprintf(stderr, "Error while reading statistics dump\n");
  
  if (feacount < 0)
    throw std::string("Invalid statistics dump\n");
  
  m_feacount += feacount;
  m_gamma += gamma;
  m_aux_gamma += aux_gamma;
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

// Diagonal covariance version of get_accumulated_second_moment
void
DiagonalStatisticsAccumulator::get_accumulated_second_moment(Vector &second_moment) const
{
  second_moment = m_second_moment;
}


void
DiagonalStatisticsAccumulator::set_accumulated_second_moment(Matrix &second_moment)
{
  for (int i = 0; i < dim(); i++)
    m_second_moment(i) = second_moment(i, i);
}


void
DiagonalStatisticsAccumulator::accumulate(int feacount, double gamma, const Vector &f)
{
  assert( feacount >= 0 );
  //assert( gamma >= 0 );
  m_feacount += feacount;
  m_gamma += gamma;
  m_accumulated = true;
  Blas_Add_Mult(m_mean, gamma, f);
  for (int i=0; i<dim(); i++)
    m_second_moment(i) += gamma * f(i) * f(i);
}


void
DiagonalStatisticsAccumulator::reset()
{
  m_feacount=0;
  m_mean=0;
  m_second_moment=0;
  m_gamma=0;
  m_aux_gamma = 0;
  m_accumulated=false;
}


void
Gaussian::accumulate(double gamma,
                     const Vector &f,
                     int accum_pos)
{
  assert((int)m_accums.size() > accum_pos);
  assert(m_accums[accum_pos] != NULL);
  
  m_accums[accum_pos]->accumulate(1, gamma, f);
}


void
Gaussian::accumulate_aux_gamma(double gamma, int accum_pos)
{
  assert((int)m_accums.size() > accum_pos);
  assert(m_accums[accum_pos] != NULL);
  m_accums[accum_pos]->accumulate_aux_gamma(gamma);
}


void
Gaussian::copy_gamma_to_aux_gamma(int source, int target)
{
  if ((int)m_accums.size() > std::max(source, target) &&
      m_accums[source] != NULL && m_accums[target] != NULL)
    m_accums[target]->set_aux_gamma(m_accums[source]->gamma());
}


void 
Gaussian::dump_statistics(std::ostream &os) const
{
  for (int i = 0; i < (int)m_accums.size(); i++)
  {
    if (m_accums[i] != NULL && m_accums[i]->accumulated()) {
      os.write((char*)&i, sizeof(int));
      m_accums[i]->dump_statistics(os);
    }
  }
  int end = -1;
  os.write((char*)&end, sizeof(int));
}

void 
Gaussian::accumulate_from_dump(std::istream &is, StatisticsMode mode)
{
  int accum_pos;  
  is.read((char*)&accum_pos, sizeof(int));

  if (m_accums.size() == 0)
    start_accumulating(mode);

  while (accum_pos >= 0) {
    if (accum_pos >= (int)m_accums.size() || m_accums[accum_pos] == NULL)
      throw str::fmt(128, "Gaussian::accumulate_from_dump:Invalid accumulator position %i", accum_pos);
    m_accums[accum_pos]->accumulate_from_dump(is);
    is.read((char*)&accum_pos, sizeof(int));
  }
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
Gaussian::ismooth_statistics(int source, int target, double smoothing)
{
  if (!accumulated(source) || !accumulated(target))
  {
    fprintf(stderr, "Warning: Could not perform I-smoothing due to missing statistics!\n");
    return;
  }
  double smooth_factor = smoothing/m_accums[source]->gamma();

  m_accums[target]->set_gamma(m_accums[target]->gamma()+smoothing);
  
  LaVectorDouble mu_source, mu_target;
  m_accums[source]->get_accumulated_mean(mu_source);
  m_accums[target]->get_accumulated_mean(mu_target);
  Blas_Add_Mult(mu_target, smooth_factor, mu_source);
  m_accums[target]->set_accumulated_mean(mu_target);

  LaGenMatDouble sigma_source, sigma_target;
  m_accums[source]->get_accumulated_second_moment(sigma_source);
  m_accums[target]->get_accumulated_second_moment(sigma_target);
  Blas_Add_Mat_Mult(sigma_target, smooth_factor, sigma_source);
  m_accums[target]->set_accumulated_second_moment(sigma_target);
}


void
Gaussian::estimate_parameters(EstimationMode mode, double minvar,
                              double covsmooth, double c1, double c2,
                              double tau, bool ml_stats_target)
{
  if ((mode == ML_EST && !accumulated(ML_BUF)) ||
      (mode == MMI_EST && (!accumulated(ML_BUF) || !accumulated(MMI_BUF))) ||
      ((mode == MPE_EST || mode == MPE_MMI_PRIOR_EST) &&
       (!accumulated(MPE_NUM_BUF) || !accumulated(MPE_DEN_BUF))))
  {
    fprintf(stderr, "Warning: Could not estimate Gaussian parameters due to missing statistics!\n");
    return;
  }

  Vector new_mean;
  Matrix new_covariance;
  int feacount;
  
  if (mode == ML_EST)
  {
    m_accums[ML_BUF]->get_mean_estimate(new_mean);
    m_accums[ML_BUF]->get_covariance_estimate(new_covariance);
    feacount = m_accums[ML_BUF]->feacount();
  }
  else
  {
    int num_buf = ML_BUF, den_buf = MMI_BUF;
    if (mode == MPE_EST || mode == MPE_MMI_PRIOR_EST)
    {
      num_buf = MPE_NUM_BUF;
      den_buf = MPE_DEN_BUF;
    }
    
    Vector old_mean;
    Matrix old_covariance;
    get_mean(old_mean);
    get_covariance(old_covariance);
    
    // c & mu~ & sigma~
    double c = m_accums[num_buf]->gamma() - m_accums[den_buf]->gamma();
    LaVectorDouble mu_tilde;
    LaVectorDouble temp_denominator_mu;
    m_accums[num_buf]->get_accumulated_mean(mu_tilde);
    m_accums[den_buf]->get_accumulated_mean(temp_denominator_mu);
    Blas_Add_Mult(mu_tilde, -1, temp_denominator_mu);

    LaGenMatDouble sigma_tilde;
    LaGenMatDouble temp_denominator_sigma;
    m_accums[num_buf]->get_accumulated_second_moment(sigma_tilde);
    m_accums[den_buf]->get_accumulated_second_moment(temp_denominator_sigma);
    Blas_Add_Mat_Mult(sigma_tilde, -1, temp_denominator_sigma);

    double min_d = 0;
    if (is_diagonal_covariance())
    {
      for (int i = 0; i < m_dim; i++)
      {
        double a2 = old_covariance(i, i);
        double a1 = sigma_tilde(i, i) +
          c*(old_covariance(i, i) + old_mean(i)*old_mean(i)) -
          2*mu_tilde(i)*old_mean(i);
        double a0 = c*sigma_tilde(i, i)-mu_tilde(i)*mu_tilde(i);
        if (a2 == 0)
        {
          assert( a1 > 0 ); // Otherwise this defines a maximum
          min_d = std::max(min_d, -a0/a1);
        }
        else
        {
          double temp = a1*a1 - (4*a0*a2);
          if (temp >= 0) // Otherwise ignore
            min_d = std::max(min_d, (-a1+sqrt(temp))/(2*a2));
        }
      }
    }
    else
    {
      // Full covariance case
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

      // Find a good D
      for (int i=0; i<eigvals.size(); i++)
        if (std::fabs(eigvals(i).i) < 0.000000001)
          min_d=std::max(min_d, eigvals(i).r);
    }

    assert(c2>1);
    double d=std::max(c1*m_accums[den_buf]->gamma() + tau, c2*min_d);

    fprintf(stderr, "num-gamma: %g  den-gamma: %g  d: %g  %s\n",
            m_accums[num_buf]->gamma(),
            m_accums[den_buf]->gamma(), d, (d>c2*min_d?"DEN":"MIN"));
    
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

    feacount = m_accums[num_buf]->feacount();
  }

  // Check that covariances are valid
  if (feacount > 1)
  {
    for (int i=0; i<dim(); i++)
      if (new_covariance(i,i) <= 0)
        fprintf(stderr, "Warning: Variance in dimension %i is %g (%i features)\n", i, new_covariance(i,i), feacount);
  }
  
  // Common tweaking
  for (int i=0; i<dim(); i++)
    if (new_covariance(i,i) < minvar)
      new_covariance(i,i) = minvar;
  
  if (covsmooth != 0)
  {
    for (int i=0; i<dim(); i++)
      for (int j=0; j<dim(); j++)
        if (i != j)
          new_covariance(i,j) *= feacount/(feacount+covsmooth);
  }

  // Set the parameters
  if (ml_stats_target)
  {
    double gamma = m_accums[ML_BUF]->gamma();
    Blas_R1_Update(new_covariance, new_mean, new_mean, 1);
    Blas_Scale(gamma, new_mean);
    Blas_Scale(gamma, new_covariance);
    m_accums[ML_BUF]->set_accumulated_mean(new_mean);
    m_accums[ML_BUF]->set_accumulated_second_moment(new_covariance);
  }
  else
  {
    set_mean(new_mean);
    set_covariance(new_covariance);
  }
}


void
Gaussian::split(Gaussian &g1, Gaussian &g2, double perturbation) const
{
  assert(dim() != 0);

  Vector mean1; get_mean(mean1);
  Vector mean2; get_mean(mean2);
  Matrix cov; get_covariance(cov);
  // After splitting new covariance matrix
  Matrix cov1; get_covariance(cov1);
  Matrix cov2; get_covariance(cov2);
  double cnst_u = perturbation;
  double cnst_beta = 0.5; 
  double cnst_alfa = 0.5;

  Matrix cholesky;
  Vector a_l; // Vector from cholesky
  Vector test;

  LinearAlgebra::cholesky_factor(cov, cholesky);  
  a_l.ref((cholesky.col(0))); // ref: Matrix --> Vector
  test.ref(cov.col(0)); // ref muuttaa Matrix --> Vector
  test(0) = sqrt(test(0));

  // Vector
  /* mean1 = mean - sqrt((1-alfa)/alfa) * u * a_l
   * mean2 = mean + sqrt(alfa/(1-alfa)) * u * a_l */

  /* Separates one column from cholesky. Let's pick the first one.
   * Points to original memoryslot */
  

  double cnst_a = -1 * sqrt((1-cnst_alfa)/cnst_alfa) * cnst_u;
  double cnst_b = sqrt(cnst_alfa/(1-cnst_alfa)) * cnst_u;

  Blas_Add_Mult(mean1, cnst_a, a_l);
  Blas_Add_Mult(mean2, cnst_b, a_l);
  
  // Matrix
  // (1-alfa)/alfa * cov + (Beta-Beta*u² - 1)*(1/alfa)*a_l*(a_l)^T + a_l*(a_l)^T

  double cnst_c = (cnst_beta - cnst_beta*cnst_u*cnst_u - 1)*1/cnst_alfa;
  Blas_Scale((1-cnst_alfa)/cnst_alfa, cov1); // Scaling matrix (1-alfa)/alfa
  Blas_R1_Update(cov1, a_l, a_l, cnst_c); // cov1 + cnst_c*a_l*a_l'
  Blas_R1_Update(cov1, a_l, a_l, 1); // ... + a_l*a_l'

  // alfa/(1-alfa)*cov + (Beta*u² - Beta - u²)*1/(1-alfa)*a_l*a_l' + a_l*a_l'
  double cnst_d = (cnst_beta*cnst_u*cnst_u - cnst_beta - cnst_u*cnst_u)*1/(1-cnst_alfa);
  Blas_Scale(cnst_alfa/(1-cnst_alfa), cov2); // Scaling matrix alfa/(1-alfa)
  Blas_R1_Update(cov2, a_l, a_l, cnst_d); // cov2 + cnst_d*a_l*a_l'
  Blas_R1_Update(cov2, a_l, a_l, 1); // ... + a_l*a_l'


  g1.set_mean(mean1);
  g2.set_mean(mean2);
  g1.set_covariance(cov1);
  g2.set_covariance(cov2);

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
  //Matrix eye = Matrix::eye(m_dim, m_dim);
  double weight_sum = 0;
  new_mean = 0;
  double even_weight = 1.0/(double)weights.size();
  bool use_even_weight = false;

  for (int i = 0; i < (int)weights.size(); i++)
    weight_sum += weights[i];
  if (weight_sum < 1e-15)
  {
    use_even_weight = true;
    weight_sum = 1;
  }
  
  for (int i = 0; i < (int)weights.size(); i++)
  {
    Matrix cur_covariance;
    Vector cur_mean;
    double cur_weight = (use_even_weight?even_weight:weights[i]);
    gaussians[i]->get_covariance(cur_covariance);
    gaussians[i]->get_mean(cur_mean);
    Blas_R1_Update(cur_covariance, cur_mean, cur_mean);
    //Blas_Mat_Mat_Mult(cur_covariance, eye, new_covariance, cur_weight, 1.0);
    Blas_Add_Mat_Mult(new_covariance, cur_weight, cur_covariance);
    Blas_Add_Mult(new_mean, cur_weight, cur_mean);
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

  // Values for the other Gaussian
  LaVectorLongInt pivots(dim(),1);
  LaGenMatDouble other_covariance; g.get_covariance(other_covariance);
  LaGenMatDouble other_precision; g.get_covariance(other_precision);
  LaVectorDouble other_mean; g.get_mean(other_mean);
  LUFactorizeIP(other_precision, pivots);
  LaLUInverseIP(other_precision, pivots);
  
  // This Gaussian
  LaGenMatDouble cov; get_covariance(cov);
  LaVectorDouble mean; get_mean(mean);

  // Temporary modifiers
  LaGenMatDouble temp_matrix(dim(), dim());
  LaVectorDouble diff; g.get_mean(diff);
  LaVectorDouble temp_vector; g.get_mean(temp_vector);

  // Precomputations
  Blas_Mat_Mat_Mult(other_precision, cov, temp_matrix, 1.0, 0.0);
  Blas_Add_Mult(diff, -1, mean); 
  Blas_Mat_Vec_Mult(other_precision, diff, temp_vector);

  // Actual KL-divergence
  double value=LinearAlgebra::spd_determinant(other_covariance)
    /LinearAlgebra::spd_determinant(cov);
  value = log(value);
  value += temp_matrix.trace();
  value += Blas_Dot_Prod(diff, temp_vector);
  value -= dim();
  value *= 0.5;

  return value;
}


void
Gaussian::draw_sample(Vector &sample)
{
  get_mean(sample);
  if (chol == NULL)
  {
    Matrix cov; get_covariance(cov);
    chol = new Matrix(dim(), dim());
    LinearAlgebra::cholesky_factor(cov, *chol);
  }
  Vector ziggies(dim());
  for (int i=0; i<dim(); i++)
    ziggies(i) = ziggurat::rnd.rnor();
  Blas_Mat_Vec_Mult(*chol, ziggies, sample, 1.0, 1.0);
}


bool
Gaussian::full_stats_accumulated(int accum_pos)
{
  if ((int)m_accums.size() <= accum_pos || m_accums[accum_pos] == NULL)
    return false;
  return m_accums[accum_pos]->full_stats_accumulated();
}


void
Gaussian::set_covariance(const Vector &covariance,
                         bool finish_statistics)
{
  Matrix cov(covariance.size(), covariance.size());
  cov=0;
  for (int i=0; i<covariance.size(); i++)
    cov(i,i)=covariance(i);
  set_covariance(cov, finish_statistics);
  if (chol != NULL)
  {
    delete chol;
    chol = NULL;
  }
}


void
Gaussian::get_covariance(Vector &covariance) const
{
  Matrix cov;
  get_covariance(cov);
  covariance.resize(cov.rows());
  for (int i=0; i<covariance.size(); i++)
    covariance(i)=cov(i,i);
}



DiagonalGaussian::DiagonalGaussian(int dim)
{
  reset(dim);
}


DiagonalGaussian::DiagonalGaussian(const DiagonalGaussian &g)
{
  reset(g.dim());
  g.get_mean(m_mean);
  m_covariance.copy(g.m_covariance);
  m_precision.copy(g.m_precision);
  m_constant = g.m_constant;
  m_valid_parameters = g.m_valid_parameters;
}


DiagonalGaussian::~DiagonalGaussian()
{
  stop_accumulating();
}


void
DiagonalGaussian::reset(int dim)
{
  m_dim=dim;
  
  m_mean.resize(dim);
  m_covariance.resize(dim);
  m_precision.resize(dim);
  m_mean=0; m_covariance=0; m_precision=0;

  m_constant=0;
  m_valid_parameters = false;
}


double
DiagonalGaussian::compute_likelihood(const Vector &f) const
{
  return exp(compute_log_likelihood(f));
}


double
DiagonalGaussian::compute_log_likelihood(const Vector &f) const
{
  double ll=0;

// Seems to be faster without BLAS, old implementation commented out
  
//   Vector diff(f);
//   Blas_Add_Mult(diff, -1, m_mean);
//   for (int i=0; i<dim(); i++)
//     ll += diff(i)*diff(i)*m_precision(i);
  
  for (int i=0; i<dim(); i++)
  {
    double d=f(i)-m_mean(i);
    ll += d*d*m_precision(i);
  }
  
  ll *= -0.5;
  ll += m_constant;

  return ll;
}


double
DiagonalGaussian::compute_likelihood_exponential(const Vector &exponential_feature) const
{
  throw std::string("compute_likelihood_exponential not implemented for DiagonalGaussian\n");
}


double
DiagonalGaussian::compute_log_likelihood_exponential(const Vector &exponential_feature) const
{
  throw std::string("compute_log_likelihood_exponential not implemented for DiagonalGaussian\n");
}


double
DiagonalGaussian::kullback_leibler(Gaussian &g) const
{
  assert(dim()==g.dim());

  // Values for the other Gaussian
  LaVectorDouble other_mean; g.get_mean(other_mean);
  LaVectorDouble other_covariance; g.get_covariance(other_covariance);

  double det=1;
  for (int i=0; i<dim(); i++)
    det *= m_covariance(i);

  double other_det=1;
  for (int i=0; i<dim(); i++)
    other_det *= other_covariance(i);

  Vector diff(other_mean);
  Blas_Add_Mult(diff, -1.0, m_mean);
  
  // kl
  double value = log(other_det/det);
  for (int i=0; i<dim(); i++)
    value += (m_covariance(i)+diff(i))/other_covariance(i);
  value -= dim();
  value *= 0.5;

  return value;
}


void
DiagonalGaussian::draw_sample(Vector &sample)
{
  get_mean(sample);
  for (int i=0; i<dim(); i++)
    sample(i) += sqrt(m_covariance(i)) * ziggurat::rnd.rnor();
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
  {
    is >> m_mean(i);
    if (is.fail())
      throw std::string("Error in reading Gaussian specifications");
  }
  for (int i=0; i<dim(); i++) {
    is >> m_covariance(i);
    if (is.fail())
      throw std::string("Error in reading Gaussian specifications");
    if (m_covariance(i) > 0)
      m_precision(i) = 1/m_covariance(i);
    else
      m_precision(i) = 0;
  }
  set_constant();
}


void
DiagonalGaussian::start_accumulating(StatisticsMode mode)
{
  if (mode & PDF_ML_STATS && !(mode & PDF_ML_FULL_STATS)) {
    m_accums.resize(ML_BUF+1, NULL);
    m_accums[ML_BUF] = new DiagonalStatisticsAccumulator(dim());
  }
  else if (mode & PDF_ML_FULL_STATS) {
    m_accums.resize(ML_BUF+1, NULL);
    m_accums[ML_BUF] = new FullStatisticsAccumulator(dim());
  }
  
  if (mode & PDF_MMI_STATS) {
    m_accums.resize(MMI_BUF+1, NULL);
    if (m_accums[ML_BUF] == NULL)
      m_accums[ML_BUF] = new DiagonalStatisticsAccumulator(dim());
    m_accums[MMI_BUF] = new DiagonalStatisticsAccumulator(dim());
  }
  if (mode & PDF_MPE_NUM_STATS) {
    m_accums.resize(MPE_NUM_BUF+1, NULL);
    m_accums[MPE_NUM_BUF] = new DiagonalStatisticsAccumulator(dim());
  }
  if (mode & PDF_MPE_DEN_STATS) {
    m_accums.resize(MPE_DEN_BUF+1, NULL);
    m_accums[MPE_DEN_BUF] = new DiagonalStatisticsAccumulator(dim());
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

  m_valid_parameters = false;
  if (finish_statistics)
    m_covariance_determinant=1;
  for (int i=0; i<dim(); i++) {
    assert(covariance(i,i)>0);
    m_covariance(i) = covariance(i,i);
    if (finish_statistics && m_covariance(i) > 0) {
      m_precision(i) = 1/m_covariance(i);
      m_covariance_determinant *= m_covariance(i);
    }
    else
      m_precision(i) = 0;
  }
  if (finish_statistics)
    set_constant();
  if (chol != NULL)
  {
    delete chol;
    chol = NULL;
  }
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

  m_valid_parameters = false;
  if (finish_statistics)
    m_covariance_determinant=1;
  for (int i=0; i<dim(); i++) {
    m_covariance(i) = covariance(i);
    if (finish_statistics && m_covariance(i) > 0) {
      m_precision(i) = 1/m_covariance(i);
      m_covariance_determinant *= m_covariance(i);
    }
    else
      m_precision(i) = 0;
  }
  if (finish_statistics)
    set_constant();
  if (chol != NULL)
  {
    delete chol;
    chol = NULL;
  }
}


void
DiagonalGaussian::set_constant(void)
{
  m_constant=1;
  for (int i=0; i<dim(); i++)
    m_constant *= m_precision(i);
  if (m_constant > 0)
  {
    m_constant = log(sqrt(m_constant));
    m_valid_parameters = true;
  }
  else
  {
    m_valid_parameters = false;
  }
}


void
DiagonalGaussian::split(Gaussian &g1, Gaussian &g2, double perturbation) const
{
  assert(dim() != 0);

  Vector mean1; get_mean(mean1);
  Vector mean2; get_mean(mean2);
  Matrix cov; get_covariance(cov);
  // After splitting new covariance matrix
  Matrix cov1; get_covariance(cov1);
  Matrix cov2; get_covariance(cov2);
  double cnst_u = perturbation;
  double cnst_beta = 0.5; 
  double cnst_alfa = 0.5;

  Vector a_l; // Vector from A, Cov = A*A'

  a_l.ref(cov.col(0)); // Matrix --> Vector
  a_l(0) = sqrt(a_l(0));


  // Vector
  /* mean1 = mean - sqrt((1-alfa)/alfa) * u * a_l
   * mean2 = mean + sqrt(alfa/(1-alfa)) * u * a_l */  

  double cnst_a = -1 * sqrt((1-cnst_alfa)/cnst_alfa) * cnst_u;
  double cnst_b = sqrt(cnst_alfa/(1-cnst_alfa)) * cnst_u;

  Blas_Add_Mult(mean1, cnst_a, a_l);
  Blas_Add_Mult(mean2, cnst_b, a_l);
  
  // Matrix
  // (1-alfa)/alfa * cov + (Beta-Beta*u² - 1)*(1/alfa)*a_l*(a_l)^T + a_l*(a_l)^T

  double cnst_c = (cnst_beta - cnst_beta*cnst_u*cnst_u - 1)*1/cnst_alfa;
  Blas_Scale((1-cnst_alfa)/cnst_alfa, cov1); // Scaling matrix (1-alfa)/alfa
  Blas_R1_Update(cov1, a_l, a_l, cnst_c); // cov1 + cnst_c*a_l*a_l'
  Blas_R1_Update(cov1, a_l, a_l, 1); // ... + a_l*a_l'

  // alfa/(1-alfa)*cov + (Beta*u² - Beta - u²)*1/(1-alfa)*a_l*a_l' + a_l*a_l'
  double cnst_d = (cnst_beta*cnst_u*cnst_u - cnst_beta - cnst_u*cnst_u)*1/(1-cnst_alfa);
  Blas_Scale(cnst_alfa/(1-cnst_alfa), cov2); // Scaling matrix alfa/(1-alfa)
  Blas_R1_Update(cov2, a_l, a_l, cnst_d); // cov2 + cnst_d*a_l*a_l'
  Blas_R1_Update(cov2, a_l, a_l, 1); // ... + a_l*a_l'


  g1.set_mean(mean1);
  g2.set_mean(mean2);
  g1.set_covariance(cov1);
  g2.set_covariance(cov2);
}


FullCovarianceGaussian::FullCovarianceGaussian(int dim)
{
  reset(dim);
}


FullCovarianceGaussian::FullCovarianceGaussian(const FullCovarianceGaussian &g)
{
  reset(g.dim());
  g.get_mean(m_mean);
  m_covariance.copy(g.m_covariance);
  m_precision.copy(g.m_precision);
  m_exponential_parameters.copy(g.m_exponential_parameters);
  m_constant = g.m_constant;
  m_exponential_normalizer = g.m_exponential_normalizer;
  m_valid_parameters = g.m_valid_parameters;
}


FullCovarianceGaussian::~FullCovarianceGaussian()
{
  stop_accumulating();
}


void
FullCovarianceGaussian::reset(int dim)
{
  m_dim=dim;
  
  m_mean.resize(dim);
  m_precision.resize(dim,dim);
  m_mean=0; m_precision=0; m_covariance=0;

  m_constant=0;
  m_valid_parameters=false;
}


double
FullCovarianceGaussian::compute_likelihood(const Vector &f) const
{
  return exp(compute_log_likelihood(f));
}


double
FullCovarianceGaussian::compute_log_likelihood(const Vector &f) const
{
  double ll;

  Vector diff(f);
  Vector t(f.size());

  Blas_Add_Mult(diff, -1, m_mean);
  Blas_Mat_Vec_Mult(m_precision, diff, t, 1, 0);
  ll = -0.5 * Blas_Dot_Prod(diff, t);
  ll += m_constant;

  return ll;
}


double
FullCovarianceGaussian::compute_likelihood_exponential(const Vector &exponential_feature) const
{
  return exp(compute_log_likelihood_exponential(exponential_feature));
}


double
FullCovarianceGaussian::compute_log_likelihood_exponential(const Vector &exponential_feature) const
{
  double ll;

  ll = Blas_Dot_Prod(exponential_feature, m_exponential_parameters);
  ll += m_exponential_normalizer;
  ll += m_constant;

  return ll;
}


void
FullCovarianceGaussian::write(std::ostream &os) const
{
  Matrix covariance;
  get_covariance(covariance);
  
  os << "full ";
  for (int i=0; i<dim(); i++)
    os << m_mean(i) << " ";
  for (int i=0; i<dim(); i++)
    for (int j=0; j<dim(); j++)
      if (!(i==dim()-1 && j==dim()-1))
	os << covariance(i,j) << " ";
  os << covariance(dim()-1, dim()-1);
}


void
FullCovarianceGaussian::read(std::istream &is)
{
  Matrix covariance(dim(), dim());
  
  for (int i=0; i<dim(); i++)
  {
    is >> m_mean(i);
    if (is.fail())
      throw std::string("Error in reading Gaussian specifications");
  }
  for (int i=0; i<dim(); i++)
  {
    for (int j=0; j<dim(); j++)
    {
      is >> covariance(i,j);
      if (is.fail())
        throw std::string("Error in reading Gaussian specifications");
    }
  }
  
  set_covariance(covariance);
}


void
FullCovarianceGaussian::start_accumulating(StatisticsMode mode)
{
  if (mode & (PDF_ML_STATS|PDF_ML_FULL_STATS)) {
    m_accums.resize(ML_BUF+1, NULL);
    m_accums[ML_BUF] = new FullStatisticsAccumulator(dim());
  }
  if (mode & PDF_MMI_STATS) {
    m_accums.resize(MMI_BUF+1, NULL);
    if (m_accums[ML_BUF] == NULL)
      m_accums[ML_BUF] = new FullStatisticsAccumulator(dim());
    m_accums[MMI_BUF] = new FullStatisticsAccumulator(dim());
  }
  if (mode & PDF_MPE_NUM_STATS) {
    m_accums.resize(MPE_NUM_BUF+1, NULL);
    m_accums[MPE_NUM_BUF] = new FullStatisticsAccumulator(dim());
  }
  if (mode & PDF_MPE_DEN_STATS) {
    m_accums.resize(MPE_DEN_BUF+1, NULL);
    m_accums[MPE_DEN_BUF] = new FullStatisticsAccumulator(dim());
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
FullCovarianceGaussian::recompute_exponential_parameters()
{
  // Constant
  Vector transformed_mean(m_mean.size());  
  Blas_Mat_Vec_Mult(m_precision, m_mean, transformed_mean);
  m_exponential_normalizer = -0.5 * Blas_Dot_Prod(transformed_mean, m_mean);

  // Linear part
  m_exponential_parameters.resize((int)(dim()*(dim()+3)/2));
  for (int i=0; i<dim(); i++)
    m_exponential_parameters(i) = transformed_mean(i);

  // Quadratic part
  Vector vectorized_precision;
  LinearAlgebra::map_m2v(m_precision, vectorized_precision);
  for (int i=0; i<vectorized_precision.size(); i++)
    m_exponential_parameters(dim()+i) = -0.5 * vectorized_precision(i);
}


void
FullCovarianceGaussian::set_mean(const Vector &mean)
{
  assert(mean.size()==dim());
  m_mean.copy(mean);
  recompute_exponential_parameters();
}


void
FullCovarianceGaussian::set_covariance(const Matrix &covariance,
                                       bool finish_statistics)
{
  assert(covariance.rows()==dim());
  assert(covariance.cols()==dim());

  m_covariance.copy(covariance); // Maintain the original covariance
  if (finish_statistics && LinearAlgebra::is_spd(covariance))
  {
    LinearAlgebra::inverse(covariance, m_precision);
    m_constant = log(sqrt(LinearAlgebra::spd_determinant(m_precision)));
    m_covariance_determinant = LinearAlgebra::spd_determinant(m_covariance);
    recompute_exponential_parameters();
    m_valid_parameters = true;
  }
  else
  {
    m_precision = 0;
    m_constant = 0;
    m_valid_parameters=false;
  }
  if (chol != NULL)
  {
    delete chol;
    chol = NULL;
  }
}


#ifdef USE_SUBSPACE_COV
PrecisionConstrainedGaussian::PrecisionConstrainedGaussian()
{
}


PrecisionConstrainedGaussian::PrecisionConstrainedGaussian(PrecisionSubspace *space)
{
  m_ps = space;
  m_coeffs.resize(space->subspace_dim());
  reset(space->feature_dim());
}

PrecisionConstrainedGaussian::PrecisionConstrainedGaussian(const PrecisionConstrainedGaussian &g)
{
  m_transformed_mean.copy(g.m_transformed_mean);
  m_coeffs.copy(g.m_coeffs);
  m_ps = g.get_subspace();
  m_constant = g.m_constant;
  m_valid_parameters = g.m_valid_parameters;
}


PrecisionConstrainedGaussian::~PrecisionConstrainedGaussian()
{
}


void
PrecisionConstrainedGaussian::reset(int feature_dim)
{
  m_dim=feature_dim;
  
  m_transformed_mean.resize(feature_dim);
  m_transformed_mean=0;
  m_coeffs=0; m_coeffs(0)=1;
  
  m_constant=0;
  m_valid_parameters = false;
}


double
PrecisionConstrainedGaussian::compute_likelihood(const Vector &f) const
{
  return exp(compute_log_likelihood(f));
}


double
PrecisionConstrainedGaussian::compute_log_likelihood(const Vector &f) const
{
  if (!m_ps->computed())
    m_ps->precompute(f);
  
  double result=m_constant
    +Blas_Dot_Prod(m_transformed_mean, f);
    +m_ps->dotproduct(m_coeffs);
  return result;
}


double
PrecisionConstrainedGaussian::compute_log_likelihood_exponential(const Vector &exponential_feature) const
{
  throw std::string("compute_log_likelihood_exponential not implemented for PrecisionConstrainedGaussian\n");
}


double
PrecisionConstrainedGaussian::compute_likelihood_exponential(const Vector &exponential_feature) const
{
  throw std::string("compute_likelihood_exponential not implemented for PrecisionConstrainedGaussian\n");
}


void
PrecisionConstrainedGaussian::write(std::ostream &os) const
{
  os << " " << subspace_dim() << " ";
  for (int i=0; i<dim(); i++)
    os << m_transformed_mean(i) << " ";
  for (int i=0; i<subspace_dim()-1; i++)
    os << m_coeffs(i) << " ";
  os << m_coeffs(subspace_dim()-1);  
}


void
PrecisionConstrainedGaussian::read(std::istream &is)
{
  int ss_dim;

  is >> ss_dim;  
  m_coeffs.resize(ss_dim);
  for (int i=0; i<dim(); i++)
  {
    is >> m_transformed_mean(i);
    if (is.fail())
      throw std::string("Error in reading Gaussian specifications");
  }
  for (int i=0; i<subspace_dim(); i++)
  {
    is >> m_coeffs(i);
    if (is.fail())
      throw std::string("Error in reading Gaussian specifications");
  }

  recompute_constant();
}


void
PrecisionConstrainedGaussian::start_accumulating(StatisticsMode mode)
{
  if (mode & (PDF_ML_STATS|PDF_ML_FULL_STATS)) {
    m_accums.resize(ML_BUF+1, NULL);
    m_accums[ML_BUF] = new FullStatisticsAccumulator(dim());
  }
  if (mode & PDF_MMI_STATS) {
    m_accums.resize(MMI_BUF+1, NULL);
    if (m_accums[ML_BUF] == NULL)
      m_accums[ML_BUF] = new FullStatisticsAccumulator(dim());
    m_accums[MMI_BUF] = new FullStatisticsAccumulator(dim());
  }
  if (mode & PDF_MPE_NUM_STATS) {
    m_accums.resize(MPE_NUM_BUF+1, NULL);
    m_accums[MPE_NUM_BUF] = new FullStatisticsAccumulator(dim());
  }
  if (mode & PDF_MPE_DEN_STATS) {
    m_accums.resize(MPE_DEN_BUF+1, NULL);
    m_accums[MPE_DEN_BUF] = new FullStatisticsAccumulator(dim());
  }
}


void
PrecisionConstrainedGaussian::get_mean(Vector &mean) const
{
  mean.resize(dim());
  Matrix covariance;
  m_ps->compute_covariance(m_coeffs, covariance);
  Blas_Mat_Vec_Mult(covariance, m_transformed_mean, mean);
}


void
PrecisionConstrainedGaussian::get_covariance(Matrix &covariance) const
{
  m_ps->compute_covariance(m_coeffs, covariance);
}


void
PrecisionConstrainedGaussian::set_mean(const Vector &mean)
{
  Matrix precision;
  m_ps->compute_precision(m_coeffs, precision);
  Blas_Mat_Vec_Mult(precision, mean, m_transformed_mean);

  // Recompute the constant
  recompute_constant();
}


void
PrecisionConstrainedGaussian::set_covariance(const Matrix &covariance,
                                             bool finish_statistics)
{
  // Save the old covariance
  Matrix old_covariance;
  m_ps->compute_covariance(m_coeffs, old_covariance);
  
  // Optimize
  m_ps->optimize_coefficients(covariance, m_coeffs);

  // Compute the new precision
  Matrix precision;
  m_ps->compute_precision(m_coeffs, precision);

  // Retransform the mean parameters
  Vector mean(m_transformed_mean);
  Blas_Mat_Vec_Mult(old_covariance, m_transformed_mean, mean);
  Blas_Mat_Vec_Mult(precision, mean, m_transformed_mean);

  // Recompute the constant
  recompute_constant();

  if (chol != NULL)
  {
    delete chol;
    chol = NULL;
  }
}


void
PrecisionConstrainedGaussian::recompute_constant()
{
  Matrix t;
  m_ps->compute_precision(m_coeffs, t);
  m_constant = log(sqrt(LinearAlgebra::spd_determinant(t)));
  m_covariance_determinant = 1/LinearAlgebra::spd_determinant(t);
  
  Vector mean(dim());

  LaVectorLongInt pivots(dim());
  LUFactorizeIP(t, pivots);
  LaLUInverseIP(t, pivots);
  
  Blas_Mat_Vec_Mult(t, m_transformed_mean, mean);
  m_constant += -0.5 * Blas_Dot_Prod(m_transformed_mean, mean);
  m_valid_parameters = true;
}


SubspaceConstrainedGaussian::SubspaceConstrainedGaussian()
{
}


SubspaceConstrainedGaussian::SubspaceConstrainedGaussian(ExponentialSubspace *space)
{
  m_es = space;
  m_coeffs.resize(space->subspace_dim());
  reset(space->feature_dim());  
}


SubspaceConstrainedGaussian::SubspaceConstrainedGaussian(const SubspaceConstrainedGaussian &g)
{
  g.get_subspace_coeffs(m_coeffs);
  m_es = g.get_subspace();
  m_valid_parameters = g.m_valid_parameters;
}


SubspaceConstrainedGaussian::~SubspaceConstrainedGaussian()
{
}


void
SubspaceConstrainedGaussian::reset(int feature_dim)
{
  m_dim=feature_dim;
  
  m_coeffs=0;
  m_coeffs(0)=1;
  
  m_constant=0;
  m_valid_parameters = false;
}


double
SubspaceConstrainedGaussian::compute_likelihood(const Vector &f) const
{
  return exp(compute_log_likelihood(f));
}


double
SubspaceConstrainedGaussian::compute_log_likelihood(const Vector &f) const
{
  if (!m_es->computed())
    m_es->precompute(f);
  
  double result=m_constant+m_es->dotproduct(m_coeffs);
  return result;
}


double
SubspaceConstrainedGaussian::compute_log_likelihood_exponential(const Vector &exponential_feature) const
{
  throw std::string("compute_log_likelihood_exponential not implemented for SubspaceConstrainedGaussian\n");
}


double
SubspaceConstrainedGaussian::compute_likelihood_exponential(const Vector &exponential_feature) const
{
  throw std::string("compute_likelihood_exponential not implemented for SubspaceConstrainedGaussian\n");
}


void
SubspaceConstrainedGaussian::write(std::ostream &os) const
{
  os << " " << subspace_dim() << " ";
  for (int i=0; i<m_coeffs.size()-1; i++)
    os << m_coeffs(i) << " ";
  os << m_coeffs(m_coeffs.size()-1);
}


void
SubspaceConstrainedGaussian::read(std::istream &is)
{
  int ss_dim;

  is >> ss_dim;
  m_coeffs.resize(ss_dim);
  for (int i=0; i<m_coeffs.size()-1; i++)
  {
    is >> m_coeffs(i);
    if (is.fail())
      throw std::string("Error in reading Gaussian specifications");
  }
  is >> m_coeffs(m_coeffs.size()-1);

  Vector psi;
  Matrix precision;
  Matrix covariance;
  
  m_es->compute_psi(m_coeffs, psi);
  m_es->compute_precision(m_coeffs, precision);
  LinearAlgebra::inverse(precision, covariance);
  m_constant = LinearAlgebra::determinant(precision);
  m_constant = log(m_constant);

  Vector t(psi);
  Blas_Mat_Vec_Mult(covariance, psi, t);
  m_constant -= Blas_Dot_Prod(psi, t);
  m_constant -= m_es->feature_dim() * log(2*3.1416);
  m_valid_parameters = true;
}


void
SubspaceConstrainedGaussian::start_accumulating(StatisticsMode mode)
{
  if (mode & (PDF_ML_STATS|PDF_ML_FULL_STATS)) {
    m_accums.resize(ML_BUF+1, NULL);
    m_accums[ML_BUF] = new FullStatisticsAccumulator(dim());
  }
  if (mode & PDF_MMI_STATS) {
    m_accums.resize(MMI_BUF+1, NULL);
    if (m_accums[ML_BUF] == NULL)
      m_accums[ML_BUF] = new FullStatisticsAccumulator(dim());
    m_accums[MMI_BUF] = new FullStatisticsAccumulator(dim());
  }
  if (mode & PDF_MPE_NUM_STATS) {
    m_accums.resize(MPE_NUM_BUF+1, NULL);
    m_accums[MPE_NUM_BUF] = new FullStatisticsAccumulator(dim());
  }
  if (mode & PDF_MPE_DEN_STATS) {
    m_accums.resize(MPE_DEN_BUF+1, NULL);
    m_accums[MPE_DEN_BUF] = new FullStatisticsAccumulator(dim());
  }
}


void
SubspaceConstrainedGaussian::get_mean(Vector &mean) const
{
  m_es->compute_mu(m_coeffs, mean);
}


void
SubspaceConstrainedGaussian::get_covariance(Matrix &covariance) const
{
  m_es->compute_covariance(m_coeffs, covariance);
}


void
SubspaceConstrainedGaussian::set_mean(const Vector &mean)
{
  std::cout << "Warning: don't use set_mean() for Scgmms, use set_parameters() instead" << std::endl;
}


void
SubspaceConstrainedGaussian::set_covariance(const Matrix &covariance,
                                            bool finish_statistics)
{
  std::cout << "Warning: don't use set_covariance() for Scgmms, use set_parameters instead" << std::endl;
}


void
SubspaceConstrainedGaussian::set_parameters(const Vector &mean, const Matrix &covariance)
{
  m_es->optimize_coefficients(mean, covariance, m_coeffs);
  m_valid_parameters = true;

  if (chol != NULL)
  {
    delete chol;
    chol = NULL;
  }
}
#endif


Mixture::Mixture()
{
  m_dim = 0;
}


Mixture::Mixture(PDFPool *pool)
{
  m_pool = pool;
  m_dim = 0; // Use m_pool to figure out the dimension
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
Mixture::compute_likelihood(const Vector &f) const
{
  double l = 0;
  for (unsigned int i=0; i< m_pointers.size(); i++) {
    l += m_weights[i]*m_pool->compute_likelihood(f, m_pointers[i]);
  }
  return l;
}


double
Mixture::compute_log_likelihood(const Vector &f) const
{
  double ll = 0;
  for (unsigned int i=0; i< m_pointers.size(); i++) {
    ll += m_weights[i]*m_pool->compute_likelihood(f, m_pointers[i]);
  }
  return util::safe_log(ll);
}


void
Mixture::start_accumulating(StatisticsMode mode)
{
  if (mode & (PDF_ML_STATS|PDF_ML_FULL_STATS)) {
    m_accums.resize(ML_BUF+1, NULL);
    m_accums[ML_BUF] = new MixtureAccumulator(size());
  }
  if (mode & PDF_MMI_STATS) {
    m_accums.resize(MMI_BUF+1, NULL);
    if (m_accums[ML_BUF] == NULL)
      m_accums[ML_BUF] = new MixtureAccumulator(size());
    m_accums[MMI_BUF] = new MixtureAccumulator(size());
  }
  if (mode & PDF_MPE_NUM_STATS) {
    m_accums.resize(MPE_NUM_BUF+1, NULL);
    m_accums[MPE_NUM_BUF] = new MixtureAccumulator(size());
  }
  if (mode & PDF_MPE_DEN_STATS) {
    m_accums.resize(MPE_DEN_BUF+1, NULL);
    m_accums[MPE_DEN_BUF] = new MixtureAccumulator(size());
  }

  if (mode != 0)
  {
    for (int i=0; i<size(); i++)
    {
      if (!get_base_pdf(i)->is_accumulating())
        get_base_pdf(i)->start_accumulating(mode);
    }
  }
}


void
Mixture::accumulate(double gamma,
		    const Vector &f,
		    int accum_pos)
{
  double total_likelihood, this_likelihood, weighted_likelihood;
  double this_gamma;

  // Compute the total likelihood for this mixture
  total_likelihood = compute_likelihood(f);

  m_accums[accum_pos]->mixture_ll += gamma*util::safe_log(total_likelihood);
  
  // Accumulate all basis distributions with some gamma
  if (total_likelihood > 0) {
    for (int i=0; i<size(); i++) {
      this_likelihood = m_pool->compute_likelihood(f, m_pointers[i]);
      weighted_likelihood = gamma * m_weights[i] * this_likelihood;
      this_gamma = weighted_likelihood / total_likelihood;
      
      m_accums[accum_pos]->gamma[i] += this_gamma;
      get_base_pdf(i)->accumulate(this_gamma, f, accum_pos);
//       get_base_pdf(i)->accumulate_aux_gamma(
//         this_gamma*util::safe_log(this_likelihood), accum_pos);
      get_base_pdf(i)->accumulate_aux_gamma(fabs(this_gamma), accum_pos);
    }
    m_accums[accum_pos]->accumulated = true;
  }
}


void
Mixture::accumulate_aux_gamma(double gamma, int accum_pos)
{
  m_accums[accum_pos]->aux_gamma += gamma;
}


void
Mixture::copy_aux_gamma(int source, int target)
{
  if ((int)m_accums.size() > std::max(source, target) &&
      m_accums[source] != NULL && m_accums[target] != NULL)
    m_accums[target]->aux_gamma = m_accums[source]->aux_gamma;
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
Mixture::dump_statistics(std::ostream &os) const
{
  os.precision(10);
  for (int a = 0; a < (int)m_accums.size(); a++)
  {
    if (m_accums[a] != NULL && m_accums[a]->accumulated)
    {
      os << a << " " << size();

      for (int i = 0; i < size(); i++)
        os << " " << m_pointers[i] << " " << m_accums[a]->gamma[i];
      os << " " << m_accums[a]->aux_gamma << " " << m_accums[a]->mixture_ll;
      os << std::endl;
    }
  }
  os << "-1" << std::endl;
}


void 
Mixture::accumulate_from_dump(std::istream &is, StatisticsMode mode)
{
  int accum_pos;
  is >> accum_pos;

  if (m_accums.size() == 0)
    start_accumulating(mode);

  while (accum_pos >= 0)
  {
    if (accum_pos >= (int)m_accums.size() || m_accums[accum_pos] == NULL)
      throw str::fmt(128, "Mixture::accumulate_from_dump: Invalid accumulator position %i", accum_pos);

    int pointer, sz;
    double acc;
    is >> sz;
    assert(sz==size());

    for (int i=0; i<size(); i++) {
      is >> pointer >> acc;    
      assert(m_pointers[i] == pointer);
      m_accums[accum_pos]->gamma[i] += acc;
    }

    double aux, ll;
    is >> aux >> ll;
    m_accums[accum_pos]->aux_gamma += aux;
    m_accums[accum_pos]->mixture_ll += ll;

    m_accums[accum_pos]->accumulated = true;
    
    is >> accum_pos;
  }
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
Mixture::estimate_parameters(EstimationMode mode)
{
  if ((mode == ML_EST && !accumulated(ML_BUF)) ||
      (mode == MMI_EST && (!accumulated(ML_BUF) || !accumulated(MMI_BUF))) ||
      ((mode == MPE_EST || mode == MPE_MMI_PRIOR_EST) &&
       (!accumulated(MPE_NUM_BUF) || !accumulated(MPE_DEN_BUF))))
  {
    fprintf(stderr, "Warning: Could not estimate mixture parameters due to missing statistics!\n");
    return;
  }
  
  if (mode == ML_EST) {    
    double total_gamma = 0;
    for (int i=0; i<size(); i++)
      total_gamma += m_accums[ML_BUF]->gamma[i];    
    for (int i=0; i<size(); i++)
      m_weights[i] = m_accums[ML_BUF]->gamma[i]/total_gamma;
  }
  else if (mode == MMI_EST || mode == MPE_EST || mode == MPE_MMI_PRIOR_EST)
  {
    // There are many alternatives for updating mixture coefficients
    // This implementation follows Woodland & Povey, '02

    int num_buf = ML_BUF, den_buf = MMI_BUF;
    if (mode == MPE_EST || mode == MPE_MMI_PRIOR_EST)
    {
      num_buf = MPE_NUM_BUF;
      den_buf = MPE_DEN_BUF;
    }

    double currfval=0, oldfval=0, diff=1;

    // Iterate until convergence
    std::vector<double> old_weights = m_weights;
    int iter=0;
    while (diff > 0.00001 && iter < 1000) {
      iter++;
      diff = 0;
      //std::vector<double> previous_weights = m_weights;

      if (size() == 1)
      {
        assert( m_weights.size() == 1 );
        m_weights[0] = 1;
        break;
      }

      // Go through every mixture weight
      for (int i=0; i<size(); i++) {

        // JPy: Moved previous weights here
        std::vector<double> previous_weights = m_weights;
        
        // Solve a quadratic equation
        // See Povey: Frame discrimination training ... 3.3 (9)
        // Note: the equation isn't given explicitly, needs some derivation
        double a, b, c, partsum, sol1, sol2;
        // a
        a=0, partsum=0;
        for (int j=0; j<size(); j++)
          if (i != j)
            partsum += previous_weights[j];
        if (partsum <= 0)
          continue;
        for (int j=0; j<size(); j++)
          if (i != j)
            a -= m_accums[den_buf]->gamma[j] * previous_weights[j] / (old_weights[j] * partsum);
        a += m_accums[den_buf]->gamma[i] / old_weights[i];
        // b
        b = -a;
        for (int j=0; j<size(); j++)
            b -= m_accums[num_buf]->gamma[j];
        // c
        c = m_accums[num_buf]->gamma[i];
        // Solve
        sol1 = (-b-sqrt(b*b-4*a*c)) / (2*a);
        sol2 = (-b+sqrt(b*b-4*a*c)) / (2*a);

        if (sol1 <= 0 || sol1 >= 1.0 || (sol2 > 0 && sol2 < 1.0))
        {
          fprintf(stderr, "Warning: Mixture size %i, iter %i, sol1 = %g, sol2 = %g, old = %g\n", size(), iter, sol1, sol2, old_weights[i]);
        }
        // else if (sol2 > 0 && sol2 < 1.0)
        // {
        //   fprintf(stderr, "Warning: Mixture size %i, alternative solution, sol1 = %g, sol2 = %g\n", size(), sol1, sol2);
        // }

        // Mixture component was removed
        // if (sol1 <= 0) {
        //   std::cout << "Warning: mixture component " << i << " was removed." << std::endl;
        //   remove_component(i);
        //   diff = 100;
        //   break;
        // }
                
        // assert(sol1 > 0); assert(sol1 <= 1.00001);
        // m_weights[i] = sol1;

        // // Renormalize weights
        // norm=0;
        // for (int i=0; i<size(); i++)
        //   norm += m_weights[i];
        // for (int i=0; i<size(); i++)
        //   m_weights[i] /= norm;

        // Heuristics: If outside permitted region, move halfway
        if (!isnan(sol1))
        {
          if (sol1 <= 0)
            m_weights[i] = m_weights[i]/2.0;
          else if (sol1 >= 1.0)
            m_weights[i] = m_weights[i] + (1-m_weights[i])/2.0;
          else
            m_weights[i] = sol1;
          m_weights[i] = std::max(m_weights[i], 1e-8); // FIXME: Minimum weight
        }
        
        // Renormalize others
        double norm_m = (1-m_weights[i]) / partsum;
        for (int j = 0; j < size(); j++)
          if (j != i)
            m_weights[j] *= norm_m;
      }
      
      // Compute function value
      oldfval = currfval;
      currfval = 0;
      for (int i=0; i<size(); i++)
        currfval += m_accums[num_buf]->gamma[i] * log(m_weights[i]) -
          m_accums[den_buf]->gamma[i] * m_weights[i] / old_weights[i];
      diff = std::fabs(oldfval-currfval);
      if (iter > 1 && oldfval > currfval)
      {
        fprintf(stderr, "Warning: Mixture size %i, iter %i, reduced function value, %g\n", size(), iter, currfval - oldfval);
      }
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
    if (is.fail())
      throw std::string("Error in reading mixture specifications");
    add_component(index, weight);
  }
  
  normalize_weights();
}


int
Mixture::component_index(int p)
{
  for (int i = 0; i < (int)m_pointers.size(); i++)
    if (m_pointers[i] == p)
      return i;
  return -1;
}


void
Mixture::update_components(const std::vector<int> &cmap)
{
  for (int i = 0; i < (int)m_pointers.size(); i++)
  {
    if (cmap[m_pointers[i]] < 0)
    {
      // Delete this component
      m_pointers.erase(m_pointers.begin()+i);
      m_weights.erase(m_weights.begin()+i);
      i--; // Continue from the next component
    }
    else
      m_pointers[i] = cmap[m_pointers[i]];
  }
  normalize_weights();
}


void
Mixture::remove_component(int index)
{
  assert( index >= 0 && index < (int)m_pointers.size() );
  m_pointers.erase(m_pointers.begin()+index);
  m_weights.erase(m_weights.begin()+index);
  normalize_weights();
}


double
Mixture::cross_entropy(Mixture &g, int samples)
{
  double ce=0;
  Vector sample;
  for (int i=0; i<samples; i++) {
    draw_sample(sample);
    g.get_pool()->reset_cache();
    ce += util::safe_log(g.compute_likelihood(sample));
  }
  return ce/samples;
}


double
Mixture::kullback_leibler(Mixture &g, int samples)
{
  double kl=0;
  Vector sample;
  for (int i=0; i<samples; i++) {
    draw_sample(sample);
    m_pool->reset_cache();
    g.get_pool()->reset_cache();
    kl += util::safe_log(compute_likelihood(sample)/g.compute_likelihood(sample));
  }
  return kl/samples;
}


double
Mixture::bhattacharyya(Mixture &g, int samples)
{
  double bh=0;
  double f_val, g_val;
  Vector sample;
  for (int i=0; i<samples; i++) {
    if (mtw::rnd.f() <0.5)
      draw_sample(sample);
    else
      g.draw_sample(sample);
    m_pool->reset_cache();
    f_val = compute_likelihood(sample);
    g_val = g.compute_likelihood(sample);
    bh += sqrt(f_val*g_val) / (0.5*(f_val+g_val));
  }
  return bh/samples;
}


void
Mixture::draw_sample(Vector &sample)
{
  double randval = mtw::rnd.f();
  double cumsum = 0;
  for (unsigned int i=0; i<m_weights.size(); i++) {
    cumsum += m_weights[i];
    if (randval <= cumsum) {
      m_pool->get_pdf(m_pointers[i])->draw_sample(sample);
      return;
    }    
  }
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
  for (unsigned int i = 0; i < m_cluster_centers.size(); i++)
    delete m_cluster_centers[i];
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
  m_use_clustering = 0;
  m_evaluate_min_clusters = 1;
  m_evaluate_min_gaussians = 1;
  m_cluster_centers.clear();
  m_ismooth_prev_prior = false;
}


PDF*
PDFPool::get_pdf(int index) const
{
  return m_pool[index];
}


void
PDFPool::set_pdf(int pdfindex, PDF *pdf)
{
  if ((unsigned int)pdfindex >= m_pool.size()) {
    m_pool.resize(pdfindex+1);
    m_likelihoods.resize(m_pool.size());
  }
  m_pool[pdfindex]=pdf;
}


int
PDFPool::add_pdf(PDF *pdf)
{
  int index = (int)m_pool.size();
  m_pool.push_back(pdf);
  m_likelihoods.resize(m_pool.size());
  return index;
}


void
PDFPool::delete_pdf(int index)
{
  m_pool.erase(m_pool.begin()+index);
  reset_cache();
  m_likelihoods.resize(m_pool.size());
}


void
PDFPool::reset_cache()
{
  while (!m_valid_likelihoods.empty()) {
    m_likelihoods[m_valid_likelihoods.back()]=-1.0;
    m_valid_likelihoods.pop_back();
  }

#ifdef USE_SUBSPACE_COV
  std::map<int, PrecisionSubspace*>::const_iterator pitr;
  for (pitr = m_precision_subspaces.begin(); pitr != m_precision_subspaces.end(); ++pitr)
    (*pitr).second->reset_cache();
  
  std::map<int, ExponentialSubspace*>::const_iterator eitr;
  for (eitr = m_exponential_subspaces.begin(); eitr != m_exponential_subspaces.end(); ++eitr)
    (*eitr).second->reset_cache();
#endif
}

double
PDFPool::compute_likelihood(const Vector &f, int index)
{
  if (m_likelihoods[index] > 0)
    return m_likelihoods[index];
  m_likelihoods[index] = m_pool[index]->compute_likelihood(f);
  m_valid_likelihoods.push_back(index);
  return m_likelihoods[index];
}


void
PDFPool::precompute_likelihoods(const Vector &f)
{
  reset_cache();

#ifdef USE_SUBSPACE_COV
  std::map<int, PrecisionSubspace*>::const_iterator pitr;
  for (pitr = m_precision_subspaces.begin(); pitr != m_precision_subspaces.end(); ++pitr)
    (*pitr).second->precompute(f);

  std::map<int, ExponentialSubspace*>::const_iterator eitr;
  for (eitr = m_exponential_subspaces.begin(); eitr != m_exponential_subspaces.end(); ++eitr)
    (*eitr).second->precompute(f);
#endif

  // Clustering not in use
  if (!use_clustering()) {
    Vector exponential_feature_vector((int)(dim()*(dim()+3)/2));
    for (int i=0; i<dim(); i++)
      exponential_feature_vector(i) = f(i);
    Matrix tmat(dim(), dim()); tmat=0;
    Blas_R1_Update(tmat, f, f, 1.0);
    Vector tvec;
    LinearAlgebra::map_m2v(tmat, tvec);
    for (int i=0; i<tvec.size(); i++)
      exponential_feature_vector(dim()+i) = tvec(i);
    
    for (int i=0; i<size(); i++) {
      FullCovarianceGaussian *fcgaussian = dynamic_cast< FullCovarianceGaussian* > (m_pool[i]);
      if (fcgaussian != NULL)
        m_likelihoods[i] = fcgaussian->compute_likelihood_exponential(exponential_feature_vector);
      else
        m_likelihoods[i] = m_pool[i]->compute_likelihood(f);
      m_valid_likelihoods.push_back(i);
    }
  }

  // Gaussian clustering in use
  else { 
    // Push the clusters to a priority queue
    ClusterLikelihoods cluster_likelihoods;
    double likelihood;
    for (int i=0; i<number_of_clusters(); i++) {
      likelihood = m_cluster_centers[i]->compute_likelihood(f);
      cluster_likelihoods.push(ClusterLikelihoodPair(i, likelihood));
    }

    // Precompute Gaussians as long as needed
    int total_clusters_evaluated=0, total_gaussians_evaluated=0, cluster_pos, gauss_pos;
    ClusterLikelihoodPair current_cluster;
    while((total_clusters_evaluated < evaluate_min_clusters()) || (total_gaussians_evaluated < evaluate_min_gaussians())) {
      current_cluster = cluster_likelihoods.top();
      cluster_pos = current_cluster.first;
      for (unsigned int j=0; j<m_cluster_to_gaussians[cluster_pos].size(); j++) {
        gauss_pos = m_cluster_to_gaussians[cluster_pos][j];
        m_likelihoods[gauss_pos] = m_pool[gauss_pos]->compute_likelihood(f);
        m_valid_likelihoods.push_back(gauss_pos);
      }
      total_clusters_evaluated++;
      total_gaussians_evaluated += m_cluster_to_gaussians[cluster_pos].size();
      cluster_likelihoods.pop();
    }

    // For the rest, use the cluster center likelihood
    while(!cluster_likelihoods.empty()) {
      current_cluster = cluster_likelihoods.top();
      cluster_pos = current_cluster.first;
      for (unsigned int j=0; j<m_cluster_to_gaussians[cluster_pos].size(); j++) {
        gauss_pos = m_cluster_to_gaussians[cluster_pos][j];
        m_likelihoods[gauss_pos] = current_cluster.second;
        m_valid_likelihoods.push_back(gauss_pos);
      }
      cluster_likelihoods.pop();
    }
    
  }
}


void
PDFPool::set_gaussian_parameters(double minvar, double covsmooth,
                                 double c1, double c2, double ismooth,
                                 double mmi_prior_ismooth)
{
  m_minvar = minvar;
  m_covsmooth = covsmooth;
  m_c1 = c1;
  m_c2 = c2;
  m_ismooth = ismooth;
  m_mmi_prior_ismooth = mmi_prior_ismooth;
}


#ifdef USE_SUBSPACE_COV
void
PDFPool::set_hcl_optimization(HCL_LineSearch_MT_d *ls,
                              HCL_UMin_lbfgs_d *bfgs,
                              std::string ls_cfg_file,
                              std::string bfgs_cfg_file)
{
  std::map<int, PrecisionSubspace*>::const_iterator pitr;
  for (pitr = m_precision_subspaces.begin(); pitr != m_precision_subspaces.end(); ++pitr) {
    (*pitr).second->set_hcl_optimization(ls, bfgs, ls_cfg_file, bfgs_cfg_file);
  }
  
  std::map<int, ExponentialSubspace*>::const_iterator eitr;
  for (eitr = m_exponential_subspaces.begin(); eitr != m_exponential_subspaces.end(); ++eitr) {
    (*eitr).second->set_hcl_optimization(ls, bfgs, ls_cfg_file, bfgs_cfg_file);
  }
}
#endif

void
PDFPool::estimate_parameters(PDF::EstimationMode mode)
{
  for (int i=0; i<size(); i++)
  {
#ifdef USE_SUBSPACE_COV
    PrecisionConstrainedGaussian *pctemp = dynamic_cast< PrecisionConstrainedGaussian* > (m_pool[i]);
    SubspaceConstrainedGaussian *sctemp = dynamic_cast< SubspaceConstrainedGaussian* > (m_pool[i]);

    if (pctemp != NULL || sctemp != NULL)
      std::cout << "Training Gaussian: " << i << "/" << size() << std::endl;
#endif

    Gaussian *temp = dynamic_cast< Gaussian* > (m_pool[i]);

    try {
      if (temp != NULL)
      {
        double tau = 0;
        if (!m_ismooth_prev_prior)
        {
          if (mode == PDF::MPE_MMI_PRIOR_EST)
          {
            if (m_mmi_prior_ismooth > 0)
              temp->ismooth_statistics(PDF::ML_BUF, PDF::ML_BUF,
                                       m_mmi_prior_ismooth);
            temp->estimate_parameters(PDF::MMI_EST, m_minvar, m_covsmooth,
                                      m_c1, m_c2, 0, true);
            temp->ismooth_statistics(PDF::ML_BUF,PDF::MPE_NUM_BUF, m_ismooth);
          }
          else if (mode == PDF::MPE_EST && m_ismooth > 0)
            temp->ismooth_statistics(PDF::ML_BUF, PDF::MPE_NUM_BUF, m_ismooth);
          else if (mode == PDF::MMI_EST && m_ismooth > 0)
            temp->ismooth_statistics(PDF::ML_BUF, PDF::ML_BUF, m_ismooth);
        }
        else
          tau = m_ismooth;
        temp->estimate_parameters(mode, m_minvar, m_covsmooth,
                                  m_c1, m_c2, tau, false);
      }
      else
        m_pool[i]->estimate_parameters(mode);
    } catch (std::string errstr) {
      std::cout << "Warning: Gaussian number " << i
                << ": " << errstr << std::endl;
    }
  }
}


void
PDFPool::read_gk(const std::string &filename)
{
  std::ifstream in(filename.c_str());
  if (!in)
    throw str::fmt(512, "PDFPool::read_gk(): could not open %s\n", filename.c_str());
  
  int pdfs = 0;
#ifdef USE_SUBSPACE_COV
  int ssid;
#endif
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
#ifdef USE_SUBSPACE_COV
      else if (type_str == "precision_subspace") {
        in >> ssid;
        PrecisionSubspace *ps = new PrecisionSubspace();
        ps->read_subspace(in);
        m_precision_subspaces[ssid]=ps;
        i--;
      }
      else if (type_str == "exponential_subspace") {
        in >> ssid;
        ExponentialSubspace *es = new ExponentialSubspace();
        es->read_subspace(in);
        m_exponential_subspaces[ssid]=es;
        i--;
      }
      else if (type_str == "pcgmm") {
        in >> ssid;
        assert(m_precision_subspaces[ssid] != NULL);
	m_pool[i]=new PrecisionConstrainedGaussian(m_precision_subspaces[ssid]);
        m_pool[i]->read(in);
      }
      else if (type_str == "scgmm") {
        in >> ssid;
        assert(m_exponential_subspaces[ssid] != NULL);
	m_pool[i]=new SubspaceConstrainedGaussian(m_exponential_subspaces[ssid]);
        m_pool[i]->read(in);
      }
#endif
      else
        throw std::string("Unknown model type\n") + type_str;
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
#ifdef USE_SUBSPACE_COV
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
#endif
    else
      throw std::string("Unknown model type\n");
  }
  
  if (!in)
    throw std::string("PDFPool::read_gk(): error reading file: ") + filename;
}


void
PDFPool::write_gk(const std::string &filename) const
{
  std::ofstream out(filename.c_str());
  if (!out)
    throw std::string("PDFPool::write_gk(): could not open ") + filename;
  
  out << m_pool.size() << " " << m_dim << " variable\n";

#ifdef USE_SUBSPACE_COV
  std::map<int, PrecisionSubspace*>::const_iterator pitr;
  for (pitr = m_precision_subspaces.begin(); pitr != m_precision_subspaces.end(); ++pitr) {
    out << "precision_subspace ";
    out << (*pitr).first << " "; 
    (*pitr).second->write_subspace(out);
  }

  std::map<int, ExponentialSubspace*>::const_iterator eitr;
  for (eitr = m_exponential_subspaces.begin(); eitr != m_exponential_subspaces.end(); ++eitr) {
    out << "exponential_subspace ";
    out << (*eitr).first << " ";
    (*eitr).second->write_subspace(out);
  }
#endif 

  for (unsigned int i=0; i<m_pool.size(); i++) {

#ifdef USE_SUBSPACE_COV
    PrecisionConstrainedGaussian *pcg = dynamic_cast< PrecisionConstrainedGaussian* > (m_pool[i]);
    if (pcg != NULL) {
      out << "pcgmm ";
      PrecisionSubspace *ps = pcg->get_subspace();
      for (pitr = m_precision_subspaces.begin(); pitr != m_precision_subspaces.end(); ++pitr) {
        if ((*pitr).second == ps)
          out << (*pitr).first;
      }      
    }

    SubspaceConstrainedGaussian *scg = dynamic_cast< SubspaceConstrainedGaussian* > (m_pool[i]);
    if (scg != NULL) {
      out << "scgmm ";
      ExponentialSubspace *es = scg->get_subspace();
      for (eitr = m_exponential_subspaces.begin(); eitr != m_exponential_subspaces.end(); ++eitr) {
        if ((*eitr).second == es)
          out << (*eitr).first; 
      }      
    }
#endif
    
    m_pool[i]->write(out);
    out << std::endl;
  }

  if (!out)
    throw std::string("PDFPool::write_gk(): error writing file: ") + filename;
}


#ifdef USE_SUBSPACE_COV
void
PDFPool::set_precision_subspace(int id, PrecisionSubspace *ps)
{
  m_precision_subspaces[id] = ps;
}


void
PDFPool::set_exponential_subspace(int id, ExponentialSubspace *es)
{
  m_exponential_subspaces[id] = es;
}


PrecisionSubspace*
PDFPool::get_precision_subspace(int id)
{
  return m_precision_subspaces[id];
}


ExponentialSubspace*
PDFPool::get_exponential_subspace(int id)
{
  return m_exponential_subspaces[id];
}


void
PDFPool::remove_precision_subspace(int id)
{
  m_precision_subspaces.erase(id);
}


void
PDFPool::remove_exponential_subspace(int id)
{
  m_exponential_subspaces.erase(id);
}
#endif

bool
PDFPool::split_gaussian(int index, int *new_index)
{
  Gaussian *gaussian = dynamic_cast< Gaussian* > (m_pool[index]);
  if (gaussian == NULL)
    return false;

  if (!gaussian->accumulated(PDF::ML_BUF))
    throw std::string("PDFPool::split_gaussian: ML statistics are required");
  
  Gaussian *new_gaussian = gaussian->copy_gaussian();
  gaussian->split(*new_gaussian);
  *new_index = add_pdf(new_gaussian);
  return true;
}


double
PDFPool::get_gaussian_occupancy(int index) const
{
  Gaussian *gaussian = dynamic_cast< Gaussian* > (m_pool[index]);
  if (gaussian != NULL && gaussian->accumulated(PDF::ML_BUF))
    return gaussian->m_accums[PDF::ML_BUF]->gamma();
  return -1;
}


void
PDFPool::get_occ_sorted_gaussians(std::vector<int> &sorted_gaussians,
                                  double minocc)
{
  for (int i = 0; i < (int)m_pool.size(); i++)
  {
    Gaussian *gaussian = dynamic_cast< Gaussian* > (m_pool[i]);
    if (gaussian != NULL)
    {
      if (gaussian->accumulated(PDF::ML_BUF))
      {
        if (gaussian->m_accums[PDF::ML_BUF]->gamma() >= minocc)
          sorted_gaussians.push_back(i);
      }
    }
  }
  if (sorted_gaussians.size() > 0)
  {
    // Sort the Gaussians according to occupancy
    std::sort(sorted_gaussians.begin(), sorted_gaussians.end(),
              PDFPool::Gaussian_occ_comp(m_pool));
  }
}


void
PDFPool::read_clustering(const std::string &filename)
{
  std::ifstream in(filename.c_str());
  if (!in)
    throw std::string("PDFPool::read_clustering(): could not open ")+filename;

  in >> m_number_of_clusters;
  m_gaussian_to_cluster.resize(size());
  m_cluster_to_gaussians.resize(m_number_of_clusters);
  m_cluster_centers.resize(m_number_of_clusters);
  
  if (m_number_of_clusters > 0.3*size())
    throw std::string("PDFPool::read_clustering(): Are you sure that this clustering makes sense?\n");

  // Read all gaussian-cluster pairs
  while (in) {
    int gauss_index;
    int cluster_index;
    in >> gauss_index >> cluster_index;
    //gauss_index--; cluster_index--;
    
    if (gauss_index >= size())
      throw std::string("PDFPool::read_clustering(): Gauss index out of bounds\n");
    if (cluster_index >= number_of_clusters())
      throw std::string("PDFPool::read_clustering(): Cluster index out of bounds\n");

    m_gaussian_to_cluster[gauss_index] = cluster_index;
    m_cluster_to_gaussians[cluster_index].push_back(gauss_index);
  }

  // Compute cluster centers
  for (int i=0; i<number_of_clusters(); i++) {

    // Collect all Gaussians that belong to this cluster
    std::vector<const Gaussian*> gaussians;
    std::vector<double> weights;
    for (unsigned int j=0; j<m_cluster_to_gaussians[i].size(); j++) {
      Gaussian *gaussian = dynamic_cast< Gaussian* > (m_pool[m_cluster_to_gaussians[i][j]]);
      if (gaussian != NULL) {
        gaussians.push_back(gaussian);
        weights.push_back(1.0);
      }
      else
        throw str::fmt(128, "PDFPool::read_clustering(): the distribution at index %i wasn't Gaussian\n", m_cluster_to_gaussians[i][j]);
    }

    // Compute the center by merging
    DiagonalGaussian *dg = new DiagonalGaussian(dim());
    dg->merge(weights, gaussians, true);
    m_cluster_centers[i] = dg;
  }
}


void
PDFPool::write_cluster_gaussians(const std::string &filename)
{
  std::ofstream out(filename.c_str());
  if (!out)
    throw std::string("PDFPool::write_cluster_gaussians(): could not open ") + filename;
  
  out << number_of_clusters() << " " << m_dim << " variable\n";
  for (int i = 0; i < number_of_clusters(); i++)
  {
    m_cluster_centers[i]->write(out);
    out << std::endl;
  }

  if (!out)
    throw std::string("PDFPool::write_cluster_gaussians(): error writing file: ") + filename;
}


void
PDFPool::inject_cluster_gaussians(PDFPool *target_pool)
{
  for (int i = 0; i < number_of_clusters(); i++)
  {
    target_pool->add_pdf(m_cluster_centers[i]);
  }
}

}
