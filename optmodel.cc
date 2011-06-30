#include <fstream>
#include <string>
#include <iostream>
#include <stdlib.h>
#include <algorithm>

#include "io.hh"
#include "str.hh"
#include "conf.hh"
#include "HmmSet.hh"
#include "LmbfgsOptimize.hh"

using namespace aku;

std::string statistics_file;
std::string out_model_name;
std::string state_file;

int info;

conf::Config config;
HmmSet model;
LmbfgsOptimize optimizer;

double min_var;
double ac_scale;
int num_frames = 0;

double ml_weight = 0;
double mmi_weight = 0;
double mpe_weight = 0;

double msmooth_tau = 0;
double gsmooth_tau = 0;

Vector gradient;


// Gets Gaussian parameters, transforms them to optimization form and sets
// the parameters in the optimizer object
void initialize_optimization_parameters(void)
{
  PDFPool *pool = model.get_pool();
  int num_params = 0;
  Vector params;
  
  // Compute first the number of parameters
  for (int i = 0; i < model.num_emission_pdfs(); i++)
    num_params += model.get_emission_pdf(i)->size();
  num_params += pool->size()*pool->dim()*2;
  
  // Allocate parameter vector and fill in the parameters
  params.resize(num_params);
  int pindex = 0;

  // Mixture components
  for (int i = 0; i < model.num_emission_pdfs(); i++)
  {
    Mixture *m = model.get_emission_pdf(i);
    for (int j = 0; j < m->size(); j++)
      params(pindex++) = util::safe_log(m->get_mixture_coefficient(j));
  }

  // Means and covariances (diagonal)
  for (int i = 0; i < pool->size(); i++)
  {
    Gaussian *pdf = dynamic_cast< Gaussian* >(pool->get_pdf(i));
    if (pdf == NULL)
      throw std::string("Only Gaussian PDFs are supported!");
    Vector temp;
    pdf->get_mean(temp);
    assert( temp.size() == pool->dim() );
    for (int j = 0; j < pool->dim(); j++)
      params(pindex++) = temp(j);
    pdf->get_covariance(temp);
    assert( temp.size() == pool->dim() );
    for (int j = 0; j < pool->dim(); j++)
    {
      if (temp(j) < 1.0001*min_var)
        temp(j) = 1.0001*min_var;
      //params(pindex++) = log(temp(j)-min_var);
      params(pindex++) = sqrt(temp(j)-min_var);
    }
  }
  assert( pindex == num_params );

  optimizer.set_parameters(params);
}


// Moves the parameters from the optimizer object back to the model
void set_model_parameters(void)
{
  int pindex = 0;
  PDFPool *pool = model.get_pool();
  Vector params;

  optimizer.get_parameters(params);
  
  // Mixture components
  for (int i = 0; i < model.num_emission_pdfs(); i++)
  {
    Mixture *m = model.get_emission_pdf(i);
    // Compute the normalization
    double norm = 0;
    for (int j = 0; j < m->size(); j++)
      norm += exp(params(pindex+j));
    for (int j = 0; j < m->size(); j++)
      m->set_mixture_coefficient(j, exp(params(pindex++))/norm);
  }

  // Means and covariances
  for (int i = 0; i < pool->size(); i++)
  {
    Gaussian *pdf = dynamic_cast< Gaussian* >(pool->get_pdf(i));
    if (pdf == NULL)
      throw std::string("Only Gaussian PDFs are supported!");
    Vector temp(pool->dim());
    for (int j = 0; j < pool->dim(); j++)
      temp(j) = params(pindex++);
    pdf->set_mean(temp);
    for (int j = 0; j < pool->dim(); j++)
    {
      //temp(j) = min_var + exp(params(pindex));
      temp(j) = min_var + params(pindex)*params(pindex);
      pindex++;
    }
    pdf->set_covariance(temp);
  }
}


// Extracts the gradient from model accumulators
void extract_gradient(void)
{
  int pindex = 0;
  PDFPool *pool = model.get_pool();
  Vector params;
  std::vector<double> temp;
  std::vector<double> gauss_mixture_aux_gamma;

  optimizer.get_parameters(params);
  gradient.resize(optimizer.get_num_parameters());

  gauss_mixture_aux_gamma.resize(pool->size(), 0);
  
  // Mixture components
  for (int i = 0; i < model.num_emission_pdfs(); i++)
  {
    Mixture *m = model.get_emission_pdf(i);
    // Compute the normalization
    double norm = 0;
    temp.resize(m->size());
    for (int j = 0; j < m->size(); j++)
      norm += exp(params(pindex+j));
    // Compute the derivatives wrt the original mixture weights
    for (int j = 0; j < m->size(); j++)
    {
      temp[j] = 0;
      if (mpe_weight != 0)
      {
        temp[j] += -mpe_weight*(ac_scale/(double)num_frames) *
          (m->get_accumulated_gamma(PDF::MPE_NUM_BUF, j) /
           m->get_mixture_coefficient(j));
      }
      if (mmi_weight != 0)
      {
        temp[j] += -mmi_weight*(ac_scale/(double)num_frames) * (
          (m->get_accumulated_gamma(PDF::ML_BUF, j) /
           m->get_mixture_coefficient(j)) -
          (m->get_accumulated_gamma(PDF::MMI_BUF, j) /
           m->get_mixture_coefficient(j)));
      }
      if (ml_weight != 0)
      {
        temp[j] += -ml_weight*(ac_scale/(double)num_frames) * 
          (m->get_accumulated_gamma(PDF::ML_BUF, j) /
           m->get_mixture_coefficient(j));
      }
      if (msmooth_tau != 0)
      {
        temp[j] += -ac_scale*msmooth_tau/
          ((m->get_accumulated_aux_gamma(PDF::MPE_NUM_BUF)+1)*
           (double)num_frames)*
          (m->get_accumulated_gamma(PDF::ML_BUF, j) /
           m->get_mixture_coefficient(j));
        gauss_mixture_aux_gamma[m->get_base_pdf_index(j)] +=
          msmooth_tau/(m->get_accumulated_aux_gamma(PDF::MPE_NUM_BUF)+1);
      }
    }
    
    // Combine to form derivatives wrt the transformed parameters
    for (int j = 0; j < m->size(); j++)
    {
      double val = 0;
      double ep = exp(params(pindex));
      for (int k = 0; k < m->size(); k++)
      {
        if (k == j)
          val += temp[k]*((ep - m->get_mixture_coefficient(k)*ep) / norm);
        else
        {
          val += temp[k]*(-m->get_mixture_coefficient(k)*ep/norm);
        }
      }

      gradient(pindex++) = val;
    }
  }
  
  // Means and covariances
  for (int i = 0; i < pool->size(); i++)
  {
    Gaussian *pdf = dynamic_cast< Gaussian* >(pool->get_pdf(i));
    if (pdf == NULL)
      throw std::string("Only Gaussian PDFs are supported!");
    Vector mpe_m1, mmi_m1, ml_m1;
    Vector mpe_m2, mmi_m2, ml_m2;
    Vector mean;
    Vector diag_cov;
    double mpe_gamma = 0, mmi_gamma = 0, ml_gamma = 0;
    double gsmooth_gamma = 0;
    pdf->get_mean(mean);
    pdf->get_covariance(diag_cov);
    if (mpe_weight != 0)
    {
      pdf->get_accumulated_mean(PDF::MPE_NUM_BUF, mpe_m1);
      pdf->get_accumulated_second_moment(PDF::MPE_NUM_BUF, mpe_m2);
      mpe_gamma = pdf->get_accumulated_gamma(PDF::MPE_NUM_BUF);
      printf("%.4f\n", mpe_gamma);
    }
    if (mmi_weight != 0 || ml_weight != 0 || msmooth_tau!=0 || gsmooth_tau!=0)
    {
      pdf->get_accumulated_mean(PDF::ML_BUF, ml_m1);
      pdf->get_accumulated_second_moment(PDF::ML_BUF, ml_m2);
      ml_gamma = pdf->get_accumulated_gamma(PDF::ML_BUF);
    }
    if (mmi_weight != 0)
    {
      pdf->get_accumulated_mean(PDF::MMI_BUF, mmi_m1);
      pdf->get_accumulated_second_moment(PDF::MMI_BUF, mmi_m2);
      mmi_gamma = pdf->get_accumulated_gamma(PDF::MMI_BUF);
    }
    if (gsmooth_tau!=0)
    {
      gsmooth_gamma = pdf->get_accumulated_aux_gamma(PDF::MPE_NUM_BUF);
    }
    for (int j = 0; j < pool->dim(); j++)
    {
      gradient(pindex) = 0;
      if (mpe_weight != 0)
      {
        gradient(pindex) +=
          -mpe_weight*(ac_scale*(mpe_m1(j)-mean(j)*mpe_gamma) /
            (diag_cov(j)*(double)num_frames));
      }
      if (mmi_weight != 0)
      {
        gradient(pindex) += -mmi_weight*(ac_scale/(double)num_frames) * (
          (ml_m1(j)-mean(j)*ml_gamma)/diag_cov(j) -
          (mmi_m1(j)-mean(j)*mmi_gamma)/diag_cov(j));
      }
      if (ml_weight != 0)
      {
        gradient(pindex) += -ml_weight*(ac_scale/(double)num_frames) *
          (ml_m1(j)-mean(j)*ml_gamma)/diag_cov(j);
      }
      if (msmooth_tau != 0)
      {
        gradient(pindex) += -ac_scale*gauss_mixture_aux_gamma[i]/
          (double)num_frames*
          (ml_m1(j)-mean(j)*ml_gamma)/diag_cov(j);
      }
      if (gsmooth_tau != 0)
      {
        gradient(pindex) += -ac_scale*gsmooth_tau/
          (gsmooth_gamma+1)/
          (double)num_frames*
          (ml_m1(j)-mean(j)*ml_gamma)/diag_cov(j);
      }

      pindex++;
    }
    for (int j = 0; j < pool->dim(); j++)
    {
//       double ep = exp(params(pindex));
//       double c = ep + min_var;
      double ep = 2*params(pindex);
      double c = params(pindex)*params(pindex) + min_var;
      gradient(pindex) = 0;
      if (mpe_weight != 0)
      {
        gradient(pindex) += -mpe_weight*(ac_scale/(double)num_frames) * (
          (mpe_m2(j)-2*mpe_m1(j)*mean(j)+mpe_gamma*mean(j)*mean(j)-mpe_gamma*c)
           / (2*c*c)) * ep;
      }
      if (mmi_weight != 0)
      {
        gradient(pindex) += -mmi_weight*(ac_scale/(double)num_frames) * (
          (ml_m2(j)-2*ml_m1(j)*mean(j)+ml_gamma*mean(j)*mean(j)-ml_gamma*c)
          / (2*c*c) -
          (mmi_m2(j)-2*mmi_m1(j)*mean(j)+mmi_gamma*mean(j)*mean(j)-mmi_gamma*c)
          / (2*c*c)) * ep;
      }
      if (ml_weight != 0)
      {
        gradient(pindex) += -ml_weight*(ac_scale/(double)num_frames) * (
          (ml_m2(j)-2*ml_m1(j)*mean(j)+ml_gamma*mean(j)*mean(j)-ml_gamma*c)/
          (2*c*c)) * ep;
      }
      if (msmooth_tau != 0)
      {
        gradient(pindex) += -ac_scale*gauss_mixture_aux_gamma[i]/
          (double)num_frames*
          ((ml_m2(j)-2*ml_m1(j)*mean(j)+ml_gamma*mean(j)*mean(j)-ml_gamma*c)/
           (2*c*c)) * ep;
      }
      if (gsmooth_tau != 0)
      {
        gradient(pindex) += -ac_scale*gsmooth_tau/
          (gsmooth_gamma+1)/
          (double)num_frames*
          ((ml_m2(j)-2*ml_m1(j)*mean(j)+ml_gamma*mean(j)*mean(j)-ml_gamma*c)/
           (2*c*c)) * ep;
      }
      
      pindex++;
    }
  }
  assert( pindex == optimizer.get_num_parameters() );

  optimizer.set_gradient(gradient);
}


// FIXME: Only MPE without smoothing!
// NOTE: Call after gradient has been computed
void set_inv_hessian(void)
{
  int pindex = 0;
  PDFPool *pool = model.get_pool();
  Vector inv_hes;
  Vector params;

  assert( gradient.size() == optimizer.get_num_parameters() );
  
  std::vector<double> gauss_gamma;
  std::vector<double> gauss_prior;

  gauss_gamma.resize(pool->size(), 0);
  gauss_prior.resize(pool->size(), 0);

  optimizer.get_parameters(params);
  inv_hes.resize(optimizer.get_num_parameters());
  // Mixture components
  for (int i = 0; i < model.num_emission_pdfs(); i++)
  {
    Mixture *m = model.get_emission_pdf(i);
    for (int j = 0; j < m->size(); j++)
    {
      inv_hes(pindex++) = 10/(ac_scale*ac_scale); // FIXME: Magic number!
      gauss_gamma[m->get_base_pdf_index(j)] += m->get_accumulated_gamma(PDF::MMI_BUF, j);
      gauss_prior[m->get_base_pdf_index(j)] += m->get_mixture_coefficient(j)*m->get_accumulated_gamma(PDF::MMI_BUF, j);
    }
  }

  for (int i = 0; i < (int)pool->size(); i++)
    gauss_prior[i] = gauss_prior[i]/gauss_gamma[i];

  std::vector<double> mu_g, mu_h, mu_h_mean;
  std::vector<double> cov_g, cov_h, cov_h_mean;
  std::vector<double> min_mu_h;
  std::vector<double> min_cov_h;
  mu_g.resize(pool->size(), 0);
  mu_h.resize(pool->size(), 0);
  mu_h_mean.resize(pool->size(), 0);
  cov_g.resize(pool->size(), 0);
  cov_h.resize(pool->size(), 0);
  cov_h_mean.resize(pool->size(), 0);
  min_mu_h.resize(pool->size(), 1e10);
  min_cov_h.resize(pool->size(), 1e10);

  // Means and covariances
  for (int i = 0; i < pool->size(); i++)
  {
    Gaussian *pdf = dynamic_cast< Gaussian* >(pool->get_pdf(i));
    if (pdf == NULL)
      throw std::string("Only Gaussian PDFs are supported!");
    Vector mpe_m1, mpe_m2;
    Vector mean;
    Vector diag_cov;
    double mpe_gamma = 0;
    pdf->get_mean(mean);
    pdf->get_covariance(diag_cov);
    if (mpe_weight != 0)
    {
      pdf->get_accumulated_mean(PDF::MPE_NUM_BUF, mpe_m1);
      pdf->get_accumulated_second_moment(PDF::MPE_NUM_BUF, mpe_m2);
      mpe_gamma = pdf->get_accumulated_gamma(PDF::MPE_NUM_BUF);
    }

    std::vector<double> mu_hes;
    std::vector<double> cov_hes;
    mu_hes.resize(pool->dim(), 0);
    cov_hes.resize(pool->dim(), 0);
    
    for (int j = 0; j < pool->dim(); j++)
    {
      double hessian = 0;
      hessian = -(ac_scale/(double)num_frames) * (
        (mpe_m2(j)-2*mpe_m1(j)*mean(j)+mpe_gamma*mean(j)*mean(j)-
         mpe_gamma*diag_cov(j)) /
        (diag_cov(j)*diag_cov(j)));
      hessian += -(ac_scale*ac_scale/(double)num_frames) * (
        -gauss_prior[i]*(mpe_m2(j)-2*mpe_m1(j)*mean(j)+mpe_gamma*mean(j)*mean(j))) / (diag_cov(j)*diag_cov(j));
      mu_hes[j] = hessian;
//       double temp = -(ac_scale/(double)num_frames) / (diag_cov(j)*diag_cov(j));
//       fprintf(stderr, "%i, mu %i: %g  [%g %g %g  %g * %g]\n", i, j, hessian,
//               mpe_m2(j)*temp,
//               -2*mpe_m1(j)*mean(j)*temp,
//               mpe_gamma*(mean(j)*mean(j)-diag_cov(j))*temp,
//               -gauss_prior[i],
//               (mpe_m2(j)-2*mpe_m1(j)*mean(j)+mpe_gamma*mean(j)*mean(j))*temp);
      if (hessian < min_mu_h[i])
        min_mu_h[i] = hessian;
    }

    for (int j = 0; j < pool->dim(); j++)
    {
      double ep = exp(params(pindex+pool->dim()+j));
      double hessian = 0;
      hessian = -(ac_scale/(double)num_frames) * (
        (3*mpe_gamma*diag_cov(j)-
         3*(mpe_m2(j)-2*mpe_m1(j)*mean(j)+mpe_gamma*mean(j)*mean(j))) /
        (2*diag_cov(j)*diag_cov(j)*diag_cov(j)));
      hessian += -(ac_scale*ac_scale/(double)num_frames) * (
        -gauss_prior[i]*(2*mpe_gamma*diag_cov(j)-
                         (mpe_m2(j)-2*mpe_m1(j)*mean(j)+
                          mpe_gamma*mean(j)*mean(j)))) /
        (2*diag_cov(j)*diag_cov(j)*diag_cov(j));
      // Gradient is already multiplied with ep
      hessian = hessian*ep*ep + gradient(pindex+pool->dim()+j);
      cov_hes[j] = hessian;
//       double temp = -(ac_scale/(double)num_frames) /
//         (2*diag_cov(j)*diag_cov(j)*diag_cov(j));
//       fprintf(stderr, "%i, cov %i: %g  [%g %g  %g * %g, gr %g]\n", i, j, hessian,
//               3*mpe_gamma*diag_cov(j)*temp,
//               -3*(mpe_m2(j)-2*mpe_m1(j)*mean(j)+mpe_gamma*mean(j)*mean(j))*temp,
//               -gauss_prior[i],
//               (2*mpe_gamma*diag_cov(j)-(mpe_m2(j)-2*mpe_m1(j)*mean(j)+
//                                         mpe_gamma*mean(j)*mean(j)))*temp,
//               gradient(pindex+pool->dim()+j));
      
      if (hessian < min_cov_h[i])
        min_cov_h[i] = hessian;
    }
    //fprintf(stderr, "\n");
    //fprintf(stderr, "Gaussian %i, min mu hes %g, min cov hes %g\n", i, min_mu_h[i], min_cov_h[i]);
    double hessian_add = (min_cov_h[i] < min_mu_h[i] ? -min_cov_h[i] : -min_mu_h[i]) + 1e-4; // FIXME: Magic number

    if (hessian_add < 0)
      hessian_add = 0;

    for (int j = 0; j < pool->dim(); j++)
    {
      double hessian = mu_hes[j] + hessian_add;
      mu_h_mean[i] += hessian;
      mu_h[i] += hessian*hessian;
      mu_g[i] += gradient(pindex)*gradient(pindex);
      inv_hes(pindex++) = 1/hessian;
    }
    for (int j = 0; j < pool->dim(); j++)
    {
      double hessian = cov_hes[j] + hessian_add;
      cov_h_mean[i] += hessian;
      cov_h[i] += hessian*hessian;
      cov_g[i] += gradient(pindex)*gradient(pindex);
      inv_hes(pindex++) = 1/hessian;
    }
  }
  int windex = 0;
  for (int i = 0; i < model.num_emission_pdfs(); i++)
  {
    Mixture *m = model.get_emission_pdf(i);
    double m_mu_g = 0, m_mu_h = 0, m_mu_h_mean = 0;
    double m_cov_g = 0, m_cov_h = 0, m_cov_h_mean = 0;
    double m_w = 0;
    double m_min_mu_h = 1e10;
    double m_min_cov_h = 1e10;
    for (int j = 0; j < m->size(); j++)
    {
      int index = m->get_base_pdf_index(j);
      m_mu_g += sqrt(mu_g[index]);
      m_mu_h += sqrt(mu_h[index]);
      m_mu_h_mean += mu_h_mean[index];
      m_cov_g += sqrt(cov_g[index]);
      m_cov_h += sqrt(cov_h[index]);
      m_cov_h_mean += cov_h_mean[index];
      m_w += gradient(windex)*gradient(windex);
      if (min_mu_h[index] < m_min_mu_h)
        m_min_mu_h = min_mu_h[index];
      if (min_cov_h[index] < m_min_cov_h)
        m_min_cov_h = min_cov_h[index];
      windex++;
    }
    fprintf(stderr, "Mixture %i:\n", i);
    fprintf(stderr, "  Mean: grad %g, hes %g, hes mean %g, min hes %g\n",
            m_mu_g/m->size(), m_mu_h/m->size(), m_mu_h_mean/m->size(), m_min_mu_h);
    fprintf(stderr, "  Cov : grad %g, hes %g, hes_mean %g, min hes %g\n",
            m_cov_g/m->size(), m_cov_h/m->size(), m_cov_h_mean/m->size(),
            m_min_cov_h);
    fprintf(stderr, "  Weights: %g\n", sqrt(m_w/m->size()));
  }
  for (int i = 0; i < pindex; i++)
    fprintf(stderr, "%g\n", inv_hes(i));
  fprintf(stderr, "\n");
  assert( pindex == optimizer.get_num_parameters() );
  optimizer.set_init_diag_inv_hessian(inv_hes);
}


// NOTE: Call after gradient has been computed
void limit_initial_step_parameter_change(double init_scale, double limit)
{
  PDFPool *pool = model.get_pool();
  Vector inv_hes;
  Vector params;
  int pindex = 0;

  optimizer.get_parameters(params);
  inv_hes.resize(optimizer.get_num_parameters());
  
  // Mixture components
  for (int i = 0; i < model.num_emission_pdfs(); i++)
  {
    Mixture *m = model.get_emission_pdf(i);
    double grad = 0;
    for (int j = 0; j < m->size(); j++)
    {
      // NOTE: Handles transformed parameters
      double temp = gradient(pindex + j);
      grad += temp*temp;
    }
    double rms_change = sqrt(grad/m->size());
    if (rms_change*init_scale > limit)
    {
      double c = limit/(rms_change*init_scale);
      for (int j = 0; j < m->size(); j++)
        inv_hes(pindex + j) = c;
    }
    else
    {
      for (int j = 0; j < m->size(); j++)
        inv_hes(pindex + j) = 1;
    }
    pindex += m->size();
  }

  // Means and covariances
  for (int i = 0; i < pool->size(); i++)
  {
    double grad = 0;
    for (int j = 0; j < pool->dim(); j++)
    {
      double temp = gradient(pindex + j);
      grad += temp*temp;
    }
    double rms_change = sqrt(grad/pool->dim());
    if (rms_change*init_scale > limit)
    {
      double c = limit/(rms_change*init_scale);
      for (int j = 0; j < pool->dim(); j++)
        inv_hes(pindex + j) = c;
    }
    else
    {
      for (int j = 0; j < pool->dim(); j++)
        inv_hes(pindex + j) = 1;
    }
    pindex += pool->dim();

    grad = 0;
    for (int j = 0; j < pool->dim(); j++)
    {
      double ep = exp(params(pindex+j));
      double temp = gradient(pindex + j)/ep; // Inverse transform
      grad += temp*temp;
    }
    rms_change = sqrt(grad/pool->dim());
    if (rms_change*init_scale > limit)
    {
      double c = limit/(rms_change*init_scale);
      for (int j = 0; j < pool->dim(); j++)
        inv_hes(pindex + j) = c;
    }
    else
    {
      for (int j = 0; j < pool->dim(); j++)
        inv_hes(pindex + j) = 1;
    }
    pindex += pool->dim();
  }
  assert( pindex == optimizer.get_num_parameters() );
  optimizer.set_init_diag_inv_hessian(inv_hes);
}


class GaussParamLimit : public LmbfgsLimitInterface {
public:
  virtual void limit_search_direction(const Vector *params,
                                      Vector *search_dir);
  virtual double limit_search_step(const Vector *params, double step) { return step; }

  void set_limit(double limit) { m_limit = limit; }

private:
  double m_limit; // Relative change of parameters
};


class BinSearchFEval {
public:
  virtual double evaluate_function(double p) const = 0;
  virtual ~BinSearchFEval() {}
};


double bin_search_max_param(double lower_bound, double low_value,
                            double upper_bound, double up_value,
                            double max_value, double accuracy,
                            const BinSearchFEval &f)
{
  double new_param = (lower_bound + upper_bound) / 2.0;
  if (new_param-lower_bound < accuracy)
    return new_param;
  double new_value = f.evaluate_function(new_param);
  if (new_value > max_value)
    return bin_search_max_param(lower_bound, low_value, new_param, new_value,
                                max_value, accuracy, f);
  else
    return bin_search_max_param(new_param, new_value, upper_bound, up_value,
                                max_value, accuracy, f);
}


class MixtureWeightKLD : public BinSearchFEval {
public:
  MixtureWeightKLD(const Vector *paramp, const Vector *searchp, int base_index,
                   int num_weights) : wp(paramp), dp(searchp),
                                      index(base_index), size(num_weights) { }
  virtual double evaluate_function(double p) const;
private:
  const Vector *wp;
  const Vector *dp;
  int index;
  int size;
};

double
MixtureWeightKLD::evaluate_function(double p) const
{
  double kld = 0;

  // Compute the normalization factors
  double new_norm = 0;
  double orig_norm = 0;
  for (int i = 0; i < size; i++)
  {
    orig_norm += exp((*wp)(index+i));
    new_norm += exp((*wp)(index+i) + p*(*dp)(index+i));
  }

  for (int i = 0; i < size; i++)
  {
    double orig_w = exp((*wp)(index+i))/orig_norm;
    double new_w = exp((*wp)(index+i)+p*(*dp)(index+i))/new_norm;
    kld += new_w*log(new_w/orig_w);
  }
  return kld;
}


class GaussianKLD : public BinSearchFEval {
public:
  GaussianKLD(const Vector *paramp, const Vector *searchp, int base_index,
              int dimension, double mv) : gp(paramp), dp(searchp),
                                          index(base_index),
                                          dim(dimension), min_var(mv) { }
  virtual double evaluate_function(double p) const;
private:
  const Vector *gp;
  const Vector *dp;
  int index;
  int dim;
  double min_var;
};

double
GaussianKLD::evaluate_function(double p) const
{
  double kld = 0;
  for (int i = 0; i < dim; i++)
  {
    double orig_m = (*gp)(index+i);
    double orig_v = min_var + (*gp)(index+dim+i)*(*gp)(index+dim+i);
    double new_m = orig_m + p*(*dp)(index+i);
    double new_v = (*gp)(index+dim+i) + p*(*dp)(index+dim+i);
    new_v = min_var + new_v*new_v;
    double dm = new_m - orig_m;
    kld += new_v/orig_v + log(orig_v/new_v) + dm*dm/orig_v;
  }
  return (kld - dim)/2.0;
}


void GaussParamLimit::limit_search_direction(const Vector *params,
                                             Vector *search_dir)
{
  PDFPool *pool = model.get_pool();
  int pindex = 0;
  double search_acc = 1e-4;

  // Mixture components
  for (int i = 0; i < model.num_emission_pdfs(); i++)
  {
    Mixture *m = model.get_emission_pdf(i);
    MixtureWeightKLD mix(params, search_dir, pindex, m->size());
    double kld = mix.evaluate_function(1);
    if (kld > m_limit)
    {
      double step = bin_search_max_param(0, 0, 1, kld, m_limit, search_acc,
                                         mix);
      fprintf(stderr, "Mixture %i limited, original KLD %.4g, step size %.4g, new KLD %.4g\n", i, kld, step, mix.evaluate_function(step));
      for (int j = 0; j < m->size(); j++)
        (*search_dir)(pindex+j) = (*search_dir)(pindex+j) * step;
    }
    pindex += m->size();

    // Old limit code commented out
    /*
    double cur_ratio = 1;
    int iter = 0;
    for (; iter < 20; iter++)
    {
      // Compute the normalization factors
      double norm = 0;
      double old_norm = 0;
      for (int j = 0; j < m->size(); j++)
      {
        norm += exp((*params)(pindex+j) + cur_ratio*(*search_dir)(pindex+j));
        old_norm += exp((*params)(pindex+j));
      }
      // Compute the maximum change
      double max_change = 0;
      int max_change_index = 0;
      for (int j = 0; j < m->size(); j++)
      {
        double old_p = exp((*params)(pindex+j))/old_norm;
        double temp = fabs(exp((*params)(pindex+j)+cur_ratio*(*search_dir)(pindex+j))/norm-old_p)/old_p;
        if (temp > max_change)
        {
          max_change = temp;
          max_change_index = j;
        }
      }
      if (iter == 0 && max_change < m_limit)
        break;
      // Compute new ratio
      double sign = ((*search_dir)(pindex+max_change_index)<0?-1:1);
      double new_ratio = log((1+sign*m_limit)*norm/old_norm)/(*search_dir)(pindex+max_change_index);
      if (new_ratio < 0)
      {
        cur_ratio = 0;
        iter++; // Ratio has changed
        break;
      }
      if (iter > 0 && fabs(new_ratio-cur_ratio)/cur_ratio < 0.001)
        break;
      cur_ratio = new_ratio;
    }
    if (iter > 0)
    {
      for (int j = 0; j < m->size(); j++)
        (*search_dir)(pindex+j) = (*search_dir)(pindex+j) * cur_ratio;
//      fprintf(stderr, "Mixture %i limit, ratio %f\n", i, cur_ratio);
    }
    pindex += m->size();
    */
  }

  // Means and covariances
  for (int i = 0; i < pool->size(); i++)
  {
    GaussianKLD g(params, search_dir, pindex, pool->dim(), min_var);
    double kld = g.evaluate_function(1);
    if (kld > m_limit)
    {
      double step = bin_search_max_param(0, 0, 1, kld, m_limit, search_acc, g);
      fprintf(stderr, "Gaussian %i limited, original KLD %.4g, step size %.4g, new KLD %.4g\n", i, kld, step, g.evaluate_function(step));
      for (int j = 0; j < 2*pool->dim(); j++)
        (*search_dir)(pindex+j) = (*search_dir)(pindex+j) * step;
    }
    pindex += 2*pool->dim();


    // Old limit code commented out
    /*
    double ratio = 1;
    double max_mean_change = 0;
    double max_cov_change = 0;
    for (int j = 0; j < pool->dim(); j++)
    {
      double c = (*params)(pindex+pool->dim())*(*params)(pindex+pool->dim()) +
        min_var;
      double temp = fabs((*search_dir)(pindex)/sqrt(c));
      if (temp > m_limit)
      {
//         if (m_limit/temp < ratio)
//           fprintf(stderr, "Gauss %i, mean limit, ratio %f (%f in %f)\n",
//                   i, m_limit/temp, (*search_dir)(pindex), sqrt(c));
        ratio = std::min(m_limit/temp, ratio);
        assert( ratio > 0 );
      }

      if (temp > max_mean_change)
        max_mean_change = temp;

      double abs_p = fabs((*params)(pindex+pool->dim()));
      double abs_g = fabs((*search_dir)(pindex+pool->dim()));

      // Limit delta_var/var <= m_limit
      if ((*params)(pindex+pool->dim())*(*search_dir)(pindex+pool->dim()) < 0)
      {
        double p2=abs_p*abs_p;
        double c_limit;
        if (p2 > c*m_limit)
          c_limit = abs_p - sqrt(p2 - c*m_limit);
        else
          c_limit = abs_p*0.999;
        // Don't let to zero
        if (abs_g > c_limit)
        {
//           if (c_limit/-(*search_dir)(pindex+pool->dim()) < ratio)
//             fprintf(stderr, "Gauss %i, cov limit, ratio %f (%f in %f)\n", i,
//                     c_limit/-(*search_dir)(pindex+pool->dim()),
//                     (*search_dir)(pindex+pool->dim()), (*params)(pindex+pool->dim()));
          ratio = std::min(c_limit/abs_g, ratio);
          assert( ratio > 0 );
        }
      }
      else
      {
        double c_limit = -abs_p + sqrt(abs_p*abs_p + c*m_limit);
        if (abs_g > c_limit)
        {
//           if (c_limit/(*search_dir)(pindex+pool->dim()) < ratio)
//             fprintf(stderr, "Gauss %i, cov limit, ratio %f (%f in %f)\n", i,
//                     c_limit/(*search_dir)(pindex+pool->dim()),
//                     (*search_dir)(pindex+pool->dim()), (*params)(pindex+pool->dim()));
          ratio = std::min(c_limit/abs_g, ratio);
          assert( ratio > 0 );
        }
      }

      temp = (*params)(pindex+pool->dim()) + (*search_dir)(pindex+pool->dim());
      temp = fabs(1 - (temp*temp+min_var)/c);
      if (temp > max_cov_change)
        max_cov_change = temp;
      
      pindex++;
    }
    if (ratio < 1)
    {
      for (int j = 0; j < pool->dim(); j++)
      {
        (*search_dir)(pindex-j-1) = (*search_dir)(pindex-j-1) * ratio;
        (*search_dir)(pindex+j) = (*search_dir)(pindex+j) * ratio;
      }
    }

//     if (info > 2)
//       printf("%i %g %g\n", i, max_mean_change, max_cov_change);

    pindex += pool->dim();
    */
  }
}

class GaussParamLimit param_limiter;


double get_msmooth_score(void)
{
  double mscore = 0;
  // Mixture components
  for (int i = 0; i < model.num_emission_pdfs(); i++)
  {
    Mixture *m = model.get_emission_pdf(i);
    mscore += ac_scale*msmooth_tau/
      (m->get_accumulated_aux_gamma(PDF::MPE_NUM_BUF)+1) *
      m->get_accumulated_mixture_ll(PDF::ML_BUF);
    fprintf(stderr, "%.15g %.15g ", msmooth_tau/(m->get_accumulated_aux_gamma(PDF::MPE_NUM_BUF)+1), m->get_accumulated_mixture_ll(PDF::ML_BUF));
  }
  fprintf(stderr, "\nMixture loglikelihood score: %g\n", mscore);
  return mscore / (double)num_frames;
}

double get_gsmooth_score(void)
{
  PDFPool *pool = model.get_pool();
  double gscore = 0;
  // Mixture components
  for (int i = 0; i < pool->size(); i++)
  {
    Gaussian *pdf = dynamic_cast< Gaussian* >(pool->get_pdf(i));
    if (pdf == NULL)
      throw std::string("Only Gaussian PDFs are supported!");

    gscore += ac_scale*gsmooth_tau/
      (pdf->get_accumulated_aux_gamma(PDF::MPE_NUM_BUF)+1)*
      pdf->get_accumulated_aux_gamma(PDF::ML_BUF);
    fprintf(stderr, "%.15g %.15g ", gsmooth_tau/(pdf->get_accumulated_aux_gamma(PDF::MPE_NUM_BUF)+1), pdf->get_accumulated_aux_gamma(PDF::ML_BUF));
  }
  fprintf(stderr, "\nGaussian loglikelihood score: %g\n", gscore);
  return gscore / (double)num_frames;
}


int
main(int argc, char *argv[])
{
  double score = 0;
  std::map< std::string, double > sum_statistics;
  std::string base_file_name;
  PDF::StatisticsMode statistics_mode = 0;
  
  try {
    config("usage: optmodel [OPTION...]\n")
      ('h', "help", "", "", "display help")
      ('b', "base=BASENAME", "arg", "", "Previous base filename for model files")
      ('g', "gk=FILE", "arg", "", "Previous mixture base distributions")
      ('m', "mc=FILE", "arg", "", "Previous mixture coefficients for the states")
      ('p', "ph=FILE", "arg", "", "Previous HMM definitions")
      ('L', "list=LISTNAME", "arg must", "", "file with one statistics file per line")
      ('F', "osf=FILE", "arg must", "", "Optimization state file")
      ('o', "out=BASENAME", "arg must", "", "base filename for output models")
      ('\0', "ml=FLOAT", "arg", "0", "ML optimization weight")
      ('\0', "mmi=FLOAT", "arg", "0", "MMI optimization weight")
      ('\0', "mpe=FLOAT", "arg", "0", "MPE optimization weight")
      ('\0', "msmooth=FLOAT", "arg", "0", "MPE mixture ML smoothing")
      ('\0', "gsmooth=FLOAT", "arg", "0", "MPE Gaussian ML smoothing")
      ('l', "initscale=SCALE", "arg", "", "Initialize with inverse Hessian scale")
      ('\0', "minvar=FLOAT", "arg", "0.09", "minimum variance (default 0.09)")
      ('\0', "limit=FLOAT", "arg", "1", "Limit parameter change")
      ('A', "ac-scale=FLOAT", "arg", "1", "acoustic scaling used in stats")
      ('P', "pher", "", "", "Use phone error instead of frame error MPE")
      ('\0', "bfgsu=INT", "arg", "4", "Number of BFGS updates (default 4)")
      ('s', "savesum=FILE", "arg", "", "save summary information")
      ('\0', "no-write", "", "", "Don't write anything")
      ('i', "info=INT", "arg", "0", "info level")
      ;
    config.default_parse(argc, argv);

    info = config["info"].get_int();
    out_model_name = config["out"].get_str();

    optimizer.set_verbosity(info);

    if (!(config["mmi"].specified || config["mpe"].specified || config["ml"].specified))
      throw std::string("Must give at least one of --ml, --mmi or --mpe");
    if (config["ml"].specified)
    {
      ml_weight = config["ml"].get_float();
      statistics_mode |= PDF_ML_STATS;
    }
    if (config["mmi"].specified)
    {
      mmi_weight = config["mmi"].get_float();
      statistics_mode |= (PDF_ML_STATS | PDF_MMI_STATS);
    }
    if (config["mpe"].specified)
    {
      mpe_weight = config["mpe"].get_float();
      statistics_mode |= (PDF_MPE_NUM_STATS|PDF_MPE_DEN_STATS);
    }

    if (config["msmooth"].specified)
    {
      if (!config["mpe"].specified)
        throw std::string("--msmooth requires --mpe");
      msmooth_tau = config["msmooth"].get_float();
      statistics_mode |= PDF_ML_STATS;
    }
    if (config["gsmooth"].specified)
    {
      if (!config["mpe"].specified)
        throw std::string("--gsmooth requires --mpe");
      gsmooth_tau = config["gsmooth"].get_float();
      statistics_mode |= PDF_ML_STATS;
    }

    // Load the previous models
    if (config["base"].specified)
    {
      model.read_all(config["base"].get_str());
      base_file_name = config["base"].get_str();
    }
    else if (config["gk"].specified && config["mc"].specified &&
             config["ph"].specified)
    {
      model.read_gk(config["gk"].get_str());
      model.read_mc(config["mc"].get_str());
      model.read_ph(config["ph"].get_str());
      base_file_name = config["gk"].get_str();
    }
    else
    {
      throw std::string("Must give either --base or all --gk, --mc and --ph");
    }

    // Open the list of statistics files
    std::ifstream filelist(config["list"].get_str().c_str());
    if (!filelist)
    {
      fprintf(stderr, "Could not open %s\n", config["list"].get_str().c_str());
      exit(1);
    }

    optimizer.set_max_bfgs_updates(config["bfgsu"].get_int());
    
    state_file = config["osf"].get_str();
    min_var = config["minvar"].get_float();
    ac_scale = config["ac-scale"].get_float();
    
    // Accumulate statistics
    model.start_accumulating(statistics_mode);
    while (filelist >> statistics_file && statistics_file != " ") {
      model.accumulate_gk_from_dump(statistics_file+".gks");
      model.accumulate_mc_from_dump(statistics_file+".mcs");
      std::string lls_file_name = statistics_file+".lls";
      std::ifstream lls_file(lls_file_name.c_str());
      while (lls_file.good())
      {
        char buf[256];
        std::string temp;
        std::vector<std::string> fields;
        lls_file.getline(buf, 256);
        temp.assign(buf);
        str::split(&temp, ":", false, &fields, 2);
        if (fields.size() == 2)
        {
          double value = strtod(fields[1].c_str(), NULL);
          if (sum_statistics.find(fields[0]) == sum_statistics.end())
            sum_statistics[fields[0]] = value;
          else
            sum_statistics[fields[0]] = sum_statistics[fields[0]] + value;
        }
      }
      lls_file.close();
    }

    if (mpe_weight != 0 &&
        sum_statistics.find("MPFE score") == sum_statistics.end())
      throw std::string("MPFE score not available");
    if (mmi_weight != 0 &&
        sum_statistics.find("MMI score") == sum_statistics.end())
      throw std::string("MMI score not available");
    if (ml_weight != 0 &&
        sum_statistics.find("Numerator loglikelihood") == sum_statistics.end())
      throw std::string("Numerator loglikelihood not available");
    if (sum_statistics.find("Number of frames") == sum_statistics.end())
      throw std::string("Number of frames not available");

    score = 0;
    num_frames = (int)sum_statistics["Number of frames"];
    if (mpe_weight != 0)
    {
      if (config["pher"].specified)
      {
        score += mpe_weight*sum_statistics["MPFE score"]/(double)num_frames;
        mpe_weight = -mpe_weight;
      }
      else
      {
        // Change the value from phone accuracy to phone error in order to
        // turn the optimization problem into minimization
        score += mpe_weight*(1-sum_statistics["MPFE score"]/(double)num_frames);
      }
    }
    if (mmi_weight != 0)
    {
      // Negative MMI score, normalized with number of frames
      score += -mmi_weight*sum_statistics["MMI score"]/(double)num_frames;
    }
    if (ml_weight != 0)
    {
      // Negative loglikelihood, normalized with number of frames
      score += -ml_weight*sum_statistics["Numerator loglikelihood"]/(double)num_frames;
    }

    if (msmooth_tau != 0)
      score -= get_msmooth_score();
    if (gsmooth_tau != 0)
      score -= get_gsmooth_score();
    
    if (config["initscale"].specified)
    {
      optimizer.set_inv_hessian_scale(config["initscale"].get_float());
      initialize_optimization_parameters();
    }
    else
    {
      // Load optimization state and model parameters
      if (!optimizer.load_optimization_state(state_file))
      {
        fprintf(stderr, "Could not read %s, start optimization with --initscale\n", state_file.c_str());
        exit(1);
      }
    }

    optimizer.set_function_value(score);
    extract_gradient();
    if (config["limit"].specified)
    {
      if (config["limit"].get_float() <= 0 || config["limit"].get_float() >= 1)
      {
        fprintf(stderr, "The limit must be 0 < limit < 1\n");
        exit(1);
      }
      param_limiter.set_limit(config["limit"].get_float());
      optimizer.set_limit_interface(&param_limiter);
    }

    // Perform the optimization step
    optimizer.optimization_step();

    if (!optimizer.converged())
    {
      if (!config["no-write"].specified)
      {
        // Write the resulting models
        set_model_parameters();
        model.write_all(out_model_name);
        // Write the optimization state
        optimizer.write_optimization_state(state_file);
      }
    }
    else
      fprintf(stderr, "The model has converged!\n");
    
    if (config["savesum"].specified  && !config["no-write"].specified) {
      std::string summary_file_name = config["savesum"].get_str();
      std::ofstream summary_file(summary_file_name.c_str(),
                                 std::ios_base::app);
      if (!summary_file)
        fprintf(stderr, "Could not open summary file: %s\n",
                summary_file_name.c_str());
      else
      {
        summary_file << base_file_name << std::endl;
        for (std::map<std::string, double>::const_iterator it =
               sum_statistics.begin(); it != sum_statistics.end(); it++)
        {
          summary_file << "  " << (*it).first << ": " << (*it).second <<
            std::endl;
        }
      }
      summary_file.close();
    }
  }
  
  catch (std::exception &e) {
    fprintf(stderr, "exception: %s\n", e.what());
    abort();
  }
  
  catch (std::string &str) {
    fprintf(stderr, "exception: %s\n", str.c_str());
    abort();
  }
}
