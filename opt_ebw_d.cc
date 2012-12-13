#include <fstream>
#include <string>
#include <iostream>
#include <stdlib.h>
#include <algorithm>

#include "io.hh"
#include "str.hh"
#include "conf.hh"
#include "HmmSet.hh"

using namespace aku;

std::string statistics_file;
std::string state_file;

int info;

conf::Config config;
HmmSet model;
HmmSet train_stats;

double ac_scale;
int num_frames = 0;

Vector gaussian_d; // Dimension: Number of Gaussians
Vector d_params; // Dimension depends on the clustering mode
Vector gradient;
Vector prev_step;
Vector prev_gradient;
Vector gaussian_min_d;
Vector gaussian_max_d;
Vector param_min_d;
Vector param_max_d;

typedef enum {CL_NONE, CL_MIXTURE, CL_PHOSTATE, CL_GLOBAL} CL_TYPE;
CL_TYPE clustering_mode = CL_NONE;

bool mpe_gradient = false;

double max_update_step = 0.182; // Max 1.2*D
double qp_max_step_increase = 2;
double qp_epsilon = 0.02;

bool relaxed_gaussian_minimum = false;
// Multiplier for true minimum D for Gaussians, usually 0.5 < gamma < 1.0
double gmin_gamma = 0;

int num_gradient = 0;
int num_hessian = 0;
int num_negative_hessian = 0;
int num_smoothed_hessian = 0;
int num_step_limit = 0;
int num_acceleration_limit = 0;
int num_min_limit = 0;
int num_max_limit = 0;
int num_discarded_gaussians = 0;


typedef enum {D_ML, D_MMI, D_MPE} OPT_TYPE;
OPT_TYPE control_mode;
OPT_TYPE train_mode;

typedef std::map<std::string, int> PhoStateMap;
PhoStateMap pho_state_index;

std::vector<int> gauss_cluster;

double log_prior_coefficient = 0;


int prepare_mixture_clustering(void)
{
  PDFPool *pool = model.get_pool();  
  gauss_cluster.resize(pool->size(), -1);
  for (int i = 0; i < model.num_emission_pdfs(); i++)
  {
    Mixture *m = model.get_emission_pdf(i);
    for (int j = 0; j < (int)m->size(); j++)
    {
      int pool_index = m->get_base_pdf_index(j);
      gauss_cluster[pool_index] = i;
    }
  }
  fprintf(stderr, "Mixture clustering, %d clusters\n",
          (int)model.num_emission_pdfs());
  return model.num_emission_pdfs();
}


int prepare_pho_state_clustering(void)
{
  PDFPool *pool = model.get_pool();  
  gauss_cluster.resize(pool->size(), -1);
  for (int i = 0; i < model.num_hmms(); i++)
  {
    Hmm &hmm = model.hmm(i);
    std::string phoneme = hmm.get_center_phone();
    for (int j = 0; j < hmm.num_states(); j++)
    {
      char buf[phoneme.size() + 10];
      sprintf(buf, "%s.%d", phoneme.c_str(), j);
      std::string label = buf;
      PhoStateMap::iterator it = pho_state_index.find(label);
      if (it == pho_state_index.end())
      {
        int index = pho_state_index.size();
        it = pho_state_index.insert(PhoStateMap::value_type(label, index)).first;
      }
      Mixture *m = model.get_emission_pdf(
        model.state(hmm.state(j)).emission_pdf);
      for (int k = 0; k < (int)m->size(); k++)
      {
        int pool_index = m->get_base_pdf_index(k);
        gauss_cluster[pool_index] = (*it).second;
      }
    }
  }
  fprintf(stderr, "Phoneme/state clustering, %d clusters\n",
          (int)pho_state_index.size());
  return pho_state_index.size();
}


int prepare_global_clustering(void)
{
  PDFPool *pool = model.get_pool();  
  gauss_cluster.resize(pool->size());
  for (int i = 0; i < pool->size(); i++)
    gauss_cluster[i] = 0;
  fprintf(stderr, "Global clustering, 1 cluster\n");
  return 1;
}


void
read_d_file(std::ifstream &d_file, int num_d, Vector &d,
            Vector &min_d, Vector &max_d)
{
  for (int i = 0; i < num_d; i++)
  {
    if (!d_file.good())
      throw std::string("Failed to read D values");
    char buf[256];
    std::string temp;
    std::vector<std::string> fields;
    d_file.getline(buf, 256);
    temp.assign(buf);
    str::split(&temp, " ", true, &fields, 4);
    if (fields.size() == 3)
    {
      double value = strtod(fields[0].c_str(), NULL);
      double mind = strtod(fields[1].c_str(), NULL);
      double maxd = strtod(fields[2].c_str(), NULL);
      // LIMITS ARE NOT ENFORCED!
      if (mind < 0 || (maxd > 0 && maxd < mind))
        throw std::string("Invalid value in D file");
      d(i) = std::max(value, 0.0);
      min_d(i) = mind;
      max_d(i) = maxd;
    }
    else
      throw std::string("Invalid format in D file");
  }
}


void
set_gaussian_parameters(void)
{
  PDFPool *pool = model.get_pool();
  for (int i = 0; i < pool->size(); i++)
  {
    int d_index = i;
    if (clustering_mode != CL_NONE)
      d_index = gauss_cluster[i];
    double d = d_params(d_index);
    if (d < gaussian_min_d(i))
    {
      if (relaxed_gaussian_minimum)
      {
        double alpha = (1-gmin_gamma)*gaussian_min_d(i);
        double beta = 1/alpha;
        gaussian_d(i) = alpha*exp(beta*(d-gaussian_min_d(i))) +
          gmin_gamma*gaussian_min_d(i);
      }
      else
      {
        gaussian_d(i) = gaussian_min_d(i);
      }
    }
    else if (gaussian_max_d(i) > 0 && d > gaussian_max_d(i))
      gaussian_d(i) = gaussian_max_d(i);
    else
      gaussian_d(i) = d;
  }
}



// Read the Gaussian specific D-values and set optimization parameters
void set_optimization_parameters(std::string &gauss_d_file_name,
                                 std::string &cluster_d_file_name)
{
  PDFPool *pool = model.get_pool();
  int num_params = 0;
  
  // Allocate parameter vector and fill in the parameters
  if (clustering_mode == CL_MIXTURE)
  {
    num_params = prepare_mixture_clustering();
  }
  else if (clustering_mode == CL_PHOSTATE)
  {
    num_params = prepare_pho_state_clustering();
  }
  else if (clustering_mode == CL_GLOBAL)
  {
    num_params = prepare_global_clustering();
  }
  else
    num_params = pool->size();
  if (clustering_mode != CL_NONE)
  {
    // Check Gaussian cluster indexes are ok
    for (int i = 0 ;i < pool->size(); i++)
      assert( gauss_cluster[i] >= 0 );
  }
  d_params.resize(num_params);
  prev_step.resize(num_params);
  gaussian_d.resize(pool->size());
  gaussian_min_d.resize(pool->size());
  gaussian_min_d = 0;
  gaussian_max_d.resize(pool->size());
  gaussian_max_d = -1;

  bool gaussian_d_initialized = false;
  if (gauss_d_file_name.size() > 0)
  {
    std::ifstream gauss_d_file(gauss_d_file_name.c_str());
    if (!gauss_d_file.is_open())
      throw std::string("Error opening file") + gauss_d_file_name;

    read_d_file(gauss_d_file, pool->size(),
                gaussian_d, gaussian_min_d, gaussian_max_d);
    gaussian_d_initialized = true;
  }

  bool set_cluster_d_from_gaussians = false;

  assert( d_params.size() == num_params );
  param_min_d.resize(num_params);
  param_max_d.resize(num_params);
  if (cluster_d_file_name.size() == 0)
    set_cluster_d_from_gaussians = true;
  else
  {
    std::ifstream cluster_d_file(cluster_d_file_name.c_str());
    if (!cluster_d_file.is_open())
      set_cluster_d_from_gaussians = true;
    else
      read_d_file(cluster_d_file, num_params,
                  d_params, param_min_d, param_max_d);
  }
  if (set_cluster_d_from_gaussians)
    fprintf(stderr, "Initializing clustered D's from Gaussian D's\n");
  
  // Set parameter limits. Use extremes if clustering the Gaussians
  if (set_cluster_d_from_gaussians)
  {
    assert( gaussian_d_initialized );
    d_params = -1;
    param_min_d = -1;
    param_max_d = -1;

    for (int i = 0; i < pool->size(); i++)
    {
      int d_index = i;
      if (clustering_mode != CL_NONE)
        d_index = gauss_cluster[i];

      if (d_params(d_index) < 0)
        d_params(d_index) = gaussian_d(i);
      else
        d_params(d_index) = std::min(d_params(d_index), gaussian_d(i));

      if (param_min_d(d_index) < 0)
        param_min_d(d_index) = gaussian_min_d(i);
      else
        param_min_d(d_index) = std::min(param_min_d(d_index),
                                        gaussian_min_d(i));
      if (param_max_d(d_index) < 0)
        param_max_d(d_index) = gaussian_max_d(i);
      else
        param_max_d(d_index) = std::max(param_max_d(d_index),
                                        gaussian_max_d(i));
    }
    
    set_gaussian_parameters();
  }
}


void write_d_file(std::string &d_file_name, Vector &d,
                  Vector &min_d, Vector &max_d)
{
  std::ofstream d_file(d_file_name.c_str(), std::ios_base::out);
  for (int i = 0; i < d.size(); i++)
  {
    d_file << d(i) << " " << min_d(i) << " " << max_d(i) << std::endl;
  }
  d_file.close();
}


// Extracts the gradient from model accumulators
void extract_gradient(void)
{
  PDFPool *pool = model.get_pool();
  PDFPool *train_pool = train_stats.get_pool();
  std::vector<double> temp;
  Vector gauss_gradient;

  gauss_gradient.resize(pool->size());

  // Means and covariances
  for (int i = 0; i < pool->size(); i++)
  {
    Gaussian *pdf = dynamic_cast< Gaussian* >(pool->get_pdf(i));
    Gaussian *train_pdf = dynamic_cast< Gaussian* >(train_pool->get_pdf(i));
    if (pdf == NULL || train_pdf == NULL)
      throw std::string("Only Gaussian PDFs are supported!");
    Vector control_m1, control_m2;
    Vector train_m1, train_m2;
    Vector train_mean, train_diag_cov;
    Vector new_mean, new_diag_cov;
    Vector mean_gradient;
    Vector cov_gradient;
    double control_gamma = 0, train_gamma = 0;
    
    train_pdf->get_mean(train_mean);
    train_pdf->get_covariance(train_diag_cov);
    
    mean_gradient.resize(pool->dim());
    cov_gradient.resize(pool->dim());
    if (control_mode == D_MPE)
    {
      pdf->get_accumulated_mean(PDF::MPE_NUM_BUF, control_m1);
      pdf->get_accumulated_second_moment(PDF::MPE_NUM_BUF, control_m2);
      control_gamma = pdf->get_accumulated_gamma(PDF::MPE_NUM_BUF);
      if (!mpe_gradient)
      {
        Vector temp;
        pdf->get_accumulated_mean(PDF::MPE_DEN_BUF, temp);
        Blas_Add_Mult(control_m1, -1, temp);
        pdf->get_accumulated_second_moment(PDF::MPE_DEN_BUF, temp);
        Blas_Add_Mult(control_m2, -1, temp);
        control_gamma -= pdf->get_accumulated_gamma(PDF::MPE_DEN_BUF);
      }
    }
    else if (control_mode == D_ML || control_mode == D_MMI)
    {
      pdf->get_accumulated_mean(PDF::ML_BUF, control_m1);
      pdf->get_accumulated_second_moment(PDF::ML_BUF, control_m2);
      control_gamma = pdf->get_accumulated_gamma(PDF::ML_BUF);
      if (control_mode == D_MMI)
      {
        Vector temp;
        pdf->get_accumulated_mean(PDF::MMI_BUF, temp);
        Blas_Add_Mult(control_m1, -1, temp);
        pdf->get_accumulated_second_moment(PDF::MMI_BUF, temp);
        Blas_Add_Mult(control_m2, -1, temp);
        control_gamma -= pdf->get_accumulated_gamma(PDF::MMI_BUF);
      }
    }
    if (train_mode == D_MPE)
    {
      train_pdf->get_accumulated_mean(PDF::MPE_NUM_BUF, train_m1);
      train_pdf->get_accumulated_second_moment(PDF::MPE_NUM_BUF, train_m2);
      train_gamma = train_pdf->get_accumulated_gamma(PDF::MPE_NUM_BUF);
      if (!mpe_gradient)
      {
        Vector temp;
        train_pdf->get_accumulated_mean(PDF::MPE_DEN_BUF, temp);
        Blas_Add_Mult(train_m1, -1, temp);
        train_pdf->get_accumulated_second_moment(PDF::MPE_DEN_BUF, temp);
        Blas_Add_Mult(train_m2, -1, temp);
        train_gamma -= train_pdf->get_accumulated_gamma(PDF::MPE_DEN_BUF);
      }
    }
    else if (train_mode == D_ML || train_mode == D_MMI)
    {
      train_pdf->get_accumulated_mean(PDF::ML_BUF, train_m1);
      train_pdf->get_accumulated_second_moment(PDF::ML_BUF, train_m2);
      train_gamma = train_pdf->get_accumulated_gamma(PDF::ML_BUF);
      if (train_mode == D_MMI)
      {
        Vector temp;
        train_pdf->get_accumulated_mean(PDF::MMI_BUF, temp);
        Blas_Add_Mult(train_m1, -1, temp);
        train_pdf->get_accumulated_second_moment(PDF::MMI_BUF, temp);
        Blas_Add_Mult(train_m2, -1, temp);
        train_gamma -= train_pdf->get_accumulated_gamma(PDF::MMI_BUF);
      }
    }

    double d_div=1;
    d_div = train_gamma + gaussian_d(i);
    if (fabs(d_div) < 1e-5)
      d_div = (d_div<0?-1:1)*1e-5; // Avoid division by zero

    new_mean = train_m1;
    Blas_Add_Mult(new_mean, gaussian_d(i), train_mean);
    Blas_Scale(1/d_div, new_mean);
    new_diag_cov = train_diag_cov;
    for (int j = 0; j < new_diag_cov.size(); j++)
      new_diag_cov(j) += train_mean(j)*train_mean(j);
    Blas_Scale(gaussian_d(i), new_diag_cov);
    Blas_Add_Mult(new_diag_cov, 1, train_m2);
    Blas_Scale(1/d_div, new_diag_cov);
    for (int j = 0; j < new_diag_cov.size(); j++)
    {
      new_diag_cov(j) -= new_mean(j)*new_mean(j);
      if (new_diag_cov(j) < 0.1) // FIXME: minvar
        new_diag_cov(j) = 0.1;
    }

    for (int j = 0; j < pool->dim(); j++)
    {
      mean_gradient(j) =
        -ac_scale*(control_m1(j)-new_mean(j)*control_gamma) / new_diag_cov(j);
      double c = new_diag_cov(j);
      cov_gradient(j) = -ac_scale * (
        (control_m2(j) - 2*control_m1(j)*new_mean(j) +
         control_gamma*new_mean(j)*new_mean(j) - control_gamma*c)
        / (2*c*c));
      if (new_diag_cov(j) < train_diag_cov(j) &&
          fabs(new_diag_cov(j)-0.1) < 1e-10) // FIXME: minvar
        cov_gradient(j) = 0;
    }

    // Apply chain rule to get the derivative with respect to Gaussian D
    d_div *= d_div;

    gauss_gradient(i) = 0;
    for (int j = 0; j < pool->dim(); j++)
    {
      double dmu = (-train_m1(j) + train_gamma*train_mean(j)) / d_div;
      double dsigma = (train_gamma*(train_mean(j)*train_mean(j) +
                                    train_diag_cov(j)) -
                       train_m2(j)) / d_div - 2*new_mean(j)*dmu;
      gauss_gradient(i) += mean_gradient(j) * dmu + cov_gradient(j) * dsigma;
    }
  }

  gradient.resize(d_params.size());
  if (clustering_mode == CL_MIXTURE)
    assert( d_params.size() == model.num_emission_pdfs() );
  else if (clustering_mode == CL_GLOBAL)
    assert( d_params.size() == 1 );
  else if (clustering_mode == CL_NONE)
    assert( d_params.size() == pool->size() );
  for (int i = 0; i < d_params.size(); i++)
      gradient(i) = 0;

  const double dtol = 0.1; // Tolerance for Gaussian/mixture D discrepency
  // Combine the cluster gradients as sums of Gaussian gradients
  for (int i = 0; i < pool->size(); i++)
  {
    int d_index = i;
    if (clustering_mode != CL_NONE)
      d_index = gauss_cluster[i];
    if (gaussian_d(i) > gaussian_min_d(i) &&
        (gaussian_max_d(i) <= 0 || gaussian_d(i) < gaussian_max_d(i)) &&
        fabs(d_params(d_index) - gaussian_d(i)) < dtol)
    {
      // Gaussian gradients already incorporate mixture weights
      gradient(d_index) += gauss_gradient(i);
    }
    else if (relaxed_gaussian_minimum && gaussian_d(i) <= gaussian_min_d(i))
    {
      double d_derivative = 1;
      if (d_params(d_index) < gaussian_min_d(i) && gaussian_min_d(i) > 0)
      {
        double beta = 1/((1-gmin_gamma)*gaussian_min_d(i));
        d_derivative = exp(beta*(d_params(d_index)-gaussian_min_d(i)));
      }
      gradient(d_index) += gauss_gradient(i) * d_derivative;
    }
    else
      num_discarded_gaussians++;
  }

  // Compute the gradient for log(D) and add prior gradient if needed
  for (int i = 0; i < d_params.size(); i++)
  {
    gradient(i) = gradient(i) * d_params(i);
    if (log_prior_coefficient != 0)
    {
      if (param_min_d(i) > 0 && param_max_d(i) > 0)
      {
        double normalized_log_prior =
          log_prior_coefficient/(double)d_params.size();
        assert( param_max_d(i) > param_min_d(i) );
        gradient(i) +=
          normalized_log_prior / ( log(param_max_d(i)) - log(param_min_d(i)) );
      }
    }
  }
  
  for (int i = 0; i < d_params.size(); i++)
  {
    printf("%g %g\n", gradient(i), d_params(i));
  }
}


bool qp_optimization_step(void)
{
  // QP operates in log(D) domain, although d_params consists real D values
  const double tiny = 1e-4;
  double update_step_2norm = 0;
  assert( d_params.size() == gradient.size() );
  assert( gradient.size() == prev_gradient.size());
  assert( gradient.size() == prev_step.size());
  for (int i = 0; i < d_params.size(); i++)
  {
    double hessian;
    double update_step = 0;
    double gradient_update = -qp_epsilon * gradient(i);
    enum {GRADIENT, HESSIAN, SMOOTHED_HESSIAN} step_mode;
    update_step = gradient_update; // Default: gradient step (log-domain)
    step_mode = GRADIENT;
    if (fabs(prev_step(i)) > tiny)
    {
      hessian = (gradient(i) - prev_gradient(i)) / prev_step(i);
      if (hessian > 0) // Looking for a minimum
      {
        update_step = -gradient(i) / hessian; // Hessian step
        step_mode = HESSIAN;
        if (gradient(i) * prev_gradient(i) > 0)
        {
          update_step += gradient_update; // Smooth with gradient
          step_mode = SMOOTHED_HESSIAN;
        }
      }
      else
      {
        num_negative_hessian++;
        if (fabs(update_step) < fabs(prev_step(i)))
          update_step = prev_step(i);
      }
    }

    if (fabs(update_step) > fabs(qp_max_step_increase * prev_step(i)))
    {
      update_step *= fabs(qp_max_step_increase * prev_step(i))/fabs(update_step);
      num_acceleration_limit++;
    }
    if (fabs(update_step) > max_update_step)
    {
      update_step *= max_update_step/fabs(update_step);
      num_step_limit++;
    }
    
    if (update_step*gradient(i) > 0 || fabs(update_step) < tiny)
    {
      update_step = gradient_update;
      step_mode = GRADIENT;
    }

    if (step_mode == GRADIENT)
      num_gradient++;
    else if (step_mode == HESSIAN)
      num_hessian++;
    else if (step_mode == SMOOTHED_HESSIAN)
      num_smoothed_hessian++;

    // Do the update
    double old_log_param = log(d_params(i));
    double old_param = d_params(i);
    d_params(i) = exp(old_log_param + update_step);
    if (d_params(i) < param_min_d(i))
    {
      d_params(i) = param_min_d(i);
      num_min_limit++;
    }
    else if (param_max_d(i) > 0 && d_params(i) > param_max_d(i))
    {
      d_params(i) = param_max_d(i);
      num_max_limit++;
    }
    prev_step(i) = log(d_params(i)) - old_log_param;
    double temp = d_params(i) - old_param;
    update_step_2norm += temp*temp;
  }

  if (update_step_2norm/d_params.size() < 0.001)
    return true;
  
  return false;
}


double log_prior_score(void)
{
  double prior_score = 0;
  assert( d_params.size() == param_min_d.size() );
  assert( d_params.size() == param_max_d.size() );
  if (log_prior_coefficient != 0)
  {
    double normalized_log_prior = log_prior_coefficient/(double)d_params.size();
    for (int i = 0; i < d_params.size(); i++)
    {
      if (param_min_d(i) > 0 && param_max_d(i) > 0)
      {
        assert( param_max_d(i) > param_min_d(i) );
        prior_score += (log(d_params(i)) - log(param_min_d(i))) /
          (log(param_max_d(i)) - log(param_min_d(i)));
      }
    }
    prior_score *= normalized_log_prior;
  }
  return prior_score;
}


void gradient_step(double step_size)
{
  for (int i = 0; i < d_params.size(); i++)
  {
    double update_step = -step_size * gradient(i);
    if (fabs(update_step) > max_update_step)
    {
      update_step *= max_update_step/fabs(update_step);
      num_step_limit++;
    }

    double old_log_param = log(d_params(i));
    d_params(i) = exp(old_log_param + update_step);
    num_gradient++;
    if (d_params(i) < param_min_d(i))
    {
      d_params(i) = param_min_d(i);
      num_min_limit++;
    }
    else if (param_max_d(i) > 0 && d_params(i) > param_max_d(i))
    {
      d_params(i) = param_max_d(i);
      num_max_limit++;
    }
    prev_step(i) = log(d_params(i)) - old_log_param;
  }
}


bool
read_vector(FILE *fp, Vector *v)
{
  int size;
  if (fread(&size, sizeof(int), 1, fp) < 1)
    return false;
  v->resize(size);
  for (int i = 0; i < v->size(); i++)
  {
    if (fread(&((*v)(i)), sizeof(double), 1, fp) < 1)
      return false;
  }
  return true;
}


void
write_vector(FILE *fp, Vector *v)
{
  int size = v->size();
  fwrite(&size, sizeof(int), 1, fp);
  for (int i = 0; i < v->size(); i++)
  {
    if (fwrite(&((*v)(i)), sizeof(double), 1, fp) < 1)
      throw std::string("Write error");
  }
}


void
write_qp_state(std::string &filename)
{
  FILE *fp;
  if ((fp=fopen(filename.c_str(), "wb")) == NULL)
    throw std::string("Could not open file ") + filename + std::string(" for writing");
  //write_vector(fp, &d_params);
  write_vector(fp, &prev_step);
  write_vector(fp, &gradient);
  fclose(fp);
}


bool
read_qp_state(std::string &filename)
{
  FILE *fp;
  if ((fp=fopen(filename.c_str(), "rb")) == NULL)
  {
    return false;
    //throw std::string("Could not open file ") + filename + std::string(" for reading");
  }
  if (//!read_vector(fp, &d_params) ||
      !read_vector(fp, &prev_step) ||
      !read_vector(fp, &prev_gradient))
  {
    //throw std::string("Read error");
    return false;
  }
  fclose(fp);
  return true;
}


int
main(int argc, char *argv[])
{
  double score = 0;
  std::map< std::string, double > sum_statistics;
  std::string base_file_name;
  PDF::StatisticsMode control_stats_mode = 0;
  PDF::StatisticsMode train_stats_mode = 0;
  std::string source_gaussian_d_file;
  
  try {
    config("usage: opt_ebw_d [OPTION...]\n")
      ('h', "help", "", "", "display help")
      ('b', "base=BASENAME", "arg", "", "Previous base filename for model files")
      ('g', "gk=FILE", "arg", "", "Previous mixture base distributions")
      ('m', "mc=FILE", "arg", "", "Previous mixture coefficients for the states")
      ('p', "ph=FILE", "arg", "", "Previous HMM definitions")
      ('L', "list=LISTNAME", "arg must", "", "Development set statistics")
      ('T', "tstats=LISTNAME", "arg must", "", "Training set statistics")
      ('F', "osf=FILE", "arg must", "", "Optimization state file")
      ('D', "ebwd=FILE", "arg", "", "EBW D values and limits (per Gaussian)")
      ('o', "gauss-out=FILE", "arg must", "", "Output file for Gaussian EBW D values and limits")
      ('\0', "cluster-d=FILE", "arg must", "", "Input/output file for clustered D values and limits")
      ('\0', "control=MODE", "arg must", "", "Control criterion: ML/MMI/MPE")
      ('\0', "train=MODE", "arg must", "", "Training criterion: ML/MMI/MPE")
      ('\0', "grad", "", "", "Gradient based statistics (with MPE)")
      ('\0', "cluster=MODE", "arg", "", "Cluster Gaussian D values: global/pho/mix")
      ('\0', "qp-eps=FLOAT", "arg", "2", "QuickProp epsilon (gradient multiplier")
      ('l', "initscale=SCALE", "arg", "1", "Initial gradient step size")
      ('I', "d-init", "", "", "Initialize Gaussian D file")
      ('P', "prior=FLOAT", "arg", "1", "Add log prior")
      ('\0', "gmin=FLOAT", "arg", "0.75", "Multiplier for relaxed Gaussian-D minimum")
      ('A', "ac-scale=FLOAT", "arg", "1", "acoustic scaling used in stats")
      ('s', "savesum=FILE", "arg", "", "save summary information")
      ('i', "info=INT", "arg", "0", "info level")
      ;
    config.default_parse(argc, argv);

    info = config["info"].get_int();

    conf::Choice criterion_choice;
    criterion_choice("ml", D_ML)("mmi", D_MMI)("mpe", D_MPE);
    std::string str = config["control"].get_str();
    std::transform(str.begin(), str.end(), str.begin(), ::tolower);
    int result = 0;
    if (!criterion_choice.parse(str, result))
      throw std::string("Invalid choice for --control: ") + str;
    control_mode = (OPT_TYPE)result;

    str = config["train"].get_str();
    std::transform(str.begin(), str.end(), str.begin(), ::tolower);
    if (!criterion_choice.parse(str, result))
      throw std::string("Invalid choice for --train: ") + str;
    train_mode = (OPT_TYPE)result;

    conf::Choice cluster_choice;
    cluster_choice("global", CL_GLOBAL)("pho", CL_PHOSTATE)
      ("mix", CL_MIXTURE);
    if (config["cluster"].specified)
    {
      str = config["cluster"].get_str();
      std::transform(str.begin(), str.end(), str.begin(), ::tolower);
      int result = 0;
      if (!cluster_choice.parse(str, result))
        throw std::string("Invalid choice for --cluster: ") + str;
      clustering_mode = (CL_TYPE)result;
    }

    if (clustering_mode != CL_NONE && !config["cluster-d"].specified)
    {
      throw std::string("Clustering requires --cluster-d");
    }

    qp_epsilon = config["qp-eps"].get_float();
    
    if (config["grad"].specified)
      mpe_gradient = true;

    if (config["prior"].specified)
      log_prior_coefficient = config["prior"].get_float();

    if (config["gmin"].specified)
    {
      relaxed_gaussian_minimum = true;
      gmin_gamma = config["gmin"].get_float();
    }
    
    if (control_mode == D_ML)
      control_stats_mode = PDF_ML_STATS;
    else if (control_mode == D_MMI)
      control_stats_mode = (PDF_ML_STATS | PDF_MMI_STATS);
    else if (control_mode == D_MPE)
    {
      control_stats_mode = PDF_MPE_NUM_STATS;
      if (!mpe_gradient)
        control_stats_mode |= PDF_MPE_DEN_STATS;
    }
    else
      throw std::string("Invalid control statistics mode");

    if (train_mode == D_ML)
      train_stats_mode = PDF_ML_STATS;
    else if (train_mode == D_MMI)
      train_stats_mode = (PDF_ML_STATS | PDF_MMI_STATS);
    else if (train_mode == D_MPE)
    {
      train_stats_mode = PDF_MPE_NUM_STATS;
      if (!mpe_gradient)
        train_stats_mode |= PDF_MPE_DEN_STATS;
    }
    else
      throw std::string("Invalid training statistics mode");

    // Load the previous models
    if (config["base"].specified)
    {
      model.read_all(config["base"].get_str());
      train_stats.read_all(config["base"].get_str());
      base_file_name = config["base"].get_str();
    }
    else if (config["gk"].specified && config["mc"].specified &&
             config["ph"].specified)
    {
      model.read_gk(config["gk"].get_str());
      model.read_mc(config["mc"].get_str());
      model.read_ph(config["ph"].get_str());
      train_stats.read_gk(config["gk"].get_str());
      train_stats.read_mc(config["mc"].get_str());
      train_stats.read_ph(config["ph"].get_str());
      base_file_name = config["gk"].get_str();
    }
    else
    {
      throw std::string("Must give either --base or all --gk, --mc and --ph");
    }

    state_file = config["osf"].get_str();
    ac_scale = config["ac-scale"].get_float();
    source_gaussian_d_file = config["ebwd"].get_str();

    std::string cluster_d_file = config["cluster-d"].get_str();
    set_optimization_parameters(source_gaussian_d_file, cluster_d_file);

    if (config["d-init"].specified)
    {
      set_gaussian_parameters();
      std::string target_gaussian_d_file = config["gauss-out"].get_str();
      write_d_file(target_gaussian_d_file, gaussian_d,
                   gaussian_min_d, gaussian_max_d);
      exit(0);
    }
    
    if (!config["initscale"].specified)
    {
      // Load optimization state and model parameters
      if (!read_qp_state(state_file))
      {
        fprintf(stderr, "Could not read %s, start optimization with --initscale\n", state_file.c_str());
        exit(1);
      }
    }


    // Open the list of statistics files
    std::ifstream filelist(config["list"].get_str().c_str());
    if (!filelist)
    {
      fprintf(stderr, "Could not open %s\n", config["list"].get_str().c_str());
      exit(1);
    }
    
    // Accumulate statistics
    model.start_accumulating(control_stats_mode);
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

    // Open the list of statistics files
    std::ifstream train_stat_list(config["tstats"].get_str().c_str());
    if (!train_stat_list)
    {
      fprintf(stderr, "Could not open %s\n", config["tstats"].get_str().c_str());
      exit(1);
    }
    // Accumulate training statistics
    train_stats.start_accumulating(train_stats_mode);
    while (train_stat_list >> statistics_file && statistics_file != " ") {
      train_stats.accumulate_gk_from_dump(statistics_file+".gks");
      train_stats.accumulate_mc_from_dump(statistics_file+".mcs");
    }


    if (control_mode == D_MPE &&
        sum_statistics.find("MPE score") == sum_statistics.end())
      throw std::string("MPE score not available");
    if (control_mode == D_MMI &&
        sum_statistics.find("MMI score") == sum_statistics.end())
      throw std::string("MMI score not available");
    if (control_mode == D_ML &&
        sum_statistics.find("Numerator loglikelihood") == sum_statistics.end())
      throw std::string("Numerator loglikelihood not available");
    if (sum_statistics.find("Number of frames") == sum_statistics.end())
      throw std::string("Number of frames not available");

    score = 0; // With QP, not used at all...
    num_frames = (int)sum_statistics["Number of frames"];
    if (control_mode == D_MPE)
    {
      // Negate the phone accuracy in order to
      // turn the optimization problem into minimization
      score += -sum_statistics["MPE score"];
    }
    if (control_mode == D_MMI)
    {
      // Negative the MMI score
      score += -sum_statistics["MMI score"];
    }
    if (control_mode == D_ML)
    {
      // Negative loglikelihood, normalized with number of frames
      score += -sum_statistics["Numerator loglikelihood"];
    }

    score += log_prior_score();

    extract_gradient();

    // Perform the optimization step
    bool converged = false;
    if (config["initscale"].specified)
    {
      double step = config["initscale"].get_double();
      if (step > 0)
        gradient_step(step);
    }
    else
      converged = qp_optimization_step();
    if (!converged)
    {
      set_gaussian_parameters();

      // Write new D values
      std::string target_gaussian_d_file = config["gauss-out"].get_str();
      write_d_file(target_gaussian_d_file, gaussian_d,
                   gaussian_min_d, gaussian_max_d);
      if (config["cluster-d"].specified)
      {
        write_d_file(cluster_d_file, d_params,
                     param_min_d, param_max_d);
      }

      // Write the optimization state
      write_qp_state(state_file);
    }
    
    if (config["savesum"].specified) {
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

    fprintf(stderr, "score: %.3f\n", score);
    fprintf(stderr, "Gradient updates: %d\n", num_gradient);
    fprintf(stderr, "Negative Hessians: %d\n", num_negative_hessian);
    fprintf(stderr, "Hessian updates: %d\n", num_hessian);
    fprintf(stderr, "Smoothed Hessian updates: %d\n", num_smoothed_hessian);
    fprintf(stderr, "Step limits: %d\n", num_step_limit);
    fprintf(stderr, "Acceleration limits: %d\n", num_acceleration_limit);
    fprintf(stderr, "Minimum limits: %d\n", num_min_limit);
    fprintf(stderr, "Maximum limits: %d\n", num_max_limit);
//    if (clustering_mode != CL_NONE)
      fprintf(stderr, "Discarded Gaussians: %d\n", num_discarded_gaussians);

    if (converged)
      return 1;
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
