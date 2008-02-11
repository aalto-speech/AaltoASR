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

typedef enum {MODE_ML, MODE_MMI, MODE_MPE} OptMode;

OptMode mode;



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
      params(pindex++) = log(temp(j)-min_var);
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
      temp(j) = min_var + exp(params(pindex++));
    pdf->set_covariance(temp);
  }
}


// Extracts the gradient from model accumulators
void extract_gradient(void)
{
  int pindex = 0;
  PDFPool *pool = model.get_pool();
  Vector params;
  Vector gradient;
  std::vector<double> temp;

  optimizer.get_parameters(params);
  gradient.resize(optimizer.get_num_parameters());
  
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
      if (mode == MODE_MPE)
      {
        temp[j] = -(ac_scale/(double)num_frames) *
          (m->get_accumulated_gamma(PDF::MPE_NUM_BUF, j) /
           m->get_mixture_coefficient(j));
      }
      else if (mode == MODE_MMI)
      {
        temp[j] = -(ac_scale/(double)num_frames) * (
          (m->get_accumulated_gamma(PDF::ML_BUF, j) /
           m->get_mixture_coefficient(j)) -
          (m->get_accumulated_gamma(PDF::MMI_BUF, j) /
           m->get_mixture_coefficient(j)));
      }
      else if (mode == MODE_ML)
      {
        temp[j] = -(ac_scale/(double)num_frames) * 
          (m->get_accumulated_gamma(PDF::ML_BUF, j) /
           m->get_mixture_coefficient(j));
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
    Vector m1, nm1;
    Vector m2, nm2;
    Vector mean;
    Vector diag_cov;
    double gamma = 0, ngamma = 0;
    pdf->get_mean(mean);
    pdf->get_covariance(diag_cov);
    if (mode == MODE_MPE)
    {
      pdf->get_accumulated_mean(PDF::MPE_NUM_BUF, m1);
      pdf->get_accumulated_second_moment(PDF::MPE_NUM_BUF, m2);
      gamma = pdf->get_accumulated_gamma(PDF::MPE_NUM_BUF);
    }
    else if (mode == MODE_MMI)
    {
      pdf->get_accumulated_mean(PDF::ML_BUF, nm1);
      pdf->get_accumulated_mean(PDF::MMI_BUF, m1);
      pdf->get_accumulated_second_moment(PDF::ML_BUF, nm2);
      pdf->get_accumulated_second_moment(PDF::MMI_BUF, m2);
      ngamma = pdf->get_accumulated_gamma(PDF::ML_BUF);
      gamma = pdf->get_accumulated_gamma(PDF::MMI_BUF);
    }
    else if (mode == MODE_ML)
    {
      pdf->get_accumulated_mean(PDF::ML_BUF, nm1);
      pdf->get_accumulated_second_moment(PDF::ML_BUF, nm2);
      ngamma = pdf->get_accumulated_gamma(PDF::ML_BUF);
    }
    for (int j = 0; j < pool->dim(); j++)
    {
      if (mode == MODE_MPE)
      {
        gradient(pindex) =
          -(ac_scale*(m1(j)-mean(j)*gamma) / (diag_cov(j)*(double)num_frames));
      }
      else if (mode == MODE_MMI)
      {
        gradient(pindex) = -(ac_scale/(double)num_frames) * (
          (nm1(j)-mean(j)*ngamma)/diag_cov(j) -
          (m1(j)-mean(j)*gamma)/diag_cov(j));
      }
      else if (mode == MODE_ML)
      {
        gradient(pindex) = -(ac_scale/(double)num_frames) *
          (nm1(j)-mean(j)*ngamma)/diag_cov(j);
      }
      pindex++;
    }
    for (int j = 0; j < pool->dim(); j++)
    {
      double ep = exp(params(pindex));
      double c = ep + min_var;
      if (mode == MODE_MPE)
      {
        gradient(pindex) = -(ac_scale/(double)num_frames) * (
          (m2(j)-2*m1(j)*mean(j)+gamma*mean(j)*mean(j)-gamma*c)/(2*c*c)) * ep;
      }
      else if (mode == MODE_MMI)
      {
        gradient(pindex) = -(ac_scale/(double)num_frames) * (
          (nm2(j)-2*nm1(j)*mean(j)+ngamma*mean(j)*mean(j)-ngamma*c)/(2*c*c) -
          (m2(j)-2*m1(j)*mean(j)+gamma*mean(j)*mean(j)-gamma*c)/(2*c*c)) * ep;
      }
      else if (mode == MODE_ML)
      {
        gradient(pindex) = -(ac_scale/(double)num_frames) * (
          (nm2(j)-2*nm1(j)*mean(j)+ngamma*mean(j)*mean(j)-ngamma*c)/(2*c*c))
          * ep;
      }
      pindex++;
    }
  }
  assert( pindex == optimizer.get_num_parameters() );

  optimizer.set_gradient(gradient);
}


int
main(int argc, char *argv[])
{
  double score = 0;
  std::map< std::string, double > sum_statistics;
  std::string base_file_name;
  
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
      ('\0', "ml", "", "", "ML optimization")
      ('\0', "mmi", "", "", "MMI optimization")
      ('\0', "mpe", "", "", "MPE optimization")
      ('l', "initscale=SCALE", "arg", "", "Initialize with inverse Hessian scale")
      ('\0', "minvar=FLOAT", "arg", "0.09", "minimum variance (default 0.09)")
      ('A', "ac-scale=FLOAT", "arg", "1", "acoustic scaling used in stats")
      ('\0', "bfgsu=INT", "arg", "4", "Number of BFGS updates (default 4)")
      ('s', "savesum=FILE", "arg", "", "save summary information")
      ('i', "info=INT", "arg", "0", "info level")
      ;
    config.default_parse(argc, argv);

    info = config["info"].get_int();
    out_model_name = config["out"].get_str();

    optimizer.set_verbosity(info);

    if (!(config["mmi"].specified^config["mpe"].specified^config["ml"].specified))
      throw std::string("Must give one of --ml, --mmi or --mpe");
    if (config["ml"].specified)
      mode = MODE_ML;
    else if (config["mmi"].specified)
      mode = MODE_MMI;
    else if (config["mpe"].specified)
      mode = MODE_MPE;
    else
      throw std::string("Invalid optimization mode");

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

    // Accumulate statistics
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

    if (mode == MODE_MPE &&
        sum_statistics.find("MPFE score") == sum_statistics.end())
      throw std::string("MPFE score not available");
    if (mode == MODE_MMI &&
        sum_statistics.find("MMI score") == sum_statistics.end())
      throw std::string("MMI score not available");
    if (mode == MODE_ML &&
        sum_statistics.find("Numerator loglikelihood") == sum_statistics.end())
      throw std::string("Numerator loglikelihood not available");
    if (sum_statistics.find("Number of frames") == sum_statistics.end())
      throw std::string("Number of frames not available");
    
    if (mode == MODE_MPE)
      score = sum_statistics["MPFE score"];
    else if (mode == MODE_MMI)
      score = sum_statistics["MMI score"];
    else if (mode == MODE_ML)
      score = sum_statistics["Numerator loglikelihood"];
    num_frames = (int)sum_statistics["Number of frames"];

    state_file = config["osf"].get_str();
    min_var = config["minvar"].get_float();
    ac_scale = config["ac-scale"].get_float();
    
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
    
    if (mode == MODE_MPE)
    {
      // Change the value from phone accuracy to phone error in order to
      // turn the optimization problem into minimization
      optimizer.set_function_value(1 - (score / (double)num_frames));
    }
    else if (mode == MODE_MMI)
    {
      // Negative MMI score, normalized with number of frames
      optimizer.set_function_value(-score / (double)num_frames);
    }
    else if (mode == MODE_ML)
    {
      // Negative loglikelihood, normalized with number of frames
      optimizer.set_function_value(-score / (double)num_frames);
    }

    extract_gradient();

    // Perform the optimization step
    optimizer.optimization_step();

    if (!optimizer.converged())
    {
      // Write the resulting models
      set_model_parameters();
      model.write_all(out_model_name);
      // Write the optimization state
      optimizer.write_optimization_state(state_file);
    }
    else
      fprintf(stderr, "The model has converged!\n");
    
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
