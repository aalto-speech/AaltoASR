#include <fstream>
#include <string>
#include <iostream>
#include <vector>

#include "io.hh"
#include "str.hh"
#include "conf.hh"
#include "HmmSet.hh"
#include "Pcgmm.hh"
#include "Scgmm.hh"

int info_flag;
int pcgmm_flag;
int scgmm_flag;
int kl_flag;

int dim;
int lambda_dim;
int basis_dim;
int num_gaussians;

conf::Config config;
HmmSet fc_model;
Pcgmm pcgmm;
Scgmm scgmm;
std::vector<float> kl_divergences;

void compress_pcgmm();
void compress_scgmm();


int
main(int argc, char *argv[])
{
  try {
    config("usage: basis_compress [OPTION...]\n")
      ('h', "help", "", "", "display help")
      ('g', "gk=FILE", "arg must", "", "full covariance gaussians")
      ('o', "out=FILE", "arg must", "", "compressed gaussians")
      ('b', "basis=FILE", "arg must", "", "basis to be used")
      ('d', "dim", "arg", "", "dimension of untied parameters, default=basis dimension")
      ('k', "kl=FILE", "arg", "", "kullback-leibler divergences for each gaussian")
      ('i', "info=INT", "arg", "0", "info level")
      ('\0', "hcl_bfgs_cfg=FILE", "arg", "", "configuration file for HCL biconjugate gradient algorithm")
      ('\0', "hcl_line_cfg=FILE", "arg", "", "configuration file for HCL line search algorithm")
      ;
    config.default_parse(argc, argv);
    
    info_flag = config["info"].get_int();
    kl_flag = config["kl"].specified;
    
    if (config["gk"].specified) {
      fc_model.read_gk(config["gk"].get_str());
      num_gaussians=fc_model.num_kernels();
      if (kl_flag)
	kl_divergences.resize(num_gaussians);     
    }
    else
      throw std::string("--gk=FILE must be specified");
    
    if (!config["out"].specified)
      throw std::string("--out=FILE must be specified");
    
    if (!config["basis"].specified)
      throw std::string("--basis=FILE must be specified");
    
  }
  catch (std::exception &e) {
    fprintf(stderr, "exception: %s\n", e.what());
    abort();
  }
  catch (std::string &str) {
    fprintf(stderr, "exception: %s\n", str.c_str());
    abort();
  }
  
  // Read parameters from the first line
  std::ifstream ibasis(config["basis"].get_str().c_str());
  std::string type_line;
  ibasis >> dim >> type_line >> basis_dim;
  ibasis.close();

  if (config["dim"].specified) {
    lambda_dim=config["dim"].get_int();
    if (lambda_dim>basis_dim)
      throw std::string("--dim is bigger than basis dimension");
  }
  else
    lambda_dim=basis_dim;

  if (type_line == "pcgmm") {
    pcgmm_flag=1; 
    scgmm_flag=0;
  }
  else if (type_line == "scgmm") {
    pcgmm_flag=0;
    scgmm_flag=1;
  }
  
  // Compress all gaussians
  if (pcgmm_flag)
    compress_pcgmm();
  else if (scgmm_flag)
    compress_scgmm();
  
  // Print kullback-leibler divergences to file if desired
  if (kl_flag) {
    std::ofstream klfile(config["kl"].get_str().c_str());
    for (int g=0; g<num_gaussians; g++)
      klfile << kl_divergences[g] << std::endl;    
  }

  return 1;
}


void compress_pcgmm() {

  pcgmm.reset(basis_dim, dim, num_gaussians);
  pcgmm.read_basis(config["basis"].get_str());  
  LaVectorDouble lambda(lambda_dim);
  LaGenMatDouble sample_cov(dim, dim);
  LaGenMatDouble model_cov(dim, dim);

  for (int k=0; k<fc_model.num_kernels(); k++) {
    
    // Fetch the next gaussian to be processed
    HmmKernel &fc_kernel=fc_model.kernel(k);
    // Fetch the target pcgmm gaussian
    Pcgmm::Gaussian &pcgmm_gaussian=pcgmm.get_gaussian(k);

    // Just copy mean    
    for (int i=0; i<dim; i++)
      pcgmm_gaussian.mu(i)=fc_kernel.center[i];

    // Copy covariance to sample_cov
    for (int i=0; i<dim; i++)
      for (int j=0; j<dim; j++)
	sample_cov(i,j)=fc_kernel.cov.full(i,j);
    
    // Initialize untied parameters
    lambda(LaIndex())=0;
    lambda(0)=1;
    
    // Optimize and set the optimized parameters
    pcgmm.optimize_lambda(sample_cov, lambda);
    pcgmm_gaussian.lambda.copy(lambda);

    // Save kullback-leibler divergences KL(sample_fc, model_fc)
    if (kl_flag) {
      pcgmm.calculate_covariance(lambda, model_cov);
      kl_divergences[k]=pcgmm.kullback_leibler_covariance(sample_cov, model_cov);
    }

    // Write pcgmm to file
    pcgmm.write_gk(config["out"].get_str());
  }
}


void compress_scgmm() {

  scgmm.reset(basis_dim, dim, num_gaussians);
  scgmm.read_basis(config["basis"].get_str());  
  LaVectorDouble lambda(lambda_dim);
  LaVectorDouble sample_mean(dim);
  LaGenMatDouble sample_cov(dim, dim);
  LaVectorDouble model_mean(dim);
  LaGenMatDouble model_cov(dim, dim);

  for (int k=0; k<fc_model.num_kernels(); k++) {
    
    // Fetch the next gaussian to be processed
    HmmKernel &fc_kernel=fc_model.kernel(k);
    // Fetch the target scgmm gaussian
    Scgmm::Gaussian &scgmm_gaussian=scgmm.get_gaussian(k);

    // Copy mean to sample_mean    
    for (int i=0; i<dim; i++)
      sample_mean(i)=fc_kernel.center[i];

    // Copy covariance to sample_cov
    for (int i=0; i<dim; i++)
      for (int j=0; j<dim; j++)
	sample_cov(i,j)=fc_kernel.cov.full(i,j);
    
    // Initialize untied parameters
    lambda(LaIndex())=0;
    lambda(0)=1;
    
    // Optimize and set the optimized parameters
    if (config["hcl_bfgs_cfg"].specified)
      scgmm.set_hcl_bfgs_cfg_file(config["hcl_bfgs_cfg"].get_str());
    if (config["hcl_line_cfg"].specified)
      scgmm.set_hcl_line_cfg_file(config["hcl_line_cfg"].get_str());

    scgmm.optimize_lambda(sample_cov, sample_mean, lambda);
    scgmm_gaussian.lambda.copy(lambda);

    // Save kullback-leibler divergences KL(sample_fc, model_fc)
    if (kl_flag) {
      scgmm.calculate_mu(lambda, model_mean);
      scgmm.calculate_covariance(lambda, model_cov);
      kl_divergences[k]=scgmm.kullback_leibler(sample_mean, sample_cov,
					       model_mean, model_cov);
    }

    // Write scgmm to file
    scgmm.write_gk(config["out"].get_str());
  }    
}


