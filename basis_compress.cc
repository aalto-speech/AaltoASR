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

#include "HCL_Rn_d.h"

int info_flag;
int pcgmm_flag;
int scgmm_flag;
int kl_flag;
int old_flag;
int affine_flag;

int dim;
int lambda_dim;
int basis_dim;
int num_gaussians;

conf::Config config;
HmmSet fc_stats;
HmmSet model_states;
Pcgmm pcgmm;
Scgmm scgmm;
std::vector<float> kl_divergences;
std::ifstream stats_file;

void compress_pcgmm();
void compress_scgmm();


int
main(int argc, char *argv[])
{
  try {
    config("usage: basis_compress [OPTION...]\n")
      ('h', "help", "", "", "display help")
      ('g', "stats=FILE", "arg must", "", "full covariance statistics (gk and mc)")
      ('o', "out=FILE", "arg must", "", "compressed gaussians (gk and mc)")
      ('p', "old=FILE", "arg", "", ".gk with old basis coefficients (gk and mc)")
      ('b', "basis=FILE", "arg", "", "basis to be used")
      ('d', "dim", "arg", "", "dimension of untied parameters, default=basis dimension")
      ('k', "kl=FILE", "arg", "", "kullback-leibler divergences for each gaussian")
      ('i', "info=INT", "arg", "0", "info level")
      ('a', "affine", "", "", "affine subspace")
      ('\0', "hcl_bfgs_cfg=FILE", "arg", "", "configuration file for HCL biconjugate gradient algorithm")
      ('\0', "hcl_line_cfg=FILE", "arg", "", "configuration file for HCL line search algorithm")
      ;
    config.default_parse(argc, argv);

    old_flag = config["old"].specified;
    info_flag = config["info"].get_int();
    kl_flag = config["kl"].specified;
    affine_flag = config["affine"].specified;
    
    if (config["stats"].specified) {
      std::string dummy;
      fc_stats.read_mc(config["stats"].get_str()+".mc");
      stats_file.open((config["stats"].get_str()+".gk").c_str());
      stats_file >> num_gaussians >> dim >> dummy;
      assert(dummy=="full_cov");
      if (kl_flag)
	kl_divergences.resize(num_gaussians);
    }
    else
      throw std::string("--gk=FILE must be specified");
    
    if (!config["out"].specified)
      throw std::string("--out=FILE must be specified");
    
    if (config["basis"].specified && config["old"].specified)
      throw std::string("you don't have to define both --basis and --old");

    if (!config["basis"].specified && !config["old"].specified)
      throw std::string("you have to define either --basis or --old");
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
  std::string type_line;
  if (config["basis"].specified) {
    std::ifstream ibasis(config["basis"].get_str().c_str());
    ibasis >> dim >> type_line >> basis_dim;
    ibasis.close();
  }
  else if (config["old"].specified) {
    std::ifstream iold((config["old"].get_str()+".gk").c_str());
    int dummy;
    iold >> dummy >> dim >> type_line >> basis_dim;
    iold.close();
  }

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
}


void compress_pcgmm() {

  pcgmm.reset(basis_dim, dim, num_gaussians);
  
  if (old_flag) {
    pcgmm.read_gk(config["old"].get_str()+".gk");
    model_states.read_mc(config["old"].get_str()+".mc");
    assert(model_states.num_states()==fc_stats.num_states());
  } else
    pcgmm.read_basis(config["basis"].get_str());  

  LaGenMatDouble sample_cov(dim, dim);
  LaGenMatDouble model_cov(dim, dim);

  // Optimization space
  HCL_RnSpace_d vs(basis_dim);
  
  // Linesearch
  HCL_LineSearch_MT_d ls;
  if (config["hcl_line_cfg"].specified)
    ls.Parameters().Merge(config["hcl_line_cfg"].get_str().c_str());
  
  // lmBFGS
  HCL_UMin_lbfgs_d bfgs(&ls);
  if (config["hcl_bfgs_cfg"].specified)
    bfgs.Parameters().Merge(config["hcl_bfgs_cfg"].get_str().c_str());
  
  // Go through every state
  int kernel_pos=0;
  for (int s=0; s<fc_stats.num_states(); s++) {
    HmmState &fc_state=fc_stats.state(s);
    HmmState &model_state=model_states.state(s);
    
    // Remove kernels so long that the fc and pcgmm state mixtures match
    if (old_flag) {
      while (fc_state.weights.size()<model_state.weights.size()) {
	scgmm.remove_gaussian(model_state.weights[model_state.weights.size()-1].kernel);
	model_states.remove_kernel(model_state.weights[model_state.weights.size()-1].kernel);
      }
    }

    // Go through every state Gaussian
    for (unsigned int k=0; k<fc_state.weights.size(); k++, kernel_pos++) {

      if (info_flag > 0)
	printf("Optimizing parameters for gaussian %i\n",kernel_pos);
      
      // Fetch the target pcgmm gaussian
      Pcgmm::Gaussian &pcgmm_gaussian=pcgmm.get_gaussian(kernel_pos);
      // Fetch the target lambda
      LaVectorDouble &lambda=pcgmm_gaussian.lambda;
      
      // Just copy mean from the file
      for (int i=0; i<dim; i++)
	stats_file >> pcgmm_gaussian.mu(i);
      
      // Copy covariance to sample_cov
      for (int i=0; i<dim; i++)
	for (int j=0; j<dim; j++)
	  stats_file >> sample_cov(i,j);
      
      // Initialize untied parameters
      if (!old_flag) {
	lambda(LaIndex())=0;
	lambda(0)=1;
      }

      PcgmmLambdaFcnl f(vs, basis_dim, pcgmm, sample_cov, affine_flag);
      HCL_RnVector_d x((HCL_RnSpace_d&)(f.Domain()));
  
      for (int i=0; i<basis_dim; i++)
	x(i+1)=lambda(i);
      
      int trypos=0;
      double trythese[10]={50,40,30,20,10,8,6,4,2,1};
    testpoint:
      try {
	bfgs.Parameters().PutValue("MaxUpdates", trythese[trypos]);
	bfgs.Minimize(f, x);
      } catch(LaException e) {
	trypos++;
	goto testpoint;
      }
      
      for (int i=0; i<basis_dim; i++)
	lambda(i)=x(i+1);
      
      // Save kullback-leibler divergences KL(sample_fc, model_fc)
      if (kl_flag) {
	pcgmm.calculate_covariance(lambda, model_cov);
	kl_divergences[kernel_pos]=pcgmm.kullback_leibler_covariance(sample_cov, model_cov);
      }
    }
  }

  // Write pcgmm to file
  pcgmm.write_gk(config["out"].get_str()+".gk");
  fc_stats.write_mc(config["out"].get_str()+".mc");
}


void compress_scgmm() {

  scgmm.reset(basis_dim, dim, num_gaussians);

  if (old_flag) {
    scgmm.read_gk(config["old"].get_str()+".gk");
    model_states.read_mc(config["old"].get_str()+".mc");
    assert(model_states.num_states()==fc_stats.num_states());
  }
  else
    scgmm.read_basis(config["basis"].get_str());

  LaVectorDouble sample_mean(dim);
  LaGenMatDouble sample_cov(dim, dim);
  LaVectorDouble model_mean(dim);
  LaGenMatDouble model_cov(dim, dim);

  // Optimization space
  HCL_RnSpace_d vs(basis_dim);

  // Linesearch
  HCL_LineSearch_MT_d ls;
  if (config["hcl_line_cfg"].specified)
    ls.Parameters().Merge(config["hcl_line_cfg"].get_str().c_str());
  
  // lmBFGS
  HCL_UMin_lbfgs_d bfgs(&ls);
  if (config["hcl_bfgs_cfg"].specified)
    bfgs.Parameters().Merge(config["hcl_bfgs_cfg"].get_str().c_str());

  // Go through every state
  int kernel_pos=0;
  for (int s=0; s<fc_stats.num_states(); s++) {
    HmmState &fc_state=fc_stats.state(s);
    HmmState &model_state=model_states.state(s);

    // Remove kernels so long that the fc and pcgmm state mixtures match
    if (old_flag) {
      while (fc_state.weights.size()<model_state.weights.size()) {
	scgmm.remove_gaussian(model_state.weights[model_state.weights.size()-1].kernel);
	model_states.remove_kernel(model_state.weights[model_state.weights.size()-1].kernel);      
      }
    }
    
    // Go through every state Gaussian
    for (unsigned int k=0; k<fc_state.weights.size(); k++, kernel_pos++) {

      if (info_flag > 0)
	printf("Optimizing parameters for gaussian %i\n",kernel_pos);
      
      // Fetch the target scgmm gaussian
      Scgmm::Gaussian &scgmm_gaussian=scgmm.get_gaussian(kernel_pos);
      // Fetch the target lambda
      LaVectorDouble &lambda=scgmm_gaussian.lambda;

      // Copy mean to sample_mean
      for (int i=0; i<dim; i++)
	stats_file >> sample_mean(i);

      // Copy covariance to sample_cov
      for (int i=0; i<dim; i++)
	for (int j=0; j<dim; j++)
	  stats_file >> sample_cov(i,j);
    
      // Initialize untied parameters
      if (!old_flag) {
	lambda(LaIndex())=0;
	lambda(0)=1;
      }

      ScgmmLambdaFcnl f(vs, basis_dim, scgmm, sample_cov, sample_mean, affine_flag);
      HCL_RnVector_d x((HCL_RnSpace_d&)(f.Domain()));
  
      for (int i=0; i<basis_dim; i++)
	x(i+1)=lambda(i);
      
      int trypos=0;
      double trythese[10]={50,40,30,20,10,8,6,4,2,1};
    testpoint:
      try {
	bfgs.Parameters().PutValue("MaxUpdates", trythese[trypos]);
	bfgs.Minimize(f, x);
      } catch(LaException e) {
	trypos++;
	goto testpoint;
      }
      
      for (int i=0; i<basis_dim; i++)
	lambda(i)=x(i+1);

      // Save kullback-leibler divergences KL(sample_fc, model_fc)
      if (kl_flag) {
	scgmm.calculate_mu(lambda, model_mean);
	scgmm.calculate_covariance(lambda, model_cov);
	kl_divergences[kernel_pos]=scgmm.kullback_leibler(sample_mean, sample_cov,
							  model_mean, model_cov);
      }
    }
  }

  // Write scgmm to file
  scgmm.write_gk(config["out"].get_str()+".gk");
  fc_stats.write_mc(config["out"].get_str()+".mc");
}


