#include <cstdio>
#include <fstream>
#include <string>
#include <iostream>


#include "io.hh"
#include "str.hh"
#include "conf.hh"
#include "HmmSet.hh"
#include "Pcgmm.hh"
#include "Scgmm.hh"

int info_flag;
int pcgmm_flag;
int scgmm_flag;
int basis_dim;
int flat;

conf::Config config;
HmmSet fc_model;
std::vector<double> counts;


void init_pcgmm();
void init_scgmm();


int
main(int argc, char *argv[])
{
  try {
    config("usage: basis_init [OPTION...]\n")
      ('h', "help", "", "", "display help")
      ('g', "gk=FILE", "arg must", "", "full covariance Gaussian kernels")
      ('m', "mc=FILE", "arg must", "", "kernel indices for states")
      ('o', "out=FILE", "arg must", "", "output filename for basis")
      ('d', "dim=INT", "arg must", "", "subspace dimension")
      ('c', "counts=FILE", "arg", "", "state feature counts")
      ('i', "info=INT", "arg", "0", "info level")
      ('\0', "pcgmm", "", "", "initialize precision basis")
      ('\0', "scgmm", "", "", "initialize exponential basis")
      ('\0', "flat", "", "", "set all gaussians to be equally important")
      ;
    config.default_parse(argc, argv);
    
    info_flag = config["info"].get_int();
    pcgmm_flag = config["pcgmm"].specified;
    scgmm_flag = config["scgmm"].specified;
    basis_dim = config["dim"].get_int();
    flat = config["flat"].specified;

    if (!pcgmm_flag && !scgmm_flag)
      throw std::string("Either --pcgmm or --scgmm should be defined");

    if (pcgmm_flag && scgmm_flag)
      throw std::string("Both --pcgmm and --scgmm can't be defined at the same time");

    if (config["mc"].specified)
      fc_model.read_mc(config["mc"].get_str());
    else
      throw std::string("--mc=FILE must be specified");

    if (config["gk"].specified)
      fc_model.read_gk(config["gk"].get_str());
    else
      throw std::string("--gk=FILE must be specified");
    
    if (config["counts"].specified) {
      FILE *counts_file=fopen((config["counts"].get_str()).c_str(), "r");
      std::string counts_line;
      bool ok;
      double total=0;

      while(str::read_line(&counts_line, counts_file, true))
	counts.push_back(str::str2float(counts_line.c_str(), &ok));
      fclose(counts_file);

      for (unsigned int i=0; i<counts.size(); i++)
	total+=counts[i];
      for (unsigned int i=0; i<counts.size(); i++)
	counts[i]/=total;

      if (fc_model.num_states() != (int)counts.size())
	throw std::string("number of states in count file is invalid");
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

  if (pcgmm_flag)
    init_pcgmm();
  
  if (scgmm_flag)
    init_scgmm();
}


void init_pcgmm()
{
  assert(basis_dim <= (fc_model.dim()*(fc_model.dim()+1))/2);

  Pcgmm pcgmm;
  std::vector<double> w;
  std::vector<LaGenMatDouble> covs;

  w.resize(fc_model.num_kernels());
  covs.resize(fc_model.num_kernels());


  unsigned int pos=0,k,kernel_index;
  int d1, d2, s;
  for (s=0; s<fc_model.num_states(); s++) {
    HmmState &state=fc_model.state(s);
    
    for (k=0; k<state.weights.size(); k++) {
      kernel_index = state.weights[k].kernel;
      HmmKernel &kernel=fc_model.kernel(kernel_index);

      // Weights
      if (flat)
	w.at(pos)=1;
      else {
	w.at(pos)=state.weights[k].weight;
	if (counts.size() != 0)
	  w.at(pos) *= counts[s];
      }
      
      // Covariances
      covs.at(pos).resize(fc_model.dim(), fc_model.dim());
      for (d1=0; d1<fc_model.dim(); d1++)
	for (d2=0; d2<fc_model.dim(); d2++)
	  covs.at(pos)(d1,d2)=kernel.cov.full(d1,d2);
      
      pos++;
    }
  }


  pcgmm.initialize_basis_pca(w, covs, basis_dim);
  pcgmm.write_basis(config["out"].get_str());
}


void init_scgmm()
{
  assert(basis_dim <= (fc_model.dim()*(fc_model.dim()+1))/2+fc_model.dim());

  Scgmm scgmm;
  std::vector<double> w;
  std::vector<LaVectorDouble> means;
  std::vector<LaGenMatDouble> covs;
  
  w.resize(fc_model.num_kernels());
  covs.resize(fc_model.num_kernels());
  means.resize(fc_model.num_kernels());


  unsigned int pos=0,k,kernel_index;
  int d1, d2, s;
  for (s=0; s<fc_model.num_states(); s++) {
    HmmState &state=fc_model.state(s);
    
    for (k=0; k<state.weights.size(); k++) {
      kernel_index = state.weights[k].kernel;
      HmmKernel &kernel=fc_model.kernel(kernel_index);

      // Weights
      if (flat)
	w.at(pos)=1;
      else {
	w.at(pos)=state.weights[k].weight;
	if (counts.size() != 0)
	  w.at(pos) *= counts[s];
      }

      // Covariances
      covs.at(pos).resize(fc_model.dim(), fc_model.dim());
      for (d1=0; d1<fc_model.dim(); d1++)
	for (d2=0; d2<fc_model.dim(); d2++)
	  covs.at(pos)(d1,d2)=kernel.cov.full(d1,d2);
      
      // Means
      means.at(pos).resize(fc_model.dim(),1);
      for (d1=0; d1<fc_model.dim(); d1++)
	means.at(pos)(d1)=kernel.center.at(d1);

      pos++;
    }
  }
    
  scgmm.initialize_basis_pca(w, covs, means, basis_dim);
  scgmm.write_basis(config["out"].get_str());
}
