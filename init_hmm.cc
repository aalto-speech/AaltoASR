#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sstream>
#include <iostream>
#include <fstream>

#include "HmmSet.hh"
#include "str.hh"
#include "conf.hh"

#define MAXLINE 4096

void read_segmodels(HmmSet &model, const std::string &filename);
void read_kernels(HmmSet &model, const std::string &cb_base, float gwidth,
                  bool diag_cov);
void make_transitions(HmmSet &model);


conf::Config config;


int main(int argc, char **argv)
{
  std::string statebind, cb_base, outbase;
  std::string outfile;
  float gwidth;
  int diag_cov = 0;
  HmmSet model;
  
  try {
    config("usage: init_hmm [OPTION...]\n")
      ('h', "help", "", "", "display help")
      ('b', "bind=FILE", "arg must", "", "model state configuration and bindings")
      ('a', "cb_base=BASENAME", "arg must", "", "base filename for kernel input")
      ('o', "out=BASENAME", "arg must", "", "base filename for output models")
      ('w', "gausswidth=FLOAT", "arg", "1.0", "width of the Gaussian kernels")
      ('d', "diag", "", "", "use diagonal covariances")
      ;
    
    config.default_parse(argc, argv);
    statebind = config["bind"].get_str();
    cb_base = config["cb_base"].get_str();
    outbase = config["out"].get_str();

    gwidth = config["gausswidth"].get_float();
    diag_cov = config["diag"].specified;

    read_segmodels(model, statebind);
    read_kernels(model, cb_base, gwidth, diag_cov);
    make_transitions(model);

    outfile = outbase + std::string(".gk");
    model.write_gk(outfile);

    outfile = outbase + std::string(".mc");
    model.write_mc(outfile);

    outfile = outbase + std::string(".ph");
    model.write_ph(outfile);
  }  catch (std::exception &e) {
    std::cerr << e.what() << std::endl;
  }
  return (0);
}


void read_segmodels(HmmSet &model, const std::string &filename)
{
  FILE *fp;
  char line[MAXLINE], *token, *label;
  int num_of_states, last_state=0;
  
  if ((fp = fopen(filename.c_str(),"r")) == NULL) {
    throw std::string("ERROR: cannot open file ") + filename +
      std::string(" for reading.");
  }
  while (fgets(line,MAXLINE,fp)) {
    if ((token = strtok(line," \t\n")) != NULL) {
      label=token;
      if ((token = strtok(NULL," \t\n")) == NULL) {
        fclose(fp);
        throw std::string("ERROR: state info missing for model '") +
          std::string(label) + std::string("' in file ") + filename;
      }
      num_of_states=atoi(token);
      Hmm &hmm = model.add_hmm(label,num_of_states);

      for (int i=0; i<num_of_states; i++) {
	if ((token = strtok(NULL," \t\n")) == NULL) {
          throw str::fmt(1024, "ERROR: state index %d/%d missing for model '%s' in file %s",i+1,num_of_states,hmm.label.c_str()) +  filename;
	  fclose(fp); exit(-1);
	}
	hmm.state(i) = atoi(token);
	if (hmm.state(i) > last_state) {
	  last_state=hmm.state(i);
	}
      }
    }
  }
  fclose(fp);
  
  model.reserve_states(last_state + 1);
  return;
}

void read_kernels(HmmSet &model, const std::string &cb_base, float gwidth,
                  bool diag_cov)
{
  std::string ifile;
  int k, num_of_kernels, d;
  int dim = 0;
  HmmCovariance::Type type;

  std::string line;
  line.reserve(4096); // for efficiency

  if (diag_cov) {
    model.set_covariance_type(HmmCovariance::DIAGONAL);
    type=HmmCovariance::DIAGONAL;
  }
  else {
    model.set_covariance_type(HmmCovariance::SINGLE);
    type=HmmCovariance::SINGLE;
  }

  for (int s=0; s<model.num_states(); s++) {

    ifile = cb_base + str::fmt(64, "_%d.cod", s);

    int width, height;
    std::string dummy;
    std::ifstream in(ifile.c_str());
    if (!in) {
      throw std::string("ERROR: cannot open file ") + ifile +
        std::string(" for reading.");
    }

    // Read header
    std::getline(in, line);
    std::istringstream buf(line);
    buf >> dim >> dummy >> width >> height >> dummy;
    if (!buf) {
      throw std::string("Invalid header in file ") + ifile;
    }
    
      // Set dimension
    if (s == 0)
      model.set_dim(dim);
    else if (model.dim() != dim) {
      throw str::fmt(256, "Conflicting dimensions %d %d",model.dim(),dim);
    }

    // Read kernel
    num_of_kernels = width * height;
    float weight = 1.0/num_of_kernels;
    for (k=0; k<num_of_kernels; k++) {
      int index = model.num_kernels();
      HmmKernel &kernel=model.add_kernel();
      HmmState &state = model.state(s);

      state.add_weight(index, weight);

      // Read center
      std::getline(in, line);
      std::istringstream buf(line);
      for (d=0; d<model.dim(); d++)
        buf >> kernel.center[d];
      if (!buf) {
        throw str::fmt(256, "Error while reading kernel %d", k);
      }

      // Set variance
      if (type==HmmCovariance::DIAGONAL)
      {
        for (d=0; d<dim; d++)
          kernel.cov.diag(d) = gwidth;
      }
      else if (type==HmmCovariance::SINGLE)
      {
        kernel.cov.var() = gwidth;
      }
    }
  }
}

void 
make_transitions(HmmSet &model)
{
  for (int h=0;h<model.num_hmms();h++) {
    Hmm &hmm=model.hmm(h);
    model.add_transition(h,-1,0,1.0, -1);
    for (int s=0;s<hmm.num_states();s++) {
      model.add_transition(h,s,s,0.8,-1);
      if (s+1==hmm.num_states()) 
	model.add_transition(h,s,-2,0.2,-1);
      else 
	model.add_transition(h,s,s+1,0.2,-1);
    }
  }
}
