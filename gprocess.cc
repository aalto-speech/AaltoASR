#include <cstdio>
#include <fstream>
#include <string>
#include <iostream>

#include "gmd.h"
#include "blas3pp.h"

#include "io.hh"
#include "str.hh"
#include "conf.hh"
#include "HmmSet.hh"
#include "FeatureModules.hh"
#include "FeatureGenerator.hh"

int info_flag;
int f2d_flag;
int d2f_flag;
int p2f_flag;
int s2f_flag;
int mllt_flag;
double prune=0;

conf::Config config;
HmmSet source_model;
HmmSet target_model;
FeatureGenerator fg;
LinTransformModule *mllt;
LaGenMatDouble H;


int
main(int argc, char *argv[])
{
  try {
    config("usage: gprocess [OPTION...]\n")
      ('h', "help", "", "", "display help")
      ('b', "base=BASENAME", "arg must", "", "filename for source models (.gk and .mc)")
      ('c', "cfg=FILE", "arg", "", "configuration file for possible MLLT")
      ('o', "out=FILE", "arg must", "", "modified model (.gk and .mc)")
      ('i', "info=INT", "arg", "0", "info level")
      ('p', "prune=DOUBLE", "arg", "", "prune gaussians with smaller mixture weights than given")
      ('\0', "d2f", "", "", "convert diagonal to full covariance")
      ('\0', "p2f", "", "", "convert pcgmm to full covariance")
      ('\0', "s2f", "", "", "convert scgmm to full covariance")
      ('\0', "f2d", "", "", "convert full to diagonal covariance")
      ;
    config.default_parse(argc, argv);
    
    info_flag = config["info"].get_int();
    f2d_flag = config["f2d"].specified;
    d2f_flag = config["d2f"].specified;
    p2f_flag = config["p2f"].specified;
    s2f_flag = config["s2f"].specified;
    mllt_flag = config["cfg"].specified;

    if (config["prune"].specified)
      prune = config["prune"].get_double();

    if (config["base"].specified) {
      source_model.read_gk(config["base"].get_str()+".gk");
      source_model.read_mc(config["base"].get_str()+".mc");
    }
    else
      throw std::string("--base=BASENAME must be specified");


    if (mllt_flag) {

      if (!d2f_flag)
	throw std::string("Use --cfg only with --d2f!");
      
      FILE *f=fopen(config["cfg"].get_str().c_str(), "r");
      fg.load_configuration(f);
      fclose(f);
      mllt=(LinTransformModule*)fg.module("transform");
      H.resize(mllt->dim(), mllt->dim());
      for (int i=0; i<mllt->dim(); i++)
	for (int j=0; j<mllt->dim(); j++)
	  H(i,j)=(*(mllt->get_transformation_matrix()))[i*mllt->dim() + j];
      LaVectorLongInt pivots(mllt->dim(),1);
      LUFactorizeIP(H, pivots);
      LaLUInverseIP(H, pivots);
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

  int dim=source_model.dim();

  target_model=HmmSet(source_model);

  // Diagonal to full covariance
  if (d2f_flag) {
    assert(source_model.covariance_type()==HmmCovariance::DIAGONAL);

    for (int k=0; k<target_model.num_kernels(); k++) {
      HmmKernel &source_kernel=source_model.kernel(k);
      HmmKernel &target_kernel=target_model.kernel(k);
      target_kernel.resize(dim, HmmCovariance::FULL);

      // MLLT
      if (mllt_flag) {
	if (dim != mllt->dim())
	  throw std::string("Feature dimension doesn't match MLLT dimension!");
	LaGenMatDouble D=LaGenMatDouble::zeros(dim);
	LaGenMatDouble t=LaGenMatDouble::zeros(dim);
	LaGenMatDouble t2=LaGenMatDouble::zeros(dim);
	LaVectorDouble t3=LaVectorDouble(dim,1);
	LaVectorDouble mean=LaVectorDouble(dim,1);

	for (int i=0; i<dim; i++) {
	  t3(i)=source_kernel.center.at(i);
	  D(i,i)=source_kernel.cov.diag(i);	  
	}
	Blas_Mat_Mat_Mult(H, D, t, 1.0, 0.0);
	Blas_Mat_Mat_Trans_Mult(t, H, t2);
	Blas_Mat_Vec_Mult(H, t3, mean);
	for (int i=0; i<dim; i++) {
	  target_kernel.center.at(i)=mean(i);
	  for (int j=0; j<dim; j++)
	    target_kernel.cov.full(i,j)=t2(i,j);
	}
      }

      // NO MLLT
      else {
	for (int i=0; i<dim; i++) {
	  target_kernel.center.at(i)=source_kernel.center.at(i);
	  target_kernel.cov.full(i,i)=source_kernel.cov.diag(i);
	}	
      }
      
    }
    target_model.set_covariance_type(HmmCovariance::FULL);
  }

  // PCGMM to full covariance
  if (p2f_flag) {
    assert(source_model.covariance_type()==HmmCovariance::PCGMM);
    
    Pcgmm &pcgmm=source_model.pcgmm;
    target_model.set_covariance_type(HmmCovariance::FULL);
    target_model.reserve_kernels(pcgmm.num_gaussians());

    LaGenMatDouble cov;    
    for (unsigned int k=0; k<pcgmm.num_gaussians(); k++) {
      HmmKernel &target_kernel=target_model.kernel(k);
      target_kernel.resize(dim, HmmCovariance::FULL);
      for (int i=0; i<dim; i++)
	target_kernel.center.at(i)=pcgmm.gaussians.at(k).mu(i);
      
      pcgmm.calculate_covariance(pcgmm.gaussians.at(k).lambda, cov);
      for (int i=0; i<dim; i++)
	for (int j=0; j<dim; j++)
	  target_kernel.cov.full(i,j)=cov(i,j);
    }
    
    pcgmm.reset(0,0,0);
  }


  // SCGMM to full covariance
  if (s2f_flag) {
    assert(source_model.covariance_type()==HmmCovariance::SCGMM);

    Scgmm &scgmm=source_model.scgmm;
    target_model.set_covariance_type(HmmCovariance::FULL);
    target_model.reserve_kernels(scgmm.num_gaussians());

    LaVectorDouble mu;
    LaGenMatDouble cov;
    for (unsigned int k=0; k<scgmm.num_gaussians(); k++) {
      HmmKernel &target_kernel=target_model.kernel(k);
      target_kernel.resize(dim, HmmCovariance::FULL);

      scgmm.calculate_mu(scgmm.gaussians.at(k).lambda, mu);
      for (int i=0; i<dim; i++)
	target_kernel.center.at(i)=mu(i);
      
      scgmm.calculate_covariance(scgmm.gaussians.at(k).lambda, cov);
      for (int i=0; i<dim; i++)
	for (int j=0; j<dim; j++)
	  target_kernel.cov.full(i,j)=cov(i,j);
    }
    
    scgmm.reset(0,0,0);
  }

  // Full to diagonal covariance
  if (f2d_flag) {
    assert(source_model.covariance_type()==HmmCovariance::FULL);

    for (int k=0; k<target_model.num_kernels(); k++) {
      HmmKernel &source_kernel=source_model.kernel(k);
      HmmKernel &target_kernel=target_model.kernel(k);
      target_kernel.cov.resize(dim, HmmCovariance::DIAGONAL);
      for (int i=0; i<dim; i++)
	target_kernel.cov.diag(i)=source_kernel.cov.full(i,i);
    }
    target_model.set_covariance_type(HmmCovariance::DIAGONAL);
  }
  
  // Prune components with small weights
  if (prune>0) {
    std::vector<int> remove_indices;
    for (int s=0; s<target_model.num_states(); s++) {
      HmmState &state=target_model.state(s);
      
      remove_indices.resize(0);
      for (unsigned int k=state.weights.size()-1; k>0; k--)
	if (state.weights[k].weight<prune)
	  remove_indices.push_back(state.weights[k].kernel);
      for (unsigned int k=0; k<remove_indices.size(); k++) {
	// Don't remove the last gaussian for this state
	if (remove_indices.at(k)==0 && state.weights.size()==1)
	  continue;
	target_model.remove_kernel(remove_indices.at(k));
      }
    }
  }

  // Write output model
  target_model.write_gk(config["out"].get_str()+".gk");
  target_model.write_mc(config["out"].get_str()+".mc");
}
