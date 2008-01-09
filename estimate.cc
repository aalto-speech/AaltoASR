#include <fstream>
#include <string>
#include <string.h>
#include <iostream>

#include "io.hh"
#include "str.hh"
#include "conf.hh"
#include "HmmSet.hh"
#include "FeatureGenerator.hh"
#include "Recipe.hh"

  
std::string stat_file;
std::string out_file;

int info;
bool transtat;
int maxg;

conf::Config config;
FeatureGenerator fea_gen;
HmmSet model;


int
main(int argc, char *argv[])
{
  std::map< std::string, double > sum_statistics;
  std::string base_file_name;
  PDF::EstimationMode mode;
  
  try {
    config("usage: estimate [OPTION...]\n")
      ('h', "help", "", "", "display help")
      ('b', "base=BASENAME", "arg", "", "Previous base filename for model files")
      ('g', "gk=FILE", "arg", "", "Previous mixture base distributions")
      ('m', "mc=FILE", "arg", "", "Previous mixture coefficients for the states")
      ('p', "ph=FILE", "arg", "", "Previous HMM definitions")
      ('c', "config=FILE", "arg", "", "feature configuration (required for MLLT)")
      ('L', "list=LISTNAME", "arg must", "", "file with one statistics file per line")
      ('C', "coeffs=NAME", "arg", "", "Precomputed precision/subspace Gaussians")
      ('o', "out=BASENAME", "arg must", "", "base filename for output models")
      ('t', "transitions", "", "", "estimate also state transitions")
      ('i', "info=INT", "arg", "0", "info level")
      ('\0', "mllt=MODULE", "arg", "", "update maximum likelihood linear transform")
      ('\0', "ml", "", "", "maximum likelihood estimation")
      ('\0', "mmi", "", "", "maximum mutual information estimation")
      ('\0', "mpe", "", "", "minimum phone error estimation")
      ('\0', "minvar=FLOAT", "arg", "0.1", "minimum variance (default 0.1)")
      ('\0', "covsmooth", "arg", "0", "covariance smoothing (default 0.0)")
      ('\0', "C1=FLOAT", "arg", "1.0", "constant \"C1\" for MMI updates (default 1.0)")
      ('\0', "C2=FLOAT", "arg", "2.0", "constant \"C2\" for MMI updates (default 2.0)")
      ('\0', "mmi-ismooth=FLOAT", "arg", "0.0", "I-smoothing constant for MMI")
      ('\0', "mpe-ismooth=FLOAT", "arg", "0.0", "I-smoothing constant for MPE")
      ('\0', "mmi-prior", "", "", "Use MMI prior when I-smoothing MPE model")
      ('\0', "delete=FLOAT", "arg", "0.0", "delete Gaussians with occupancies below the threshold")
      ('\0', "mremove=FLOAT", "arg", "0.0", "remove mixture components below the weight threshold")
      ('\0', "split=FLOAT", "arg", "0.0", "split a Gaussian if the occupancy exceeds the threshold")
      ('\0', "maxg=INT", "arg", "0", "maximum number of Gaussians per state for splitting")
      ('s', "savesum=FILE", "arg", "", "save summary information")
      ('\0', "hcl-bfgs-cfg=FILE", "arg", "", "configuration file for HCL biconjugate gradient algorithm")
      ('\0', "hcl-line-cfg=FILE", "arg", "", "configuration file for HCL line search algorithm")
      ;
    config.default_parse(argc, argv);

    transtat = config["transitions"].specified;    
    info = config["info"].get_int();
    out_file = config["out"].get_str();
    maxg = config["maxg"].get_int();

    int count = 0;
    if (config["ml"].specified) {
      count++;
      mode = PDF::ML_EST;
    }
    if (config["mmi"].specified) {
      count++;
      mode = PDF::MMI_EST;
    }
    if (config["mpe"].specified) {
      count++;
      mode = PDF::MPE_EST;
    }
    if (count != 1)
      throw std::string("Define exactly one of --ml, --mmi and --mpe!");

    if (config["mmi-ismooth"].specified &&
        (!config["mmi"].specified && !config["mmi-prior"].specified))
      fprintf(stderr, "Warning: --mmi-ismooth ignored without --mmi or --mmi-prior\n");
    if (config["mpe-ismooth"].specified && mode != PDF::MPE_EST)
        fprintf(stderr, "Warning: --mpe-ismooth ignored without --mpe\n");
    if (config["mmi-prior"].specified)
    {
      if (mode == PDF::MPE_EST)
        mode = PDF::MPE_MMI_PRIOR_EST;
      else
        fprintf(stderr, "Warning: --mmi-prior ignored without --mpe\n");
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

    if (config["mllt"].specified)
    {
      if (!config["ml"].specified)
        throw std::string("--mllt is only supported with --ml");
    }
    
    if (config["config"].specified) {
      fea_gen.load_configuration(io::Stream(config["config"].get_str()));
    }
    else if (config["mllt"].specified) {
      throw std::string("Must specify configuration file with MLLT");      
    }
    
    // Open the list of statistics files
    std::ifstream filelist(config["list"].get_str().c_str());
    if (!filelist)
    {
      fprintf(stderr, "Could not open %s\n", config["list"].get_str().c_str());
      exit(1);
    }

    // Accumulate statistics
    while (filelist >> stat_file && stat_file != " ") {
      model.accumulate_gk_from_dump(stat_file+".gks");
      model.accumulate_mc_from_dump(stat_file+".mcs");
      if (transtat)
        model.accumulate_ph_from_dump(stat_file+".phs");
      std::string lls_file_name = stat_file+".lls";
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

    // Estimate parameters
    model.set_gaussian_parameters(config["minvar"].get_double(),
                                  config["covsmooth"].get_double(),
                                  config["C1"].get_double(),
                                  config["C2"].get_double(),
                                  config["mmi-ismooth"].get_double(),
                                  config["mpe-ismooth"].get_double());

    // Linesearch for subspace models
    HCL_LineSearch_MT_d ls;
    if (config["hcl-line-cfg"].specified)
      ls.Parameters().Merge(config["hcl-line-cfg"].get_str().c_str());

    // lmBFGS for subspace models
    HCL_UMin_lbfgs_d bfgs(&ls);
    if (config["hcl-bfgs-cfg"].specified)
      bfgs.Parameters().Merge(config["hcl-bfgs-cfg"].get_str().c_str());

    model.set_hcl_optimization(&ls, &bfgs, config["hcl-line-cfg"].get_str(), config["hcl-bfgs-cfg"].get_str());

    // Load precomputed coefficients
    if (config["coeffs"].specified) {
      std::ifstream coeffs_files(config["coeffs"].get_str().c_str());
      std::string coeff_file_name;
      while (coeffs_files >> coeff_file_name) {
        std::ifstream coeff_file(coeff_file_name.c_str());
        int g;
        while (coeff_file >> g) {
          PrecisionConstrainedGaussian *pc = dynamic_cast< PrecisionConstrainedGaussian* > (model.get_pool_pdf(g));
          if (pc != NULL) {
            pc->read(coeff_file);
          }
          
          SubspaceConstrainedGaussian *sc = dynamic_cast< SubspaceConstrainedGaussian* > (model.get_pool_pdf(g));
          if (sc != NULL) {
            sc->read(coeff_file);
          }
        }
      }

      // Re-estimate only mixture parameters in this case
      model.estimate_parameters(mode, false, true);
    }

    // Normal training, FIXME: MLLT + precomputed coefficients?
    else {

      if (transtat)
        model.estimate_transition_parameters();
      if (config["mllt"].specified)
        model.estimate_mllt(fea_gen, config["mllt"].get_str());
      else
        model.estimate_parameters(mode);
      
      // Delete Gaussians
      if (config["delete"].specified)
        model.delete_gaussians(config["delete"].get_double());
      
      // Remove mixture components
      if (config["mremove"].specified)
        model.remove_mixture_components(config["mremove"].get_double());
      
      // Split Gaussians if desired
      if (config["split"].specified)
        model.split_gaussians(config["split"].get_double(), maxg);
      
      model.stop_accumulating();
    }
    
    // Write final models
    model.write_all(out_file);
    if (config["config"].specified) {
      fea_gen.write_configuration(io::Stream(out_file + ".cfg","w"));
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
