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
double minocc;

conf::Config config;
FeatureGenerator fea_gen;
HmmSet model;


int
main(int argc, char *argv[])
{
  double total_log_likelihood = 0;
  try {
    config("usage: estimate [OPTION...]\n")
      ('h', "help", "", "", "display help")
      ('b', "base=BASENAME", "arg", "", "Previous base filename for model files")
      ('g', "gk=FILE", "arg", "", "Previous mixture base distributions")
      ('m', "mc=FILE", "arg", "", "Previous mixture coefficients for the states")
      ('p', "ph=FILE", "arg", "", "Previous HMM definitions")
      ('c', "config=FILE", "arg", "", "feature configuration (required for MLLT)")
      ('L', "list=LISTNAME", "arg must", "", "file with one statistics file per line")
      ('o', "out=BASENAME", "arg must", "", "base filename for output models")
      ('t', "transitions", "", "", "estimate also state transitions")
      ('i', "info=INT", "arg", "0", "info level")
      ('\0', "mllt=MODULE", "arg", "", "update maximum likelihood linear transform")
      ('\0', "ml", "", "", "maximum likelihood estimation")
      ('\0', "mmi", "", "", "maximum mutual information estimation")
      ('\0', "minvar=FLOAT", "arg", "0.1", "minimum variance (default 0.1)")
      ('\0', "covsmooth", "arg", "0", "covariance smoothing (default 0.0)")
      ('\0', "C1=FLOAT", "arg", "1.0", "constant \"C1\" for MMI updates (default 1.0)")
      ('\0', "C2=FLOAT", "arg", "2.0", "constant \"C2\" for MMI updates (default 2.0)")
      ('\0', "split", "", "", "split (every gaussian)")
      ('\0', "minocc=FLOAT", "arg", "0.0", "minimum occupancy count for splitting")
      ('\0', "maxg=INT", "arg", "0", "maximum number of Gaussians per state for splitting")
      ('s', "savesum=FILE", "arg", "", "save summary information (loglikelihood)")
      ;
    config.default_parse(argc, argv);

    transtat = config["transitions"].specified;    
    info = config["info"].get_int();
    out_file = config["out"].get_str();
    maxg = config["maxg"].get_int();
    minocc = config["minocc"].get_double();
    
    if (config["mmi"].specified && config["ml"].specified)
      throw std::string("Don't define both --ml and --mmi!");
    
    if (!config["mmi"].specified && !config["ml"].specified)
      throw std::string("Define either --ml and --mmi!");
      
    // Load the previous models
    if (config["base"].specified)
    {
      model.read_all(config["base"].get_str());
    }
    else if (config["gk"].specified && config["mc"].specified &&
             config["ph"].specified)
    {
      model.read_gk(config["gk"].get_str());
      model.read_mc(config["mc"].get_str());
      model.read_ph(config["ph"].get_str());
    }
    else
    {
      throw std::string("Must give either --base or all --gk, --mc and --ph");
    }
    if (config["ml"].specified)
      model.set_estimation_mode(PDF::ML);
    else
      model.set_estimation_mode(PDF::MMI);
    if (config["mllt"].specified)
      model.set_full_stats(true);
    
    if (config["config"].specified) {
      fea_gen.load_configuration(io::Stream(config["config"].get_str()));
    }
    else if (config["mllt"].specified) {
      throw std::string("Must specify configuration file with MLLT");      
    }
    
    model.start_accumulating();

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
      if (lls_file)
      {
        double temp;
        lls_file >> temp;
        lls_file.close();
        total_log_likelihood += temp;
      }
    }

    // Estimate parameters
    model.set_gaussian_parameters(config["minvar"].get_double(),
                                  config["covsmooth"].get_double(),
                                  config["C1"].get_double(),
                                  config["C2"].get_double());

    if (transtat)
      model.estimate_transition_parameters();
    if (config["mllt"].specified)
      model.estimate_mllt(fea_gen, config["mllt"].get_str());
    else
      model.estimate_parameters();

    // Split Gaussians if desired
    if (config["split"].specified)
      model.split_gaussians(minocc, maxg);

    model.stop_accumulating();
    
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
        summary_file << total_log_likelihood << std::endl;
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
