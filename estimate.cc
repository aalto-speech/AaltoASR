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

conf::Config config;
HmmSet ml_model;
HmmSet mmi_model;


int
main(int argc, char *argv[])
{

  try {
    config("usage: estimate [OPTION...]\n")
      ('h', "help", "", "", "display help")
      ('b', "base=BASENAME", "arg", "", "Previous base filename for model files")
      ('g', "gk=FILE", "arg", "", "Previous mixture base distributions")
      ('m', "mc=FILE", "arg", "", "Previous mixture coefficients for the states")
      ('p', "ph=FILE", "arg", "", "Previous HMM definitions")
      ('l', "list=LISTNAME", "arg must", "", "file with one statistics file per line")      
      ('o', "out=BASENAME", "arg must", "", "base filename for output models")
      ('t', "transitions", "", "", "estimate also state transitions")
      ('\0', "ml", "", "", "maximum likelihood estimation")
      ('\0', "mmi", "", "", "maximum mutual information")
      ('i', "info=INT", "arg", "0", "info level")
      ;
    config.default_parse(argc, argv);

    transtat = config["transitions"].specified;    
    info = config["info"].get_int();
    out_file = config["out"].get_str();

    if (config["mmi"].specified && config["ml"].specified)
      throw std::string("Don't define both --ml and --mmi!");
    
    if (!config["mmi"].specified && !config["ml"].specified)
      throw std::string("Define either --ml and --mmi!");
      
    // Load the previous models
    if (config["base"].specified)
      {
	ml_model.read_all(config["base"].get_str());
      }
    else if (config["gk"].specified && config["mc"].specified &&
             config["ph"].specified)
      {
	ml_model.read_gk(config["gk"].get_str());
	ml_model.read_mc(config["mc"].get_str());
	ml_model.read_ph(config["ph"].get_str());
      }
    else
      {
	throw std::string("Must give either --base or all --gk, --mc and --ph");
      }
    ml_model.start_accumulating();

    // Load also mmi if needed
    if (config["mmi"].specified) {
      if (config["base"].specified)
      {
        mmi_model.read_all(config["base"].get_str());
      }
      else if (config["gk"].specified && config["mc"].specified &&
	       config["ph"].specified)
	{
	  mmi_model.read_gk(config["gk"].get_str());
	  mmi_model.read_mc(config["mc"].get_str());
	  mmi_model.read_ph(config["ph"].get_str());
	} 
      mmi_model.start_accumulating();
    }

    // Open the list of statistics files
    std::ifstream filelist(config["list"].get_str().c_str());
    if (!filelist)
      fprintf(stderr, "Could not open %s\n", config["list"].get_str().c_str());

    while (filelist >> stat_file) {

      // Accumulate ML statistics
      ml_model.accumulate_from_dump(stat_file);

      // Accumulate MMI statistics if needed
      if (config["mmi"].specified)
	mmi_model.accumulate_from_dump(stat_file+"_mmi");
    }

    // Estimate parameters
    ml_model.stop_accumulating();
    if (config["mmi"].specified)
      mmi_model.stop_accumulating();
    
    // Write final models
    ml_model.write_all(out_file);
    if (config["mmi"].specified)
      mmi_model.write_all(out_file+"_mmi");
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
