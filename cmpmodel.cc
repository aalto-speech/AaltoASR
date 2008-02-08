#include <fstream>
#include <string>
#include <string.h>
#include <iostream>

#include "io.hh"
#include "conf.hh"
#include "HmmSet.hh"
#include "Distributions.hh"


int
main(int argc, char *argv[])
{
  conf::Config config;
  HmmSet first_model;
  HmmSet second_model;
  
  try {
    config("usage: cmpmodel [OPTION...]\n")
      ('h', "help", "", "", "display help")
      ('\0', "base1=BASENAME", "arg", "", "base filename for the first model")
      ('\0', "gk1=FILE", "arg", "", "Mixture base distributions for the first model")
      ('\0', "mc1=FILE", "arg", "", "Mixture coefficients for the states of the first model")
      ('\0', "ph1=FILE", "arg", "", "HMM definitions for the first model")
      ('\0', "base2=BASENAME", "arg", "", "base filename for the second model")
      ('\0', "gk2=FILE", "arg", "", "Mixture base distributions for the second model")
      ('\0', "mc2=FILE", "arg", "", "Mixture coefficients for the states of the second model")
      ('\0', "ph2=FILE", "arg", "", "HMM definitions for the second model")
      ('k', "kl", "", "", "Kullback-Leibler divergence from the first to the second model")
      ('s', "skl", "", "", "Symmetrized Kullback-Leibler divergence")
      ('i', "info=INT", "arg", "0", "info level")
      ;
    config.default_parse(argc, argv);
    
    // Initialize the HMM model
    if (config["info"].get_int())
      std::cout << "Loading HMMs\n";

    // Sanity check
    if (!(config["kl"].specified || config["skl"].specified))
      throw std::string("Must give either --kl or --skl (or both)");

    // Load the first model
    if (config["base1"].specified)
      first_model.read_all(config["base1"].get_str());
    else if (config["gk1"].specified && config["mc1"].specified &&
             config["ph1"].specified)
    {
      first_model.read_gk(config["gk1"].get_str());
      first_model.read_mc(config["mc1"].get_str());
      first_model.read_ph(config["ph1"].get_str());
    }
    else
      throw std::string("Must give either --base1 or all --gk1, --mc1 and --ph1");

    // Load the second model
    if (config["base2"].specified)
      second_model.read_all(config["base2"].get_str());
    else if (config["gk2"].specified && config["mc2"].specified &&
             config["ph2"].specified)
    {
      second_model.read_gk(config["gk2"].get_str());
      second_model.read_mc(config["mc2"].get_str());
      second_model.read_ph(config["ph2"].get_str());
    }
    else
      throw std::string("Must give either --base2 or all --gk2, --mc2 and --ph2");

    // Model similarity check
    if (first_model.num_states() != second_model.num_states())
      throw std::string("Both models should have the same number of states");

    // Compute Kullback-Leiblers
    double kl=0;
    for (int i=0; i<first_model.num_states(); i++) {
      HmmState &first_model_state = first_model.state(i);
      Mixture *first_model_mixture = first_model.get_emission_pdf(first_model_state.emission_pdf);
      
      HmmState &second_model_state = second_model.state(i);
      Mixture *second_model_mixture = second_model.get_emission_pdf(second_model_state.emission_pdf);

      kl = first_model_mixture->kullback_leibler(*second_model_mixture, 10000);
      if (config["kl"].specified)
        std::cout << "kl-divergence, state " << i << ": "  << kl << std::endl;
      if (config["skl"].specified) {
        kl += second_model_mixture->kullback_leibler(*first_model_mixture, 10000);
        std::cout << "symmetric kl-divergence, state " << i << ": "  << kl << std::endl;
      }
    }
  }
  
  // Handle errors
  catch (HmmSet::UnknownHmm &e) {
    fprintf(stderr, 
	    "Unknown HMM in transcription, "
	    "writing incompletely taught models\n");
    abort();
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
