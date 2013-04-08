#include <fstream>
#include <string>
#include <iostream>

#include "io.hh"
#include "str.hh"
#include "conf.hh"
#include "HmmSet.hh"

using namespace aku;

conf::Config config;
HmmSet model;


int
main(int argc, char *argv[])
{
  std::string stat_file;
  std::map< std::string, double > sum_statistics;

  try {
    config("usage: combine_stats [OPTION...]\n")
      ('h', "help", "", "", "display help")
      ('b', "base=BASENAME", "arg", "", "Base filename for model files")
      ('g', "gk=FILE", "arg", "", "Gaussian distributions")
      ('m', "mc=FILE", "arg", "", "Mixture coefficients for the states")
      ('p', "ph=FILE", "arg", "", "HMM definitions")
      ('L', "list=LISTNAME", "arg must", "", "File with one statistics file per line")
      ('o', "out=BASENAME", "arg must", "", "Base filename for output statistics")
      ;
    config.default_parse(argc, argv);

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

    std::string out_file = config["out"].get_str();
    model.dump_statistics(out_file);
    if (sum_statistics.size() > 0)
    {
      std::string lls_file_name = out_file+".lls";
      std::ofstream lls_file(lls_file_name.c_str());
      if (lls_file)
      {
        lls_file.precision(12); 
        for (std::map<std::string, double>::const_iterator it =
               sum_statistics.begin(); it != sum_statistics.end(); it++)
        {
          lls_file << (*it).first << ": " << (*it).second <<
            std::endl;
        }
        lls_file.close();
      }
      else
      {
        fprintf(stderr, "Could not open lls-file\n");
        exit(1);
      }
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
