#ifndef PHONEPROBSTOOLBOX_HH
#define PHONEPROBSTOOLBOX_HH

#include <string>
#include <cstring>
#include "conf.hh"
#include "FeatureGenerator.hh"
#include "HmmSet.hh"
#include "Recipe.hh"

class PPToolbox {
public:
  void read_models(const std::string &base);
  void read_configuration(const std::string &cfgname);
  void set_clustering(const std::string &clfile_name, double eval_minc, double eval_ming);
  void generate_to_fd(const int in, const int out, const bool raw_flag);
  void generate_from_file_to_fd(const std::string &input_name, const int out, const bool raw_flag);
  void generate(const std::string &input_name, const std::string &output_name, const bool raw_flag);
  //set_lnabytes(int x);
private:
  conf::Config config;
  aku::FeatureGenerator gen;
  aku::HmmSet model;
  std::vector<float> obs_log_probs;

  void write_int(FILE *fp, unsigned int i);
};

#endif
