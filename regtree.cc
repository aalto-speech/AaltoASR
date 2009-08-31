#include "HmmSet.hh"
#include "io.hh"
#include "str.hh"
#include "conf.hh"
#include "RegClassTree.hh"
#include <fstream>

conf::Config config;
HmmSet model;
RegClassTree rtree;

int
main(int argc, char *argv[])
{

  try {
    config("usage: regtree [OPTION...]\n")
      ('h', "help", "", "", "display help")
      ('b', "base=BASENAME", "arg", "", "base filename for model files")
      ('g', "gk=FILE", "arg", "", "Mixture base distributions")
      ('m', "mc=FILE", "arg", "", "Mixture coefficients for the states")
      ('p', "ph=FILE", "arg", "", "HMM definitions")
      ('s', "mcs=FILE", "arg must", "", "Mixture statistics file")
      ('t', "terminalnodes=INT", "arg", "16", "Number of maximum terminal nodes")
      ('u', "unit=STRING", "arg", "PHONE", "PHONE|MIX|GAUSSIAN type of units. Don't use MIX in case of shared gaussians between mixtures")
      ('o', "out=FILE", "arg", "", "File to write the regression tree to. If omitted, output to stdout")
      ;
    config.default_parse(argc, argv);

    if (config["base"].specified) {
      model.read_all(config["base"].get_str());
    }
    else if (config["gk"].specified && config["mc"].specified
        && config["ph"].specified) {
      model.read_gk(config["gk"].get_str());
      model.read_mc(config["mc"].get_str());
      model.read_ph(config["ph"].get_str());
    }
    else {
      throw std::string("Must give either --base or all --gk, --mc and --ph");
    }
    model.accumulate_mc_from_dump(config["mcs"].get_str());

    rtree.set_unit_mode(RegClassTree::UNIT_PHONE);
    if (config["unit"].get_str() == "MIX") rtree.set_unit_mode(
        RegClassTree::UNIT_MIX);
    if (config["unit"].get_str() == "GAUSSIAN") rtree.set_unit_mode(
        RegClassTree::UNIT_GAUSSIAN);

    rtree.initialize_root_node(&model);
    rtree.build_tree(config["terminalnodes"].get_int());

    if (config["out"].specified) {
      std::ofstream out(config["out"].get_c_str());
      rtree.write(&out);
      out.close();
    }
    else {
      rtree.write(&std::cout);
    }

  }
  catch (HmmSet::UnknownHmm &e) {
    fprintf(stderr, "Unknown HMM in transcription\n");
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
