#include "misc/conf.hh"
#include "misc/io.hh"
#include "misc/str.hh"
#include "fsalm/LM.hh"

using namespace fsalm;

conf::Config config;
LM lm;

int
main(int argc, char *argv[])
{
  try {
    config("usage: fsalm-convert [OPTION...]\n")
      ('h', "help", "", "", "display help")
      ('\0', "arpa=FILE", "arg", "", "read ARPA language model")
      ('\0', "bin=FILE", "arg", "", "read binary fsa model")
      ('\0', "out-bin", "arg", "", "write binary fsa model")
      ;
    config.default_parse(argc, argv);
    if (config.arguments.size() != 0)
      config.print_help(stderr, 1);

    // Read the language model
    //
    if (config["arpa"].specified) {
      if (config["bin"].specified) {
        fprintf(stderr, "options --arpa and --blm not allowed together\n");
        exit(1);
      }
      lm.read_arpa(io::Stream(config["arpa"].get_str(), "r").file, true);
      lm.trim();
    }
    else if (config["bin"].specified) {
      lm.read(io::Stream(config["bin"].get_str(), "r").file);
    }
    else {
      fprintf(stderr, "option --arpa or --bin required\n");
      exit(1);
    }
    fprintf(stderr, "model order %d\n", lm.order());

    // Write models
    //
    if (config["out-bin"].specified) {
      fprintf(stderr, "writing binary fsa model: %s\n", 
              config["out-bin"].get_c_str()); 
      lm.write(io::Stream(config["out-bin"].get_str(), "w").file);
    }
  }
  catch (std::string &str) {
    fprintf(stderr, "exception: %s\n", str.c_str());
    exit(1);
  }
  catch (std::exception &e) {
    fprintf(stderr, "exception: %s\n", e.what());
    exit(1);
  }
}

