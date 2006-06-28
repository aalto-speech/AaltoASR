#include "io.hh"
#include "conf.hh"
#include "FeatureGenerator.hh"
#include "SpeakerConfig.hh"

conf::Config config;
FeatureGenerator gen;

int
main(int argc, char *argv[])
{
  assert(sizeof(float) == 4);
  assert(sizeof(int) == 4);

  try {
    config("usage: feadot [OPTION...]\n")
      ('h', "help", "", "", "display help")
      ('c', "config=FILE", "arg must", "", "read feature configuration")
      ('o', "output=FILE", "arg", "-", "write dot graph")
      ;
    config.default_parse(argc, argv);
    if (config.arguments.size() != 0)
      config.print_help(stderr, 1);

    gen.load_configuration(io::Stream(config["config"].get_str()));
    gen.print_dot_graph(io::Stream(config["output"].get_str(), "w"));
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
