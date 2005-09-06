#include "MorphSet.hh"
#include "io.hh"
#include "conf.hh"

int
main(int argc, char *argv[])
{
  /* Parse command line. */
  conf::Config config;
  config("usage: morph-lattice MORPHSET TEXT\n")
    ('h', "help", "", "", "display help")
    ('v', "verbosity=INT", "arg", "0", "verbosity level (default 0)")
    ('C', "config=FILE", "arg", "", "configuration file");
  config.parse(argc, argv);
  if (config["help"].specified) {
    fputs(config.help_string().c_str(), stdout);
    exit(0);
  }
  if (config["config"].specified)
    config.read(io::Stream(config["config"].get_str(), "r").file);
  config.check_required();
  if (config.arguments.size() != 2) {
    fputs(config.help_string().c_str(), stderr);
    exit(1);
  }

  /* Run morfessor. */
  MorphSet morph_set;
  {
    io::Stream morph_stream(config.arguments[0], "r");
    morph_set.read(morph_stream.file);
  }

  morph_set.write(stdout);
}
