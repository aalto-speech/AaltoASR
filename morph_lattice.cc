#include "Latticer.hh"
#include "MorphSet.hh"
#include "io.hh"
#include "conf.hh"

int
main(int argc, char *argv[])
{
  /* Parse command line. */
  conf::Config config;
  config("usage: morph-lattice MORPHSET [INPUT [OUTPUT]]\n")
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
  if (config.arguments.size() < 1 || config.arguments.size() > 3) {
    fputs(config.help_string().c_str(), stderr);
    exit(1);
  }

  /* Read morph set. */
  MorphSet morph_set;
  {
    io::Stream morph_stream(config.arguments[0], "r");
    morph_set.read(morph_stream.file);
  }

  /* Create lattice */
  {
    std::string input("-");
    if (config.arguments.size() == 2)
      input = config.arguments[1];

    std::string output("-");
    if (config.arguments.size() == 3)
      output = config.arguments[2];

    Latticer latticer;
    latticer.morph_set = &morph_set;
    io::Stream input_stream(input, "r");
    io::Stream output_stream(output, "w");
    latticer.create_lattice(input_stream.file, output_stream.file);
  }
}
