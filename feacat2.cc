#include "io.hh"
#include "conf.hh"
#include "FeatureGenerator.hh"

conf::Config config;
FeatureGenerator gen;
bool raw = false;

void
print_feature(ConstFeatureVec &fea)
{
  // Raw output
  if (raw) {
    for (int i = 0; i < fea.dim(); i++) {
      float tmp = fea[i];
      fwrite(&tmp, sizeof(float), 1, stdout);
    }
  }

  // ASCII output
  else {
    for (int i=0; i < fea.dim(); i++)
      printf("%8.2f ", fea[i]);
    printf("\n");
  }
}

int
main(int argc, char *argv[])
{
  assert(sizeof(float) == 4);

  try {
    config("usage: feacat [OPTION...] FILE\n")
      ('h', "help", "", "", "display help")
      ('F', "features=FILE", "arg must", "", "read feature configuration")
      ('r', "raw", "", "", "raw float output")
      ('s', "sample-rate=INT", "arg", "0", "set sample rate for raw audio")
      ;
    config.default_parse(argc, argv);
    if (config.arguments.size() != 1)
      config.print_help(stderr, 1);
    raw = config["raw"].specified;

    gen.load_configuration(io::Stream(config["features"].get_str()));
    gen.open(config.arguments[0], config["sample-rate"].get_int());

    int start_frame = 0;
    int end_frame = INT_MAX;
    if (config["start-frame"].specified)
      start_frame = config["start-frame"].get_int();
    if (config["end-frame"].specified)
      end_frame = config["end-frame"].get_int();
    for (int f = start_frame; f < end_frame; f++) {
      ConstFeatureVec fea = gen.generate(f);
      if (gen.eof())
	break;
      print_feature(fea);
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
