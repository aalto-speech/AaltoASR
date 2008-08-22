#include "misc/conf.hh"
#include "misc/io.hh"
#include "misc/str.hh"
#include "Search.hh"

using namespace fsalm;

conf::Config config;
LM lm;

int
main(int argc, char *argv[])
{
  try {
    config("usage: morpheus [OPTION...]\n")
      ('h', "help", "", "", "display help")
      ('\0', "arpa=FILE", "arg", "", "read ARPA language model")
      ('\0', "bin=FILE", "arg", "", "read binary fsa model")
      ('s', "start=INT", "arg", "1", "start from word (1 = first)")
      ('e', "end=INT", "arg", "0", "end after word")
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

    int start = config["start"].get_int();
    int end = config["end"].get_int();

    mrf::Search s(&lm);
    Str line;
    int line_no = 1;
    while (str::read_line(line, stdin, true)) {
      str::clean(line, " \n\t0123456789");
      if (line.empty())
        continue;

      if (line_no < start)
        continue;
      if (end > 0 && line_no > end)
        break;
      line_no++;

      try {
        s.reset(line);
        for (int i = 0; i < line.length(); i++) {
          s.process_pos(i);
        }
        mrf::TokenPtrVec &vec = s.active_tokens.at(line.length());
        assert(vec.size() == 1);
        puts(s.str(vec.back()).c_str());
      }
      catch (mrf::NoSeg &e) {
        printf("- %s\n", line.c_str());
      }
    }
    fprintf(stderr, "exiting\n");
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

