#include "misc/conf.hh"
#include "misc/io.hh"
#include "misc/str.hh"
#include "Morpheus.hh"

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
      ('\0', "fsa=FILE", "arg", "", "read binary fsa model")
      ('\0', "preserve-id", "", "", "preserve trn id in parenthesis")
      ('p', "prob", "", "", "print also LM probability")
      ('s', "start=INT", "arg", "1", "start from line (1 = first)")
      ('e', "end=INT", "arg", "0", "end after line")
      ;
    config.default_parse(argc, argv);
    if (config.arguments.size() != 0)
      config.print_help(stderr, 1);

    // Read the language model
    //
    if (config["arpa"].specified) {
      if (config["fsa"].specified) {
        fprintf(stderr, "options --arpa and --fsa not allowed together\n");
        exit(1);
      }
      lm.read_arpa(io::Stream(config["arpa"].get_str(), "r").file, true);
      lm.trim();
    }
    else if (config["fsa"].specified) {
      lm.read(io::Stream(config["fsa"].get_str(), "r").file);
    }
    else {
      fprintf(stderr, "option --arpa or --fsa required\n");
      exit(1);
    }
    fprintf(stderr, "model order %d\n", lm.order());

    int start = config["start"].get_int();
    int end = config["end"].get_int();

    mrf::Morpheus m(&lm);
    Str line;
    int line_no = 1;
    while (str::read_line(line, stdin, true)) {
      str::clean(line);
      if (line.empty())
        continue;

      if (line_no < start)
        continue;
      if (end > 0 && line_no > end)
        break;
      line_no++;

      try {
        StrVec words = str::split(line, " \t", true);

        Str id;
        if (config["preserve-id"].specified &&
            words.back()[0] == '(') 
        {
          id = words.back();
          words.pop_back();
        }

        m.reset();
        m.add_symbol(m.sentence_start_str);
        m.add_symbol(m.word_boundary_str);
        FOR(w, words) {
          m.add_string(words[w]);
          m.add_symbol(m.word_boundary_str);
        }
        m.add_symbol(m.sentence_end_str);
        if (config["prob"].specified)
          printf("%g ", m.score());
        printf("%s", m.str().c_str());
        if (config["preserve-id"].specified && !id.empty())
          printf(" %s", id.c_str());
        fputs("\n", stdout);
      }
      catch (mrf::NoSeg &e) {
        printf("NO SEGMENTATION: %s\n", line.c_str());
      }
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

