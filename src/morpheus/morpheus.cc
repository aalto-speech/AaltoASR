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
      ('\0', "fsa=FILE", "arg", "", "read binary fsa model")
      ('s', "start=INT", "arg", "1", "start from line (1 = first)")
      ('e', "end=INT", "arg", "0", "end after line")
      ('\0', "sentences", "", "", "segment one sentence per line")
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

    mrf::Search s(&lm);
    Str line;
    int line_no = 1;
    bool sentences = config["sentences"].specified;
    while (str::read_line(line, stdin, true)) {
      str::clean(line, " \n\t0123456789");
      if (line.empty())
        continue;

      if (line_no < start)
        continue;
      if (end > 0 && line_no > end)
        break;
      line_no++;

      if (sentences)
        puts(s.segment_sentence(line).c_str());
      else
        puts(s.segment_word(line).c_str());
    }
    for (mrf::Search::WordMap::iterator it = s.unsegmented_words.begin();
         it != s.unsegmented_words.end(); it++)
    {
      fprintf(stderr, "UNSEGMENTED: %s %d\n", it->first.c_str(),
              it->second);
    }
    fprintf(stderr, "UNSEGMENTED TOKENS: %d\n", 
            s.unsegmented_word_tokens);
    fprintf(stderr, "UNSEGMENTED TYPES: %d\n", 
            s.unsegmented_word_types);
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

