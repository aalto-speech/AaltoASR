#include "misc/conf.hh"
#include "misc/io.hh"
#include "misc/str.hh"
#include "Morpheus.hh"

using namespace std;
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
        ('b', "soft-prob", "", "", "print LM probability over all segmentations (no effect on segmentation)")
        ('s', "start=INT", "arg", "1", "start from line (1 = first)")
        ('e', "end=INT", "arg", "0", "end after line")
        ('\0', "no-wb", "", "", "do not add word boundary morphs <w>")
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
        bool no_wb = config["no-wb"].specified;

        mrf::Morpheus m(&lm);
        string line;
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
                vector<string> words = str::split(line, " \t", true);

                string id;
                if (config["preserve-id"].specified &&
                        words.back()[0] == '(')
                {
                    id = words.back();
                    words.pop_back();
                }

                m.reset();
                m.add_symbol(m.sentence_start_str, false);
                if (!no_wb) m.add_symbol(m.word_boundary_str, false);
                for (auto wit = words.begin(); wit != words.end(); ++wit) {
                    m.add_string(*wit);
                    if (!no_wb) m.add_symbol(m.word_boundary_str);
                }
                m.add_symbol(m.sentence_end_str);

                if (config["prob"].specified)
                    printf("%.6f ", m.score());
                if (config["soft-prob"].specified)
                    printf("%.6f ", m.soft_score());
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
    catch (string &str) {
        fprintf(stderr, "exception: %s\n", str.c_str());
        exit(1);
    }
    catch (exception &e) {
        fprintf(stderr, "exception: %s\n", e.what());
        exit(1);
    }
}
