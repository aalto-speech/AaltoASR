#include <math.h>
#include "Toolbox.hh"

float global_beam = 200;

const char 
  *hmm_file = "tri_tied3_30.9.2004_12.ph",
  *dur_file = "tri_tied_3.dur",
  *lex_file = "tiny.lex",
  *ngram_file = "2gram.bin",
  *lookahead_file = "2gram.bin",
  *lna_file = "tri_tied_3.lna";

int
main(int argc, char *argv[])
{
  try {
    int end_frame = 250;
    if (argc > 1)
      end_frame = atoi(argv[1]);

    // Create decoder
    Toolbox t;
    t.set_cross_word_triphones(true);
    t.set_lm_lookahead(1);
    t.select_decoder(0); // Token-pass

    // Load files
    t.hmm_read(hmm_file);
    t.lex_read(lex_file);
    TPLexPrefixTree::Node *node = (TPLexPrefixTree::Node *)0x1374f00;
    TPLexPrefixTree &lex = t.debug_get_tp_lex();
    lex.debug_prune_dead_ends(lex.root()); 
  
    // Read recognition files
    t.duration_read(dur_file);
    t.ngram_read(ngram_file, 1);
    t.read_lookahead_ngram(lookahead_file);

    // Settings
    t.set_verbose(0);
    t.set_print_probs(0);
    t.set_print_indices(0);
    t.set_print_frames(0);
    t.set_word_boundary("<w>");
  
    // Scales
    t.set_duration_scale(3);
    t.set_transition_scale(1);
    t.set_lm_scale(30);

    // Beams
    // t.prune_lm_lookahead_buffers(0, 4); // min_delta, max_depth
    t.set_global_beam(global_beam);
    t.set_word_end_beam(global_beam * 0.9);
    t.set_token_limit(30000);
    t.set_prune_similar(3);
    t.set_generate_word_graph(true);
  
    // Recognize
    t.lna_open(lna_file, 1024);
    t.reset(75); // Start frame
    bool quit = false;
    int count = 0;
    while (!quit) {
      fprintf(stderr, "%d ", t.frame());
      if (t.frame() == end_frame)
	break;
      if (!t.run())
	break;
    }

    t.write_word_graph("tiny.wg");
  }
  catch (std::exception &e) {
    fprintf(stderr, "exception: %s\n", e.what());
  }
}
