#include <iomanip>
#include <fstream>

#include <errno.h>

#include "Toolbox.hh"

using namespace std;

int
main(int argc, char *argv[])
{
  Toolbox t;

  t.select_decoder(0);
  t.hmm_read("/home/jpylkkon/MORPH2/hmms/tri_tied3_30.9.2004_10.ph");
  t.lex_read("/share/puhe/jpylkkon/tying/26_all.lex");
  t.ngram_read("/share/puhe/jpylkkon/mb/4gram.bin", 1);
  t.lna_open("/share/puhe/jpylkkon/tying/tri_tied_3.lna", 1024);
  t.set_global_beam(210);
  t.set_word_end_beam(180);
  t.set_token_limit(30000);
  t.set_prune_similar(2);

  t.set_verbose(1);
  t.set_print_probs(0);
  t.set_print_indices(0);
  t.set_print_frames(0);
  t.set_word_boundary("<w>");
  t.set_lm_scale(30);
  t.reset(0);
  t.set_end(11172);
  while (t.run());
}

