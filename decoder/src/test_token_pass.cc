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
  t.hmm_read("/share/puhe/hammaspuhe/models/speecon_all_multicondition_mmi_kld0.002_6.ph");
  t.duration_read("/share/puhe/hammaspuhe/models/speecon_all_multicondition_mmi_kld0.002_6.dur");
  t.set_lm_lookahead(1);
  t.set_word_boundary("<w>");
  t.lex_read("/share/puhe/hammaspuhe/models/free-dictation.lex");
  t.ngram_read("/share/puhe/hammaspuhe/models/free-dictation.6bo.bin", 1);
  t.read_lookahead_ngram("/share/puhe/hammaspuhe/models/free-dictation.2bo.bin");
  t.prune_lm_lookahead_buffers(0, 4);
  t.set_global_beam(210);
  t.set_word_end_beam(120);
  t.set_token_limit(30000);
  t.set_prune_similar(2);
  t.set_eq_depth_beam(162);
  t.set_eq_word_count_beam(165);
  t.set_fan_in_beam(167);
  t.set_duration_scale(3);
  t.set_transition_scale(1);
  t.set_lm_scale(28);
  
  t.set_verbose(1);
  t.set_print_probs(0);
  t.set_print_indices(0);
  t.set_print_frames(0);

  t.lna_open("/fs/project/puhe/ammansik/audio/fi/speecon_eval/SA456S01.lna", 1024);
  t.reset(0);
  t.set_end(-1);
  while (t.run());
  t.lna_open("/fs/project/puhe/ammansik/audio/fi/speecon_eval/SA456S02.lna", 1024);
  t.reset(0);
  t.set_end(-1);
  while (t.run());
  t.lna_open("/fs/project/puhe/ammansik/audio/fi/speecon_eval/SA456S03.lna", 1024);
  t.reset(0);
  t.set_end(-1);
  while (t.run());
  t.lna_open("/fs/project/puhe/ammansik/audio/fi/speecon_eval/SA456S04.lna", 1024);
  t.reset(0);
  t.set_end(-1);
  while (t.run());
  t.lna_open("/fs/project/puhe/ammansik/audio/fi/speecon_eval/SA456S05.lna", 1024);
  t.reset(0);
  t.set_end(-1);
  while (t.run());
}

