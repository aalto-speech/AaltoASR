#include "FstWithPhoneLoop.hh"

void test_fstsearch(std::string fst_fname, std::string lna_name, std::string hmm_name, std::string dur_name) {
  fprintf(stderr, "Test fstsearch\n");
  FstSearch fsts(fst_fname.c_str(), hmm_name.c_str(), dur_name.c_str());

  fsts.set_beam(200.0f);
  fsts.set_token_limit(1000);
  fsts.set_duration_scale(3.0f);
  fsts.set_transition_scale(1.0f);

  fsts.lna_open(lna_name.c_str(), 1024);
  fsts.init_search();
  fsts.run();
  std::string result(fsts.get_best_final_hypo_string());
  fprintf(stderr, "%s\n", result.c_str());
}

void test_fstwithphone(std::string fst_fname, std::string open_phoneloop_fst, 
                       std::string lna_name, std::string hmm_name, std::string dur_name) {
  fprintf(stderr, "Test fstwithphone\n");
  FstWithPhoneLoop fsts(fst_fname.c_str(),open_phoneloop_fst.c_str(), hmm_name.c_str(), dur_name.c_str());

  fsts.set_duration_scale(3.0f);
  fsts.set_transition_scale(1.0f);

  fsts.set_grammar_beam(200.0f);
  fsts.set_grammar_token_limit(1000);
  fsts.set_phone_beam(20.0f);
  fsts.set_phone_token_limit(5);

  fsts.lna_open(lna_name.c_str(), 1024);
  fsts.init_search();
  fsts.run();
  float confidence=0.0f;
  std::string result(fsts.get_best_final_hypo_string_and_confidence(confidence));
  fprintf(stderr, "%.4f: %s\n", confidence, result.c_str());
  fsts.lna_close();
}

int main(int argc, char *argv[]) {
  //std::string fst_fname("/home/vsiivola/Code/fst_grammar_network/work/speechdat-final.fst");
  std::string fst_fname("speechdat-final.fst");
  std::string lna_name("A10481C4.lna");
  //std::string lna_name("A12588M1.lna");
  std::string hmm_base("/home/siivola/Models/asr_models/am/phone/lsphone_speechdat_elisa_spuh-small-ml_20");
  std::string open_phoneloop_fst("ploop-final.fst");

  std::string hmm_name(hmm_base+".ph");
  std::string dur_name(hmm_base+".dur");

  test_fstsearch(fst_fname, lna_name, hmm_name, dur_name);
  test_fstwithphone(fst_fname, open_phoneloop_fst, lna_name, hmm_name, dur_name);
}
