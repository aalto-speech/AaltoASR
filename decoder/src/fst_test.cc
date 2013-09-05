#include "FstSearch.hh"

int main(int argc, char *argv[]) {
  std::string fst_fname("/home/vsiivola/Code/fst_grammar_network/work/speechdat-final.fst");
  std::string lna_name("A10481C4.lna");
  //std::string lna_name("A12588M1.lna");
  std::string hmm_base("/home/vsiivola/Models/asr_models/am/phone/lsphone_speechdat_elisa_spuh-small-ml_20");

  //std::string fst_fname("/home/vsiivola/Code/fst_grammar_network/work/final.fst");
  //std::string lna_name("asiakaspalvelu.lna");
  //std::string hmm_base("/home/vsiivola/Models/asr_models/am/yle_matti/lsacu_all_fft400_30.8.2012_20");

  std::string hmm_name(hmm_base+".ph");
  std::string dur_name(hmm_base+".dur");

  FstSearch fsts(fst_fname.c_str(), hmm_name.c_str(), dur_name.c_str());

  fsts.lna_open(lna_name.c_str(), 1024);
  fsts.beam=200.0f;
  fsts.token_limit=1000;
  fsts.duration_scale=3.0f;
  fsts.transition_scale=1.0f;
  std::string result(fsts.run());
  fprintf(stderr, "%s\n", result.c_str());
}
