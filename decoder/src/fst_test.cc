#include "FstSearch.hh"

int main(int argc, char *argv[]) {
  std::string hmm_base("/home/vsiivola/Models/asr_models/am/yle_matti/lsacu_all_fft400_30.8.2012_20");
  std::string hmm_name(hmm_base+".ph");
  std::string dur_name(hmm_base+".dur");

  std::string fst_fname("/home/vsiivola/Code/fst_grammar_network/work/final.fst");
  FstSearch fsts(fst_fname.c_str(), hmm_name.c_str(), dur_name.c_str());
  
}
