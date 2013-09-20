#include <cstdlib>
#include <cassert>
#include <cmath>
#include <sstream>
#include "NowayHmmReader.hh"

int main(int argc, char *argv[]) {

  // Read the phone specs
  if (argc != 3) {
    fprintf(stderr, "Use: %s model.ph Hout.fst\n", argv[0]);
    exit(-1);
  }

  std::string ph_fname(argv[1]);
  std::ifstream in(ph_fname.c_str());

  NowayHmmReader hmm_reader;
  if (!in) {
    fprintf(stderr, "Can't open %s. Exit.\n", ph_fname.c_str());
    exit(-1);
  }
  hmm_reader.read(in);
  fprintf(stderr, "Read %d models\n", hmm_reader.num_models());
  
  // Loop through the hmms
  bool print_state_info(false);
  FILE *fstout = fopen(argv[2], "w");
  int last_state_idx=1;
  bool create_closure(false);
  
  fprintf(fstout, "#FSTBasic MaxPlus\n");
  fprintf(fstout, "I 0\n");
  if (create_closure) {
    fprintf(fstout, "F 0\n");
  }

  for (auto hmm : hmm_reader.hmms()) {
    if (print_state_info) {
      fprintf(stderr, "label %s\n", hmm.label.c_str());
      fprintf(stderr, "states:\n");
    }
    assert(hmm.is_source(0));
    assert(hmm.is_sink(1));
    for (auto state_idx = 0 ; state_idx < hmm.states.size(); ++ state_idx) {
      auto &state = hmm.states[state_idx];
      if (print_state_info) {
        fprintf(stderr, " %d(%d)", state_idx, state.model);
        if (hmm.is_source(state_idx)) {
          fprintf(stderr, " (source)");
        } else if (hmm.is_sink(state_idx)) {
          fprintf(stderr, " (sink)");
        }
        fprintf(stderr, " trans: [");
      }
      if (state_idx == 1) { // sink
        assert(state.transitions.size()==0);
        if (create_closure) {
          fprintf(fstout, "T %ld 0 , , 0.00000\n", last_state_idx - 3 + hmm.states.size());
        } else {
          fprintf(fstout, "F %ld\n", last_state_idx - 3 + hmm.states.size());
        }
      }

      for (auto trans_idx = 0; trans_idx < state.transitions.size(); ++ trans_idx ) {
        assert (state_idx != 1 && trans_idx <2); // No transitions fro the sink state,  2 transitions 
        auto &trans = state.transitions[trans_idx];

        if (print_state_info) {
          fprintf(stderr, " %d %.6f", trans.target, trans.log_prob /*pow(10, trans.log_prob)*/);
        }

        if (state_idx ==0) {
          assert(trans.target==2);
          fprintf(fstout, "T 0 %d %d %s %.6f\n", last_state_idx++, hmm.states[2].model, hmm.label.c_str(), 
                  trans.log_prob /*pow(10, trans.log_prob)*/);
          continue;
        }

        if  ( trans_idx==0 ) { // self transition
          assert(trans.target = state_idx);
          fprintf(fstout, "T %d %d %d , %.6f\n", last_state_idx-1, last_state_idx-1, state.model, 
                  trans.log_prob /*pow(10, trans.log_prob)*/);
          continue;
        }

        if ( trans_idx == 1 ) { // transit to next state
          if (state_idx == hmm.states.size()) { // last state, transit back to sink
            fprintf(fstout, "T %d 0 , , %.6f\n", last_state_idx-1, trans.log_prob /*pow(10, trans.log_prob)*/);
            continue;
          }         
          // trans to next state
          std::string emission_idx_str;
          if (state_idx < hmm.states.size()-1) { 
            std::ostringstream ss;
            ss << hmm.states[state_idx+1].model;
            emission_idx_str = ss.str();
          } else { // Last state, no emission from this triphone
            emission_idx_str = ",";
          }
          fprintf(fstout, "T %d %d %s , %.6f\n", last_state_idx-1, last_state_idx, emission_idx_str.c_str(), 
                  trans.log_prob /*pow(10, trans.log_prob)*/);
          last_state_idx++;
        }

      }
      if (print_state_info) fprintf(stderr, " ]\n");
    }
    if (print_state_info) fprintf(stderr, "\n");
  }
  fclose(fstout);

  return 0;
}
