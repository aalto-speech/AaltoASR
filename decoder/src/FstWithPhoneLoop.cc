#include "FstWithPhoneLoop.hh"

FstWithPhoneLoop::FstWithPhoneLoop(const char *grammar_fst_name, const char * phone_loop_fst_name,
                                   const char *hmm_path, const char * dur_path):
  SearchModelReader(hmm_path, dur_path), m_grammar_fst(grammar_fst_name), 
  m_phone_fst(phone_loop_fst_name)
 {
   m_grammar_fst.set_dur_tables(m_a_table, m_b_table);
   m_phone_fst.set_dur_tables(m_a_table, m_b_table);
}


void FstWithPhoneLoop::run() {
  while (m_acoustics->go_to(m_frame)) {
    m_phone_fst.propagate_tokens();
    m_grammar_fst.propagate_tokens();
    m_frame++;
  }
  fprintf(stderr, "%s\n", m_grammar_fst.tokens_at_final_states().c_str());
  fprintf(stderr, "%s\n", m_grammar_fst.best_tokens().c_str());
}

bytestype FstWithPhoneLoop::get_best_final_hypo_string_and_confidence(float &confidence_retval) {
  float grammar_logprob=0.0f;
  bytestype res_string(m_grammar_fst.get_best_final_hypo_string_and_logprob(grammar_logprob));
  float ploop_logprob(m_phone_fst.get_best_token_logprob());
  fprintf(stderr, "Got %.4f %.4f for '%s'\n", grammar_logprob, ploop_logprob, res_string.c_str());

  return res_string;
}
