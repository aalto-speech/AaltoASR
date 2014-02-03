#include "FstConfidence.hh"
#include "levenshtein.hh"

FstConfidence::FstConfidence(const char *grammar_fst_name, const char *hmm_fname, const char *dur_fname): FstSearch_base<FstConfidenceToken>(grammar_fst_name), m_logprob_conf_weight(2.0f), m_logprob_conf_hysteresis(100.0f), m_fst_acoustics(hmm_fname, dur_fname) {
  set_acoustics(&m_fst_acoustics);
}

void FstConfidence::grammar_token_and_best_acu_confidence(float *gt_conf, float *ba_conf) {
  // NOTE: Tokens at the same state get pruned, if only one final state in network this can be quite unreliable
  bool check_only_final_nodes=false;
  bool reject_same_prefix=false;

  float best_final_token_logprob;
  std::vector<std::string> best_final_token_symbols;
  for (const auto t: this->m_new_tokens) {
    if (this->m_fst.nodes[t.node_idx].end_node) {
      best_final_token_logprob = t.logprob;
      best_final_token_symbols = t.unemitted_words;
      //fprintf(stderr, "Best %s\n", t.str().c_str());
      break;
    }
  }

  *ba_conf = 1.5f- 0.25f*(-best_final_token_logprob + m_best_acu_score)/m_cur_frame;
  //*ba_conf = m_best_acu_score/best_final_token_logprob;

  if (best_final_token_symbols.size()==0) {
    fprintf(stderr, "Emptiness\n");
    *gt_conf = -9999999.9f;
    return;
  }

  float best_different_hypo_logprob=-9999999.9f;
  for (const auto t:this->m_new_tokens) {
    //fprintf(stderr, "Tokening %s\n", t.str().c_str());
    if (check_only_final_nodes && this->m_fst.nodes[t.node_idx].end_node == false) continue;

    if (t.unemitted_words.size() > best_final_token_symbols.size()) {
      //fprintf(stderr, "size\n");
      best_different_hypo_logprob = t.logprob;
      break;
    }

    // Check for the same prefix
    if (reject_same_prefix) {
      for (auto i=0; i<t.unemitted_words.size(); ++i) {
        if (t.unemitted_words[i] != best_final_token_symbols[i]) {
          best_different_hypo_logprob = t.logprob;
          //fprintf(stderr,"Diff hypo: ");
          //for (const auto w: t.unemitted_words) {
          //  fprintf(stderr, " %s", w.c_str());
          //}
          //fprintf(stderr, "\n");
          goto out;
        }
      }
    } else {
      if (t.unemitted_words != best_final_token_symbols) {
        best_different_hypo_logprob = t.logprob;
        goto out;
      }
    }
  }

 out:
  // Log difference
  //float token_dist(best_final_token_logprob-best_different_hypo_logprob);
  //float first_term = std::min(token_dist, 0.0f)*logf(static_cast<float>(m_frame))/1000.0f;
  //fprintf(stderr, "%.4f %.4f\n", token_dist, first_term);
  //return std::max(0.0f, 1.0f + first_term);

  // Log division
  //fprintf(stderr, "%.4f %.4f %.4f\n", best_different_hypo_logprob, best_final_token_logprob, best_different_hypo_logprob/(m_logprob_conf_weight*best_final_token_logprob-m_logprob_conf_hysteresis)); //3000.0f*logf(static_cast<float> (m_frame))));
  //*gt_conf = std::min(1.0f, best_different_hypo_logprob/(m_logprob_conf_weight*best_final_token_logprob-m_logprob_conf_hysteresis)); //-3000.0f*logf(static_cast<float> (m_frame))));

  *gt_conf = std::max(0.0f, std::min(1.0f, 0.2f- 5.0f*(-best_final_token_logprob + best_different_hypo_logprob)/m_cur_frame));
}

float FstConfidence::get_best_frame_acu_prob() {
  float best_prob = -999999999.9f;
  for (int i=0; i < m_fst_acoustics.num_models(); ++i) {
    float model_prob = m_fst_acoustics.log_prob(i);
    if (model_prob>best_prob) best_prob = model_prob;
  }
  return best_prob;
}

FstConfidenceWithPhoneLoop::FstConfidenceWithPhoneLoop(
    const char *grammar_fst_name, const char * phone_loop_fst_name,
    const char *hmm_fname, const char * dur_fname):
  FstConfidence(grammar_fst_name, hmm_fname, dur_fname), m_ploop_logprob_weight(0.8f),
  m_phone_fst(phone_loop_fst_name) 
{
  m_phone_fst.verbose=1;
  m_phone_fst.set_acoustics(&m_fst_acoustics);
}

void FstConfidenceWithPhoneLoop::run() {
  m_best_acu_score = 0.0f;
  m_cur_frame=0;
  while (m_fst_acoustics.next_frame()) {
    m_phone_fst.propagate_tokens();
    propagate_tokens();
    m_best_acu_score += get_best_frame_acu_prob(); 
    m_cur_frame++;
  }
  //fprintf(stderr, "%s\n", tokens_at_final_states().c_str());
  //fprintf(stderr, "%s\n", best_tokens().c_str());
}

std::string remove_junk(const std::string a) {
  std::string b;
  char prev_token = ' ';

  for (auto c : a) {
    if (c == ' ' || c == prev_token) continue;
    prev_token = c;
    b+=c;
  }
  return b;
}

float FstConfidenceWithPhoneLoop::levenshtein_confidence(const std::string &grammar_s, const std::string &ploop_s) {
  const std::string clean_grammar_s(remove_junk(grammar_s));
  const std::string clean_ploop_s(remove_junk(ploop_s));
  int ldist = levenshtein_distance(clean_grammar_s, clean_ploop_s);
  fprintf(stderr, "Ldist %d for '%s' vs '%s'\n", 
          ldist, clean_grammar_s.c_str(), clean_ploop_s.c_str());
  return std::max(0.0f, 1.0f-float(ldist)/clean_grammar_s.size());
}

bytestype FstConfidenceWithPhoneLoop::result_and_confidence(float *confidence_retval) {
  float grammar_logprob, ploop_logprob;
  bytestype res_string(get_result_and_logprob(grammar_logprob));
  bytestype ploop_string(m_phone_fst.get_result_and_logprob(ploop_logprob));

  m_ploop_conf = std::min(1.0f, 1.0f- 0.25f*(-grammar_logprob + ploop_logprob)/m_cur_frame);
  fprintf(stderr, "pl_lp %.2f, gr_lp %.2f, len %d\n", ploop_logprob, grammar_logprob, m_cur_frame);
  grammar_token_and_best_acu_confidence(&m_token_conf, &m_best_acu_conf);
  m_edit_conf = levenshtein_confidence(res_string, ploop_string);
  *confidence_retval = (std::min( 1.0f, m_ploop_conf) + 20.0f*std::min( 1.0f, m_token_conf) 
                        + 5.0f*std::min(1.0f, m_edit_conf) + std::min( 1.0f, m_best_acu_conf))/27.0f;
  
  if (verbose) {
    fprintf(stderr, "%s:\n\tToken: %.4f\n", res_string.c_str(), m_token_conf);
    fprintf(stderr, "\tPloop: %.4f\n", m_ploop_conf);
    fprintf(stderr, "\tEdit: %.4f\n", m_edit_conf);
    fprintf(stderr, "\tBacu: %.4f\n", m_best_acu_conf);
    fprintf(stderr, "\tTotal: %.4f\n\n", *confidence_retval);
  }

  return res_string;
}


