#ifndef FSTCONFIDENCE
#define FSTCONFIDENCE

#include "FstSearch.hh"

struct FstConfidenceToken: public FstToken {
  FstConfidenceToken(): dist_to_best_acu(0.0f) {};

  float dist_to_best_acu;
};

class FstConfidence: public FstSearch_base<FstConfidenceToken> {
public:
  FstConfidence(const char *grammar_fst_name, const char *hmm_fname, const char *dur_fname);

  void set_logprob_conf_weight(float lpw) {m_logprob_conf_weight = lpw;}
  void set_logprob_conf_hysteresis(float h) { m_logprob_conf_hysteresis = h; }

  inline void run() {
    m_best_acu_score = 0.0f;
    m_cur_frame = 0 ;
    while (m_fst_acoustics.next_frame()) {
      propagate_tokens();
      get_best_frame_acu_prob();
      m_cur_frame++;
    }
  }

  virtual inline bytestype result_and_confidence(float *confidence_retval) {
    float gt_conf, ba_conf;
    grammar_token_and_best_acu_confidence(&gt_conf, &ba_conf);
    *confidence_retval = 0.5* (gt_conf + ba_conf);
    return get_result();
  }

protected:
  float m_logprob_conf_weight;
  float m_logprob_conf_hysteresis;

  float m_best_acu_score;

  FstAcoustics m_fst_acoustics;
  void grammar_token_and_best_acu_confidence(float *, float *);

  float get_best_frame_acu_prob();

  // For length normalization
  int m_cur_frame;
};

class FstConfidenceWithPhoneLoop : public FstConfidence {
public:
  FstConfidenceWithPhoneLoop(const char *grammar_fst_name, const char *phone_loop_fst_name,
                             const char *hmm_fname, const char *dur_fname);

  void set_phone_beam(float pb) {m_phone_fst.set_beam(pb);}
  void set_phone_token_limit(float ptl) {m_phone_fst.set_token_limit(ptl);}
  void set_phone_duration_scale(float pds) {m_phone_fst.set_duration_scale(pds);}

  void set_phone_loop_logprob_weight(float w) { m_ploop_logprob_weight=w;}

  // Overridden funcs
  void run();
  bytestype result_and_confidence(float *confidence_retval);

  void init_search() {
    FstSearch_base<FstConfidenceToken>::init_search();
    m_phone_fst.init_search();
  }

  float get_ploop_conf() {return m_ploop_conf;}
  float get_token_conf() {return m_token_conf;}
  float get_edit_conf() {return m_edit_conf;}
  float get_best_acu_conf() {return m_best_acu_conf;}

private:
  float m_ploop_logprob_weight;
  FstSearch m_phone_fst;

  // Store last values for debug
  float m_ploop_conf;
  float m_token_conf;
  float m_edit_conf;
  float m_best_acu_conf;

  float levenshtein_confidence(const std::string &grammar_s, const std::string &ploop_s);
};

#endif
