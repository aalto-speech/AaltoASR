#ifndef MPEEVALUATOR_HH
#define MPEEVALUATOR_HH

#include "HmmSet.hh"
#include "HmmNetBaumWelch.hh"

class MPEEvaluator : public HmmNetBaumWelch::CustomDataQuery {
private:
  int m_first_frame;
  int m_cur_frame;
  std::vector< std::vector<HmmNetBaumWelch::ArcInfo>* > m_ref_segmentation;
  
public:
  typedef enum { MPEM_MONOPHONE_LABEL, // Correct monophone label
                 MPEM_MONOPHONE_STATE, // Correct monophone label+state index
                 MPEM_CONTEXT_LABEL, // Correct context phone label
                 MPEM_STATE, // Correct state
                 MPEM_CONTEXT_PHONE_STATE, // A state of the correct CP
                 MPEM_HYP_CONTEXT_PHONE_STATE, // A correct state in hypothesis CP
  } MPEMode;
  
  MPEEvaluator() { m_first_frame = -1; m_mode = MPEM_MONOPHONE_LABEL; m_model = NULL; m_mpfe_insertion_penalty = 0; m_phone_error = false; m_ignore_silence = false; m_binary_mpfe = false; }

  // CustomDataQuery interface
  virtual ~MPEEvaluator() { }
  virtual double custom_data_value(int frame, HmmNetBaumWelch::Arc &arc,
                                   int extra);

  // Other public methods
  void set_mode(MPEMode mode) { m_mode = mode; }
  void set_model(HmmSet *model) { m_model = model; }
  void set_phone_error(bool phone_error) { m_phone_error = phone_error; }
  void set_ignore_silence(bool silence) { m_ignore_silence = silence; }
  void fetch_frame_info(HmmNetBaumWelch *seg);
  void reset(void);

  int non_silence_frames(void) { return m_non_silence_frames; }
  double non_silence_occupancy(void) { return m_non_silence_occupancy; }
  int frames(void) { return (int)m_ref_segmentation.size(); }

private:
  std::string extract_center_phone(const std::string &label);
  std::string extract_context_phone(const std::string &label);
  int extract_state(const std::string &label);

private:
  MPEMode m_mode;
  HmmSet *m_model;
  int m_non_silence_frames;
  double m_non_silence_occupancy;
  double m_mpfe_insertion_penalty;
  bool m_phone_error;
  bool m_ignore_silence;
  bool m_binary_mpfe;
};

#endif // MPEEVALUATOR_HH
