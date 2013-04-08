#ifndef ONEFRAMEACOUSTICS_HH
#define ONEFRAMEACOUSTICS_HH

#include "Acoustics.hh"

class OneFrameAcoustics : public Acoustics {
public:
  OneFrameAcoustics();
  virtual ~OneFrameAcoustics();
  virtual bool go_to(int frame);

  /** Set the probabilities for a given frame.  Set empty vector for
   * end of acoustics. */
  void set(int frame, const std::vector<float> &log_probs);
  
protected:
  int m_frame;
  std::vector<float> m_log_probs;
};

#endif /* ONEFRAMEACOUSTICS_HH */
