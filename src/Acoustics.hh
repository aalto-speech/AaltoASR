#ifndef ACOUSTICS_HH
#define ACOUSTICS_HH

#include <vector>

class Acoustics {
public:
  inline Acoustics() : m_log_prob(NULL), m_eof_frame(-1) { }
  virtual ~Acoustics() { }
  virtual bool go_to(int frame) = 0;
  inline double log_prob(int model) const { return m_log_prob[model]; }
  inline int eof_frame() const { return m_eof_frame; }
protected:
  double *m_log_prob;
  int m_eof_frame;
};

#endif /* ACOUSTICS_HH */
