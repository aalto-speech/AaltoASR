#ifndef ACOUSTICS_HH
#define ACOUSTICS_HH

#include <vector>

class Acoustics {
public:
  inline Acoustics() : m_log_prob(NULL) { }
  virtual ~Acoustics() { }
  virtual bool go_to(int frame) = 0;
  inline double log_prob(int model) const { return m_log_prob[model]; }
protected:
  double *m_log_prob;
};

#endif /* ACOUSTICS_HH */
