#ifndef ACOUSTICS_HH
#define ACOUSTICS_HH

#include <vector>

class Acoustics {
public:
  inline Acoustics() { }
  inline Acoustics(int num_models) : m_log_prob(num_models) { }
  virtual ~Acoustics() { }
  virtual bool go_to(int frame) = 0;
  inline double log_prob(int model) const { return m_log_prob[model]; }
  inline const std::vector<double> &log_prob() const { return m_log_prob; }
protected:
  std::vector<double> m_log_prob;
};

#endif /* ACOUSTICS_HH */
