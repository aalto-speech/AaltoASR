#include <math.h>

#include "Hmm.hh"


StateDuration::StateDuration()
  : a(0), b(0), mode(0)
{
}

void StateDuration::set_parameters(float a, float b)
{
  float temp;
  
  this->a = a;
  this->b = b;
  mode = 0;
  if (a > 0)
  {
    const_term = -a*logf(b)-lgammaf(a);
    temp = b*(a-1); // Mode of the gamma distribution
    mode = floor(temp);
    if (get_log_prob(mode) < get_log_prob(mode+1))
      mode++;
  }
}

float StateDuration::get_log_prob(int duration) const
{
  if (a > 0)
    return (a-1)*logf(duration)-duration/b+const_term;
  return 0; // No duration penalty
}
