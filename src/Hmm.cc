#include <math.h>

#include "Hmm.hh"


StateDuration::StateDuration()
  : a(0), b(0)
{
}

void StateDuration::set_parameters(float a, float b)
{
  this->a = a;
  this->b = b;
  const_term = -a*logf(b)-lgammaf(a);
}

float StateDuration::get_log_prob(int duration) const
{
  if (a > 0)
    return (a-1)*logf(duration)-duration/b+const_term;
  return 0; // No duration penalty
}
