#ifdef _MSC_VER
#include <boost/math/tr1.hpp>
using namespace boost::math::tr1;
#else
#include <math.h>
#endif

#include "Hmm.hh"


StateDuration::StateDuration()
  : a(0), b(0), a0(0), mode(0)
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
    mode = (int)floor(temp);
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


void StateDuration::set_sr_parameters(float a0, float a1, float b0, float b1)
{
  this->a0 = a0;
  this->a1 = a1;
  this->b0 = b0;
  this->b1 = b1;
}


float StateDuration::get_sr_comp_log_prob(int duration, float sr) const
{
  if (a0 > 0)
  {
    float ia = a0 + sr*a1;
    float ib = b0 + sr*b1;
    return (ia-1)*logf((float)duration)-(float)duration/ib-ia*logf(ib)-
      lgammaf(ia);
  }
  else
  {
    return get_log_prob(duration);
  }
}
