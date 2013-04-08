#ifndef TIMER_HH
#define TIMER_HH

#include <cstddef>  // NULL
#include <time.h>
#include <sys/times.h>

class Timer {
public:
  Timer();
  void start();
  void stop();
  void reset();
  clock_t ticks() const;
  float sec() const;
protected:
  clock_t m_ticks_total;
  clock_t m_ticks_start;
};

inline
Timer::Timer()
  : m_ticks_total(0)
{
}

inline
void
Timer::start()
{
  m_ticks_start = times(NULL);
}

inline
void
Timer::stop()
{
  int ticks_end = times(NULL);
  m_ticks_total += ticks_end - m_ticks_start;
}

inline
void
Timer::reset()
{
  m_ticks_total = 0;
}

inline
clock_t
Timer::ticks() const
{
  return m_ticks_total;
}

inline
float
Timer::sec() const
{
  return (float)m_ticks_total / (float)CLOCKS_PER_SEC;
}

#endif /* TIMER_HH */
