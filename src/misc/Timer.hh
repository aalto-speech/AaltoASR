#ifndef TIMER_HH
#define TIMER_HH

#include <assert.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <sys/times.h>
#include <stdio.h>

/** A class for measuring process time in ticks or seconds.  If ticks
 * or seconds are needed while the timer is running, call snapshot()
 * first, and then request ticks or seconds WHILE the timer is still
 * running.  After stop() call, the functions always return the stop
 * time.
 */
class Timer {
public:
  /** Default constructor. */
  Timer();

  /** Starts the clock without resetting it. */
  void start();

  /** Stops the clock (can be restarted) */
  void stop();

  /** Records the current time between start and this function
   * call. */
  void snapshot();

  /** Reset the clock. */
  void reset();

  /** Real time in ticks. */
  clock_t real_ticks() const;

  /** CPU time in ticks. */
  clock_t user_ticks() const;

  /** System time in ticks. */
  clock_t sys_ticks() const;

  /** Total (CPU plus system) time in ticks */
  clock_t user_sys_ticks() const;

  /** Real time in seconds. */
  float real_sec() const;

  /** CPU time in seconds. */
  float user_sec() const;

  /** System time in seconds. */
  float sys_sec() const;

  /** Total (CPU plus sytem) time in seconds. */
  float user_sys_sec() const;

protected:
  clock_t m_final_real_ticks;
  clock_t m_final_user_ticks;
  clock_t m_final_sys_ticks;
  clock_t m_temp_real_ticks;
  clock_t m_temp_user_ticks;
  clock_t m_temp_sys_ticks;
  clock_t m_real_start;
  clock_t m_user_start;
  clock_t m_sys_start;
  long m_ticks_per_sec;
  bool m_running;
};

inline
Timer::Timer()
{
  m_running = false;
  reset();
  m_ticks_per_sec = sysconf(_SC_CLK_TCK);
  if (m_ticks_per_sec <= 0) {
    perror("ERROR: sysconf(_SC_CLK_TCK) failed");
    exit(1);
  }
}

inline
void
Timer::start()
{
  assert(!m_running);
  m_running = true;

  struct tms tms;
  clock_t ret = times(&tms);
  if (ret == (clock_t)-1) {
    perror("ERROR: times() failed");
    exit(1);
  }
  m_real_start = ret;
  m_user_start = tms.tms_utime;
  m_sys_start = tms.tms_stime;
  m_temp_real_ticks = m_user_start;
  m_temp_user_ticks = m_user_start;
  m_temp_sys_ticks = m_sys_start;
}

inline
void
Timer::stop()
{
  assert(m_running);
  m_running = false;

  struct tms tms;
  clock_t ret = times(&tms);
  if (ret == (clock_t)-1) {
    perror("ERROR: times() failed");
    exit(1);
  }
  m_final_real_ticks += ret - m_real_start;
  m_final_user_ticks += tms.tms_utime - m_user_start;
  m_final_sys_ticks += tms.tms_stime - m_sys_start;
}

inline
void
Timer::snapshot()
{
  assert(m_running);

  struct tms tms;
  clock_t ret = times(&tms);
  if (ret == (clock_t)-1) {
    perror("ERROR: times() failed");
    exit(1);
  }
  m_temp_real_ticks = ret - m_real_start;
  m_temp_user_ticks = tms.tms_utime - m_user_start;
  m_temp_sys_ticks = tms.tms_stime - m_sys_start;
}

inline
void
Timer::reset()
{
  assert(!m_running);
  m_final_real_ticks = 0;
  m_final_user_ticks = 0;
  m_final_sys_ticks = 0;
  m_temp_real_ticks = 0;
  m_temp_user_ticks = 0;
  m_temp_sys_ticks = 0;
  m_real_start = 0;
  m_user_start = 0;
  m_sys_start = 0;
}

inline
clock_t
Timer::real_ticks() const
{
  if (m_running)
    return m_temp_real_ticks;
  return m_final_real_ticks;
}

inline
clock_t
Timer::user_ticks() const
{
  if (m_running)
    return m_temp_user_ticks;
  return m_final_user_ticks;
}

inline
clock_t
Timer::sys_ticks() const
{
  if (m_running)
    return m_temp_sys_ticks;
  return m_final_sys_ticks;
}

inline
clock_t
Timer::user_sys_ticks() const
{
  if (m_running)
    return m_temp_user_ticks + m_temp_sys_ticks;
  return m_final_user_ticks + m_final_sys_ticks;
}

inline
float
Timer::real_sec() const
{
  if (m_running)
    return (float)m_temp_real_ticks / (float)m_ticks_per_sec;
  return (float)m_final_real_ticks / (float)m_ticks_per_sec;
}

inline
float
Timer::user_sec() const
{
  if (m_running)
    return (float)m_temp_user_ticks / (float)m_ticks_per_sec;
  return (float)m_final_user_ticks / (float)m_ticks_per_sec;
}

inline
float
Timer::sys_sec() const
{
  if (m_running)
    return (float)m_temp_sys_ticks / (float)m_ticks_per_sec;
  return (float)m_final_sys_ticks / (float)m_ticks_per_sec;
}

inline
float
Timer::user_sys_sec() const
{
  return user_sys_ticks() / (float)m_ticks_per_sec;
}

#endif /* TIMER_HH */
