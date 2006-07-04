#include <assert.h>
#include "OneFrameAcoustics.hh"

OneFrameAcoustics::OneFrameAcoustics() 
  : m_frame(-1)
{
}

OneFrameAcoustics::~OneFrameAcoustics() 
{ 
}
  
bool 
OneFrameAcoustics::go_to(int frame)
{
  assert(m_frame == frame);
  if (m_log_probs.empty())
    return false;
  return true;
}

void 
OneFrameAcoustics::set(int frame, const std::vector<float> &log_probs)
{
  m_frame = frame;
  m_num_models = log_probs.size();
  m_log_probs = log_probs;
  m_log_prob = &m_log_probs[0];
}
