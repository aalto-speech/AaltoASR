#include <cstddef>  // NULL
#include "StateProbReader.hh"

StateProbReader::StateProbReader()
  : m_file(NULL),
    m_first_frame(0),
    m_last_frame(1),
    m_frames(1)
{
}

StateProbReader::~StateProbReader()
{
}

void
StateProbReader::open(FILE *file, int frames)
{
  size_t items_read;
  m_file = file;

  // Check the file format
  char tmp[3];
  items_read = fread(tmp, 3, 1, file);
  if (items_read != 1 || strcmp(tmp, "SPS") != 0)
    throw InvalidFormat();

  // Read the number of states
  items_read = fread(&m_num_states, sizeof(float), 1, file);
  if (items_read != 1)
    throw InvalidFormat();

  // Reserve memory for read buffer;
  m_frames = frames;
  m_read_buffer.reserve(m_frames * m_num_states);
  m_first_frame = 0;
  m_last_frame = 1;
}

bool
StateProbReader::go_to(int frame)
{
  // Read frames upto requested frame
  while (m_last_frame < frame) {
    size_t floats_read = fread(frame(m_last_frame), 
			       sizeof(float), m_num_states, m_file);
    if (floats_read == 0)
      return false; // End of file
    if (floats_read != m_num_states)
      throw ShortFrame();

    m_last_frame++;
  }

  // Check if the past frame is still in the buffer.
  if (frame <= m_last_frame - m_frames)
    return ForgottenFrame();

  m_log_prob = frame(frame);
  return true;
}
