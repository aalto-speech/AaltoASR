#include "LnaReaderCircular.hh"

LnaReaderCircular::LnaReaderCircular()
  : m_fin(),
    m_in(NULL),
    m_num_models(0),
    m_buffer_size(0),
    m_first_index(0),
    m_frames_read(0),
    m_log_prob_buffer(0),
    m_eof_frame(-1),
    m_read_buffer(0)
{
}

void
LnaReaderCircular::open(const char *file, int num_models, int size)
{
  // Open file
  if (m_fin.is_open())
    m_fin.close();
  m_fin.clear();
  m_fin.open(file);
  if (!m_fin)
    throw OpenError();
  m_in = &m_fin;

  init(m_fin, num_models, size);
}

void
LnaReaderCircular::init(std::istream &in, int num_models, int size)
{
  m_in = &in;

  // Set variables
  m_num_models = num_models;
  m_buffer_size = size;
  m_first_index = 0;
  m_frames_read = 0;
  m_eof_frame = -1;

  // Initialize buffers
  m_log_prob_buffer.clear();
  m_read_buffer.clear();
  m_log_prob_buffer.resize(num_models * size);
  m_read_buffer.resize(num_models + 1);
}
    
void
LnaReaderCircular::close()
{
  if (m_fin.is_open())
    m_fin.close();
  m_in = NULL;
}

bool
LnaReaderCircular::go_to(int frame)
{
  if (m_in == NULL)
    throw NotOpened();

  if (m_eof_frame > 0 && frame >= m_eof_frame)
    return false;

  while (m_frames_read <= frame) {

    // Read a frame
    m_in->read(&m_read_buffer[0], m_num_models + 1);
    if (m_in->bad())
      throw ReadError();

    // Check frame length and id
    if (m_in->fail()) { // EOF or short read
      int bytes_read = m_in->gcount();
      if (bytes_read == 0) {
	m_eof_frame = m_frames_read;
	return false;
      }
      if (bytes_read != m_num_models + 1)
	throw ShortFrame();
      if (m_read_buffer[0] != 0 && (unsigned char)m_read_buffer[0] != 0x80)
	throw InvalidFrameId();
    }

    // Parse frame to the circular buffer
    for (int i = 0; i < m_num_models; i++) {
      double tmp = (unsigned char)m_read_buffer[i + 1] / -24.0;
      m_log_prob_buffer[m_first_index] = tmp;
      m_first_index++;
      if (m_first_index >= m_log_prob_buffer.size())
	m_first_index = 0;
    }

    m_frames_read++;
  }

  if (frame < m_frames_read - m_buffer_size)
    throw FrameForgotten();

  int index = m_first_index - (m_frames_read - frame) * m_num_models;
  if (index < 0)
    index += m_log_prob_buffer.size();
    
  m_log_prob = &m_log_prob_buffer[index];

  return true;
}
