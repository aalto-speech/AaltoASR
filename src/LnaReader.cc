#include <cstddef>  // NULL
#include <memory>

#include <assert.h>
#include <math.h>

#include "LnaReader.hh"

LnaReader::LnaReader()
  : Acoustics(),
    m_two_byte(false),
    m_num_models(0),
    m_frame_size(0),
    m_frames(0),
    m_block_size(256),
    m_eof(0),
    m_fin(),
    m_in(NULL),
    m_read_buffer()
{
}

LnaReader::LnaReader(std::istream &in, int num_models)
  : Acoustics(num_models),
    m_two_byte(false),
    m_num_models(num_models),
    m_frame_size(num_models + 1),
    m_frames(0),
    m_block_size(256),
    m_eof(0),
    m_fin(),
    m_in(&in),
    m_read_buffer()
{
}

LnaReader::LnaReader(const char *file, int num_models)
  : Acoustics(num_models),
    m_two_byte(false),
    m_num_models(num_models),
    m_frame_size(num_models + 1),
    m_frames(0),
    m_block_size(256),
    m_eof(0),
    m_fin(file),
    m_in(&m_fin),
    m_read_buffer()
{
  if (!m_fin)
    throw OpenError();
}

void 
LnaReader::open(const char *file, int num_models, book two_byte)
{
  m_log_prob.resize(num_models);
  if (two_byte) 
    m_frames = num_models * 2 + 1;
  else
    m_frame_size = num_models + 1;
  m_frames = 0;
  m_eof = 0;
  if (m_fin.is_open())
    m_fin.close();
  m_fin.clear();
  m_fin.open(file);
  if (!m_fin)
    throw OpenError();
  m_in = &m_fin;
  m_num_models = num_models;
  m_log_prob.resize(num_models);
  m_read_buffer.resize(0);
}

void
LnaReader::close()
{
  
}

// PRECONDITIONS:
// - read_buffer has full frames
// POSTCONDITIONS:
// - the buffer contains 'frame'th frame or the last frame in the file
// - read buffer has full frames
void
LnaReader::read_to(int frame)
{
  assert(m_in != NULL);

  int bytes_read = m_frames * m_frame_size;
  int bytes_end = (frame + 1) * m_frame_size;
  m_read_buffer.resize(bytes_end);

  // Read more if the frame is not in the buffer
  while (bytes_read < bytes_end && !m_eof) {

    // FIXME: is it portable to write directly in std::vector?!
    bool ok = 
      m_in->read(&m_read_buffer[bytes_read], bytes_end - bytes_read);

    if (m_in->bad())
      throw ReadError();
    bytes_read += m_in->gcount();

    // Early eof
    if (!ok) {
      if (bytes_read % m_frame_size != 0)
	throw ShortFrame();
      m_eof = true;
    }
  }
  m_frames = bytes_read / m_frame_size;
}

bool
LnaReader::go_to(int frame)
{
  if (frame >= m_frames)
    read_to(frame + m_block_size - 1);
  if (frame >= m_frames)
    return false;
  parse_frame(frame);
  return true;
}

// PRECONDITIONS:
// - m_read_buffer has enough data
void
LnaReader::parse_frame(int frame)
{
  int start = frame * m_frame_size;

  assert(m_read_buffer.size() >= start + m_frame_size);

  // Check frame id
  //
  if (m_read_buffer[start] == 0)
    m_sentence_end = false;
  else if ((unsigned char)m_read_buffer[start] == 0x80)
    m_sentence_end = true;
  else
    throw InvalidFrameId();

  // Got frame, parse it
  //
  if (m_two_byte) {
    for (int i = 1; i < m_frame_size; i += 2) {
      float tmp = (unsigned char)m_read_buffer[i + start] * 256 + 
	(unsigned char)m_read_buffer[i + start + 1];
      tmp = tmp / -1820.0;

      assert(tmp <= 0);
      m_log_prob[i-1] = tmp;
    }
  }
  else {
    for (int i = 1; i < m_frame_size; i++) {
      float tmp = (unsigned char)m_read_buffer[i + start] / -24.0;
      assert(tmp <= 0);
      m_log_prob[i-1] = tmp;
    }
  }
}
