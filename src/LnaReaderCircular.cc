#include <cstddef>  // NULL
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <cassert>
#include "LnaReaderCircular.hh"

LnaReaderCircular::LnaReaderCircular()
  : m_file(NULL),
    m_header_size(5),
    m_buffer_size(0),
    m_first_index(0),
    m_frames_read(0),
    m_log_prob_buffer(0),
    m_frame_size(0),
    m_read_buffer(0),
    m_lna_bytes(1)
{
}

int
LnaReaderCircular::read_int()
{
  assert(sizeof(unsigned int) == 4);
  unsigned char buf[4];
  if (fread(buf, 4, 1, m_file) != 1) {
    fprintf(stderr, "LnaReaderCircular::open(): read error: %s\n", 
	    strerror(errno));
    exit(1);
  }

  int tmp = buf[3] + (buf[2] << 8) + (buf[1] << 16) + (buf[0] << 24);
  return tmp;
}

void
LnaReaderCircular::open(const char *filename, int buf_size)
{
  if (m_file != NULL)
    close();

  m_file = fopen(filename, "r");
  if (m_file == NULL) {
    fprintf(stderr, "LnaReaderCircular::open(): could not open %s: %s\n",
	    filename, strerror(errno));
    exit(1);
  }

  // Read header
  m_num_models = read_int();
  if (m_num_models <= 0) {
    fprintf(stderr, "LnaReaderCircular::open(): invalid number of states %d\n",
	    m_num_models);
    exit(1);
  }
  int bytes = fgetc(m_file);
  if (bytes== EOF) {
    fprintf(stderr, "LnaReaderCircular::open(): read error: %s\n",
	    strerror(errno));
    exit(1);
  }

  // Set some variables
  m_lna_bytes = bytes;
  m_buffer_size = buf_size;
  m_first_index = 0;
  m_frames_read = 0;
  m_eof_frame = -1;
  if (m_lna_bytes == 4)
    m_frame_size = m_num_models * 4;
  else if (m_lna_bytes == 2)
    m_frame_size = m_num_models * 2;
  else if (m_lna_bytes == 1)
    m_frame_size = m_num_models;
  else
  {
    fprintf(stderr, "LnaReaderCircular::open(): invalid LNA byte number %d\n",
            m_lna_bytes);
    exit(1);
  }

  // Initialize buffers
  m_log_prob_buffer.clear();
  m_read_buffer.clear();
  m_log_prob_buffer.resize(m_num_models * buf_size);
  m_read_buffer.resize(m_frame_size);
}
    
void
LnaReaderCircular::close()
{
  if (m_file != NULL)
    fclose(m_file);
  m_file = NULL;
}

void
LnaReaderCircular::seek(int frame)
{
  if (m_file == NULL) {
    fprintf(stderr, "LnaReaderCircular::seek(): file not opened yet\n");
    exit(1);
  }

  if (fseek(m_file, m_header_size + m_frame_size * frame, SEEK_SET) < 0) {
    fprintf(stderr, "LnaReaderCircular::seek(): seek error %s\n", 
	    strerror(errno));
    exit(1);
  }

  m_frames_read = frame;
  m_first_index = 0;
}

bool
LnaReaderCircular::go_to(int frame)
{
  if (m_file == NULL) {
    fprintf(stderr, "LnaReaderCircular::go_to(): file not opened yet\n");
    exit(1);
  }

  if (m_eof_frame > 0 && frame >= m_eof_frame)
    return false;

  if (frame < m_frames_read - m_buffer_size)
    seek(frame);

  // FIXME: do we want to seek forward if skip is great?  Currently we
  // just read until the desired frame is reached.

  while (m_frames_read <= frame) {

    // Read a frame
    size_t ret = fread(&m_read_buffer[0], m_frame_size, 1, m_file);

    // Did we get the frame?
    if (ret != 1) {
      assert(ret == 0);

      // Check errors
      if (ferror(m_file)) {
	fprintf(stderr, "LnaReaderCircular::go_to(): read error on frame "
		"%d: %s\n", m_frames_read, strerror(errno));
	exit(1);
      }

      // Otherwise we have EOF
      m_eof_frame = m_frames_read;
      return false;
    }

    // Parse frame to the circular buffer
    if (m_lna_bytes == 4) {
      for (int i = 0; i < m_num_models; i++) {
        float tmp;
        unsigned char *p = (unsigned char*)&tmp;
        for (int j = 0; j < 4; j++)
          p[j] = m_read_buffer[i*4+j];
        m_log_prob_buffer[m_first_index] = tmp;
        m_first_index++;
        if (m_first_index >= m_log_prob_buffer.size())
          m_first_index = 0;
      }
    }
    else if (m_lna_bytes == 2) {
      for (int i = 0; i < m_num_models; i++) {
	float tmp = ((unsigned char)m_read_buffer[i*2] * 256 +
		     (unsigned char)m_read_buffer[i*2 + 1]) / -1820.0;
	m_log_prob_buffer[m_first_index] = tmp;
        m_first_index++;
        if (m_first_index >= m_log_prob_buffer.size())
          m_first_index = 0;
      }
    }
    else {
      for (int i = 0; i < m_num_models; i++) {
	float tmp = (unsigned char)m_read_buffer[i] / -24.0;
	m_log_prob_buffer[m_first_index] = tmp;
        m_first_index++;
        if (m_first_index >= m_log_prob_buffer.size())
          m_first_index = 0;
      }
    }
    m_frames_read++;
  }

  int index = m_first_index - (m_frames_read - frame) * m_num_models;
  if (index < 0)
    index += m_log_prob_buffer.size();
    
  m_log_prob = &m_log_prob_buffer[index];

  return true;
}
