#ifndef LNAREADERCIRCULAR_HH
#define LNAREADERCIRCULAR_HH

#include <iostream>
#include <fstream>
#include <stdio.h>

#include "Acoustics.hh"

class LnaReaderCircular : public Acoustics {
public:
  LnaReaderCircular();
  void open(const char *filename, int buf_size);
  void close();
  void seek(int frame);
  
  virtual bool go_to(int frame);

private:
  int read_int();

  FILE *m_file;

  // INVARIANTS
  //
  // - m_frames_read contains the number of read frames (from the
  // beginning of the file)
  //
  // - m_first_index'th position in m_buffer contains the first value
  // of the earliest frame (also the first value of the frame which is
  // read next)
  //
  // - m_first_index'th position in m_log_prob_buffer contains frame
  // (m_frames - m_log_prob_buffer.size())
  //
  // - (m_index-m_num_models)'th position in m_buffer contains frame 
  // (m_frames - 1)

  int m_header_size;  // The size of file header in bytes.
  int m_buffer_size;  // Size of buffer in frames 
  int m_first_index;  // Index of the last value in the last frame
  int m_frames_read;  // Frames read
  std::vector<float> m_log_prob_buffer;
  int m_eof_frame;
  int m_frame_size;

  // Temporary buffers
  std::vector<char> m_read_buffer;

  int m_lna_bytes;
};

#endif /* LNAREADERCIRCULAR_HH */
