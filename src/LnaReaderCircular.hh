#ifndef LNAREADERCIRCULAR_HH
#define LNAREADERCIRCULAR_HH

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <fcntl.h>
#include <string.h>

// Use io.h in Visual Studio varjokal 17.3.2010
#ifdef _MSC_VER
#include <io.h>
#else
#include <unistd.h>
#endif

#include <errno.h>

#include "Acoustics.hh"

// O_BINARY is only defined in Windows
#ifndef O_BINARY
#define O_BINARY 0
#endif

class LnaReaderCircular : public Acoustics {
public:
  LnaReaderCircular();
  inline void open_file(const char *filename, int buf_size) {

#ifdef _MSC_VER
  int fd = _open(filename, _O_RDONLY|_O_BINARY);
#else
  int fd = open(filename, O_RDONLY|O_BINARY);
#endif

  if (fd < 0) {
      fprintf(stderr, "LnaReaderCircular::open(): could not open %s: %s\n",
	      filename, strerror(errno));
      exit(1);
    }
    open_fd(fd, buf_size);
  }
  void open_fd(const int fd, int buf_size);
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
