#ifndef LNAREADERCIRCULAR_HH
#define LNAREADERCIRCULAR_HH

#include <iostream>
#include <fstream>

#include "Acoustics.hh"

class LnaReaderCircular : public Acoustics {
public:
  LnaReaderCircular();
  void open(const char *file, int num_models, int size);
  void init(std::istream &in, int num_models, int size);
  void close();
  void seek(int frame);
  
  virtual bool go_to(int frame);

  struct NotOpened : public std::exception {
    virtual const char *what() const throw()
      { return "LnaReaderCircular: not opened"; }
  };

  struct OpenError : public std::exception {
    virtual const char *what() const throw()
      { return "LnaReaderCircular: open error"; }
  };

  struct ShortFrame : public std::exception {
    virtual const char *what() const throw()
      { return "LnaReaderCircular: short frame"; }
  };

  struct FrameForgotten : public std::exception {
    virtual const char *what() const throw()
      { return "LnaReaderCircular: frame forgotten"; }
  };

  struct ReadError : public std::exception {
    virtual const char *what() const throw()
      { return "LnaReaderCircular: read error"; }
  };

  struct InvalidFrameId : public std::exception {
    virtual const char *what() const throw()
      { return "LnaReaderCircular: invalid frame id"; }
  };

  struct CannotSeek : public std::exception {
    virtual const char *what() const throw()
      { return "LnaReaderCircular: cannot seek"; }
  };

private:
  std::ifstream m_fin;
  std::istream *m_in;

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

  int m_buffer_size;  // Size of buffer in frames 
  int m_first_index;  // Index of the last value in the last frame
  int m_frames_read;  // Frames read
  std::vector<float> m_log_prob_buffer;
  int m_eof_frame;

  // Temporary buffers
  std::vector<char> m_read_buffer;
};

#endif /* LNAREADERCIRCULAR_HH */
