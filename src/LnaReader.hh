#ifndef LNAREADER_HH
#define LNAREADER_HH

#include <fstream>
#include <iostream>
#include <exception>

#include "Acoustics.hh"

// IMPLEMENTATION NOTES:
//
// - Currently LnaReader keeps read frames in a buffer and never
// discards past frames.


class LnaReader : public Acoustics {
public:
  LnaReader();
  LnaReader(std::istream &in, int num_models);
  LnaReader(const char *file, int num_models);
  
  void open(const char *file, int num_models);
  void close();
  bool sentence_end() const { return m_sentence_end; }

  virtual bool go_to(int frame);

  struct ReadError : public std::exception {
    virtual const char *what() const throw()
      { return "LnaReader: read error"; }
  };

  struct OpenError : public std::exception {
    virtual const char *what() const throw()
      { return "LnaReader: open error"; }
  };

  struct ShortFrame : public std::exception {
    virtual const char *what() const throw() 
      { return "LnaReader: short frame"; }
  };

  struct InvalidFrameId : public std::exception {
    virtual const char *what() const throw() 
      { return "LnaReader: invalid frame id"; }
  };

protected:
  void read_to(int frame);
  void parse_frame(int frame);

  bool m_two_byte;
  int m_num_models;
  int m_frame_size; // Frame size in the lna file (usually models+1)
  int m_frames;

  int m_block_size;

  bool m_eof;
  bool m_sentence_end;

  std::ifstream m_fin;
  std::istream *m_in;
  std::vector<char> m_read_buffer;
};

#endif /* LNAREADER_HH */
