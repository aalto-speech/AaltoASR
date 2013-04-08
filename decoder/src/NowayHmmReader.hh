#ifndef NOWAYHMMREADER_HH
#define NOWAYHMMREADER_HH

#include <iostream>
#include <vector>
#include <map>

#include "Hmm.hh"

class NowayHmmReader {
public:
  NowayHmmReader();
  void read(std::istream &in);
  void read_durations(std::istream &in);
  
  std::map<std::string, int> &hmm_map()
    { return m_hmm_map; }
  std::vector<Hmm> &hmms()
    { return m_hmms; }
  int num_models() const { return m_num_models; }

  struct InvalidFormat : public std::exception {
    virtual const char *what() const throw()
      { return "NowayHmmReader: invalid format"; }
  };
  struct StateOutOfRange : public std::exception {
    virtual const char *what() const throw()
    { return "NowayHmmReader: state number of our range"; }
  };

private:
  void read_hmm(std::istream &in, Hmm &hmm);

  std::vector<Hmm> m_hmms;
  std::map<std::string, int> m_hmm_map;
  int m_num_models;
};

#endif /* NOWAYHMMREADER_HH */
