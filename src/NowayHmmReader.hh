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

#ifdef STATE_DURATION_PROBS
  void read_durations(std::istream &in);
#endif
  
  const std::map<std::string, int> &hmm_map() const 
    { return m_hmm_map; }
  const std::vector<Hmm> &hmms() const 
    { return m_hmms; }
  int num_models() const { return m_num_models; }

  struct InvalidFormat : public std::exception {
    virtual const char *what() const throw()
      { return "NowayHmmReader: invalid format"; }
  };

private:
  void read_hmm(std::istream &in, Hmm &hmm);

  std::vector<Hmm> m_hmms;
  std::map<std::string, int> m_hmm_map;
  int m_num_models;
};

#endif /* NOWAYHMMREADER_HH */
