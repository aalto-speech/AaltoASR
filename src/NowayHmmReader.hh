#ifndef NOWAYHMMREADER_HH
#define NOWAYHMMREADER_HH

#include <iostream>
#include <vector>
#include <map>

#include "Hmm.hh"

class NowayHmmReader {
public:
  void read_hmm(std::istream &in, Hmm &hmm);
  void read(std::istream &in);
  const std::map<std::string, int> &hmm_map() const 
    { return m_hmm_map; }
  const std::vector<Hmm> &hmms() const 
    { return m_hmms; }

  struct InvalidFormat : public std::exception {
    virtual const char *what() const throw()
      { return "NowayHmmReader: invalid format"; }
  };

private:
  std::vector<Hmm> m_hmms;
  std::map<std::string, int> m_hmm_map;
};

#endif /* NOWAYHMMREADER_HH */
