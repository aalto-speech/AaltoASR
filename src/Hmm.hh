#ifndef HMM_HH
#define HMM_HH

#include <fstream>
#include <istream>
#include <vector>
#include <string>

struct HmmTransition {
  int target;
  double log_prob;
};

struct HmmState {
public:
  int model;
  std::vector<HmmTransition> transitions;
};


class Hmm {
public:
  std::vector<HmmState> states;
  std::string label;

  inline HmmState &state(int state) { return states[state]; }
  inline bool is_source(int state) const { return state == 0; }
  inline bool is_sink(int state) const { return state == 1; }

  struct InvalidInput : public std::exception {
    virtual const char *what() const throw()
      { return "Hmm: invalid input"; }
  };

  struct ReadError : public std::exception {
    virtual const char *what() const throw()
      { return "Hmm: read error"; }
  };
};

#endif /* HMM_HH */
