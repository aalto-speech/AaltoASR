#ifndef HMM_HH
#define HMM_HH

#include <fstream>
#include <iostream>
#include <vector>
#include <string>

class StateDuration {
public:
  StateDuration();
  void set_parameters(float a, float b);
  float get_log_prob(int duration) const;
  int get_mode(void) const { return mode; }

private:
  float a,b;
  float const_term;
  int mode;
};

struct HmmTransition {
  int target;
  float log_prob;
};

struct HmmState {
public:
  int model;
  std::vector<HmmTransition> transitions;
  StateDuration duration;
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
