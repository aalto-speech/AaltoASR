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
  void set_sr_parameters(float a0, float a1, float b0, float b1);
  float get_sr_comp_log_prob(int duration, float sr) const;
  bool is_valid_duration_model(void) { return (a>0); }

private:
  float a,b;
  float a0,a1,b0,b1;
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

  void set_pho_dur_stat(float m, float s) { pho_mean_dur = m; pho_dur_std = s;}
  float mean_dur(void) const { return pho_mean_dur; }
  float dur_std(void) const { return pho_dur_std; }

  struct InvalidInput : public std::exception {
    virtual const char *what() const throw()
      { return "Hmm: invalid input"; }
  };

  struct ReadError : public std::exception {
    virtual const char *what() const throw()
      { return "Hmm: read error"; }
  };

private:
  float pho_mean_dur, pho_dur_std;
};

#endif /* HMM_HH */
