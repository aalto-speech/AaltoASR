#ifndef NGRAM_HH
#define NGRAM_HH

#include <cstdio>
#include <deque>
#include <vector>
#include <assert.h>
#include "Vocabulary.hh"

template <typename KT> class ClusterMap;

class NGram : public Vocabulary {
public:
  typedef std::deque<int> Gram; //Compability with old treegram
  enum Type { BACKOFF=0, INTERPOLATED=1 };

  NGram(): m_last_order(0), m_order(0), m_type(BACKOFF) {}
  virtual ~NGram() {};
  int order() { return m_order; }
  int last_order() { return m_last_order; }
  void set_type(Type type) { m_type = type; }
  Type get_type() { return(m_type); }
  virtual void read(FILE *in, bool binary=false)=0;
  virtual void write(FILE *out, bool binary=false)=0;
  virtual bool using_clmap() = 0;

  inline float log_prob(std::vector<int> &gram) {
    assert(gram.size() > 0);
    switch (m_type) {
    case BACKOFF:
      if (!using_clmap()) return(log_prob_bo(gram));
      return(log_prob_bo_cl(gram));
    case INTERPOLATED:
      if (!using_clmap()) return(log_prob_i(gram));
      return(log_prob_i_cl(gram));
    default:
      assert(false);
    }
    return(0);
  }

  inline float log_prob(Gram &gram) {
    assert(gram.size() > 0);
    switch (m_type) {
    case BACKOFF:
      if (!using_clmap()) return(log_prob_bo(gram));
      return(log_prob_bo_cl(gram));
    case INTERPOLATED:
      if (!using_clmap()) return(log_prob_i(gram));
      return(log_prob_i_cl(gram));
    default:
      assert(false);
    }
    return(0);
  }

  inline float log_prob(std::vector<unsigned short> &gram) {
    assert(gram.size() > 0);
    switch (m_type) {
    case BACKOFF:
      if (!using_clmap()) return(log_prob_bo(gram));
      return(log_prob_bo_cl(gram));
    case INTERPOLATED:
      if (!using_clmap()) return(log_prob_i(gram));
      return(log_prob_i_cl(gram));
    default:
      assert(false);
    }
    return(0);
  }

  virtual float log_prob_bo(const std::vector<int> &gram)=0; // backoff, default
  virtual float log_prob_bo_cl(const std::vector<int> &gram)=0; // Clustered backoff
  virtual float log_prob_i(const std::vector<int> &gram)=0; // Interpolated
  virtual float log_prob_i_cl(const std::vector<int> &gram)=0; //Interpolated backoff

  virtual float log_prob_bo(const std::vector<unsigned short> &gram)=0; // backoff, default
  virtual float log_prob_bo_cl(const std::vector<unsigned short> &gram)=0; // Clustered backoff
  virtual float log_prob_i(const std::vector<unsigned short> &gram)=0; // Interpolated
  virtual float log_prob_i_cl(const std::vector<unsigned short> &gram)=0; //Interpolated backoff

  virtual float log_prob_bo(const Gram &gram)=0; // Keep this version lean and mean
  virtual float log_prob_bo_cl(const Gram &gram)=0; // Clustered backoff
  virtual float log_prob_i(const Gram &gram)=0; // Interpolated
  virtual float log_prob_i_cl(const Gram &gram)=0; //Interpolated backoff

protected:
  int m_last_order;
  std::vector<int> m_counts;
  int m_order;
  Type m_type;
};

#endif
