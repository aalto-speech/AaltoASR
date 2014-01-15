#ifndef UTIL_HH
#define UTIL_HH

#include <cstdio>
#include <algorithm>
#include <vector>

// Visual studio math.h doesn't have log1p and log1pf functions varjokal 24.3.2010
// This needs to be defined for VS and M_LN10 constant varjokal 24.3.2010
#ifdef _MSC_VER
#include <boost/math/tr1.hpp>
using namespace boost::math::tr1;
#define _USE_MATH_DEFINES
#endif

#include <math.h>

/** Common utility functions. */
namespace util {

  /** Function evaluation object for search function(s) */
  class FuncEval {
  public:
    virtual double evaluate_function(double p) = 0;
    virtual ~FuncEval() {}
  };
  /** Binary search for finding the parameter value with which the
   * given function evaluates to max_value within the given accuracy. */
  double bin_search_param_max_value(double lower_bound, double low_value,
                                    double upper_bound, double up_value,
                                    double max_value, double value_acc,
                                    double param_acc, FuncEval &f);


  /** The square of the value. */
  template <typename T>
  T sqr(T a)
  {
    return a * a;
  }

  /** Median of the values in a vector. 
   * \note For odd number (2n + 1) of values, the n'th value is returned.
   */
  template <typename T>
  T
  median(std::vector<T> v)
  {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
  }

  /** Absolute value. */
  template <typename T>
  T
  abs(const T &value)
  {
    if (value < 0)
      return -value;
    return value;
  }

  /** Maximum of two values. */
  template <typename T>
  T
  max(const T &a, const T &b)
  {
    if (a < b)
      return b;
    return a;
  }

  inline float
  log10addf(float a, float b)
  {
    const float LOG10TOe = M_LN10;
    const float LOGeTO10 = 1.0 / M_LN10;

    a = a * LOG10TOe;
    b = b * LOG10TOe;

    float delta = a - b;
    if (delta > 0) {
      b = a;
      delta = -delta;
    }
    return (b + log1pf(exp(delta))) * LOGeTO10;
  }

  inline double
  log10add(double a, double b)
  {
    const double LOG10TOe = M_LN10;
    const double LOGeTO10 = 1.0 / M_LN10;

    a = a * LOG10TOe;
    b = b * LOG10TOe;

    double delta = a - b;
    if (delta > 0) {
      b = a;
      delta = -delta;
    }
    return (b + log1p(exp(delta))) * LOGeTO10;
  }


  inline float
  logaddf(float a, float b)
  {
    float delta = a - b;
    if (delta > 0) {
      b = a;
      delta = -delta;
    }
    return b + log1pf(expf(delta));
  }

  inline double
  logadd(double a, double b)
  {
    double delta = a - b;
    if (delta > 0) {
      b = a;
      delta = -delta;
    }
    return b + log1p(exp(delta));
  }

  // return log(exp(a)-exp(b)), a>b
  inline double
  logsub(double a, double b)
  {
    double delta = b - a;
    if (delta > 0) {
      fprintf(stderr, "invalid call to logsub, a should be bigger than b (a=%f,b=%f)\n",a,b);
      exit(1);
    }
    return a + log1p(-exp(delta));
  }

  static const double tiny_for_log = 1e-50;
  inline double safe_log(double x)
  {
    if (x < tiny_for_log)
      return log(tiny_for_log);
    else
      return log(x);
  }

  /** Compute modulo of two values so that negative arguments are
   * handled correctly. */
  inline int modulo(int a, int b) 
  {
    int result = a % b;
    if (result < 0)
      result += b;
    return result;
  }

  inline float sinc(float x)
  {
    if (fabs(x) < 1e-8)
      return 1;
    double y = M_PI*x;
    return sin(y)/y;
  }

};

#endif /* UTIL_HH */
