#ifndef UTIL_HH
#define UTIL_HH

#include <algorithm>
#include <vector>
#include <math.h>

/** Common utility functions. */
namespace util {

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
  log10add(float a, float b)
  {
    const float LOG10TOe = M_LN10;
    const float LOGeTO10 = 1.0 / M_LN10;

    a = a * LOG10TOe;
    b = b * LOG10TOe;

    float delta = a - b;
    if (delta > 64.0) {
      b += 64;
      delta = -delta;
    }
    return (b + log1pf(exp(delta))) * LOGeTO10;
  }

  inline float
  logadd(float a, float b)
  {
    float delta = a - b;
    if (delta > 64.0) {
      b += 64;
      delta = -delta;
    }
    return b + log1pf(exp(delta));
  }

  static const float tiny_for_log = (float)1e-16;
  inline double safe_log(double x)
  {
    if (x < tiny_for_log)
      return logf(tiny_for_log);
    else
      return logf(x);
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

};

#endif /* UTIL_HH */
