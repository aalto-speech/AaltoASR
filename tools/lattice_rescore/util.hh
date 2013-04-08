#ifndef UTIL_HH
#define UTIL_HH

#include <algorithm>
#include <vector>

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

};

#endif /* UTIL_HH */
