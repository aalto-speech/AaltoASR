#ifndef BIT_ALGO_HH
#define BIT_ALGO_HH

#include "misc/types.hh"

namespace fsalm {

  /** Find the maximum element from an array. */
  template <class A, typename I>
  I max(const A &array)
  {
    I max = 0;
    for (typename A::index_type i = 0; i < array.num_elems(); i++) {
      I value = array.get(i);
      if (value > max)
        max = value;
    }
    return max;
  }

  /** Find the first index of a value from a sorted array.
   * 
   * \param array = the array to be examined
   * \param value = the value to be searched
   * \param first = the first index in the range considered
   * \param limit = the index limiting the range (not included in the range)
   * \return the index of the value or \c limit if not found
   * \throw Error if range is negative
   * \throw Error if range exceeds array size
   */
  template <class A, typename I>
  typename A::index_type 
  binary_search(const A &array, I value, typename A::index_type first, 
                typename A::index_type limit)
  {
     typename A::index_type bound = lower_bound(array, value, first, limit);
     if (bound < limit && array.get(bound) == value)
       return bound;
     return limit;
  }

  /** Find the smallest index at which \c value could be inserted
   * maintaining \c array in sorted order.
   * 
   * \param array = the array to be examined
   * \param value = the candidate value
   * \param first = the first index in the range considered
   * \param limit = the index limiting the range (not included in the range)
   * \return the index at which the value should be inserted
   * \throw Error if range is negative
   * \throw Error if range exceeds array size
   */
  template <class A, typename I>
  typename A::index_type 
  lower_bound(const A &array, I value, typename A::index_type first, 
              typename A::index_type limit)
  {
    if (first > limit)
      throw Error("bit::lower_bound(): negative range");
    if (limit > array.num_elems())
      throw Error("bit::lower_bound(): out of range");

    typename A::index_type middle;
    typename A::index_type half;
    typename A::index_type len = limit - first;

    while (len > 5) { // NOTE: magic threshold to do linear search
      half = len / 2;
      middle = first + half;

      // Equal
      I cur_value = array.get(middle);

      // First half
      if (cur_value >= value) {
        limit = middle;
        len = limit - first;
      }

      // Second half
      else {
        first = middle + 1;
        len = limit - first;
      }
    }

    while (first < limit) {
      I cur_value = array.get(first);
      if (cur_value >= value)
        break;
      first++;
    }

    return first;
  }


};

#endif /* BIT_ALGO_HH */
