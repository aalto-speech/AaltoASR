#ifndef ITERATORRANGE_HH
#define ITERATORRANGE_HH

template<class Iter>
class IteratorRange {
public:
  IteratorRange(Iter begin_arg, Iter end_arg)
  : begin_(begin_arg), end_(end_arg)
  {}

  Iter begin() const { return begin_; }
  Iter end() const { return end_; }

private:
  Iter begin_;
  Iter end_;
};

#endif
