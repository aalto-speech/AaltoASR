#include <math.h>
#include "util.hh"

namespace util {

double bin_search_param_max_value(double lower_bound, double low_value,
                                  double upper_bound, double up_value,
                                  double max_value, double value_acc,
                                  double param_acc, FuncEval &f)
{
  double new_param = (lower_bound + upper_bound) / 2.0;
  double new_value = f.evaluate_function(new_param);
  if ((new_value <= max_value && max_value - new_value <= value_acc) ||
      new_param-lower_bound < param_acc)
  {
    if (low_value < up_value)
    {
      if (up_value <= max_value)
        return upper_bound;
      else if (new_value > max_value)
        return lower_bound;
    }
    else if (low_value > up_value)
    {
      if (low_value <= max_value)
        return lower_bound;
      else if (new_value > max_value)
        return upper_bound;
    }
    return new_param;
  }
  bool new_upper_bound = (new_value > max_value);
  if (low_value > up_value)
    new_upper_bound = !new_upper_bound;
  if (new_upper_bound)
    return bin_search_param_max_value(lower_bound, low_value, new_param,
                                      new_value, max_value, value_acc,
                                      param_acc, f);
  else
    return bin_search_param_max_value(new_param, new_value, upper_bound,
                                      up_value, max_value, value_acc,
                                      param_acc, f);
}

}
