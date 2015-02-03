#include <fstream>
#include <string>
#include <iostream>
#include <stdlib.h>
#include <algorithm>

#include "io.hh"
#include "str.hh"
#include "conf.hh"
#include "HmmSet.hh"

using namespace aku;
using namespace std;

string statistics_file;
string out_model_name;
string state_file;

int info;

conf::Config config;
HmmSet model;

double min_var;
double weight_kld_limit;
double mean_kld_limit;
double cov_kld_limit;

double mixture_criterion_kld_ratio = 0;
double mean_criterion_kld_ratio = 0;
double cov_criterion_kld_ratio = 0;

bool criterion_relative_ratio = false;
double criterion_value = 0;

double mixture_max_objective_function = 0;

int global_num_below_kld = 0;
double global_sum_objective = 0;
int global_num_negative_objective = 0;


double mpe_smooth = 800;


vector<double> gaussian_weights; // Mixture weights for each Gaussian
bool weighted_gaussian_kld_ratios = false;


bool global_debug_flag = false;
bool global_debug_flag2 = false;


typedef enum {MODE_MMI, MODE_MPE} OPTIMIZATION_MODE;
OPTIMIZATION_MODE opt_mode = MODE_MMI;


class FuncEval {
public:
  virtual double evaluate_function(double p) const = 0;
  virtual ~FuncEval() {}
};


double maximize_function(double lower_bound, double upper_bound,
                         double accuracy, const FuncEval &f)
{
  const double r = (sqrt(5)-1)/2;
  double x0 = lower_bound;
  double x1 = lower_bound + (1-r)*(upper_bound - lower_bound);
  double x2 = lower_bound + r*(upper_bound - lower_bound);
  double x3 = upper_bound;
  double f0 = f.evaluate_function(x0);
  double f1 = f.evaluate_function(x1);
  double f2 = f.evaluate_function(x2);
  double f3 = f.evaluate_function(x3);

  for (;;)
  {
    //fprintf(stderr, "    %-12g %-12g %-12g %-12g\n", x0, x1, x2, x3);
    bool finish = false;
    if (x2-x0 < accuracy)
      finish = true;
    if (f1 >= f2)
    {
      //fprintf(stderr, "    %-12g %-9g(*) %-12g %-12g\n", f0, f1, f2, f3);
      if (finish)
      {
        if (f1 > f0)
          return x1;
        return x0;
      }
      x3 = x2;
      f3 = f2;
      x2 = x1;
      f2 = f1;
      x1 = x0 + (1-r)*(x3-x0);
      f1 = f.evaluate_function(x1);
    }
    else
    {
      //fprintf(stderr, "    %-12g %-12g %-9g(*) %-12g\n", f0, f1, f2, f3);
      if (finish)
      {
        if (f3 > f2)
          return x3;
        return x2;
      }
      x0 = x1;
      f0 = f1;
      x1 = x2;
      f1 = f2;
      x2 = x0 + r*(x3-x0);
      f2 = f.evaluate_function(x2);
    }
  }
}


//bool bin_search_debug = false;


// For monotonous functions! Assumes lower_bound < upper_bound.
double bin_search_max_param(double lower_bound, double low_value,
                            double upper_bound, double up_value,
                            double max_value, double accuracy,
                            const FuncEval &f)
{
  // if (bin_search_debug)
  // {
  //   fprintf(stderr, "[%g, %g] -> [%g, %g]\n", lower_bound, upper_bound,
  //           low_value, up_value);
  // }
  double new_param = (lower_bound + upper_bound) / 2.0;
  if (new_param-lower_bound <= accuracy)
    return new_param;
  double new_value = f.evaluate_function(new_param);
  bool new_upper_bound = (new_value > max_value);
  if (low_value > up_value)
    new_upper_bound = !new_upper_bound;
  if (new_upper_bound)
    return bin_search_max_param(lower_bound, low_value, new_param, new_value,
                                max_value, accuracy, f);
  else
    return bin_search_max_param(new_param, new_value, upper_bound, up_value,
                                max_value, accuracy, f);
}


// For monotonous functions! Assumes lower_bound < upper_bound.
double bin_search_param_value_acc(double lower_bound, double low_value,
                                  double upper_bound, double up_value,
                                  double target_value, double value_acc,
                                  double param_acc, const FuncEval &f)
{
  // fprintf(stderr, "[%g, %g] -> [%g, %g]\n", lower_bound, upper_bound,
  //         low_value, up_value);
  double new_param = (lower_bound + upper_bound) / 2.0;
  double new_value = f.evaluate_function(new_param);
  if (fabs(new_value-target_value) <= value_acc ||
      new_param-lower_bound < param_acc)
  {
    if (global_debug_flag2)
    {
      fprintf(stderr, "SUM: [%g, %g, %g] -> [%g, %g, %g]\n", lower_bound, new_param,
              upper_bound, low_value, new_value, up_value);

      // global_debug_flag = true;
      // fprintf(stderr, "--- SUM Lower bound ---\n");
      // f.evaluate_function(lower_bound);
      // fprintf(stderr, "--- SUM Midway ---\n");
      // f.evaluate_function(new_param);
      // fprintf(stderr, "--- SUM Upper bound ---\n");
      // f.evaluate_function(upper_bound);
      // fprintf(stderr, "--- SUM finished ---\n");
      // global_debug_flag = false;
    }

    double la = fabs(low_value - target_value);
    double na = fabs(new_value - target_value);
    double ua = fabs(up_value - target_value);
    if (la < na && la < ua)
      return lower_bound;
    if (ua < na)
      return upper_bound;
    return new_param;
  }
  bool new_upper_bound = (new_value > target_value);
  if (low_value > up_value)
    new_upper_bound = !new_upper_bound;
  if (new_upper_bound)
    return bin_search_param_value_acc(lower_bound, low_value, new_param,
                                      new_value, target_value, value_acc,
                                      param_acc, f);
  else
    return bin_search_param_value_acc(new_param, new_value, upper_bound,
                                      up_value, target_value, value_acc,
                                      param_acc, f);
}


// For monotonous functions! Assumes lower_bound < upper_bound.
double bin_search_max_param_value_acc(double lower_bound, double low_value,
                                      double upper_bound, double up_value,
                                      double max_value, double value_acc,
                                      double param_acc, const FuncEval &f)
{
  // fprintf(stderr, "[%g, %g] -> [%g, %g]\n", lower_bound, upper_bound,
  //         low_value, up_value);
  double new_param = (lower_bound + upper_bound) / 2.0;
  double new_value = f.evaluate_function(new_param);
  if ((new_value <= max_value && max_value - new_value <= value_acc) ||
      new_param-lower_bound < param_acc)
  {
    // fprintf(stderr, "[%g, %g, %g] -> [%g, %g, %g]\n", lower_bound, new_param,
    //         upper_bound, low_value, new_value, up_value);
    // global_debug_flag = true;
    // fprintf(stderr, "--- Lower bound ---\n");
    // f.evaluate_function(lower_bound);
    // fprintf(stderr, "--- Upper bound ---\n");
    // f.evaluate_function(upper_bound);
    // global_debug_flag = false;
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
    return bin_search_max_param_value_acc(lower_bound, low_value, new_param,
                                          new_value, max_value, value_acc,
                                          param_acc, f);
  else
    return bin_search_max_param_value_acc(new_param, new_value, upper_bound,
                                          up_value, max_value, value_acc,
                                          param_acc, f);
}




double search_lambda(double initial_value, double limit,
                     const FuncEval &f)
{
  double low_value, up_value;
  double low_bound, up_bound;
  int safeguard_counter = 0;

  double constraint = f.evaluate_function(initial_value);
  if (fabs(constraint-limit) < 1e-6) // Accuracy for KLD limit
    return initial_value;
  if (constraint < limit)
  {
    double cur_value = initial_value;
    while (constraint < limit && cur_value > 0)
    {
      if (global_debug_flag)
        fprintf(stderr, "  lambda = %g, C = %g\n", cur_value, constraint);
      up_value = constraint;
      up_bound = cur_value;
      cur_value /= 2.0;
      if (cur_value < 1e-20)
        cur_value = 0;
      constraint = f.evaluate_function(cur_value);
      if (++safeguard_counter > 100)
        return cur_value;
    }
    if (constraint < limit)
      return cur_value;
    low_value = constraint;
    low_bound = cur_value;
  }
  else // constraint > limit
  {
    double cur_value = initial_value;
    while (constraint > limit)
    {
      if (global_debug_flag)
        fprintf(stderr, "  lambda = %g, C = %g\n", cur_value, constraint);
      low_value = constraint;
      low_bound = cur_value;
      if (cur_value > 0)
        cur_value *= 2.0;
      else
        cur_value = 1;
      constraint = f.evaluate_function(cur_value);
      if (++safeguard_counter > 100)
        return cur_value;
    }
    up_value = constraint;
    up_bound = cur_value;
  }
  if (global_debug_flag)
    fprintf(stderr, "  binary search [%g, %g], values [%g, %g]\n",
            low_bound, up_bound, low_value, up_value);
  return bin_search_max_param_value_acc(low_bound, low_value,
                                        up_bound, up_value,
                                        limit, 1e-6,
                                        1e-12*(up_bound-low_bound), f);
}


// Solver for a mixture weight, using critical point method
class CriticalMixtureWeightSolver : public FuncEval {
public:
  CriticalMixtureWeightSolver(double orig_weight, double weight_gamma,
                              double w_abs_gamma,
                              double lambda, double constraint) :
    weight0(orig_weight), cur_gamma(weight_gamma), abs_gamma(w_abs_gamma),
    lambda(lambda), c(constraint) { }
  // Evaluate Lagrangian which should equate 0. Given new mixture weight.
  virtual double evaluate_function(double p) const;
  double get_derivative(double p) const; // Derivative of previous
  bool solve_weight(double &weight) const;
private:
  double weight0;
  double cur_gamma;
  double abs_gamma; // For EBW style estimation
  double lambda;
  double c;
};

double CriticalMixtureWeightSolver::evaluate_function(double p) const
{
  //return -p*sum_gamma - lambda*(p*log(p/weight0)-p*c) + cur_gamma;

  // Normal CLS equation:
  // return cur_gamma/p - lambda*(log(p/weight0) + 1) - c;

  // Alternative formulation (EBW style estimation):
  // This is a derivative of:
  //  ((cur_gamma+abs_gamma)*log(p) - (abs_gamma-cur_gamma)*p/weight0)/2.0
  // plus derivatives of constraints.
  return ((abs_gamma+cur_gamma)/p - (abs_gamma-cur_gamma)/weight0)/2.0
    -lambda*(log(p/weight0) + 1) - c;
}

double CriticalMixtureWeightSolver::get_derivative(double p) const
{
  //return -sum_gamma + lambda*(c-1) - lambda*log(p/weight0);

  // Normal CLS equation: 
  // return -cur_gamma/(p*p) - lambda/p;

  // Alternative formulation (EBW style estimation):
  return -(cur_gamma+abs_gamma)/(2*p*p) - lambda/p;
}

bool CriticalMixtureWeightSolver::solve_weight(double &weight) const
{
  double search_acc = 1e-8;
  double min_weight = 1e-4;
  if (lambda == 0)
  {
    if (global_debug_flag)
      fprintf(stderr, "CriticalMixtureWeightSolver::solve_weight: lambda == 0, c = %g\n", c);
    // Normal CLS equation:
    // weight = min(max(cur_gamma/c, min_weight), 1.0);

    // Alternative formulation (EBW style estimation):
    if (abs_gamma - cur_gamma + 2*c*weight0 <= 0)
      weight = 1.0;
    else
      weight = min(max(weight0*(abs_gamma+cur_gamma)/
                                 (abs_gamma-cur_gamma+2*c*weight0), min_weight),
                        1.0);
    return true;
  }
  // else // Normal CLS branch
  // {
  //   double extreme_point = -cur_gamma/lambda;

  //   // Extreme point (f' = 0) is a maximum, (f'' < 0 in that point and
  //   // f' has only one zero).  We are only interested in maximum
  //   // points of the Lagrangian, i.e. those zeros of f where f' < 0.
  //   // So look to the right side from the extreme point.

  //   if (extreme_point < min_weight)
  //   {
  //     // if (global_debug_flag)
  //     //   fprintf(stderr, "CriticalMixtureWeightSolver::solve_weight: extreme_point < min_weight (%g < %g)\n", extreme_point, min_weight);

  //     double lower_f = evaluate_function(min_weight);
  //     // if (global_debug_flag)
  //     //   fprintf(stderr, "CriticalMixtureWeightSolver::solve_weight: lower_f = %g\n", lower_f);
      
  //     if (lower_f < 0)
  //       weight = min_weight;
  //     else
  //     {
  //       double upper_f = evaluate_function(1.0);
  //       // if (global_debug_flag)
  //       //   fprintf(stderr, "CriticalMixtureWeightSolver::solve_weight: upper_f = %g\n", upper_f);
  //       weight = bin_search_max_param(min_weight, lower_f,
  //                                     1.0, upper_f, 0, search_acc, *this);
  //     }
  //     // if (global_debug_flag)
  //     //   fprintf(stderr, "CriticalMixtureWeightSolver::solve_weight: weight = %g\n", weight);

  //   }
  //   else if (extreme_point < 1)
  //   {
  //     // if (global_debug_flag)
  //     //   fprintf(stderr, "CriticalMixtureWeightSolver::solve_weight: extreme_point < 1 (%g)\n", extreme_point);

  //     double lower_f = evaluate_function(extreme_point);
  //     //double min_weight_f = evaluate_function(min_weight);
  //     // if (global_debug_flag)
  //     //   fprintf(stderr, "CriticalMixtureWeightSolver::solve_weight: min_weight_f = %g, lower_f = %g\n", min_weight_f, lower_f);

  //     if (lower_f < 0)
  //     {
  //       // No maximum, with this sum constraint
  //       weight = 0;
  //       return true;
  //     }
  //     else
  //     {
  //       double upper_f = evaluate_function(1.0);
  //       // if (global_debug_flag)
  //       //   fprintf(stderr, "CriticalMixtureWeightSolver::solve_weight: upper_f = %g\n", upper_f);

  //       weight = bin_search_max_param(extreme_point, lower_f,
  //                                     1.0, upper_f, 0, search_acc, *this);
  //     }
  //     // if (global_debug_flag)
  //     //   fprintf(stderr, "CriticalMixtureWeightSolver::solve_weight: weight = %g\n", weight);
  //   }
  //   else
  //   {
  //     // if (global_debug_flag)
  //     //   fprintf(stderr, "CriticalMixtureWeightSolver::solve_weight: extreme_point over 1 (%g)\n", extreme_point);

  //     // If there is a zero in ]0,1], it refers to a minimum
  //     //weight = min_weight; // ???
  //     return false; // Failed because lambda is too small
  //   }

  //   // Ensure weight limits
  //   weight = min(max(weight, min_weight), 1.0);
  //   // if (global_debug_flag)
  //   //   fprintf(stderr, "CriticalMixtureWeightSolver::solve_weight: Final weight = %g\n", weight);
    
  //   return true;
  // }
  else // EBW branch
  {
    double lower_f = evaluate_function(min_weight);
    double upper_f = evaluate_function(1.0);
    if (lower_f < upper_f)
    {
      fprintf(stderr, "  Warning: lower_f = %g, upper_f = %g, weight0 = %g, gamma = %g, abs_gamma = %g, lambda = %g, c = %g\n", lower_f, upper_f, weight0, cur_gamma, abs_gamma, lambda, c);
      abort();
    }
    if (lower_f < 0)
    {
      // fprintf(stderr, "  Warning: lower_f = %g, upper_f = %g, weight0 = %g, gamma = %g, abs_gamma = %g, lambda = %g, c = %g\n", lower_f, upper_f, weight0, cur_gamma, abs_gamma, lambda, c);
      weight = min_weight;
    }
    else if (upper_f > 0)
    {
      // fprintf(stderr, "  Warning: lower_f = %g, upper_f = %g, weight0 = %g, gamma = %g, abs_gamma = %g, lambda = %g, c = %g\n", lower_f, upper_f, weight0, cur_gamma, abs_gamma, lambda, c);
      weight = 1.0;
    }
    else
    {
      if (global_debug_flag)
        fprintf(stderr, "CriticalMixtureWeightSolver: bin search [%g, %g] -> [%g, %g]\n",
                min_weight, 1.0, lower_f, upper_f);
      weight = bin_search_max_param(min_weight, lower_f,
                                    1.0, upper_f, 0, search_acc, *this);
      if (global_debug_flag)
        fprintf(stderr, "CriticalMixtureWeightSolver::solve_weight: weight = %g\n", weight);
    }
    weight = min(max(weight, min_weight), 1.0);
    if (global_debug_flag)
      fprintf(stderr, "CriticalMixtureWeightSolver::solve_weight: Final weight = %g\n", weight);
    return true;
  }
}


// KLD constraint for mixture weights
class MixtureKLDConstraint : public FuncEval {
public:
  MixtureKLDConstraint(const Vector &orig_weights, double kldr) :
    weights0(orig_weights), k_ratio(kldr), eval_kld(true) { }
  virtual double evaluate_function(double p) const; // Constraint, given lambda
  virtual bool solve_weights(double lambda, Vector &new_weights) const = 0;
  virtual double evaluate_objective_function(const Vector &weights) const = 0;
  void set_kld_evaluation(bool e) { eval_kld = e; }
protected:
  const Vector &weights0;
  double k_ratio;
  bool eval_kld;
};

double MixtureKLDConstraint::evaluate_function(double p) const
{
  Vector new_weights;
  double kld = 0;
  if (global_debug_flag)
    fprintf(stderr, "MixtureKLDConstraint::evaluate_function(%g)\n", p);
  if (!solve_weights(p, new_weights)) // Failed?
  {
    if (!eval_kld)
    {
      if (!global_debug_flag)
      {
        fprintf(stderr, "Warning: Weight solving failed when optimizing criterion/KLD ratio!\n");
        fprintf(stderr, "Function: MixtureKLDConstraint::evaluate_function(%g)\n", p);
        fprintf(stderr, "******** This is potentially dangerous, enabling debug mode *******\n");
        global_debug_flag = true;
      }
    }
    //assert( eval_kld ); // Shouldn't fail when optimizing criterion/KLD ratio
    kld = weights0.size(); // Overestimate KLD
  }
  else
  {
    // Compute KLD
    for (int i = 0; i < weights0.size(); i++)
      kld += new_weights(i)*log(new_weights(i)/weights0(i));
    if (!eval_kld)
    {
      // Compute criterion function change
      double f_change = evaluate_objective_function(new_weights) -
        evaluate_objective_function(weights0);
      // Combine with KLD
      kld = k_ratio*kld - f_change;
    }
  }
  
  return kld;
}


// Solver for critical point mixture weights
class CriticalMixtureSolver : public MixtureKLDConstraint {
public:
  CriticalMixtureSolver(const Vector &orig_weights, const Vector &weight_gammas,
                        const Vector &weight_abs_gammas,
                        double target_constraint, double kldr);
  virtual bool solve_weights(double lambda, Vector &new_weights) const;
  virtual double evaluate_objective_function(const Vector &weights) const;

protected:
  bool solve_new_weights(double lambda, double sum_constraint,
                         Vector &new_weights, double &norm) const;

  class SumEval : public FuncEval {
  private:
    double lambda;
    const CriticalMixtureSolver *parent;
    double invalid_value;
  public:
    SumEval(double l, const CriticalMixtureSolver *p) :  lambda(l), parent(p) { invalid_value = 0; }
    void set_invalid_value(double iv) { invalid_value = iv; }
    
    virtual double evaluate_function(double p) const {
      Vector weights;
      double norm = 0;
      weights.resize(parent->weights0.size());
      bool temp = parent->solve_new_weights(lambda, p, weights, norm);
      assert( temp ); // Dangerous if this fails during iterative solving
      if (norm == 0)
        norm = invalid_value;
      return norm;
    }
  };

  friend class SumEval;

private:
  const Vector &gammas;
  const Vector &abs_gammas;
  double c; // Target constraint
  double sum_gamma;
};

CriticalMixtureSolver::CriticalMixtureSolver(const Vector &orig_weights,
                                             const Vector &weight_gammas,
                                             const Vector &weight_abs_gammas,
                                             double target_constraint,
                                             double kldr)
  : MixtureKLDConstraint(orig_weights, kldr), gammas(weight_gammas),
    abs_gammas(weight_abs_gammas), c(target_constraint)
{
  // Compute sum of the gammas
  sum_gamma = 0;
  for (int i = 0; i < weight_gammas.size(); i++)
    sum_gamma += weight_gammas(i);
}


bool
CriticalMixtureSolver::solve_new_weights(double lambda, double sum_constraint,
                                         Vector &new_weights,
                                         double &norm) const
{
  norm = 0;
  for (int i = 0; i < weights0.size(); i++)
  {
    CriticalMixtureWeightSolver w(weights0(i), gammas(i), abs_gammas(i),
                                  lambda, sum_constraint);
    if (!w.solve_weight(new_weights(i)))
    {
      if (global_debug_flag)
        fprintf(stderr, "CriticalMixtureSolver::solve_new_weights: Estimating weight %i failed\n", i);
      return false; // Failed because of too small a lambda
    }
    if (new_weights(i) == 0) // Invalid weight, invalid sum_constraint
    {
      if (global_debug_flag)
        fprintf(stderr, "CriticalMixtureSolver::solve_new_weights: Weight %i is zero, failed\n", i);
      norm = 0;
      return true;
    }
    norm += new_weights(i);
  }
  return true;
}



double
CriticalMixtureSolver::evaluate_objective_function(const Vector &weights) const
{
  double f = 0;
  for (int i = 0; i < weights0.size(); i++)
  {
    // Normal CLS equation:
    // f += gammas(i) * log(weights(i));

    // Alternative formulation (EBW style estimation):
    f += ((gammas(i)+abs_gammas(i))*log(weights(i)) -
          (abs_gammas(i)-gammas(i))*weights(i)/weights0(i))/2.0;
  }
  return f;
}


bool CriticalMixtureSolver::solve_weights(double lambda,
                                          Vector &new_weights) const
{
  bool local_debug_flag = global_debug_flag;
  global_debug_flag = false;
  double cur_sum_constraint = 0;
  double lower_value, upper_value;
  double lower_bound = 0, upper_bound = 0;
  double sum_value_inf = 1e10;

  // FIXME: lambda == 0 case is dependent on the weight solver!
  // CLS case:
  // if (lambda == 0)
  //   lower_bound = upper_bound = sum_gamma;
  // EBW case: No special treatment
  
  new_weights.resize(weights0.size());
  double norm0 = 0;
  if (!solve_new_weights(lambda, lower_bound, new_weights, norm0))
  {
    if (local_debug_flag)
    {
      fprintf(stderr, "CriticalMixtureSolver::solve_weights: Initial estimation failed\n");
      global_debug_flag = local_debug_flag;
    }

    return false;
  }
  lower_value = upper_value = norm0;
  SumEval f(lambda, this);

  if (local_debug_flag && lambda == 0)
    fprintf(stderr, "  init = %g, norm = %g\n", lower_bound, norm0);

  if (norm0 != 1)
  {
    double norm = 0;
    bool positive = false;
    bool negative = false;
    double init;
    if (norm0 == 0)
      negative = true;
    for (init = 1; init < 1e20; init *= 2.0)
    {
      if (local_debug_flag)
        fprintf(stderr, "  SUM iteration, init = %g (pos = %d, neg = %d)\n",
                init, (positive?1:0), (negative?1:0));
      double cur_c;
      norm = 0;
      if (!positive)
      {
        // CLS case:
        // cur_c = (lambda == 0 ? sum_gamma/(2.0*init) : -init);
        // EBW case:
        cur_c = -init;
        if (!solve_new_weights(lambda, cur_c, new_weights, norm))
        {
          global_debug_flag = local_debug_flag;
          return false;
        }
        if (local_debug_flag && lambda == 0)
          fprintf(stderr, "    neg: norm = %g\n", norm);
        if (norm0 == 0)
        {
          if (norm > 0)
          {
            if (upper_value == 0)
            {
              // First non-zero norm, values should be feasible further on
              upper_bound = cur_c;
              upper_value = norm;
              continue;
            }
            if ((upper_value < norm && upper_value > 1) ||
                (upper_value > norm && upper_value < 1))
            {
              // May occur only immediately after the previous statement!
              // Either the solution is between the last two values,
              // or it doesn't exist at all.
              lower_value = upper_value;
              lower_bound = upper_bound;
              upper_bound /= 2.0;
              if (lower_value < 1)
              {
                upper_value = sum_value_inf;
                f.set_invalid_value(sum_value_inf);
              }
              else
              {
                upper_value = 0;
                f.set_invalid_value(0);
              }
              break;
            }
            norm0 = upper_value; // Reinitialize
          }
          else
          {
            assert( upper_value == 0 ); // Zero should not reappear
            continue;
          }
        }
        
        assert( norm > 0 ); // Zero should not reappear
        if ((norm0 < 1 && norm > norm0) || (norm0 > 1 && norm < norm0))
          negative = true;
        if ((norm0-1)*(norm-1) < 0) // Bracketing has crossed 1
        {
          lower_bound = cur_c;
          lower_value = norm;
          break;
        }
        if (negative)
        {
          upper_bound = cur_c;
          upper_value = norm;
        }
      }
      if (!negative)
      {
        // CLS case:
        // cur_c = (lambda == 0 ? sum_gamma*2.0*init : init);
        // EBW case:
        cur_c = init;
        if (!solve_new_weights(lambda, cur_c, new_weights, norm))
        {
          global_debug_flag = local_debug_flag;
          return false;
        }
        if (local_debug_flag && lambda == 0)
          fprintf(stderr, "    pos: norm = %g\n", norm);
        if (norm == 0)
        {
          // We may find an upper limit of sum constraint
          upper_bound = cur_c;
          if (norm0 < 1)
          {
            upper_value = sum_value_inf;
            f.set_invalid_value(sum_value_inf);
          }
          else
          {
            upper_value = 0;
            f.set_invalid_value(0);
          }
          break;
        }
        if ((norm0 < 1 && norm > norm0) || (norm0 > 1 && norm < norm0))
          positive = true;
        if ((norm0-1)*(norm-1) < 0) // Bracketing has crossed 1
        {
          upper_bound = cur_c;
          upper_value = norm;
          break;
        }
        if (positive)
        {
          lower_bound = cur_c;
          lower_value = norm;
        }
      }
    }
    if (init >= 1e20)
    {
      global_debug_flag = local_debug_flag;
      return false; // Failed
    }
  }


  if (upper_bound < lower_bound)
  {
    // Swap (can happen with lambda == 0)
    double temp = lower_bound;
    lower_bound = upper_bound;
    upper_bound = temp;
    temp = lower_value;
    lower_value = upper_value;
    upper_value = temp;
  }

  if (local_debug_flag)
    fprintf(stderr, "  Sum constraint search [%g, %g], values [%g, %g]\n",
            lower_bound, upper_bound, lower_value, upper_value);
  global_debug_flag2 = local_debug_flag;
  cur_sum_constraint = bin_search_param_value_acc(
    lower_bound, lower_value, upper_bound, upper_value, 1, 1e-3,
    1e-12*(upper_bound-lower_bound), f);
  if (local_debug_flag)
    fprintf(stderr, "  Optimum: %g\n", cur_sum_constraint);

  global_debug_flag2 = false;

  // Get final weights
  global_debug_flag = local_debug_flag;
  double norm = 0;
  if (!solve_new_weights(lambda, cur_sum_constraint, new_weights, norm))
  {
    if (global_debug_flag)
      fprintf(stderr, "CriticalMixtureSolver::solve_weights: Final estimation failed\n");
    return false;
  }

  // Normalize weights (if reasonably close, otherwise the solution
  // will be rejected)
  // if (fabs(1-norm) <= 0.01)
  // {
  //   for (int i = 0; i < new_weights.size(); i++)
  //     new_weights(i) = new_weights(i) / norm;
  // }
  // else
  // {
  //   fprintf(stderr, "Weight normalization failed, norm %g\n", norm);
  //   return false;
  // }

  // Always normalize
  for (int i = 0; i < new_weights.size(); i++)
    new_weights(i) = new_weights(i) / norm;
  if (fabs(1-norm) > 0.01)
  {
    if (global_debug_flag)
      fprintf(stderr, "  Bad weight normalization, norm %g\n", norm);
    return false; // FIXME: Shouldn't fail on final weight estimation call!
  }
  
  return true;
}


// Solver for linearly modeled mixture weights
class LinearMixtureSolver : public MixtureKLDConstraint {
public:
  LinearMixtureSolver(const Vector &orig_weights, const Vector &gradient,
                      double kldr) :
    MixtureKLDConstraint(orig_weights, kldr), grad(gradient) { }
  virtual bool solve_weights(double lambda, Vector &new_weights) const;
  virtual double evaluate_objective_function(const Vector &weights) const;

protected:

  double solve_new_weights(double lambda, double sum_constraint,
                           Vector &new_weights) const;

  class SumEval : public FuncEval {
  private:
    double lambda;
    const LinearMixtureSolver *parent;
  public:
    SumEval(double l, const LinearMixtureSolver *p) :  lambda(l), parent(p) { }
    virtual double evaluate_function(double p) const {
      Vector weights;
      weights.resize(parent->weights0.size());
      return parent->solve_new_weights(lambda, p, weights);
    }
  };

  friend class SumEval;
  
private:
  const Vector &grad;
};

double
LinearMixtureSolver::evaluate_objective_function(const Vector &weights) const
{
  double f = 0;
  for (int i = 0; i < weights0.size(); i++)
    f += weights(i)*grad(i);
  return f;
}

double
LinearMixtureSolver::solve_new_weights(double lambda, double sum_constraint,
                                          Vector &new_weights) const
{
  double norm = 0;
  for (int i = 0; i < weights0.size(); i++)
  {
    new_weights(i) = max(min(weights0(i)*exp((grad(i)-sum_constraint)/lambda-1), 1.0), 1e-8);
    norm += new_weights(i);
  }
  return norm;
}

bool LinearMixtureSolver::solve_weights(double lambda,
                                        Vector &new_weights) const
{ 
  double sum_search_acc = 1e-4;
  double norm = 0;
  double cur_sum_constraint = 0;
  double low_value, up_value;
  double low_bound = 0, up_bound = 0;
  int safeguard_counter = 0;

  new_weights.resize(weights0.size());
  norm = solve_new_weights(lambda, 0, new_weights);
  low_value = up_value = norm;
  if (norm < 1)
  {
    cur_sum_constraint = -1;
    norm = solve_new_weights(lambda, cur_sum_constraint, new_weights);
    while (norm - 1 < -sum_search_acc)
    {
      up_value = norm;
      up_bound = cur_sum_constraint;
      cur_sum_constraint *= 2.0;
      norm = solve_new_weights(lambda, cur_sum_constraint, new_weights);
      if (++safeguard_counter > 100)
        abort();
    }
    low_value = norm;
    low_bound = cur_sum_constraint;
  }
  else if (norm > 1)
  {
    cur_sum_constraint = 1;
    norm = solve_new_weights(lambda, cur_sum_constraint, new_weights);
    while (norm - 1 > sum_search_acc)
    {
      low_value = norm;
      low_bound = cur_sum_constraint;
      cur_sum_constraint *= 2.0;
      norm = solve_new_weights(lambda, cur_sum_constraint, new_weights);
      if (++safeguard_counter > 100)
        abort();
    }
    up_value = norm;
    up_bound = cur_sum_constraint;
  }

  // FIXME: Accuracy
  SumEval f(lambda, this);
  cur_sum_constraint = bin_search_max_param(low_bound, low_value,
                                            up_bound, up_value,
                                            1, 1e-8*(up_bound-low_bound), f);
  // Get final weights
  norm = solve_new_weights(lambda, cur_sum_constraint, new_weights);
  
  // Normalize weights
  if (fabs(1-norm) > 0.01)
    fprintf(stderr, "Warning: Normalization deviates from 1: %g\n", norm);
  for (int i = 0; i < new_weights.size(); i++)
    new_weights(i) = new_weights(i) / norm;

  return true;
}



// KLD constraint for mean
class GaussianMeanKLDConstraint : public FuncEval {
public:
  GaussianMeanKLDConstraint(const Vector &orig_mean, const Vector &orig_cov) :
    mean0(orig_mean), cov0(orig_cov) { }
  virtual double evaluate_function(double p) const; // Constraint, given lambda
  virtual void solve_mean(double lambda, Vector &new_mean) const = 0;
protected:
  const Vector &mean0;
  const Vector &cov0;
};

double GaussianMeanKLDConstraint::evaluate_function(double p) const
{
  Vector mean;
  solve_mean(p, mean);

  // Evaluate the KLD constraint
  int dim = mean0.size();
  double kld = 0;
  for (int i = 0; i < dim; i++)
  {
    double d = mean(i)-mean0(i);
    kld += d*d/cov0(i);
  }
  return kld/2.0;
}


// Solver for critical mean
class CriticalMeanSolver : public GaussianMeanKLDConstraint {
public:
  CriticalMeanSolver(const Vector &orig_mean, const Vector &orig_cov,
                     double m0_statistics,
                     const Vector &m1_statistics) :
    GaussianMeanKLDConstraint(orig_mean, orig_cov),
    m0_stats(m0_statistics), m1_stats(m1_statistics) { }
  void solve_mean(double lambda, Vector &new_mean) const;
private:
  double m0_stats;
  const Vector &m1_stats;
};

void CriticalMeanSolver::solve_mean(double lambda, Vector &new_mean) const
{
  int dim = mean0.size();
  new_mean.resize(dim);

  // NOTE! When the critical point is a maximum, m0_stats+lambda > 0.
  // That's why it is limited to a small positive constant, so that the
  // function works also at the limit.
  for (int i = 0; i < dim; i++)
    new_mean(i) = (m1_stats(i)+lambda*mean0(i))/max(m0_stats+lambda,1e-20);
}


class MeanSolver : public FuncEval {
public:
  MeanSolver(const Vector &orig_mean, const Vector &orig_cov,
             double m0_statistics, double abs_m0, const Vector &m1_statistics,
             double k) :
    mean0(orig_mean), cov0(orig_cov), m0_stats(m0_statistics),
    abs_gamma(abs_m0),
    m1_stats(m1_statistics), k_ratio(k) { }
  virtual double evaluate_function(double p) const; // Constraint, given lambda
  void solve_mean(double lambda, Vector &new_mean) const;

protected:
  const Vector &mean0;
  const Vector &cov0;
  double m0_stats;
  double abs_gamma;
  const Vector &m1_stats;
  double k_ratio; // Function/KLD ratio
};


double MeanSolver::evaluate_function(double p) const
{
  Vector new_mean;
  solve_mean(p, new_mean);
  
  // Evaluate the KLD constraint
  int dim = mean0.size();
  double kld = 0;
  for (int i = 0; i < dim; i++)
  {
    double d = new_mean(i)-mean0(i);
    kld += d*d/cov0(i);
  }
  kld /= 2.0;
  // Evaluate function change
  double f_change = 0;
  for (int j = 0; j < dim; j++)
  {
    // double t = new_mean(j) - mean0(j);
    // double t2 = new_mean(j)*new_mean(j) - mean0(j)*mean0(j);
    // f_change += (t*m1_stats(j)-m0_stats*t2/2.0)/cov0(j);

    // 11.1.2011: Corrected:
    double t = new_mean(j) - mean0(j);
    double t2 = t*t;
    f_change += (t*(m1_stats(j)-m0_stats*mean0(j))-t2*m0_stats/2.0)/cov0(j);

 }
  double cur_ratio = k_ratio;
  // New experiment: Relate criterion/KLD ratio to occupancies as in EBW
  //cur_ratio *= 2*(abs_gamma+mpe_smooth); // - m0_stats;
  //cur_ratio *= 2*(4*abs_gamma+mpe_smooth); // Additional *4 for abs_gamma
  return cur_ratio*kld - f_change; // Should be below zero
}


void MeanSolver::solve_mean(double lambda, Vector &new_mean) const
{
  int dim = mean0.size();
  new_mean.resize(dim);

  // NOTE! When the critical point is a maximum, m0_stats+lambda > 0.
  // That's why it is limited to a small positive constant, so that the
  // function works also at the limit.
  for (int i = 0; i < dim; i++)
    new_mean(i) = (m1_stats(i)+lambda*mean0(i))/max(m0_stats+lambda,1e-20);
}


// Solver for linearly modeled mean
class LinearMeanSolver : public GaussianMeanKLDConstraint {
public:
  LinearMeanSolver(const Vector &orig_mean, const Vector &orig_cov,
                   const Vector &gradient) :
    GaussianMeanKLDConstraint(orig_mean, orig_cov),
    grad(gradient) { }
  void solve_mean(double lambda, Vector &new_mean) const;
private:
  const Vector &grad;
};

void LinearMeanSolver::solve_mean(double lambda, Vector &new_mean) const
{
  int dim = mean0.size();
  new_mean.resize(dim);
  for (int i = 0; i < dim; i++)
    new_mean(i) = mean0(i) + grad(i)*cov0(i)/lambda;
}


// KLD constraint for covariance
class GaussianCovKLDConstraint : public FuncEval {
public:
  GaussianCovKLDConstraint(const Vector &orig_cov) :
    cov0(orig_cov) { }
  virtual double evaluate_function(double p) const; // Constraint, given lambda
  virtual void solve_cov(double lambda, Vector &new_cov) const = 0;
protected:
  const Vector &cov0;
};

double GaussianCovKLDConstraint::evaluate_function(double p) const
{
  Vector cov;
  solve_cov(p, cov);

  // Evaluate the KLD constraint
  int dim = cov0.size();
  double kld = 0;
  for (int i = 0; i < dim; i++)
    kld += cov(i)/cov0(i) + log(cov0(i)/cov(i));
  return (kld - dim)/2.0;
}


// Solver for critical covariance
class CriticalCovSolver : public GaussianCovKLDConstraint {
public:
  CriticalCovSolver(const Vector &orig_mean, const Vector &orig_cov,
                    double m0_statistics, const Vector &m1_statistics,
                    const Vector &m2_statistics, double min_var) :
    GaussianCovKLDConstraint(orig_cov),
    mean0(orig_mean), m0_stats(m0_statistics), m1_stats(m1_statistics),
    m2_stats(m2_statistics), minv(min_var) { }
  void solve_cov(double lambda, Vector &new_cov) const;
private:
  const Vector &mean0;
  double m0_stats;
  const Vector &m1_stats;
  const Vector &m2_stats;
  double minv;
};

void CriticalCovSolver::solve_cov(double lambda, Vector &new_cov) const
{
  int dim = mean0.size();
  new_cov.resize(dim);
  for (int i = 0; i < dim; i++)
  {
    double temp = m2_stats(i) - 2*m1_stats(i)*mean0(i) +
      m0_stats*mean0(i)*mean0(i);
    if (lambda == 0)
    {
      new_cov(i) = temp/m0_stats;
    }
    else
    {
      double m0_l = -m0_stats + lambda;
      double l_c = lambda/cov0(i);
      // Avoid numerical problems with sqrt
      double temp2 = sqrt(max(m0_l*m0_l+4*l_c*temp, 0.0)); 
      new_cov(i) = (m0_l + temp2)/(2*l_c);
    }
    new_cov(i) = max(new_cov(i), minv);
  }
}


// Solver for linearly modeled covariance
class LinearCovSolver : public GaussianCovKLDConstraint {
public:
  LinearCovSolver(const Vector &orig_cov, const Vector &gradient,
                  double min_var) :
    GaussianCovKLDConstraint(orig_cov),
    grad(gradient), minv(min_var) { }
  void solve_cov(double lambda, Vector &new_cov) const;
private:
  const Vector &grad;
  double minv;
};

void LinearCovSolver::solve_cov(double lambda, Vector &new_cov) const
{
  int dim = cov0.size();
  new_cov.resize(dim);
  for (int i = 0; i < dim; i++)
  {
    new_cov(i) = lambda*cov0(i)/(lambda-2*cov0(i)*grad(i));
    new_cov(i) = max(new_cov(i), minv);
  }
}


// General solver for covariance
class CovSolver : public FuncEval {
public:
  CovSolver(const Vector &orig_mean, const Vector &orig_cov,
            double m0_statistics, double abs_gamma, const Vector &m1_statistics,
            const Vector &m2_statistics, double min_var, double k_ratio);
  
  virtual double evaluate_function(double p) const; // Constraint, given lambda
  void solve_cov(double lambda, Vector &new_cov) const;

  enum SolverType { MAX = 0, LINEAR = 1 };
  void set_solver(SolverType solver_type) { m_solver = solver_type; }
  enum EvalType { KLD = 0, RATIO = 1 };
  void set_evaluation(EvalType eval_type) { m_eval = eval_type; }

protected:
  double evaluate_cov_kld(const Vector &cov) const;
  double evaluate_criterion(const Vector &cov) const;
  
private:
  const Vector &m_mean0;
  const Vector &m_cov0;
  double m_m0_stats;
  double m_abs_gamma;
  const Vector &m_m1_stats;
  const Vector &m_m2_stats;
  double m_minv;
  double m_k_ratio;
  Vector m_grad0;
  SolverType m_solver;
  EvalType m_eval;
};

CovSolver::CovSolver(const Vector &orig_mean, const Vector &orig_cov,
                     double m0_statistics, double abs_gamma,
                     const Vector &m1_statistics,
          const Vector &m2_statistics, double min_var, double k_ratio) :
  m_mean0(orig_mean), m_cov0(orig_cov), m_m0_stats(m0_statistics),
  m_abs_gamma(abs_gamma),
  m_m1_stats(m1_statistics), m_m2_stats(m2_statistics), m_minv(min_var),
  m_k_ratio(k_ratio)
{
  m_solver = MAX;
  m_eval = KLD;
  int dim = m_cov0.size();
  m_grad0.resize(dim);
  for (int i = 0; i < dim; i++)
    m_grad0(i) = (m_m2_stats(i)-2*m_m1_stats(i)*m_mean0(i)+
                  m_m0_stats*m_mean0(i)*m_mean0(i)-m_m0_stats*m_cov0(i))/
      (2*m_cov0(i)*m_cov0(i));
}


double CovSolver::evaluate_function(double p) const
{
  Vector cov;
  solve_cov(p, cov);

  double kld = evaluate_cov_kld(cov);
  if (m_eval == KLD)
    return kld;
  else // m_eval == RATIO
  {
    double f_change = evaluate_criterion(cov) - evaluate_criterion(m_cov0);
    double cur_ratio = m_k_ratio;
    // New experiment: Relate criterion/KLD ratio to occupancies as in EBW
    //cur_ratio *= 2*(m_abs_gamma+mpe_smooth); // - m_m0_stats;
    //cur_ratio *= 2*(4*m_abs_gamma+mpe_smooth); // Additional *4 for abs_gamma
    return cur_ratio*kld - f_change; // Should be below zero
  }
}

void CovSolver::solve_cov(double lambda, Vector &new_cov) const
{
  int dim = m_mean0.size();
  new_cov.resize(dim);
  if (m_solver == MAX)
  {
    for (int i = 0; i < dim; i++)
    {
      double temp = m_m2_stats(i) - 2*m_m1_stats(i)*m_mean0(i) +
        m_m0_stats*m_mean0(i)*m_mean0(i);
      if (lambda == 0)
      {
        new_cov(i) = temp/m_m0_stats;
      }
      else
      {
        double m0_l = -m_m0_stats + lambda;
        double l_c = lambda/m_cov0(i);
        // Avoid numerical problems with sqrt
        double temp2 = sqrt(max(m0_l*m0_l+4*l_c*temp, 0.0)); 
        new_cov(i) = (m0_l + temp2)/(2*l_c);
      }
    }
  }
  else // m_solver == LINEAR
  {
    for (int i = 0; i < dim; i++)
      new_cov(i) = lambda*m_cov0(i)/(lambda-2*m_cov0(i)*m_grad0(i));

  }
  // Ensure minimum variance
  for (int i = 0; i < dim; i++)
    new_cov(i) = max(new_cov(i), m_minv);
}


double
CovSolver::evaluate_cov_kld(const Vector &cov) const
{
  int dim = m_cov0.size();
  double kld = 0;
  for (int i = 0; i < dim; i++)
    kld += cov(i)/m_cov0(i) + log(m_cov0(i)/cov(i));
  return (kld - dim)/2.0;
}


double
CovSolver::evaluate_criterion(const Vector &cov) const
{
  int dim = m_cov0.size();
  double f = 0;

  // FIXME! Is it sensible that there are two forms of criterion depending
  // on the type of solver used?
  if (m_solver == MAX)
  {
    for (int i = 0; i < dim; i++)
      f -= ((m_m2_stats(i) - 2*m_m1_stats(i)*m_mean0(i) +
             m_m0_stats*m_mean0(i)*m_mean0(i))/cov(i) +
            m_m0_stats*log(cov(i))) / 2.0;
  }
  else // m_solver == LINEAR
  {
    for (int i = 0; i < dim; i++)
      f += m_grad0(i)*cov(i);
  }
  return f;
}



class MixtureWeightKLD : public FuncEval {
public:
  MixtureWeightKLD(const Vector *paramp, const Vector *searchp,
                   int num_weights) : wp(paramp), dp(searchp),
                                      size(num_weights) { }
  virtual double evaluate_function(double p) const;
private:
  const Vector *wp;
  const Vector *dp;
  int size;
};

double
MixtureWeightKLD::evaluate_function(double p) const
{
  double kld = 0;

  // Compute the normalization factors
  double new_norm = 0;
  double orig_norm = 0;
  for (int i = 0; i < size; i++)
  {
    orig_norm += exp((*wp)(i));
    new_norm += exp((*wp)(i) + p*(*dp)(i));
  }

  for (int i = 0; i < size; i++)
  {
    double orig_w = exp((*wp)(i))/orig_norm;
    double new_w = exp((*wp)(i)+p*(*dp)(i))/new_norm;
    kld += new_w*log(new_w/orig_w);
  }
  return kld;
}


class GaussianKLD : public FuncEval {
public:
  GaussianKLD(const Vector *mean, const Vector *cov,
              const Vector *mean_search_dir, const Vector *cov_search_dir,
              int dimension, double mv) : gmp(mean), gcp(cov),
                                          dmp(mean_search_dir),
                                          dcp(cov_search_dir),
                                          dim(dimension), min_var(mv) { }
  virtual double evaluate_function(double p) const;
private:
  const Vector *gmp;
  const Vector *gcp;
  const Vector *dmp;
  const Vector *dcp;
  int dim;
  double min_var;
};

double
GaussianKLD::evaluate_function(double p) const
{
  double kld = 0;
  for (int i = 0; i < dim; i++)
  {
    double orig_m = (*gmp)(i);
    double orig_v = exp((*gcp)(i)) + min_var;
    double new_m = orig_m + p*(*dmp)(i);
    double new_v = exp((*gcp)(i) + p*(*dcp)(i)) + min_var;
    double dm = new_m - orig_m;
    kld += new_v/orig_v + log(orig_v/new_v) + dm*dm/orig_v;
  }
  return (kld - dim)/2.0;
}


double gaussian_mean_parameter_kld(double dmean, double cov)
{
  return dmean*dmean/(2*cov);
}

class GaussianMeanKLD : public FuncEval {
public:
  GaussianMeanKLD(const Vector *cov, const Vector *mean_search_dir,
                  int dimension) : gcp(cov), dmp(mean_search_dir),
                                   dim(dimension) { }
  virtual double evaluate_function(double p) const;
private:
  const Vector *gcp;
  const Vector *dmp;
  int dim;
};

double
GaussianMeanKLD::evaluate_function(double p) const
{
  double kld = 0;
  for (int i = 0; i < dim; i++)
  {
    double dm = p*(*dmp)(i);
    kld += dm*dm/(*gcp)(i);
  }
  return kld/2.0;
}


class GaussianCovParameterKLD : public FuncEval {
public:
  GaussianCovParameterKLD(double cov, double search_dir, double mv) :
    orig_cov(cov), change(search_dir), min_var(mv) { }
  virtual double evaluate_function(double p) const;
private:
  double orig_cov;
  double change;
  double min_var;
};

double GaussianCovParameterKLD::evaluate_function(double p) const
{
  double old_v = max(exp(orig_cov), min_var);
  double new_v = max(exp(orig_cov + p*change), min_var);
  return (new_v/old_v + log(old_v/new_v) - 1) / 2.0;
}


class GaussianCovKLD : public FuncEval {
public:
  GaussianCovKLD(const Vector *cov, const Vector *cov_search_dir,
                 int dimension, double mv) : gcp(cov), dcp(cov_search_dir),
                                             dim(dimension), min_var(mv) { }
  virtual double evaluate_function(double p) const;
private:
  const Vector *gcp;
  const Vector *dcp;
  int dim;
  double min_var;
};

double
GaussianCovKLD::evaluate_function(double p) const
{
  double kld = 0;
  for (int i = 0; i < dim; i++)
  {
    double orig_v = max(exp((*gcp)(i)), min_var);
    double new_v = max(exp((*gcp)(i) + p*(*dcp)(i)), min_var);
    kld += new_v/orig_v + log(orig_v/new_v);
  }
  return (kld - dim)/2.0;
}


class GaussianCovPartialChangeKLD : public FuncEval {
public:
  GaussianCovPartialChangeKLD(const Vector *cov, const Vector *cov_search_dir,
                              const vector<int> &increase_indicator,
                              int dimension, double mv) :
    gcp(cov), dcp(cov_search_dir), indicator(increase_indicator),
    dim(dimension), min_var(mv) { }
  virtual double evaluate_function(double p) const;
private:
  const Vector *gcp;
  const Vector *dcp;
  const vector<int> &indicator;
  int dim;
  double min_var;
};

double GaussianCovPartialChangeKLD::evaluate_function(double p) const
{
  double kld = 0;
  for (int i = 0; i < dim; i++)
  {
    double orig_v = max(exp((*gcp)(i)), min_var);
    double new_v;
    if (indicator[i])
      new_v = max(exp((*gcp)(i) + p*(*dcp)(i)), min_var);
    else
      new_v = max(exp((*gcp)(i) + (*dcp)(i)), min_var);
    kld += new_v/orig_v + log(orig_v/new_v);
  }
  return (kld - dim)/2.0;
}


void original_cls_mixture_step(void)
{
  for (int i = 0; i < model.num_emission_pdfs(); i++)
  {
    Mixture *m = model.get_emission_pdf(i);
    Vector orig_weights;
    Vector new_weights;
    Vector search_dir;
    orig_weights.resize(m->size());
    new_weights.resize(m->size());
    double norm = 0;
    bool pos = true, neg = true;
    for (int j = 0; j < m->size(); j++)
    {
      orig_weights(j) = m->get_mixture_coefficient(j);
      if (opt_mode == MODE_MMI)
        new_weights(j) = m->get_accumulated_gamma(PDF::ML_BUF, j)
          - m->get_accumulated_gamma(PDF::MMI_BUF, j);
      else if (opt_mode == MODE_MPE)
        new_weights(j) = m->get_accumulated_gamma(PDF::MPE_NUM_BUF, j);
      else
        throw "Invalid optimization mode";
      if (new_weights(j) >= 0)
        neg = false;
      if (new_weights(j) <= 0)
        pos = false;
      norm += new_weights(j);
    }
    
    for (int j = 0; j < m->size(); j++)
      new_weights(j) = new_weights(j) / norm;

    if (pos && !neg) // Critical point is a maximum
    {
      search_dir = new_weights;
      Blas_Add_Mult(search_dir, -1, orig_weights);
      if (info > 0)
        fprintf(stderr, "Mixture %i, MAX update\n", i);
    }
    else if (neg && !pos) // Critical point is a minimum
    {
      search_dir = orig_weights;
      Blas_Add_Mult(search_dir, -1, new_weights);
      if (info > 0)
        fprintf(stderr, "Mixture %i, MIN update\n", i);
    }
    else
    {
      // Extract the gradient
      search_dir.resize(m->size());
      double projection = 0;
      double normal_c = 1.0/sqrt((double)m->size());
      for (int j = 0; j < m->size(); j++)
      {
        if (opt_mode == MODE_MMI)
          search_dir(j) = m->get_accumulated_gamma(PDF::ML_BUF, j)
            - m->get_accumulated_gamma(PDF::MMI_BUF, j);
        else if (opt_mode == MODE_MPE)
          search_dir(j) = m->get_accumulated_gamma(PDF::MPE_NUM_BUF, j);
        search_dir(j) /= m->get_mixture_coefficient(j);
        projection += search_dir(j)*normal_c;
      }
      // Project to feasible region
      for (int j = 0; j < m->size(); j++)
        search_dir(j) = search_dir(j) - projection*normal_c;

      if (info > 0)
        fprintf(stderr, "Mixture %i, gradient update\n", i);
    }

    // Compute step size for approximative KLD constraint
    Vector temp;
    temp.resize(m->size());
    for (int j = 0; j < m->size(); j++)
      temp(j) = search_dir(j) / orig_weights(j);
    double step_size = Blas_Dot_Prod(search_dir, temp);
    if (pos && !neg && step_size < weight_kld_limit)
      step_size = 1; // With maximum and below KLD limit, keep the scaling

    if (step_size > 0)
    {
      step_size = sqrt(weight_kld_limit/step_size);
      double original_step_size = step_size;
      new_weights = orig_weights;
      Blas_Add_Mult(new_weights, step_size, search_dir);
      bool rescale = false;
      for (int j = 0; j < m->size(); j++)
      {
        if (new_weights(j) <= 1e-6)
        {
          step_size = min(step_size,
                               (1.0e-6 - orig_weights(j))/search_dir(j));
          rescale = true;
        }
        else if (new_weights(j) > 1)
        {
          step_size = min(step_size,
                               (1.0 - orig_weights(j))/search_dir(j));
          rescale = true;
        }
      }
      if (rescale)
      {
        if (step_size < 0)
        {
          if (info > 0)
            fprintf(stderr, "Warning: Negative step size (%g), truncating\n",
                    step_size);
          step_size = 0;
        }
        new_weights = orig_weights;
        Blas_Add_Mult(new_weights, step_size, search_dir);
        if (info > 0)
          fprintf(stderr, "  Rescaling, %g -> %g\n",
                  original_step_size, step_size);
      }
      
      norm = Blas_Norm1(new_weights);

      if (fabs(norm - 1.0) > 0.01 && info > 0)
        fprintf(stderr, "Warning: Bad normalization for mixture %i (%g)\n",
                i, norm);
      
      // Set the new mixture parameters
      for (int j = 0; j < m->size(); j++)
        m->set_mixture_coefficient(
          j, max(min(new_weights(j)/norm, 1.0), 1e-6));

      // Compute KLD
      double kld = 0;
      for (int i = 0; i < orig_weights.size(); i++)
        kld += new_weights(i)*log(new_weights(i)/orig_weights(i));
      if (info > 0)
        fprintf(stderr, "  KLD: %.4f (step size %g)\n", kld, step_size);
    }
    else
    {
      if (info > 0)
        fprintf(stderr, "Warning: No update for mixture %i\n", i);
    }
  }
}


void original_cls_mean_cov_step(void)
{
  PDFPool *pool = model.get_pool();

  // Update means and covariances (diagonal) of all Gaussians
  for (int i = 0; i < pool->size(); i++)
  {
    Gaussian *pdf = dynamic_cast< Gaussian* >(pool->get_pdf(i));
    if (pdf == NULL)
      throw string("Only Gaussian PDFs are supported!");
    Vector mean;
    Vector cov;
    Vector target_mean;
    Vector target_cov;
    Vector mean_search_dir;
    Vector cov_search_dir;
    int dim = pool->dim();
    pdf->get_mean(mean);
    assert( mean.size() == dim );
    pdf->get_covariance(cov);
    assert( cov.size() == dim );

    target_mean.resize(dim);
    target_cov.resize(dim);
    mean_search_dir.resize(dim);
    cov_search_dir.resize(dim);

    Vector d_m1, d_m2;
    double d_gamma;
    if (opt_mode == MODE_MMI)
    {
      Vector temp;
      pdf->get_accumulated_mean(PDF::ML_BUF, d_m1);
      pdf->get_accumulated_second_moment(PDF::ML_BUF, d_m2);
      pdf->get_accumulated_mean(PDF::MMI_BUF, temp);
      Blas_Add_Mult(d_m1, -1, temp);
      pdf->get_accumulated_second_moment(PDF::MMI_BUF, temp);
      Blas_Add_Mult(d_m2, -1, temp);
      d_gamma = pdf->get_accumulated_gamma(PDF::ML_BUF) -
        pdf->get_accumulated_gamma(PDF::MMI_BUF);
    }
    else if (opt_mode == MODE_MPE)
    {
      pdf->get_accumulated_mean(PDF::MPE_NUM_BUF, d_m1);
      pdf->get_accumulated_second_moment(PDF::MPE_NUM_BUF, d_m2);
      d_gamma = pdf->get_accumulated_gamma(PDF::MPE_NUM_BUF);
    }
    else
      throw "Invalid optimization mode";
    bool pos = true; // Whether covariance has a critical point or not

    //////////////////////
    // Mean update
    //////////////////////
    
    if (d_gamma == 0)
    {
      pos = false;
      // Extract gradient for the mean
      for (int j = 0; j < dim; j++)
        mean_search_dir(j) = (d_m1(j)-mean(j)*d_gamma)/cov(j);

      double scale = 0;
      for (int j = 0; j < dim; j++)
        scale += mean_search_dir(j)*mean_search_dir(j)/cov(j);
      if (scale > 0)
      {
        // Note! For the original CLS, do not use --original-limits!
        scale = sqrt(mean_kld_limit/scale);
        for (int j = 0; j < dim; j++)
          mean_search_dir(j) = scale * mean_search_dir(j);
      }
      if (info > 0)
        fprintf(stderr, "Mean %i, gradient update, scale %g\n", i, scale);
    }
    else
    {
      // Set target parameters
      for (int j = 0; j < dim; j++)
        target_mean(j) = d_m1(j) / d_gamma;
      double sign = (d_gamma>0?1:-1);
      for (int j = 0; j < dim; j++)
        mean_search_dir(j) = sign*(target_mean(j) - mean(j));

      // Check for KLD limitation
      double scale = 0;
      for (int j = 0; j < dim; j++)
        scale += gaussian_mean_parameter_kld(mean_search_dir(j), cov(j))*2;
      if ((d_gamma < 0 && scale > 0) || scale >= mean_kld_limit)
      {
        scale = sqrt(mean_kld_limit/scale);
        for (int j = 0; j < dim; j++)
          mean_search_dir(j) = scale * mean_search_dir(j);
      }
      else
        scale = 1;

      if (sign > 0)
        fprintf(stderr, "Mean %i, MAX update, scale %g\n", i, scale);
      else
        fprintf(stderr, "Mean %i, MIN update, scale %g\n", i, scale);
    }

    // Update the mean
    for (int j = 0; j < pool->dim(); j++)
      target_mean(j) = mean(j) + mean_search_dir(j);
    pdf->set_mean(target_mean);

    // Compute KLD
    double kld = 0;
    for (int j = 0; j < pool->dim(); j++)
      kld += gaussian_mean_parameter_kld(target_mean(j) - mean(j), cov(j))*2;
    if (info > 0)
      fprintf(stderr, "  KLD: %.4f\n", kld);

    //////////////////////
    // Covariance update
    //////////////////////
    
    // Transform parameters, determine the existence of the critical point,
    // and set the target parameters
    for (int j = 0; j < dim; j++)
    {
      // CLS paper version: Critical mean point assumed:
      if (d_gamma*d_m2(j) < d_m1(j)*d_m1(j))
        pos = false;

      cov(j) = util::safe_log(max(min_var, cov(j)));
      if (pos)
      {
        target_cov(j) = d_m2(j)/d_gamma -
          d_m1(j)*d_m1(j)/(d_gamma*d_gamma);
        target_cov(j) = util::safe_log(target_cov(j));
      }
    }

    if (pos)
    {
      if (info > 0 && d_gamma < 0)
        fprintf(stderr, "NOTE: Cov %i, incorrect precondition (MAX update, O(1) = %g\n", i, d_gamma);

      for (int j = 0; j < dim; j++)
        cov_search_dir(j) = target_cov(j) - cov(j);

      // Check approximative KLD
      double scale = 0;
      for (int j = 0; j < dim; j++)
        scale += cov_search_dir(j)*cov_search_dir(j);
      if (scale > cov_kld_limit)
        Blas_Scale(sqrt(cov_kld_limit/scale), cov_search_dir);
      else
        scale = cov_kld_limit; // Just for printing the scale
      
      if (info > 0)
        fprintf(stderr, "Cov %i, MAX update, scale %g\n", i,
                sqrt(cov_kld_limit/scale));
    }
    else
    {
      // Extract gradient for the covariance.
      // Note: The gradient is for log(sigma^2)
      for (int j = 0; j < dim; j++)
      {
        // CLS paper version: Assuming the mean is in the critical point
        if (d_gamma != 0)
        {
          cov_search_dir(j) = (d_m2(j)-d_m1(j)*d_m1(j)/d_gamma-d_gamma*exp(cov(j))) / (2*exp(cov(j)));
        }
        else
        {
          cov_search_dir(j) = (d_m2(j)-2*d_m1(j)*mean(j)+d_gamma*mean(j)*mean(j)-d_gamma*exp(cov(j))) / (2*exp(cov(j)));
        }
      }

      // Scale to KLD limit
      double scale = 0;
      for (int j = 0; j < dim; j++)
        scale += cov_search_dir(j)*cov_search_dir(j);
      if (scale > 0)
        Blas_Scale(sqrt(cov_kld_limit/scale), cov_search_dir);

      if (info > 0)
        fprintf(stderr, "Cov %i, gradient update, scale %g\n", i,
                sqrt(cov_kld_limit/scale));
    }
    
    // Update the covariance
    for (int j = 0; j < pool->dim(); j++)
      target_cov(j) = max(min_var, exp(cov(j) + cov_search_dir(j)));
    pdf->set_covariance(target_cov);

    // Check KLD
    kld = -pool->dim();
    for (int j = 0; j < pool->dim(); j++)
      kld += target_cov(j)/exp(cov(j))+cov(j)-util::safe_log(target_cov(j));
    if (info > 0)
      fprintf(stderr, "  KLD: %.4f\n", kld);
  }
}


void soft_max_mixture_cls_step(void)
{
  double search_acc = 1e-4;
  
  // Go through all the mixtures and update their components
  for (int i = 0; i < model.num_emission_pdfs(); i++)
  {
    Mixture *m = model.get_emission_pdf(i);
    Vector orig_weights;
    Vector new_weights;
    Vector search_dir;
    orig_weights.resize(m->size());
    new_weights.resize(m->size());
    double norm = 0;
    bool pos = true, neg = true;
    for (int j = 0; j < m->size(); j++)
    {
      orig_weights(j) = m->get_mixture_coefficient(j);
      new_weights(j) = m->get_accumulated_gamma(PDF::MPE_NUM_BUF, j);
      if (new_weights(j) >= 0)
        neg = false;
      if (new_weights(j) <= 0)
        pos = false;
      norm += new_weights(j);
    }
    for (int j = 0; j < m->size(); j++)
      new_weights(j) = new_weights(j) / norm;

    // Apply a soft-max transformation to the mixture weights
    for (int j = 0; j < m->size(); j++)
    {
      orig_weights(j) = util::safe_log(orig_weights(j));
      new_weights(j) = util::safe_log(new_weights(j));
    }

    if (pos && !neg) // Critical point is a maximum
    {
      search_dir = new_weights;
      Blas_Add_Mult(search_dir, -1, orig_weights);
    }
    else if (neg && !pos) // Critical point is a minimum
    {
      search_dir = orig_weights;
      Blas_Add_Mult(search_dir, -1, new_weights);
    }
    else
    {
      // Extract gradient wrt the original parameters
      search_dir.resize(m->size());
      vector<double> temp;
      temp.resize(m->size());  
      for (int j = 0; j < m->size(); j++)
      {
        temp[j] = (m->get_accumulated_gamma(PDF::MPE_NUM_BUF, j) /
                   m->get_mixture_coefficient(j));
      }
      // Combine to form derivatives wrt the transformed parameters
      for (int j = 0; j < m->size(); j++)
      {
        double val = 0;
        double ep = m->get_mixture_coefficient(j);
        for (int k = 0; k < m->size(); k++)
        {
          if (k == j)
            val += temp[k]*(ep - m->get_mixture_coefficient(k)*ep);
          else
          {
            val += temp[k]*(-m->get_mixture_coefficient(k)*ep);
          }
        }
        search_dir(j) = val;
      }
      if (info > 0)
        fprintf(stderr, "Gradient update for mixture %i\n", i);
    }

    // Update parameters, ensure KLD restriction
    MixtureWeightKLD mix(&orig_weights, &search_dir, m->size());
    double init_step = 1;
    double kld = mix.evaluate_function(init_step);

    while ((isnan(kld) || isinf(kld)) && init_step > 1e-30)
    {
      init_step /= 2;
      kld = mix.evaluate_function(init_step);
    }

    // Note! KLD of mean and variance are computed properly so that
    // their limits are actually twice that pointed by the parameter,
    // compared to CLS-paper. To make similar adjustment for mixture
    // weights, the limit parameters is here doubled in mixture case.

    // 22.12.2009: After testing the effect of individual parameter types,
    // it was noted that mixture weight changes dominated the model
    // changes. Therefore the KLD limit has been halved from what it
    // used to be, equating the actual KLD limit defined.
    
    norm = 0;
    if (init_step > 1e-30)
    {
      if (!pos)
      {
        // Critical point is not a maximum, find the maximum step
        while (kld < weight_kld_limit && init_step < 1000)
        {
          init_step *= 2;
          kld = mix.evaluate_function(init_step);
        }
        if (isnan(kld) || isinf(kld))
        {
          init_step /= 2;
          kld = mix.evaluate_function(init_step);
        }
      }

      if (kld > weight_kld_limit)
      {
        double step = bin_search_max_param(0, 0, init_step, kld, weight_kld_limit,
                                           search_acc*init_step, mix);
        //fprintf(stderr, "Mixture %i limited, original KLD %.4g, step size %.4g, new KLD %.4g\n", i, kld, step, mix.evaluate_function(step));
        for (int j = 0; j < m->size(); j++)
          search_dir(j) = search_dir(j) * step;
      }
      else
      {
        for (int j = 0; j < m->size(); j++)
          search_dir(j) = search_dir(j) * init_step;
      }
      
      // Compute the new transformed parameters and the normalization
      for (int j = 0; j < m->size(); j++)
      {
        new_weights(j) = orig_weights(j) + search_dir(j);
        norm += exp(new_weights(j));
      }
    }

    if (norm > 0 && !isnan(norm) && !isinf(norm))
    {
      // Set the new mixture parameters
      for (int j = 0; j < m->size(); j++)
        m->set_mixture_coefficient(j, exp(new_weights(j))/norm);
    }
    else
    {
      if (info > 0)
      {
        fprintf(stderr, "Warning: Invalid search direction, no update for mixture %i\n", i);
        fprintf(stderr, "init_step = %g, kld = %g\n", init_step, kld);
      }
    }
  }
}


void combined_mean_covariance_cls_update(void)
{
  PDFPool *pool = model.get_pool();
  double search_acc = 1e-4;

  // Update means and covariances (diagonal) of all Gaussians
  for (int i = 0; i < pool->size(); i++)
  {
    Gaussian *pdf = dynamic_cast< Gaussian* >(pool->get_pdf(i));
    if (pdf == NULL)
      throw string("Only Gaussian PDFs are supported!");

    Vector mean;
    Vector cov;
    Vector target_mean;
    Vector target_cov;
    Vector mean_search_dir;
    Vector cov_search_dir;
    int dim = pool->dim();
    pdf->get_mean(mean);
    assert( mean.size() == dim );
    pdf->get_covariance(cov);
    assert( cov.size() == dim );

    target_mean.resize(dim);
    target_cov.resize(dim);
    mean_search_dir.resize(dim);
    cov_search_dir.resize(dim);

    Vector mpe_m1, mpe_m2;
    pdf->get_accumulated_mean(PDF::MPE_NUM_BUF, mpe_m1);
    pdf->get_accumulated_second_moment(PDF::MPE_NUM_BUF, mpe_m2);
    double mpe_gamma = pdf->get_accumulated_gamma(PDF::MPE_NUM_BUF);


    bool pos = true; // Whether covariance has a critical point or not
    if (mpe_gamma == 0)
    {
      pos = false;
      // Extract gradient for the mean
      for (int j = 0; j < dim; j++)
        mean_search_dir(j) = (mpe_m1(j)-mean(j)*mpe_gamma)/cov(j);
      
      // Scale the gradient to achieve the KLD limit of the mean
      GaussianMeanKLD g(&cov, &mean_search_dir, dim);
      double step = 1e-6;
      double kld = g.evaluate_function(step);
      while (kld < mean_kld_limit && step < 10)
      {
        step *= 2;
        kld = g.evaluate_function(step);
      }
      if (kld > mean_kld_limit)
      {
        step = bin_search_max_param(0, 0, step, kld, mean_kld_limit, search_acc*step, g);
      }
      //fprintf(stderr, "Gaussian %i mean gradient initialized with step %.4g, KLD %.4g\n", i, step, g.evaluate_function(step));
      for (int j = 0; j < dim; j++)
        mean_search_dir(j) = step*mean_search_dir(j);
      if (info > 0)
        fprintf(stderr, "Gradient update for mean %i\n", i);
    }
    else
    {
      // Set target parameters
      for (int j = 0; j < dim; j++)
      {
        target_mean(j) = mpe_m1(j) / mpe_gamma;
        target_cov(j) = (mpe_m2(j) - 2*mpe_m1(j)*mean(j) +
                         mpe_gamma*mean(j)*mean(j)) / mpe_gamma;
      }

      double sign = (mpe_gamma>0?1:-1);
      for (int j = 0; j < dim; j++)
        mean_search_dir(j) = sign*(target_mean(j) - mean(j));
    }

    // Transform covariances and determine the existence of the critical point
    for (int j = 0; j < dim; j++)
    {
      cov(j) = util::safe_log(max(1.0001*min_var, cov(j)) - min_var);
      if (pos)
        target_cov(j) = util::safe_log(max(1.0001*min_var, target_cov(j))-
                                       min_var);
      if (2*mpe_m1(j)*mean(j)-mpe_m2(j) >= mean(j)*mean(j))
        pos = false;
    }

    if (pos)
    {
      for (int j = 0; j < dim; j++)
        cov_search_dir(j) = target_cov(j) - cov(j);
    }
    else
    {
      // Extract gradient for the covariance
      for (int j = 0; j < dim; j++)
        cov_search_dir(j) = (mpe_m2(j)-2*mpe_m1(j)*mean(j)+
                             mpe_gamma*mean(j)*mean(j)-mpe_gamma*cov(j)) /
          (2*cov(j)*cov(j))*(cov(j)-min_var); // Gradient is for log(sigma^2-MV)
      
      // Scale the gradient to achieve the KLD limit of the covariance
      GaussianCovKLD g(&cov, &cov_search_dir, dim, min_var);
      double step = 1e-9;
      double kld = g.evaluate_function(step);
      while ((isnan(kld) || isinf(kld)) && step > 1e-50)
      {
        step /= 2;
        kld = g.evaluate_function(step);
      }
      if (step > 1e-50)
      {
        while (kld < cov_kld_limit && step < 10)
        {
          step *= 2;
          kld = g.evaluate_function(step);
        }
        if (kld > cov_kld_limit)
        {
          step = bin_search_max_param(0, 0, step, kld, cov_kld_limit, search_acc*step, g);
        }
        //fprintf(stderr, "Gaussian %i covariance gradient initialized with step %.4g, KLD %.4g\n", i, step, g.evaluate_function(step));
        for (int j = 0; j < dim; j++)
          cov_search_dir(j) = step*cov_search_dir(j);
      }
      else
      {
        // No covariance update
        if (info > 0)
          fprintf(stderr, "Warning: No covariance update for Gaussian %i\n", i);
        for (int j = 0; j < dim; j++)
          cov_search_dir(j) = 0;
      }
      if (info > 0)
        fprintf(stderr, "Gradient update for covariance %i\n", i);
    }

    // KLD limit the whole Gaussian
    GaussianKLD g(&mean, &cov, &mean_search_dir, &cov_search_dir, dim, min_var);
    double kld = g.evaluate_function(1);
    if (kld > mean_kld_limit+cov_kld_limit) // FIXME: Another limit here?
    {
      double step = bin_search_max_param(0, 0, 1, kld, mean_kld_limit+cov_kld_limit, search_acc, g);
      //fprintf(stderr, "Gaussian %i limited, original KLD %.4g, step size %.4g, new KLD %.4g\n", i, kld, step, g.evaluate_function(step));
      for (int j = 0; j < dim; j++)
      {
        mean_search_dir(j) = mean_search_dir(j) * step;
        cov_search_dir(j) = cov_search_dir(j) * step;
      }
    }

    // Update the parameters
    for (int j = 0; j < pool->dim(); j++)
    {
      target_mean(j) = mean(j) + mean_search_dir(j);
      target_cov(j) = exp(cov(j) + cov_search_dir(j)) + min_var;
    }
    pdf->set_mean(target_mean);
    pdf->set_covariance(target_cov);
  }
}


void separate_mean_covariance_cls_update(void)
{
  PDFPool *pool = model.get_pool();
  double search_acc = 1e-4;

  // Update means and covariances (diagonal) of all Gaussians
  for (int i = 0; i < pool->size(); i++)
  {
    Gaussian *pdf = dynamic_cast< Gaussian* >(pool->get_pdf(i));
    if (pdf == NULL)
      throw string("Only Gaussian PDFs are supported!");

    Vector mean;
    Vector cov;
    Vector target_mean;
    Vector target_cov;
    Vector mean_search_dir;
    Vector cov_search_dir;
    int dim = pool->dim();
    pdf->get_mean(mean);
    assert( mean.size() == dim );
    pdf->get_covariance(cov);
    assert( cov.size() == dim );

    target_mean.resize(dim);
    target_cov.resize(dim);
    mean_search_dir.resize(dim);
    cov_search_dir.resize(dim);

    Vector mpe_m1, mpe_m2;
    pdf->get_accumulated_mean(PDF::MPE_NUM_BUF, mpe_m1);
    pdf->get_accumulated_second_moment(PDF::MPE_NUM_BUF, mpe_m2);
    double mpe_gamma = pdf->get_accumulated_gamma(PDF::MPE_NUM_BUF);
    bool pos = true; // Whether covariance has a critical point or not

    //////////////////////
    // Mean update
    //////////////////////
    
    if (mpe_gamma == 0)
    {
      pos = false;
      // Extract gradient for the mean
      for (int j = 0; j < dim; j++)
        mean_search_dir(j) = (mpe_m1(j)-mean(j)*mpe_gamma)/cov(j);

      double scale = 0;
      for (int j = 0; j < dim; j++)
        scale += mean_search_dir(j)*mean_search_dir(j)/cov(j);
      if (scale > 0)
      {
        scale = sqrt(2*mean_kld_limit/scale);
        for (int j = 0; j < dim; j++)
          mean_search_dir(j) = scale * mean_search_dir(j);
      }
      if (info > 0)
        fprintf(stderr, "Gradient update for mean %i, scale %g\n", i, scale);

//      printf("G\n");
    }
    else
    {
      // Set target parameters
      for (int j = 0; j < dim; j++)
      {
        target_mean(j) = mpe_m1(j) / mpe_gamma;
        // CLS paper version: Assume critical mean point
        target_cov(j) = mpe_m2(j)/mpe_gamma -
          mpe_m1(j)*mpe_m1(j)/(mpe_gamma*mpe_gamma);

        // "Correct" version: use previous mean
        // target_cov(j) = (mpe_m2(j)-2*mpe_m1(j)*mean(j)+
        //                  mpe_gamma*mean(j)*mean(j))/mpe_gamma;
      }

      double sign = (mpe_gamma>0?1:-1);
      for (int j = 0; j < dim; j++)
        mean_search_dir(j) = sign*(target_mean(j) - mean(j));

      // Check for KLD limitation
      double scale = 0;
      for (int j = 0; j < dim; j++)
        scale += gaussian_mean_parameter_kld(mean_search_dir(j), cov(j));      
      if ((mpe_gamma < 0 && scale > 0) || scale >= mean_kld_limit)
      {
        scale = sqrt(mean_kld_limit/scale);
//        fprintf(stderr, "Limiting mean %i KLD, scale %.4g\n", i, scale);
        for (int j = 0; j < dim; j++)
          mean_search_dir(j) = scale * mean_search_dir(j);
//          if (mpe_gamma < 0)
//            printf("M\n");
//          else
//            printf("L\n");
      }
//       else
//         printf("C\n");

      if (info > 0)
      {
        GaussianMeanKLD g(&cov, &mean_search_dir, dim);        
        fprintf(stderr, "Mean %i, KLD %.4f\n", i,
                g.evaluate_function(1));
      }
    }

    // Update the mean
    for (int j = 0; j < pool->dim(); j++)
      target_mean(j) = mean(j) + mean_search_dir(j);
    pdf->set_mean(target_mean);


    //////////////////////
    // Covariance update
    //////////////////////
    
    // Transform covariances and determine the existence of the critical point
    for (int j = 0; j < dim; j++)
    {
      // CLS paper version: Critical mean point assumed:
      if (mpe_gamma*mpe_m2(j) < mpe_m1(j)*mpe_m1(j))
        pos = false;

      // "Correct" version: use the previous mean
      // if (mpe_gamma*mpe_m2(j) <
      //     mpe_gamma*mean(j)*(2*mpe_m1(j)-mpe_gamma*mean(j)))
      //   pos = false;

      cov(j) = util::safe_log(max(min_var, cov(j)));
      if (pos)
        target_cov(j) = util::safe_log(max(min_var, target_cov(j)));
    }

    if (mpe_gamma == 0)
      pos = false; // Fallback to gradient

    if (pos)
    {
      // CLS paper version: no minimum
      double sign = 1;
      // "Correct" version: A minimum covariance critical point is possible
      // double sign = (mpe_gamma>0?1:-1);
      
      for (int j = 0; j < dim; j++)
        cov_search_dir(j) = sign*(target_cov(j) - cov(j));
    }
    else
    {
      // Extract gradient for the covariance.
      // Note: The gradient is for log(sigma^2)
      for (int j = 0; j < dim; j++)
      {
        // Not assuming known mean:
        // cov_search_dir(j) = (mpe_m2(j)-2*mpe_m1(j)*mean(j)+
        //                      mpe_gamma*mean(j)*mean(j)-mpe_gamma*exp(cov(j)))/
        //   (2*exp(cov(j)));
        // CLS paper version: Assuming the mean is in the critical point
        if (mpe_gamma != 0)
        {
          cov_search_dir(j) = (mpe_m2(j)-mpe_m1(j)*mpe_m1(j)/mpe_gamma-mpe_gamma*exp(cov(j)))/(2*exp(cov(j)));
        }
        else
        {
          cov_search_dir(j) = (mpe_m2(j)-2*mpe_m1(j)*mean(j)+mpe_gamma*mean(j)*mean(j)-mpe_gamma*exp(cov(j))) / (2*exp(cov(j)));
        }
      }
      //fprintf(stderr, "Gradient update for covariance %i\n", i);
    }
    
    GaussianCovKLD gc(&cov, &cov_search_dir, dim, min_var);
    double step = 1;
    double kld = gc.evaluate_function(step);

    // CLS paper version uses 1/2 of the KLD limit.
    // Previously tested 1/4 of the KLD limit

    if (!pos || mpe_gamma < 0)
    {
      // Scale the search direction to achieve the KLD limit of the covariance
      while ((isnan(kld) || isinf(kld) || kld>100*cov_kld_limit) && step > 1e-50)
      {
        step /= 2;
        kld = gc.evaluate_function(step);
      }
      if (step > 1e-50)
      {
        while (kld < cov_kld_limit && step < 10)
        {
          step *= 2;
          kld = gc.evaluate_function(step);
        }
        if (isnan(kld) || isinf(kld))
        {
          step /= 2;
          kld = gc.evaluate_function(step);
          if (info > 0)
            fprintf(stderr, "Covariance %i: gradient fallback to step %.4g, KLD %.4g\n", i, step, kld);
        }
      }
      else
      {
        step = 0;
        if (info > 0)
          fprintf(stderr, "Warning: No covariance update for Gaussian %i\n", i);
      }
    }

    // Check for KLD limitation
    if (step > 0)
    {
      if (kld > cov_kld_limit)
      {
        step = bin_search_max_param(0, 0, step, kld, cov_kld_limit, search_acc*step, gc);
        if (pos)
        {
          if (mpe_gamma > 0)
          {
            if (info > 0)
              fprintf(stderr, "Covariance %i: Critical point update, step %.4g, KLD %.4g\n", i, step, gc.evaluate_function(step));
          }
          else
          {
            if (info > 0)
              fprintf(stderr, "Covariance %i: Minimum point update, step %.4g, KLD %.4g\n", i, step, gc.evaluate_function(step));
          }
        }
        else
        {
          if (info > 0)
            fprintf(stderr, "Covariance %i: Gradient update, step %.4g, KLD %.4g\n", i, step, gc.evaluate_function(step));
        }

//         if (pos)
//         {
//           if (mpe_gamma < 0)
//             printf("M\n");
//           else
//             printf("L\n");
//         }
//         else
//           printf("G\n");
      }
      else
      {
        if (pos)
        {
          if (info > 0)
            fprintf(stderr, "Covariance %i: Critical point update, step 1, KLD %.4g\n", i, kld);
        }

//        printf("C\n");
      }
      // Update the covariance
      for (int j = 0; j < pool->dim(); j++)
        target_cov(j) = max(min_var, exp(cov(j) + step*cov_search_dir(j)));
      pdf->set_covariance(target_cov);
    }
//     else
//       printf("-\n");
  }
}


void kld_constrained_mixture_update(void)
{
  double avg_mixture_max_lambda = 1;
  int num_mixture_max_update = 0;
  double avg_mixture_linear_lambda = 1;
  int num_mixture_linear_update = 0;

  // Initialize Gaussian weights
  gaussian_weights.clear();
  gaussian_weights.resize(model.get_pool()->size(), 0);
  
  // Go through all the mixtures and update their components
  for (int i = 0; i < model.num_emission_pdfs(); i++)
  {
    Mixture *m = model.get_emission_pdf(i);
    Vector orig_weights;
    Vector new_weights;
    Vector weight_gamma;
    Vector weight_abs_gamma;
    Vector weight_gradient;
    orig_weights.resize(m->size());
    new_weights.resize(m->size());
    weight_gamma.resize(m->size());
    weight_gradient.resize(m->size());
    weight_abs_gamma.resize(m->size());
    bool mixture_max_update = true;

    fprintf(stderr, "Mixture %i\n", i);

    for (int j = 0; j < m->size(); j++)
    {
      orig_weights(j) = m->get_mixture_coefficient(j);
      if (opt_mode == MODE_MMI)
        weight_gamma(j) = m->get_accumulated_gamma(PDF::ML_BUF,j) -
          m->get_accumulated_gamma(PDF::MMI_BUF,j);
      else if (opt_mode == MODE_MPE)
        weight_gamma(j) = m->get_accumulated_gamma(PDF::MPE_NUM_BUF, j);
      else
        throw "Invalid optimization mode";
      // if (weight_gamma(j) <= 0) // Check the original function has a maximum
      //   mixture_max_update = false;
      weight_gradient(j) = weight_gamma(j) / orig_weights(j);
      // FIXME! Assuming Gaussian abs gamma equals mixture weight abs gamma
      Gaussian *g = dynamic_cast< Gaussian* >(m->get_base_pdf(j));
      if (opt_mode == MODE_MMI)
        weight_abs_gamma(j) = g->get_accumulated_aux_gamma(PDF::ML_BUF) +
          g->get_accumulated_aux_gamma(PDF::MMI_BUF);
      else if (opt_mode == MODE_MPE)
        weight_abs_gamma(j) = g->get_accumulated_aux_gamma(PDF::MPE_NUM_BUF);
      gaussian_weights[m->get_base_pdf_index(j)] += orig_weights(j);
    }

    double mix_ratio = mixture_criterion_kld_ratio;

    if (criterion_relative_ratio)
    {
      PDFPool *pool = model.get_pool();
      // Assuming diagonal covariance!
      double num_parameters = pool->size()*(pool->dim()*2+1);
      mix_ratio *= criterion_value*m->size()/num_parameters;
    }

    mix_ratio *= m->size();
    fprintf(stderr, "  Mixture: Ratio: %g\n", mix_ratio);

    if (mixture_max_update)
    {
      fprintf(stderr, "Critical point update\n");

      // Try critical point update
      CriticalMixtureSolver mixture_solver(orig_weights, weight_gamma,
                                           weight_abs_gamma,
                                           weight_kld_limit, mix_ratio);
      double lambda = 0;
      if (mixture_solver.evaluate_function(0) > weight_kld_limit)
      {
        //fprintf(stderr, "  Lambda = %g\n", avg_mixture_max_lambda);
        lambda = search_lambda(avg_mixture_max_lambda, weight_kld_limit,
                               mixture_solver);
      }
      if (!mixture_solver.solve_weights(lambda, new_weights))
        mixture_max_update = false;
      fprintf(stderr, "  Final lambda = %g\n", lambda);

      // Check the normalization constraint
      double norm = 0;
      for (int j = 0; j < m->size(); j++)
        norm += new_weights(j);
      if (fabs(1-norm) > 0.01)
        mixture_max_update = false;
      
      double final_kld = mixture_solver.evaluate_function(lambda);
      fprintf(stderr, "  init_k = %g\n", final_kld);
      // Check the KLD
      if (final_kld > weight_kld_limit)
      {
        fprintf(stderr, "Warning: Final mixture weight evaluation failed\n");
        mixture_max_update = false;
      }

      // Check the objective function is increased
      double d = mixture_solver.evaluate_objective_function(new_weights) - mixture_solver.evaluate_objective_function(orig_weights);
      if (d < 0)
      {
        fprintf(stderr, "Warning: Decreasing objective function %g -> %g (%g)\n", mixture_solver.evaluate_objective_function(orig_weights), mixture_solver.evaluate_objective_function(new_weights), d);
        // for (int j = 0; j < m->size(); j++)
        //   fprintf(stderr, "  %g -> %g (A: %g, G: %g, T: %g)\n", orig_weights(j), new_weights(j), weight_abs_gamma(j), weight_gamma(j),
        //           orig_weights(j)*(weight_abs_gamma(j)+weight_gamma(j))/
        //           (weight_abs_gamma(j)-weight_gamma(j)));
          
        mixture_max_update = false;
      }
      
      if (mixture_max_update)
      {
        mixture_max_objective_function += d;
        avg_mixture_max_lambda = (avg_mixture_max_lambda*num_mixture_max_update+
                                  lambda) / (num_mixture_max_update + 1);
        num_mixture_max_update++;

        if (mix_ratio > 0)
        {
          mixture_solver.set_kld_evaluation(false);
          if (mixture_solver.evaluate_function(lambda) > 0)
          {
            double old_lambda = lambda;
            lambda = search_lambda(lambda, 0, mixture_solver);
            assert( lambda >= old_lambda );
            fprintf(stderr, "  Mixture: Increasing lambda %g -> %g\n",
                    old_lambda, lambda);
            if (!mixture_solver.solve_weights(lambda, new_weights))
            {
              fprintf(stderr, "Warning: Mixture weight evaluation failed after KLD ratio\n");
              mixture_max_update = false;
            }
            else
            {
              mixture_solver.set_kld_evaluation(true);
              final_kld = mixture_solver.evaluate_function(lambda);
            }
          }
          mixture_solver.set_kld_evaluation(true);
          global_debug_flag = false;
        }
        if (mixture_max_update)
          fprintf(stderr, "Mixture KLD %.6f\n", final_kld);
      }
    }

    // EBW: Skip update if EBW equations do not produce valid a update
    if (!mixture_max_update)
    {
      fprintf(stderr, "Warning: No update\n");
      continue;
    }
    
    if (!mixture_max_update)
    {
      fprintf(stderr, "Linear update\n");
      LinearMixtureSolver mixture_solver(orig_weights, weight_gradient,
                                         mix_ratio);
      double lambda = 0;
      // For linear search lambda = 0 is meaningless
//      if (mixture_solver.evaluate_function(0) > weight_kld_limit)
      {
        lambda = search_lambda(avg_mixture_linear_lambda, weight_kld_limit,
                               mixture_solver);
        avg_mixture_linear_lambda =
          (avg_mixture_linear_lambda*num_mixture_linear_update+
           lambda) / (num_mixture_linear_update + 1);
        num_mixture_linear_update++;
      }
      mixture_solver.solve_weights(lambda, new_weights);

      if (mix_ratio > 0)
      {
        mixture_solver.set_kld_evaluation(false);
        if (mixture_solver.evaluate_function(lambda) > 0)
        {
          double old_lambda = lambda;
          lambda = search_lambda(lambda, 0, mixture_solver);
          assert( lambda >= old_lambda );
          fprintf(stderr, "  Mixture: Increasing lambda %g -> %g\n",
                  old_lambda, lambda);
          mixture_solver.solve_weights(lambda, new_weights);
        }
      }
      
      fprintf(stderr, "  Final lambda = %g\n", lambda);
      mixture_solver.set_kld_evaluation(true);
      fprintf(stderr, "Mixture KLD %.6f\n", mixture_solver.evaluate_function(lambda));
    }

    // Set the new mixture parameters
    for (int j = 0; j < m->size(); j++)
      m->set_mixture_coefficient(j, new_weights(j));
  }
}


void kld_constrained_mean_covariance_update(void)
{
  PDFPool *pool = model.get_pool();
  int dim = pool->dim();
  double avg_mean_lambda = 1;
  int num_mean_update = 0;
  double avg_cov_max_lambda = 1;
  int num_cov_max_update = 0;
  double avg_cov_linear_lambda = 1;
  int num_cov_linear_update = 0;
  
  // Assuming diagonal covariance!
  double param_ratio = (double)pool->dim()/(pool->size()*(pool->dim()*2+1.0));

  // Update means and covariances (diagonal) of all Gaussians
  for (int i = 0; i < pool->size(); i++)
  {
    Gaussian *pdf = dynamic_cast< Gaussian* >(pool->get_pdf(i));
    if (pdf == NULL)
      throw string("Only Gaussian PDFs are supported!");

    Vector mean;
    Vector cov;
    Vector mean_gradient;
    Vector cov_gradient;
    Vector target_mean;
    Vector target_cov;
    pdf->get_mean(mean);
    assert( mean.size() == dim );
    pdf->get_covariance(cov);
    assert( cov.size() == dim );
    mean_gradient.resize(dim);
    cov_gradient.resize(dim);
    target_mean.resize(dim);
    target_cov.resize(dim);

    Vector d_m1, d_m2;
    double d_gamma = 0;
    double abs_gamma = 0;

    if (opt_mode == MODE_MMI)
    {
      Vector temp;
      pdf->get_accumulated_mean(PDF::ML_BUF, d_m1);
      pdf->get_accumulated_second_moment(PDF::ML_BUF, d_m2);
      pdf->get_accumulated_mean(PDF::MMI_BUF, temp);
      Blas_Add_Mult(d_m1, -1, temp);
      pdf->get_accumulated_second_moment(PDF::MMI_BUF, temp);
      Blas_Add_Mult(d_m2, -1, temp);
      d_gamma = pdf->get_accumulated_gamma(PDF::ML_BUF) -
        pdf->get_accumulated_gamma(PDF::MMI_BUF);
      abs_gamma = pdf->get_accumulated_aux_gamma(PDF::ML_BUF) +
        pdf->get_accumulated_aux_gamma(PDF::MMI_BUF);
    }
    else if (opt_mode == MODE_MPE)
    {
      pdf->get_accumulated_mean(PDF::MPE_NUM_BUF, d_m1);
      pdf->get_accumulated_second_moment(PDF::MPE_NUM_BUF, d_m2);
      d_gamma = pdf->get_accumulated_gamma(PDF::MPE_NUM_BUF);
      abs_gamma = pdf->get_accumulated_aux_gamma(PDF::MPE_NUM_BUF);
    }
    else
      throw "Invalid optimization mode";

    double gaussian_weight = 0;
    if ((int)gaussian_weights.size() > i)
    {
      gaussian_weight = gaussian_weights[i];
      if (gaussian_weight <= 0)
        fprintf(stderr, "Warning: Invalid Gaussian weight %g\n", gaussian_weight);
    }


    if (d_gamma == 0)
    {
      fprintf(stderr,"Warning: Skipping Gaussian %i update, gamma = 0\n", i);
      continue;
    }

    //fprintf(stderr, "Gammas: %g %g %g\n", d_gamma, abs_gamma, fabs(d_gamma)/abs_gamma);

    ///////////////
    // Mean update
    ///////////////

    // Extract gradient for the mean
    for (int j = 0; j < dim; j++)
      mean_gradient(j) = (d_m1(j)-mean(j)*d_gamma)/cov(j);

    // Compute maximum KLD change based on EBW update
    // double cur_mean_kld_limit = 0;
    // for (int j = 0; j < dim; j++)
    // {
    //   double d = (d_m1(j)-mean(j)*d_gamma)/(mpe_smooth + abs_gamma);
    //   cur_mean_kld_limit += d*d/cov(j);
    // }
    // cur_mean_kld_limit = min(mean_kld_limit, cur_mean_kld_limit/2.0);
    // fprintf(stderr, "Mean %i: Maximum KLD %g\n", i, cur_mean_kld_limit);
    double cur_mean_kld_limit = mean_kld_limit;

    double min_mean_lambda = max(-d_gamma, 0.0);
    if (info > 0)
      fprintf(stderr, "Mean %i, minimum lambda limit: > %g\n", i,
              min_mean_lambda);

    CriticalMeanSolver mean_solver(mean, cov, d_gamma, d_m1);
    double lambda = min_mean_lambda;
    if (mean_solver.evaluate_function(min_mean_lambda) < cur_mean_kld_limit)
    {
      assert( d_gamma > 0 );
    }
    else
    {
      lambda = search_lambda(max(avg_mean_lambda, min_mean_lambda),
                             cur_mean_kld_limit, mean_solver);
      avg_mean_lambda = (avg_mean_lambda*num_mean_update +
                         lambda) / (num_mean_update + 1);
      num_mean_update++;
      assert( lambda >= min_mean_lambda );
    }
    mean_solver.solve_mean(lambda, target_mean);
    //fprintf(stderr, "  Final lambda = %g\n", lambda);

    if (mean_criterion_kld_ratio > 0)
    {
      double mean_ratio = mean_criterion_kld_ratio;
      if (criterion_relative_ratio)
        mean_ratio *= criterion_value*param_ratio;
      if (weighted_gaussian_kld_ratios)
        mean_ratio *= gaussian_weight;
      if (mean_ratio != mean_criterion_kld_ratio)
        fprintf(stderr, "  Mean: Ratio: %g\n", mean_ratio);
        
      MeanSolver ratio_mean_solver(mean, cov, d_gamma, abs_gamma, d_m1,
                                   mean_ratio);
      if (ratio_mean_solver.evaluate_function(lambda) > 0)
      {
        //fprintf(stderr, "Mean: KLD ratio not sufficient\n");
        double old_lambda = lambda;
        lambda = search_lambda(lambda, 0, ratio_mean_solver);
        assert( lambda >= old_lambda );
        fprintf(stderr, "  Mean: Increasing lambda %g -> %g\n", old_lambda, lambda);
        ratio_mean_solver.solve_mean(lambda, target_mean);
      }
      else
        fprintf(stderr, "  Lambda = %g\n", lambda);
    }
    else
      fprintf(stderr, "  Lambda = %g\n", lambda);
    
    pdf->set_mean(target_mean);

    if (info > 0)
    {
      Vector mean_search_dir;
      mean_search_dir.resize(dim);
      for (int j = 0; j < dim; j++)
        mean_search_dir(j) = target_mean(j) - mean(j);
      GaussianMeanKLD g(&cov, &mean_search_dir, dim);        
      fprintf(stderr, "Mean KLD %.6f\n", g.evaluate_function(1));
    }

    /////////////////////
    // Covariance update
    /////////////////////

    fprintf(stderr, "Cov %i\n", i);
    bool max_cov_update = true;

    // This check is too strict, but the effect seems to be small
    // if (d_gamma <= 0)
    //   max_cov_update = false;
    // else
    // {
    //   for (int j = 0; j < dim; j++)
    //   {
    //     // Note: Assumes d_gamma > 0 (checked above)
    //     if (d_m2(j) <= mean(j)*(2*d_m1(j)-d_gamma*mean(j)))
    //     {
    //       max_cov_update = false;
    //       break;
    //     }
    //   }
    // }

    double cov_ratio = cov_criterion_kld_ratio;
    if (criterion_relative_ratio)
      cov_ratio *= criterion_value*param_ratio;
    if (weighted_gaussian_kld_ratios)
      cov_ratio *= gaussian_weight;
    if (cov_ratio != cov_criterion_kld_ratio)  
      fprintf(stderr, "  Cov: Ratio: %g\n", cov_ratio);

    double cur_cov_kld_limit = cov_kld_limit;
    
    // Compute the KLD change based on EBW update
    // cur_cov_kld_limit = 0;
    // for (int j = 0; j < dim; j++)
    // {
    //   double d = (d_m2(j) - d_gamma*cov(j) +
    //        (mpe_smooth+abs_gamma-d_gamma)*mean(j)*mean(j)) /
    //     (mpe_smooth + abs_gamma) - target_mean(j)*target_mean(j);
    //   double new_cov = max(cov(j) + d, min_var);
    //   cur_cov_kld_limit += new_cov/cov(j) + log(cov(j)/new_cov) - 1;
    // }
    // cur_cov_kld_limit = min(cov_kld_limit, cur_cov_kld_limit/2.0);
    // fprintf(stderr, "Cov %i: Maximum KLD %g\n", i, cur_cov_kld_limit);
    
    CovSolver cov_solver(mean, cov, d_gamma, abs_gamma, d_m1, d_m2,
                         min_var, cov_ratio);

    if (max_cov_update)
    {
      // Determine lambda limits
      double min_lambda = 0;
      for (int j = 0; j < dim; j++)
      {
        double c = d_m2(j) - 2*d_m1(j)*mean(j) +
          d_gamma*mean(j)*mean(j);
        double d = 4*c/cov(j) - 2*d_gamma;
        d = d*d - 4*d_gamma*d_gamma;
        if (d > 0)
        {
          double lim1, lim2;
          lim1 = (2*d_gamma - 4*c/cov(j) - sqrt(d)) / 2;
          lim2 = (2*d_gamma - 4*c/cov(j) + sqrt(d)) / 2;
          if (lim2 > min_lambda)
            min_lambda = lim2;
        }
      }
      fprintf(stderr, "  Minimum lambda: %g\n", min_lambda);

      // Try critical point update
      double max_kld = cov_solver.evaluate_function(min_lambda);
      fprintf(stderr, "  Maximum KLD: %g\n", max_kld);

      lambda = min_lambda;
      if (max_kld > cur_cov_kld_limit)
      {
        lambda = search_lambda(max(min_lambda, avg_cov_max_lambda),
                               cur_cov_kld_limit, cov_solver);
        assert( lambda >= min_lambda );
      }
      cov_solver.solve_cov(lambda, target_cov);

      // Check the type of the critical point
      for (int j = 0; j < dim; j++)
      {
        double f_2nd = -(d_m2(j) - 2*d_m1(j)*mean(j) +
                         d_gamma*mean(j)*mean(j) -
                         d_gamma*target_cov(j)/2.0) /
          (target_cov(j)*target_cov(j)*target_cov(j));
        double k_2nd = 1/(2*target_cov(j)*target_cov(j));
        double d = f_2nd - lambda*k_2nd;
        //fprintf(stderr, "    Dim %i, hessian %g (f = %g, k = %g)\n", j, d, f_2nd, k_2nd);
        if (d >= 0)
        {
//          fprintf(stderr, "f_2nd = %g, k_2nd = %g, d = %g\n", f_2nd, k_2nd, d);
          max_cov_update = false;
        }
      }
      if (max_cov_update)
      {
        avg_cov_max_lambda = (avg_cov_max_lambda*num_cov_max_update + lambda) /
          (num_cov_max_update + 1);
        num_cov_max_update++;
      }
    }

    if (!max_cov_update)
    {
      cov_solver.set_solver(CovSolver::LINEAR);
      if (info > 0)
        fprintf(stderr, "Cov %i, gradient update\n", i);
      lambda = search_lambda(avg_cov_linear_lambda, cur_cov_kld_limit, cov_solver);
      avg_cov_linear_lambda =
        (avg_cov_linear_lambda*num_cov_linear_update + lambda) /
        (num_cov_linear_update + 1);
      num_cov_linear_update++;
      cov_solver.solve_cov(lambda, target_cov);
      fprintf(stderr, "  Final lambda = %g\n", lambda);
    }

    if (cov_ratio > 0)
    {
      cov_solver.set_evaluation(CovSolver::RATIO);
      if (cov_solver.evaluate_function(lambda) > 0)
      {
        double old_lambda = lambda;
        lambda = search_lambda(lambda, 0, cov_solver);
        assert( lambda >= old_lambda );
        fprintf(stderr, "  Cov: Increasing lambda %g -> %g\n",
                old_lambda, lambda);
        cov_solver.solve_cov(lambda, target_cov);
      }
    }
    
    pdf->set_covariance(target_cov);

    if (info > 0)
    {
      Vector cov_search_dir;
      Vector lcov;
      cov_search_dir.resize(dim);
      lcov.resize(dim);
      for (int j = 0; j < dim; j++)
      {
        lcov(j) = util::safe_log(cov(j));
        cov_search_dir(j) = util::safe_log(target_cov(j)) - lcov(j);
      }

      GaussianCovKLD gc(&lcov, &cov_search_dir, dim, min_var);
      fprintf(stderr, "Cov KLD %.6f\n", gc.evaluate_function(1));
    }
  }
}


void ebw_mixture_update(void)
{
  for (int i = 0; i < model.num_emission_pdfs(); i++)
  {
    Mixture *m = model.get_emission_pdf(i);
    vector<double> num_gamma;
    vector<double> den_gamma;
    vector<double> weights;

    num_gamma.resize(m->size());
    den_gamma.resize(m->size());
    weights.resize(m->size());
    for (int j = 0; j < (int)m->size(); j++)
    {
      Gaussian *g = dynamic_cast< Gaussian* >(m->get_base_pdf(j));
      if (opt_mode == MODE_MPE)
      {
        num_gamma[j] = (m->get_accumulated_gamma(PDF::MPE_NUM_BUF, j) +
                        g->get_accumulated_aux_gamma(PDF::MPE_NUM_BUF))/2.0;
        den_gamma[j] = (-m->get_accumulated_gamma(PDF::MPE_NUM_BUF, j) +
                        g->get_accumulated_aux_gamma(PDF::MPE_NUM_BUF))/2.0;
      }
      else
      {
        num_gamma[j] = m->get_accumulated_gamma(PDF::ML_BUF, j);
        den_gamma[j] = m->get_accumulated_gamma(PDF::MMI_BUF, j);
      }
      weights[j] = m->get_mixture_coefficient(j);
    }

    double currfval=0, oldfval=0, diff=1;

    // Iterate until convergence
    vector<double> old_weights = weights;
    int iter=0;
    while (diff > 0.00001 && iter < 1000)
    {
      iter++;
      diff = 0;

      if (m->size() == 1)
      {
        weights[0] = 1;
        break;
      }

      // Go through every mixture weight
      for (int w = 0; w < m->size(); w++)
      {
        vector<double> previous_weights = weights;
        
        // Solve a quadratic equation
        // See Povey: Frame discrimination training ... 3.3 (9)
        // Note: the equation isn't given explicitly, needs some derivation
        double a, b, c, partsum, sol1, sol2;
        // a
        a=0;
        partsum=0;
        for (int j = 0; j < m->size(); j++)
          if (w != j)
            partsum += previous_weights[j];
        if (partsum <= 0)
          continue;
        for (int j = 0; j < m->size(); j++)
          if (w != j)
            a -= den_gamma[j] * previous_weights[j] / (old_weights[j] * partsum);
        a += den_gamma[w] / old_weights[w];
        // b
        b = -a;
        for (int j = 0; j < m->size(); j++)
          b -= num_gamma[j];
        // c
        c = num_gamma[w];
        // Solve
        sol1 = (-b-sqrt(b*b-4*a*c)) / (2*a);
        sol2 = (-b+sqrt(b*b-4*a*c)) / (2*a);
        
        if (sol1 <= 0 || sol1 >= 1.0 || (sol2 > 0 && sol2 < 1.0))
        {
          fprintf(stderr, "Warning: Mixture size %i, iter %i, sol1 = %g, sol2 = %g, old = %g\n", m->size(), iter, sol1, sol2, old_weights[w]);
        }

        // Heuristics: If outside permitted region, move halfway
        if (!isnan(sol1))
        {
          if (sol1 <= 0)
            weights[w] = weights[w]/2.0;
          else if (sol1 >= 1.0)
            weights[w] = weights[w] + (1-weights[w])/2.0;
          else
            weights[w] = sol1;
          weights[w] = max(weights[w], 1e-8); // FIXME: Minimum weight
        }
        
        // Renormalize others
        double norm_m = (1-weights[w]) / partsum;
        for (int j = 0; j < m->size(); j++)
          if (j != w)
            weights[j] *= norm_m;
      }
      
      // Compute function value
      oldfval = currfval;
      currfval = 0;
      for (int w = 0; w < m->size(); w++)
        currfval += num_gamma[w] * log(weights[w]) -
          den_gamma[w] * weights[w] / old_weights[w];
      diff = fabs(oldfval-currfval);
      if (iter > 1 && oldfval > currfval)
      {
        fprintf(stderr, "Warning: Mixture size %i, iter %i, reduced function value, %g\n", m->size(), iter, currfval - oldfval);
      }
    }
    for (int j = 0; j < (int)m->size(); j++)
      m->set_mixture_coefficient(j, weights[j]);
  }
}


void cls_step(bool kldcs) //const string &in_grad_file, const string &out_grad_file)
{
  if (!kldcs)
  {
    // soft_max_mixture_cls_step();
    // separate_mean_covariance_cls_update();
    original_cls_mixture_step();
    //ebw_mixture_update();
    original_cls_mean_cov_step();
  }
  else
  {
    kld_constrained_mixture_update();
    kld_constrained_mean_covariance_update();
  }
}


int
main(int argc, char *argv[])
{
  map< string, double > sum_statistics;
  string base_file_name;
  PDF::StatisticsMode statistics_mode = 0;
  
  try {
    config("usage: clsstep [OPTION...]\n")
      ('h', "help", "", "", "display help")
      ('b', "base=BASENAME", "arg", "", "Previous base filename for model files")
      ('g', "gk=FILE", "arg", "", "Previous mixture base distributions")
      ('m', "mc=FILE", "arg", "", "Previous mixture coefficients for the states")
      ('p', "ph=FILE", "arg", "", "Previous HMM definitions")
      ('L', "list=LISTNAME", "arg must", "", "file with one statistics file per line")
      ('o', "out=BASENAME", "arg must", "", "base filename for output models")
      ('M', "mode=MODE", "arg must", "", "optimization mode (MMI or MPE)")
      ('\0', "minvar=FLOAT", "arg", "0.09", "minimum variance (default 0.09)")
      ('\0', "limit=FLOAT", "arg", "0.1", "Global KLD limit for parameter change")
      ('\0', "original-limits", "", "", "Reduced KLD limits for means and covs")
      ('\0', "weight-kld", "arg", "0.1", "KLD limit for mixture weights")
      ('\0', "mean-kld", "arg", "0.1", "KLD limit for Gaussian means")
      ('\0', "cov-kld", "arg", "0.1", "KLD limit for Gaussian covariances")
      ('\0', "kldcs", "", "", "Generalized KLD constrained search")
      ('\0', "ckratio=FLOAT", "arg", "0", "Minimum criterion change/KLD ratio")
      ('\0', "mixture-ratio=FLOAT", "arg", "0", "Specify change/KLD ratio for mixtures")
      ('\0', "cov-ratio=FLOAT", "arg", "0", "Specify change/KLD ratio for covariances")
      ('\0', "crel-ratio=NAME", "arg", "", "Ratios are relative to criterion NAME")
      ('\0', "weighted-ratio", "", "", "Gaussian KLD ratios weighted by mixture weights")
      ('s', "savesum=FILE", "arg", "", "save summary information")
      ('\0', "no-write", "", "", "Don't write anything")
      ('i', "info=INT", "arg", "0", "info level")
      ;
    config.default_parse(argc, argv);

    info = config["info"].get_int();
    out_model_name = config["out"].get_str();

    string mode_str = config["mode"].get_str();
    transform(mode_str.begin(), mode_str.end(), mode_str.begin(),
                   ::tolower);
    if (mode_str == "mmi")
    {
      opt_mode = MODE_MMI;
      statistics_mode |= (PDF_ML_STATS|PDF_MMI_STATS);
    }
    else if (mode_str == "mpe")
    {
      opt_mode = MODE_MPE;
      statistics_mode |= PDF_MPE_NUM_STATS; // And PDF_MPE_DEN_STATS?!?
    }
    else
    {
      throw string("Invalid optimization mode: ") + config["mode"].get_str();
    }
      
    // Load the previous models
    if (config["base"].specified)
    {
      model.read_all(config["base"].get_str());
      base_file_name = config["base"].get_str();
    }
    else if (config["gk"].specified && config["mc"].specified &&
             config["ph"].specified)
    {
      model.read_gk(config["gk"].get_str());
      model.read_mc(config["mc"].get_str());
      model.read_ph(config["ph"].get_str());
      base_file_name = config["gk"].get_str();
    }
    else
    {
      throw string("Must give either --base or all --gk, --mc and --ph");
    }

    // Open the list of statistics files
    ifstream filelist(config["list"].get_str().c_str());
    if (!filelist)
    {
      fprintf(stderr, "Could not open %s\n", config["list"].get_str().c_str());
      exit(1);
    }

    min_var = config["minvar"].get_float();

    // Accumulate statistics
    model.start_accumulating(statistics_mode);
    while (filelist >> statistics_file && statistics_file != " ") {
      model.accumulate_gk_from_dump(statistics_file+".gks");
      model.accumulate_mc_from_dump(statistics_file+".mcs");
      string lls_file_name = statistics_file+".lls";
      ifstream lls_file(lls_file_name.c_str());
      while (lls_file.good())
      {
        char buf[256];
        string temp;
        vector<string> fields;
        lls_file.getline(buf, 256);
        temp.assign(buf);
        str::split(&temp, ":", false, &fields, 2);
        if (fields.size() == 2)
        {
          double value = strtod(fields[1].c_str(), NULL);
          if (sum_statistics.find(fields[0]) == sum_statistics.end())
            sum_statistics[fields[0]] = value;
          else
            sum_statistics[fields[0]] = sum_statistics[fields[0]] + value;
        }
      }
      lls_file.close();
    }

    if (config["crel-ratio"].specified)
    {
      if (sum_statistics.find(config["crel-ratio"].get_str()) ==
          sum_statistics.end())
      {
        fprintf(stderr, "Invalid criterion name %s\n",
                config["crel-ratio"].get_str().c_str());
        exit(1);
      }
      criterion_relative_ratio = true;
      criterion_value = sum_statistics[config["crel-ratio"].get_str()];
      fprintf(stderr, "Using criterion relative ratio, value = %g\n",
              criterion_value);
    }
    
    if (config["limit"].get_float() <= 0 ||
        config["weight-kld"].get_float() <= 0 ||
        config["mean-kld"].get_float() <= 0 ||
        config["cov-kld"].get_float() <= 0)
    {
      fprintf(stderr, "The KLD limits must be greater than zero\n");
      exit(1);
    }

    weight_kld_limit = mean_kld_limit = cov_kld_limit = config["limit"].get_float();
    mean_criterion_kld_ratio = config["ckratio"].get_float();
    // FIXME: Optimize the ratio coefficient?
    mixture_criterion_kld_ratio = mean_criterion_kld_ratio;
    cov_criterion_kld_ratio = mean_criterion_kld_ratio;

    if (config["mixture-ratio"].specified)
      mixture_criterion_kld_ratio = config["mixture-ratio"].get_float();
    if (config["cov-ratio"].specified)
      cov_criterion_kld_ratio = config["cov-ratio"].get_float();

    weighted_gaussian_kld_ratios = config["weighted-ratio"].specified;
    
    // CLS paper version limits
    if (config["original-limits"].specified)
    {
      mean_kld_limit /= 2.0;
      cov_kld_limit /= 2.0; //4.0;
    }

    if (config["weight-kld"].specified)
      weight_kld_limit = config["weight-kld"].get_float();
    if (config["mean-kld"].specified)
      mean_kld_limit = config["mean-kld"].get_float();
    if (config["cov-kld"].specified)
      cov_kld_limit = config["cov-kld"].get_float();
    
    // Perform the optimization step
    cls_step(config["kldcs"].specified);

    if (!config["no-write"].specified)
    {
      // Write the resulting models
      model.write_all(out_model_name);
    }
    
    if (config["savesum"].specified  && !config["no-write"].specified) {
      string summary_file_name = config["savesum"].get_str();
      ofstream summary_file(summary_file_name.c_str(),
                                 ios_base::app);
      if (!summary_file)
        fprintf(stderr, "Could not open summary file: %s\n",
                summary_file_name.c_str());
      else
      {
        summary_file << base_file_name << endl;
        for (map<string, double>::const_iterator it =
               sum_statistics.begin(); it != sum_statistics.end(); it++)
        {
          summary_file << "  " << (*it).first << ": " << (*it).second <<
            endl;
        }
      }
      summary_file.close();
    }

    // Print statistics
   printf("Sum of mixture MAX objective functions: %g\n", mixture_max_objective_function);
    printf("\nSum of objective functions: %g\n", global_sum_objective);
    printf("%i mixtures below KLD limit\n", global_num_below_kld);
    printf("%i negative objective functions\n", global_num_negative_objective);
  }
  
  catch (exception &e) {
    fprintf(stderr, "exception: %s\n", e.what());
    abort();
  }
  
  catch (string &str) {
    fprintf(stderr, "exception: %s\n", str.c_str());
    abort();
  }
}
