#ifndef LMBFGSOPTIMIZE_HH
#define LMBFGSOPTIMIZE_HH

#include <vector>

#include "LinearAlgebra.hh"


class LmbfgsLimitInterface {
public:
  virtual ~LmbfgsLimitInterface() { };
  virtual void limit_search_direction(const Vector *params,
                                      Vector *search_dir) = 0;
  virtual double limit_search_step(const Vector *params, double step) = 0;
};


class LmbfgsOptimize {
public:
  LmbfgsOptimize();
  ~LmbfgsOptimize();
  
  void set_function_value(double value);
  void set_parameters(const Vector &parameters);
  void set_gradient(const Vector &gradient);
  void set_inv_hessian_scale(double inv_hes);
  void set_init_diag_inv_hessian(const Vector &inv_hes_vect);

  void set_limit_interface(LmbfgsLimitInterface *li) { m_limit_callback = li; }

  int get_num_parameters(void) { return m_num_params; }
  void get_parameters(Vector &parameters);

  void set_max_bfgs_updates(int max_updates) { m_max_bfgs_updates = max_updates; }
  void set_verbosity(int verbosity) { m_verbosity = verbosity; }
  
  bool load_optimization_state(std::string &filename);
  void write_optimization_state(std::string &filename);
  void optimization_step(void);

  bool converged(void) { return m_converged; }

private:
  bool stopping_test(void);
  void update_bfgs(void);
  void compute_search_direction(void);
  void init_bracket(void);
  bool read_vector(FILE *fp, Vector *v);
  void write_vector(FILE *fp, Vector *v);
    
private:
  double m_inv_hessian_scale;
  int m_max_bfgs_updates;
  
  // Settings
  double m_min_step;
  double m_max_step;
  double m_func_dec_tol;
  double m_grad_tol;
  double m_slope_dec_tol;
  int m_max_line_search_iter;

  bool m_converged;

  LmbfgsLimitInterface *m_limit_callback;


  // Optimization state
  int m_num_params;
  Vector *m_cur_params;
  Vector *m_prev_params;
  Vector *m_cur_gradient;
  Vector *m_prev_gradient;
  Vector **m_bfgs_updates_x;
  Vector **m_bfgs_updates_grad;
  Vector *m_search_dir;
  Vector *m_init_inv_hessian_diag_vect;
  
  std::vector<double> m_bfgs_rho;
  int m_num_bfgs_updates;
  
  int m_cur_line_search_iter;
  
  double m_func_val;
  double m_search_init_val;
  double m_prev_val;
  
  double m_search_init_slope;
  
  double m_cur_mu;
  double m_prev_mu;
  double m_min_mu;
  double m_max_mu;
  
  // Bracketing variables
  double m_bracket_mu_low;
  double m_bracket_mu_diff;
  double m_bracket_mu_incr;
  double m_bracket_val_low;
  double m_bracket_val_high;

  enum OPTSTATE { OSTATE_INIT, OSTATE_LINE_FIRST, OSTATE_LINE_INCREASE,
                  OSTATE_LINE_BRACKET, OSTATE_LINE_BACKTRACKED } m_opt_state;

  int m_verbosity;
};

#endif // LMBFGSOPTIMIZE_HH
