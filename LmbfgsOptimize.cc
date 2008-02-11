#include <fstream>
#include <string>
#include <iostream>

#include "LmbfgsOptimize.hh"


LmbfgsOptimize::LmbfgsOptimize()
{
  // Set default settings
  m_min_step = 1e-10;
  m_max_step = 1e10;
  m_func_dec_tol = 1e-4;
  m_grad_tol = 1e-2;
  m_slope_dec_tol = 0.9;
  m_max_line_search_iter = 6;
  m_max_bfgs_updates = 4;

  // Initialize optimization state
  m_num_params = 0;
  m_cur_params = NULL;
  m_prev_params = NULL;
  m_cur_gradient = NULL;
  m_prev_gradient = NULL;
  m_num_bfgs_updates = 0;
  m_bfgs_updates_x = NULL;
  m_bfgs_updates_grad = NULL;
  m_search_dir = NULL;

  m_opt_state = OSTATE_INIT;
  m_converged = false;
  m_inv_hessian_scale = 1;
}


LmbfgsOptimize::~LmbfgsOptimize()
{
  if (m_cur_params != NULL)
    delete m_cur_params;
  if (m_prev_params != NULL)
    delete m_prev_params;
  if (m_cur_gradient != NULL)
    delete m_cur_gradient;
  if (m_prev_gradient != NULL)
    delete m_prev_gradient;
  if (m_num_bfgs_updates > 0)
  {
    assert( m_bfgs_updates_x != NULL );
    assert( m_bfgs_updates_grad != NULL );
    for (int i = 0; i < m_num_bfgs_updates; i++)
    {
      delete m_bfgs_updates_x[i];
      delete m_bfgs_updates_grad[i];
    }
    delete m_bfgs_updates_x;
    delete m_bfgs_updates_grad;
  }
  if (m_search_dir != NULL)
    delete m_search_dir;
}


void
LmbfgsOptimize::set_function_value(double value)
{
  m_func_val = value;
}

void
LmbfgsOptimize::set_parameters(const Vector &parameters)
{
  assert( m_num_params == 0 );
  assert( m_cur_params == NULL );
  m_cur_params = new Vector(parameters);
  m_num_params = parameters.size();
}

void
LmbfgsOptimize::set_gradient(const Vector &gradient)
{
  assert( m_num_params > 0 );
  assert( m_num_params == gradient.size() );
  assert( m_cur_gradient == NULL );
  m_cur_gradient = new Vector(gradient);
}

void
LmbfgsOptimize::set_inv_hessian_scale(double inv_hes)
{
  assert( inv_hes > 0 );
  m_inv_hessian_scale = inv_hes;
}

void
LmbfgsOptimize::get_parameters(Vector &parameters)
{
  parameters.copy(*m_cur_params);
}


// Loads optimization state and parameters
// Returns false if the state could not be read
bool
LmbfgsOptimize::load_optimization_state(std::string &filename)
{
  FILE *fp;
  if ((fp=fopen(filename.c_str(),"rb")) == NULL)
    return false;
  if (fread(&m_num_params, sizeof(int), 1, fp) < 1 ||
      fread(&m_opt_state, sizeof(OPTSTATE), 1, fp) < 1 ||
      fread(&m_inv_hessian_scale, sizeof(double), 1, fp) < 1 ||
      fread(&m_num_bfgs_updates, sizeof(int), 1, fp) < 1 ||
      fread(&m_cur_line_search_iter, sizeof(int), 1, fp) < 1 ||
      fread(&m_search_init_val, sizeof(double), 1, fp) < 1 ||
      fread(&m_prev_val, sizeof(double), 1, fp) < 1 ||
      fread(&m_search_init_slope, sizeof(double), 1, fp) < 1 ||
      fread(&m_cur_mu, sizeof(double), 1, fp) < 1 ||
      fread(&m_prev_mu, sizeof(double), 1, fp) < 1 ||
      fread(&m_min_mu, sizeof(double), 1, fp) < 1 ||
      fread(&m_max_mu, sizeof(double), 1, fp) < 1 ||
      fread(&m_bracket_mu_low, sizeof(double), 1, fp) < 1 ||
      fread(&m_bracket_mu_diff, sizeof(double), 1, fp) < 1 ||
      fread(&m_bracket_mu_incr, sizeof(double), 1, fp) < 1 ||
      fread(&m_bracket_val_low, sizeof(double), 1, fp) < 1 ||
      fread(&m_bracket_val_high, sizeof(double), 1, fp) < 1)
    return false;
  if (m_num_bfgs_updates > m_max_bfgs_updates)
    throw std::string("Maximum number of BFGS updates is too low!");
  m_cur_params = new Vector(m_num_params);
  read_vector(fp, m_cur_params);
  int flag;
  if (fread(&flag, sizeof(flag), 1, fp) < 1)
    return false;
  if (flag)
  {
    m_prev_params = new Vector(m_num_params);
    m_prev_gradient = new Vector(m_num_params);
    if (!read_vector(fp, m_prev_params) ||
        !read_vector(fp, m_prev_gradient))
      return false;
  }
  m_search_dir = new Vector(m_num_params);
  if (!read_vector(fp, m_search_dir))
    return false;
  
  m_bfgs_updates_x = new Vector*[m_max_bfgs_updates];
  m_bfgs_updates_grad = new Vector*[m_max_bfgs_updates];
  m_bfgs_rho.resize(m_num_bfgs_updates);
  for (int i = 0; i < m_num_bfgs_updates; i++)
  {
    m_bfgs_updates_x[i] = new Vector(m_num_params);
    m_bfgs_updates_grad[i] = new Vector(m_num_params);
    if (!read_vector(fp, m_bfgs_updates_x[i]) ||
        !read_vector(fp, m_bfgs_updates_grad[i]) ||
        fread(&m_bfgs_rho[i], sizeof(double), 1, fp) < 1)
      return false;
  }
  return true;
}


void
LmbfgsOptimize::write_optimization_state(std::string &filename)
{
  FILE *fp;
  int flag;
  if ((fp=fopen(filename.c_str(), "wb")) == NULL)
    throw std::string("Could not open file ") + filename + std::string(" for writing");
  fwrite(&m_num_params, sizeof(int), 1, fp);
  fwrite(&m_opt_state, sizeof(OPTSTATE), 1, fp);
  fwrite(&m_inv_hessian_scale, sizeof(double), 1, fp);
  fwrite(&m_num_bfgs_updates, sizeof(int), 1, fp);
  fwrite(&m_cur_line_search_iter, sizeof(int), 1, fp);
  fwrite(&m_search_init_val, sizeof(double), 1, fp);
  fwrite(&m_prev_val, sizeof(double), 1, fp);
  fwrite(&m_search_init_slope, sizeof(double), 1, fp);
  fwrite(&m_cur_mu, sizeof(double), 1, fp);
  fwrite(&m_prev_mu, sizeof(double), 1, fp);
  fwrite(&m_min_mu, sizeof(double), 1, fp);
  fwrite(&m_max_mu, sizeof(double), 1, fp);
  fwrite(&m_bracket_mu_low, sizeof(double), 1, fp);
  fwrite(&m_bracket_mu_diff, sizeof(double), 1, fp);
  fwrite(&m_bracket_mu_incr, sizeof(double), 1, fp);
  fwrite(&m_bracket_val_low, sizeof(double), 1, fp);
  fwrite(&m_bracket_val_high, sizeof(double), 1, fp);
  write_vector(fp, m_cur_params);
  flag = (m_prev_params == NULL ? 0 : 1 );
  fwrite(&flag, sizeof(flag), 1, fp);
  if (flag)
  {
    write_vector(fp, m_prev_params);
    write_vector(fp, m_prev_gradient);
  }
  write_vector(fp, m_search_dir);
  for (int i = 0; i < m_num_bfgs_updates; i++)
  {
    write_vector(fp, m_bfgs_updates_x[i]);
    write_vector(fp, m_bfgs_updates_grad[i]);
    fwrite(&m_bfgs_rho[i], sizeof(double), 1, fp);
  }
}


// The main function to perform an optimization step
void
LmbfgsOptimize::optimization_step(void)
{
  bool line_search_finished = false;
  
  if (stopping_test())
  {
    m_converged = true;
    return;
  }

  if (m_verbosity > 0)
    fprintf(stderr, "Current function value: %g\n", m_func_val);

  if (m_opt_state != OSTATE_INIT)
  {
    // Check if line search has finished to an acceptable point
    if (m_func_val <=
        m_search_init_val + m_func_dec_tol*m_cur_mu*m_search_init_slope)
    {
      // Acceptable decrease in function value
      double slope = Blas_Dot_Prod(*m_cur_gradient, *m_search_dir);
      if (slope < m_slope_dec_tol*m_search_init_slope)
      {
        // The slope has not increased sufficiently
        if (m_opt_state == OSTATE_LINE_FIRST ||
            m_opt_state == OSTATE_LINE_INCREASE)
        {
          if (m_cur_mu < 0.989*m_max_mu)
          {
            m_prev_mu = m_cur_mu;
            m_cur_mu *= 2;
            if (m_cur_mu >= m_max_mu)
              m_cur_mu = 0.99*m_max_mu;
            if (m_verbosity > 1)
              fprintf(stderr, "Line Search: Slope did not increase, increasing mu %g -> %g\n", m_prev_mu, m_cur_mu);
          }
          else
          {
            // Maximum step, line search failed
            if (m_verbosity > 0)
              fprintf(stderr, "Line search failed, maximum step taken\n");
            exit(1);
          }
          m_opt_state = OSTATE_LINE_INCREASE;
        }
        else if (m_opt_state == OSTATE_LINE_BACKTRACKED)
        {
          init_bracket();
          m_opt_state = OSTATE_LINE_BRACKET;
        }
      }
      else
      {
        line_search_finished = true;
      }
    }
    else if (m_opt_state != OSTATE_LINE_BRACKET)
    {
      // Function value did not decrease sufficiently
      if (m_opt_state == OSTATE_LINE_FIRST)
      {
        // Quadratic backtrack on the first step
        double mu_temp = -m_search_init_slope*m_cur_mu*m_cur_mu/
          (2*(m_func_val-m_search_init_val-m_cur_mu*m_search_init_slope));
        m_prev_mu = m_cur_mu;
        m_cur_mu = std::max(0.1*m_cur_mu,
                            std::max(m_min_mu,std::min(0.5*m_cur_mu,mu_temp)));
        if (m_verbosity > 1)
          fprintf(stderr, "Line Search: Quadratic backtrack, changing mu %g -> %g\n", m_prev_mu, m_cur_mu);
        m_opt_state = OSTATE_LINE_BACKTRACKED;
      }
      else if (m_opt_state == OSTATE_LINE_BACKTRACKED)
      {
        // Cubic backtrack
        if (m_verbosity > 1)
        {
          fprintf(stderr, "Line Search: Cubic backtrack\n");
          fprintf(stderr, "func_val = %g, prev_val = %g\n",
                  m_func_val, m_prev_val);
          fprintf(stderr, "cur_mu = %g, prev_mu = %g\n", m_cur_mu, m_prev_mu);
          fprintf(stderr, "search_init_val = %g, search_init_slope = %g\n",
                  m_search_init_val, m_search_init_slope);
        }
        double mu_temp;
        double t1 = m_func_val-m_search_init_val-m_cur_mu*m_search_init_slope;
        double t2 = m_prev_val-m_search_init_val-m_prev_mu*m_search_init_slope;
        if (fabs(m_cur_mu-m_prev_mu) < m_min_step ||
            fabs(m_cur_mu*m_cur_mu) < m_min_step ||
            fabs(m_prev_mu*m_prev_mu) < m_min_step)
        {
          if (m_verbosity > 1)
            fprintf(stderr, "mu fallback 1: %g, %g, %g\n",
                    m_cur_mu-m_prev_mu, m_cur_mu*m_cur_mu,m_prev_mu*m_prev_mu);
          mu_temp = 0.5*m_cur_mu;
        }
        else
        {
          double t3, v1, v2;
          double a, b, disc;
          t3 = 1.0/(m_cur_mu-m_prev_mu);
          v1 = t1/(m_cur_mu*m_cur_mu);
          v2 = t2/(m_prev_mu*m_prev_mu);
          a = t3*(v1-v2);
          b = t3*(m_cur_mu*v2-m_prev_mu*v1);
          disc = b*b - 3.0*a*m_search_init_slope;
          if (disc < 0)
          {
            mu_temp = 0.5*m_cur_mu;
            if (m_verbosity > 1)
              fprintf(stderr, "mu fallback 2: %g, mu = %g\n", b, mu_temp);
          }
          else if (fabs(a) < m_min_step)
          {
            mu_temp = -m_search_init_slope / (2.0*b);
            if (m_verbosity > 1)
              fprintf(stderr, "mu fallback 3: %g, mu = %g\n", a, mu_temp);
          }
          else
          {
            mu_temp = (sqrt(disc) - b) / (3.0*a);
            if (6*a*mu_temp+2*b < 0)
            {
              if (m_verbosity > 1)
                fprintf(stderr, "Changing solution from %g\n", mu_temp);
              mu_temp = (-sqrt(disc) - b) / (3.0*a);
            }
            if (m_verbosity > 1)
              fprintf(stderr, "Normal mu: %g\n", mu_temp);
          }
        }
        m_prev_mu = m_cur_mu;
        m_cur_mu = std::max(0.1*m_cur_mu,
                            std::max(m_min_mu,std::min(0.5*m_cur_mu,mu_temp)));
        if (m_verbosity > 1)
          fprintf(stderr, "Line Search: Cubic backtrack, changing mu %g -> %g\n",
                  m_prev_mu, m_cur_mu);
      }
      else if (m_opt_state == OSTATE_LINE_INCREASE)
      {
        init_bracket();
        m_opt_state = OSTATE_LINE_BRACKET;
      }
    }
    
    if (!line_search_finished && m_opt_state == OSTATE_LINE_BRACKET)
    {
      double slope = Blas_Dot_Prod(*m_cur_gradient, *m_search_dir);
      if (m_bracket_mu_diff < 0) // The first iteration of bracketing
        m_bracket_mu_diff = fabs(m_cur_mu - m_prev_mu);
      else
      {
        if (m_func_val >
            m_search_init_val + m_func_dec_tol*m_cur_mu*m_search_init_slope)
        {
          m_bracket_mu_diff = m_bracket_mu_incr;
          m_bracket_val_high = m_func_val;
        }
        else
        {
          m_bracket_mu_low = m_cur_mu;
          m_bracket_mu_diff = m_bracket_mu_diff - m_bracket_mu_incr;
          m_bracket_val_low = m_func_val;
        }
      }

      if (m_bracket_mu_diff < m_min_mu)
      {
        // Line search failed
        if (m_verbosity > 0)
          fprintf(stderr, "Line search failed, bracketing did not converge\n");
        exit(1);
      }
      
      double tmp =
        2.0*(m_bracket_val_high-(m_bracket_val_low+slope*m_bracket_mu_diff));
      if (fabs(tmp) < m_min_step)
      {
        m_bracket_mu_incr = 0.2*m_bracket_mu_diff;
      }
      else
      {
        m_bracket_mu_incr =
          std::max(0.2*m_bracket_mu_diff,
                   -slope*m_bracket_mu_diff*m_bracket_mu_diff/tmp);
      }

      m_prev_mu = m_cur_mu;
      m_cur_mu = m_bracket_mu_low + m_bracket_mu_incr;
      if (m_verbosity > 1)
        fprintf(stderr, "Line Search: Bracketing [%g, %g], mu %g -> %g\n",
                m_bracket_mu_low, m_bracket_mu_low+m_bracket_mu_diff,
                m_prev_mu, m_cur_mu);
    }

    if (!line_search_finished)
    {
      if (m_cur_line_search_iter >= m_max_line_search_iter)
      {
        if (m_verbosity > 0)
          fprintf(stderr, "Line search failed, maximum number of iterations reached\n");
        exit(1);
      }

      // Update the model
      m_cur_params->copy(*m_prev_params);
      Blas_Add_Mult(*m_cur_params, m_cur_mu, *m_search_dir);
      m_prev_val = m_func_val;
      m_cur_line_search_iter++;
    }
  }

  if (m_opt_state == OSTATE_INIT || line_search_finished)
  {
    if (m_opt_state != OSTATE_INIT)
      update_bfgs();
    compute_search_direction();
    m_opt_state = OSTATE_LINE_FIRST;
    // Update the model
    if (m_prev_params != NULL)
      delete m_prev_params;
    m_prev_params = new Vector(*m_cur_params);
    if (m_prev_gradient != NULL)
      delete m_prev_gradient;
    m_prev_gradient = new Vector(*m_cur_gradient);
    Blas_Add_Mult(*m_cur_params, m_cur_mu, *m_search_dir);
    m_cur_line_search_iter = 1;
  }
}


bool
LmbfgsOptimize::stopping_test(void)
{
  double gnorm = Blas_Norm2(*m_cur_gradient);
  double xnorm = Blas_Norm2(*m_cur_params);
  double rel_grad_norm =
    std::max(xnorm,1.0)*gnorm/std::max(abs(m_func_val),1.0);
  if (m_verbosity > 0)
    fprintf(stderr, "Relative gradient norm: %g\n", rel_grad_norm);
  if (m_num_bfgs_updates > 0 && rel_grad_norm < m_grad_tol)
  {
    return true;
  }
  return false;
}


// Updates the BFGS updates with the new gradient
void
LmbfgsOptimize::update_bfgs(void)
{
  assert( m_prev_params != NULL );
  assert( m_prev_gradient != NULL );

  if (m_bfgs_updates_x == NULL)
    m_bfgs_updates_x = new Vector*[m_max_bfgs_updates];
  if (m_bfgs_updates_grad == NULL)
    m_bfgs_updates_grad = new Vector*[m_max_bfgs_updates];

  if (m_num_bfgs_updates == m_max_bfgs_updates)
  {
    delete m_bfgs_updates_x[0];
    delete m_bfgs_updates_grad[0];
    for (int i = 0; i < m_max_bfgs_updates-1; i++)
    {
      m_bfgs_rho[i] = m_bfgs_rho[i+1];
      m_bfgs_updates_x[i] = m_bfgs_updates_x[i+1];
      m_bfgs_updates_grad[i] = m_bfgs_updates_grad[i+1];
    }
    m_bfgs_rho.pop_back();
    m_num_bfgs_updates--;
  }
  int index = m_num_bfgs_updates++;
  m_bfgs_updates_x[index] = new Vector(*m_cur_params);
  m_bfgs_updates_grad[index] = new Vector(*m_cur_gradient);
  Blas_Add_Mult(*(m_bfgs_updates_x[index]), -1.0, *m_prev_params);
  Blas_Add_Mult(*(m_bfgs_updates_grad[index]), -1.0, *m_prev_gradient);
  m_bfgs_rho.push_back(1.0/Blas_Dot_Prod(*(m_bfgs_updates_x[index]),
                                         *(m_bfgs_updates_grad[index])));
  m_inv_hessian_scale =
    1.0/(m_bfgs_rho[index]*Blas_Dot_Prod(*(m_bfgs_updates_grad[index]),
                                         *(m_bfgs_updates_grad[index])));

  if (m_verbosity > 0)
    fprintf(stderr, "Current inverse Hessian scale: %.2f\n", m_inv_hessian_scale);
}


void
LmbfgsOptimize::compute_search_direction(void)
{
  if (m_search_dir == NULL)
    m_search_dir = new Vector(m_num_params);
  // Compute the search direction using the BFGS updates
  if (m_num_bfgs_updates == 0)
  {
    Blas_Mult(*m_search_dir, -m_inv_hessian_scale, *m_cur_gradient);
  }
  else
  {
    std::vector<double> alpha;
    alpha.resize(m_num_bfgs_updates);
    m_search_dir->copy(*m_cur_gradient);
    for (int i = m_num_bfgs_updates-1; i >= 0; i--)
    {
      alpha[i] = m_bfgs_rho[i]*Blas_Dot_Prod(*(m_bfgs_updates_x[i]),
                                             *m_search_dir);
      Blas_Add_Mult(*m_search_dir, -alpha[i], *(m_bfgs_updates_grad[i]));
    }
    Blas_Scale(m_inv_hessian_scale, *m_search_dir);
    for (int i = 0; i < m_num_bfgs_updates; i++)
    {
      double beta =
        m_bfgs_rho[i]*Blas_Dot_Prod(*(m_bfgs_updates_grad[i]), *m_search_dir);
      Blas_Add_Mult(*m_search_dir, alpha[i]-beta, *(m_bfgs_updates_x[i]));
    }
    Blas_Scale(-1.0, *m_search_dir);
  }
  
  // Initialize the search step
  double step_len = Blas_Norm2(*m_search_dir);
  if (step_len < m_min_step)
  {
    if (m_verbosity > 0)
      fprintf(stderr, "Search direction ambiguous (length %g)\n", step_len);
    exit(1);
  }
  if (step_len > m_max_step)
  {
    Blas_Scale(m_max_step/step_len, *m_search_dir);
    step_len = m_max_step;
  }
  double slope = Blas_Dot_Prod(*m_cur_gradient, *m_search_dir);
  if (slope >= 0)
  {
    if (m_verbosity > 0)
      fprintf(stderr, "Slope is nonnegative (%g)!\n", slope);
    exit(1);
  }

  m_search_init_slope = slope;
  m_search_init_val = m_func_val;

  m_min_mu = m_min_step / step_len;
  m_max_mu = m_max_step / step_len;
  m_cur_mu = 1.0;
  m_cur_mu = std::min(m_cur_mu, m_max_mu);
  m_cur_mu = std::max(m_cur_mu, m_min_mu);
  if (m_verbosity > 0)
    fprintf(stderr, "Starting line search, cur_mu = %g\n", m_cur_mu);
}


void
LmbfgsOptimize::init_bracket(void)
{
  m_bracket_mu_low = std::min(m_cur_mu, m_prev_mu);
  m_bracket_mu_diff = -1; // Indicate the first iteration
  if (m_cur_mu < m_prev_mu)
  {
    m_bracket_val_low = m_func_val;
    m_bracket_val_high = m_prev_val;
  }
  else
  {
    m_bracket_val_low = m_prev_val;
    m_bracket_val_high = m_func_val;
  }
}


bool
LmbfgsOptimize::read_vector(FILE *fp, Vector *v)
{
  for (int i = 0; i < v->size(); i++)
  {
    if (fread(&((*v)(i)), sizeof(double), 1, fp) < 1)
      return false;
  }
  return true;
}


void
LmbfgsOptimize::write_vector(FILE *fp, Vector *v)
{
  for (int i = 0; i < v->size(); i++)
  {
    if (fwrite(&((*v)(i)), sizeof(double), 1, fp) < 1)
      throw std::string("Write error");
  }
}
