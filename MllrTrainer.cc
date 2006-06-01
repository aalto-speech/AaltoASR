#include <math.h>

#include "str.hh"
#include "MllrTrainer.hh"


MllrTrainer::MllrTrainer(HmmSet &model, FeatureGenerator &feagen) :
  m_model(model),
  m_feagen(feagen)
{
  m_dim = m_model.dim();
  
  // create vector and matrix arrays
  k_array = new d_vector*[m_dim];
  G_array = new d_matrix*[m_dim];
  for (int i = 0; i < m_dim; i++)
  {
    k_array[i] = new d_vector(m_dim+1);
    G_array[i] = new d_matrix(m_dim+1, m_dim+1);
  }

  // set statistics to zero
  clear_stats();
}

MllrTrainer::~MllrTrainer()
{
  // Free allocated memory
  for (int i = 0; i < m_dim; i++)
  {
    delete k_array[i];
    delete G_array[i];
  }

  delete [] k_array;
  delete [] G_array;
}


void MllrTrainer::find_probs(HmmState *state, FeatureVec &feature)
{
  int k_index; // kernel index
  int kernelcount = (state->weights).size();
  float probs[kernelcount]; 
  float prob_sum = 0;

  // check dim

  if (m_dim != m_feagen.dim())
  {
    throw std::string("MllrTrainer: Model and feature dimensions disagree");
  }
    
  // get probabilities

  for (int k = 0; k < kernelcount; k++)
  {  
    k_index = (state->weights[k]).kernel;
    
    if (((m_model.kernel(k_index)).cov).type() != HmmCovariance::DIAGONAL)
    {
      throw std::string("MllrTrainer: Expected diagonal covariance matrix");
    }

    probs[k] = m_model.compute_kernel_likelihood(k_index, feature);
    prob_sum = prob_sum + probs[k];
    
  }
  
  if (prob_sum != 0)
  {  
    for (int k = 0; k < kernelcount; k++)
      probs[k] = probs[k]/prob_sum;
  }
  else
  {
    m_zero_prob_count++;
  }

  update_stats(state, probs, feature);
}


void MllrTrainer::calculate_transform(LinTransformModule *mllr_mod)
{
  int i, j; // rows, columns
  d_matrix A(m_dim, m_dim);
  float detA = 1;
  dense1D<int> pivots(m_dim);
  dense1D<int> pivots2(m_dim+1);
  d_matrix cofactors(m_dim, m_dim);
  d_matrix G(m_dim+1, m_dim+1);
  d_matrix G_inv(m_dim+1, m_dim+1);
  d_vector p(m_dim+1);
  d_vector w(m_dim+1);
  d_matrix trans(m_dim, m_dim+1);
  double alpha;
  
  // check that we have probabilities
  if (m_beta == 0)
  {
    throw std::string("ERROR: No probabilities. See if .seg-file empty.");
  }

  if (m_zero_prob_count > 0)
  {
    printf("Warning: There were %i zero probability sample(s)\n",
           m_zero_prob_count);
  }

  // Initialize the transform with identity matrix
  mtl::set(trans, 0);
  for (i = 0; i < m_dim; i++)
    trans(i,i+1) = 1;
  
  // calculate inverse of G
  for (i = 0; i < m_dim; i++)
  {
    copy(*(G_array[i]), G);
    lu_factor(G, pivots2);
    lu_inverse(G, pivots2, G_inv);
    copy(G_inv, *(G_array[i]));
  }

  // calculate transformation
  int row = 0;
  for (int round = 0; round < 20 * m_dim; round++)
  {
    if (row == m_dim)
      row = 0; // next iteration

    // matrix A
    detA = 1;
    for (i = 0; i < m_dim; i++)
    {
      for (j = 0; j < m_dim; j++)
	A(i, j) = trans(i, j+1);
    }
    
    // calculate cofactors
    transpose(A);
    lu_factor(A, pivots);
    lu_inverse(A, pivots, cofactors);
    
    for (i = 0; i < m_dim; i++)
      detA *= A(i, i);

    scale(cofactors, detA);
    
    // cofactor vector
    p[0] = 0;
    for (j = 0; j < m_dim; j++)
      p[j+1] = cofactors[row][j];
    
    // alpha parameter    
    alpha = calculate_alpha(*(G_array[row]), p, *(k_array[row]), m_beta);
    
    // calculate ith row of transformation matrix
    scale(p, alpha);
    add(*(k_array[row]), p);
    transpose(*(G_array[row]));
    mult(*(G_array[row]), p, w);
    for(j = 0; j < m_dim+1; j++)
      trans(row, j) = w[j];
    
    transpose(*(G_array[row]));
    
    row++; // next row 
  }


  // Accumulate the new matrix with the previous one
  d_matrix A1(m_dim, m_dim);
  d_vector b1(m_dim);
  d_matrix A2(m_dim, m_dim);
  d_vector b2(m_dim);
  const std::vector<float> *bias = mllr_mod->get_transformation_bias();
  const std::vector<float> *matrix = mllr_mod->get_transformation_matrix();
  
  for (i = 0; i < m_dim; i++)
  {
    b1[i] = (*bias)[i];
    b2[i] = trans(i, 0);
    for (j = 0; j < m_dim; j++)
    {
      A1(i, j) = (*matrix)[i*m_dim + j];
      A2(i, j) = trans(i, j + 1);
    }
  }

  // calculate A x + b =  A2 (A1 x + b1) + b2
  d_vector b(m_dim);
  mult(A2, A1, A);
  mtl::set(A,0);
  mult(A2, A1, A);
  mult(A2, b1, b); 
  add(b2, b);

  // Set the transformation
  std::vector<float> new_bias;
  std::vector<float> new_matrix;
  new_bias.resize(m_dim, 0);
  new_matrix.resize(m_dim*m_dim, 0);
  for (i = 0; i < m_dim; i++)
  {
    new_bias[i] = b2[i];
    for (j = 0; j < m_dim; j++)
    {
      new_matrix[i*m_dim+j] = A2(i, j);
    }
  }
  
  mllr_mod->set_transformation_matrix(new_matrix);
  mllr_mod->set_transformation_bias(new_bias);

  clear_stats();
}


void MllrTrainer::clear_stats()
{
  for(int i = 0; i < m_dim; i++)
  {
    mtl::set(*(k_array[i]), 0);
    mtl::set(*(G_array[i]), 0);
  }
  m_beta = 0;
  m_zero_prob_count = 0;
}

void MllrTrainer::update_stats(HmmState *state, float *probs,
                              FeatureVec &feature)
{
  typedef double d;
  int i, j; // rows, columns
  int k_index, kernelcount;
  float k_mean, k_var;

  d_matrix matrix(m_dim+1, m_dim+1); 

  // extended feature vector

  d_vector fea_vector(m_dim+1); fea_vector[0] = 1; 

  for(i = 1; i < m_dim+1; i++) fea_vector[i] = (double)feature[i-1];

  // k array

  kernelcount = (state->weights).size();

  for (int k = 0; k < kernelcount; k++)
  {
    k_index = (state->weights[k]).kernel;

    for (int i = 0; i < m_dim; i++)
    {
      // get kernel statistics
      k_mean = (m_model.kernel(k_index)).center[i];
      k_var = ((m_model.kernel(k_index)).cov).diag(i);
      
      // update k(i)
      add( scaled(fea_vector, (d)(k_mean/k_var * probs[k])), *(k_array[i]));
    }
  }

  // G array

  for (i = 0; i < m_dim+1; i++){
    for (j = 0; j < m_dim+1; j++)
      matrix[i][j] = fea_vector[i]*fea_vector[j];
  }

  for (int k = 0; k < kernelcount; k++)
  {
    k_index = (state->weights[k]).kernel;

    for(int i = 0; i < m_dim; i++)
    {
      // get kernel statistics
      k_var = ((m_model.kernel(k_index)).cov).diag(i);

      // update G(i)
      add( scaled(matrix, (d)(1/k_var * probs[k])), *(G_array[i]));
    }
  }

  for (int k = 0; k < kernelcount; k++)
    m_beta += (d)probs[k];
}

double MllrTrainer::calculate_alpha(d_matrix &Gi, d_vector &p, 
				 d_vector &k, double beta)
{

  double c2 = quadratic(p, Gi, p);
  double c1 = quadratic(p, Gi, k);

  // solve quadratic equation c2 x^2 + c1 x - beta

  double a1 = (-c1 + sqrt(c1*c1 + 4*c2*beta))/(2*c2);
  double a2 = (-c1 - sqrt(c1*c1 + 4*c2*beta))/(2*c2);

  double m1 = beta*log10(fabs(a1*c2 + c1))-(c2/2)*a1*a1;
  double m2 = beta*log10(fabs(a2*c2 + c1))-(c2/2)*a2*a2;

  // select the maximizing value

  if (m1 > m2)
    return a1;

  return a2;

}

double MllrTrainer::quadratic(d_vector &x, d_matrix &A, d_vector &y)
{

  double sum = 0;

  for (int i = 0; i < (int)A.nrows(); i++)
  {
    for (int j = 0; j < (int)A.ncols(); j++)
    {
      sum += A[i][j] * x[i] * y[j];
    }
  }
  return sum;
}
