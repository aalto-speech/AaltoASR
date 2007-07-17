#include <math.h>

#include "str.hh"
#include "MllrTrainer.hh"


MllrTrainer::MllrTrainer(HmmSet &model, FeatureGenerator &feagen) :
  m_model(model),
  m_feagen(feagen)
{
  m_dim = m_model.dim();
  
  // create vector and matrix arrays
  k_array.resize(m_dim);
  G_array.resize(m_dim);
  for (int i = 0; i < m_dim; i++)
  {
    k_array[i] = new Vector(m_dim+1);
    G_array[i] = new Matrix(m_dim+1, m_dim+1);
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
}


void MllrTrainer::find_probs(HmmState *state, const FeatureVec &feature)
{
  // get the mixture
  Mixture *mixture = m_model.get_emission_pdf(state->emission_pdf);
  int gaussiancount = mixture->size();

  double probs[gaussiancount]; 
  double prob_sum = 0;
  
  // check dim
  if (m_dim != m_feagen.dim())
    throw std::string("MllrTrainer: Model and feature dimensions disagree");
  
  // get probabilities
  for (int g = 0; g < gaussiancount; g++)
  {  
    DiagonalGaussian *gaussian =
      dynamic_cast< DiagonalGaussian* > (mixture->get_base_pdf(g));
    if (gaussian == NULL)
      throw std::string("Warning: MLLR not supported for non-diagonal models");
    
    probs[g] = gaussian->compute_likelihood(feature);
    prob_sum += probs[g];
  }
  
  if (prob_sum != 0)
    for (int g = 0; g < gaussiancount; g++)
      probs[g] = probs[g]/prob_sum;
  else
    m_zero_prob_count++;
  
  update_stats(state, probs, feature);
}

void MllrTrainer::calculate_transform(LinTransformModule *mllr_mod)
{
  int i, j; // rows, columns
  Matrix A(m_dim, m_dim);
  double detA = 1;
  LaVectorLongInt pivots(m_dim);
  LaVectorLongInt pivots2(m_dim+1);
  Matrix cofactors(m_dim, m_dim);
  Matrix G(m_dim+1, m_dim+1);
  Matrix G_inv(m_dim+1, m_dim+1);
  Vector p(m_dim+1);
  Vector w(m_dim+1);
  Matrix trans(m_dim, m_dim+1);
  LaGenMatDouble identity = LaGenMatDouble::eye(m_dim);
  LaGenMatDouble identity_extended = LaGenMatDouble::eye(m_dim+1);
  double alpha;
  
  // check that we have probabilities
  if (m_beta == 0)
    throw std::string("ERROR: No probabilities. See if .seg-file empty.");

  if (m_zero_prob_count > 0)
    printf("Warning: There were %i zero probability sample(s)\n",
           m_zero_prob_count);
  
  // Initialize the transform with identity matrix
  trans=0;
  for (i = 0; i < m_dim; i++)
    trans(i,i+1) = 1;
  
  // calculate inverse of G
  for (i = 0; i < m_dim; i++)
  {
    G.copy(*(G_array[i]));
    LUFactorizeIP(G, pivots2);
    LaLUInverseIP(G, pivots2);
    (*(G_array[i])).copy(G);
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

    // transpose A
    Matrix temp_matrix(A);
    Blas_Mat_Trans_Mat_Mult(temp_matrix, identity, A);
    cofactors.copy(A);
    LUFactorizeIP(cofactors, pivots);

//    for (i = 0; i < m_dim; i++)
//      detA *= A(i, i);
    for (i = 0; i < m_dim; i++)
      detA *= cofactors(i, i);

    LaLUInverseIP(cofactors, pivots); 
    Blas_Scale(detA, cofactors);

    // cofactor vector
    p(0) = 0;
    for (j = 0; j < m_dim; j++)
      p(j+1) = cofactors(row,j);
    
    // alpha parameter
    alpha = calculate_alpha(*(G_array[row]), p, *(k_array[row]), m_beta);    
    
    // calculate ith row of transformation matrix
    Blas_Scale(alpha, p);
    Blas_Add_Mult(p, 1.0, *(k_array[row]));
    temp_matrix.copy(*(G_array[row]));
    Blas_Mat_Trans_Mat_Mult(temp_matrix, identity_extended, *(G_array[row]));

    Blas_Mat_Vec_Mult(*(G_array[row]), p, w);

    for(j = 0; j < m_dim+1; j++)
      trans(row, j) = w(j);
    
    temp_matrix.copy(*(G_array[row]));
    Blas_Mat_Trans_Mat_Mult(temp_matrix, identity_extended, *(G_array[row]));
    
    row++; // next row 
  }


  // Accumulate the new matrix with the previous one
  Matrix A1(m_dim, m_dim);
  Vector b1(m_dim);
  Matrix A2(m_dim, m_dim);
  Vector b2(m_dim);
  const std::vector<float> *bias = mllr_mod->get_transformation_bias();
  const std::vector<float> *matrix = mllr_mod->get_transformation_matrix();
  
  for (i = 0; i < m_dim; i++)
  {
    b1(i) = (*bias)[i];
    b2(i) = trans(i, 0);
    for (j = 0; j < m_dim; j++)
    {
      A1(i, j) = (*matrix)[i*m_dim + j];
      A2(i, j) = trans(i, j + 1);
    }
  }

  // calculate A x + b =  A2 (A1 x + b1) + b2
  Vector b(m_dim);
  b=0;
  A=0;
  Blas_Mat_Mat_Mult(A2, A1, A, 1.0, 0.0);
  Blas_Mat_Vec_Mult(A2, b1, b);
  Blas_Add_Mult(b, 1.0, b2);

  // Set the transformation
  std::vector<float> new_bias;
  std::vector<float> new_matrix;
  new_bias.resize(m_dim, 0);
  new_matrix.resize(m_dim*m_dim, 0);
  for (i = 0; i < m_dim; i++)
  {
    new_bias[i] = b(i);
    for (j = 0; j < m_dim; j++)
    {
      new_matrix[i*m_dim+j] = A(i, j);
    }
  }
  
  mllr_mod->set_transformation_matrix(new_matrix);
  mllr_mod->set_transformation_bias(new_bias);

  clear_stats();
}

void MllrTrainer::restore_identity(LinTransformModule *mllr_mod)
{
  std::vector<float> new_bias;
  std::vector<float> new_matrix;
  int dim = mllr_mod->dim();
  new_bias.resize(dim, 0);
  new_matrix.resize(dim * dim, 0);
  for (int i = 0; i < dim; i++)
  {
    new_bias[i] = 0;
    new_matrix[i*dim+i] = 1;
  }
  mllr_mod->set_transformation_matrix(new_matrix);
  mllr_mod->set_transformation_bias(new_bias);
}


void MllrTrainer::clear_stats()
{
  for(int i = 0; i < m_dim; i++)
  {
    *(k_array[i]) = 0;
    *(G_array[i]) = 0;
  }
  m_beta = 0;
  m_zero_prob_count = 0;
}

void MllrTrainer::update_stats(HmmState *state, double *probs,
                              const FeatureVec &feature)
{
  int i; // rows, columns
  int gaussiancount;

  Matrix matrix(m_dim+1, m_dim+1); 

  // extended feature vector
  Vector fea_vector(m_dim+1); fea_vector(0) = 1;
  for (i = 1; i < m_dim+1; i++) fea_vector(i) = feature[i-1];
  
  // k array
  Mixture *mixture = m_model.get_emission_pdf(state->emission_pdf);
  gaussiancount = mixture->size();
    
  for (int g = 0; g < gaussiancount; g++)
  {
    DiagonalGaussian *gaussian =
      dynamic_cast< DiagonalGaussian* > (mixture->get_base_pdf(g));
    if (gaussian == NULL)
      throw std::string("Warning: MLLR not supported for non-diagonal models");

    // get gaussian statistics
    Vector g_mean;
    Vector g_var;
    gaussian->get_mean(g_mean);
    gaussian->get_covariance(g_var);
    
    // update k(i)
    for (int i = 0; i < m_dim; i++)
      Blas_Add_Mult(*(k_array[i]), g_mean(i)/g_var(i)*probs[g], fea_vector);
  }
  
  // G array
  matrix=0;
  Blas_R1_Update(matrix, fea_vector, fea_vector);

  for (int g = 0; g < gaussiancount; g++)
  {
    DiagonalGaussian *gaussian =
      dynamic_cast< DiagonalGaussian* > (mixture->get_base_pdf(g));
    if (gaussian == NULL)
      fprintf(stderr, "Warning: MLLR not supported for non-diagonal models");

    // get gaussian statistics
    Vector g_var;
    gaussian->get_covariance(g_var);
    
    for (int i = 0; i < m_dim; i++)
      Blas_Add_Mat_Mult(*(G_array[i]), 1/g_var(i) * probs[g], matrix);
  }

  for (int g = 0; g < gaussiancount; g++)
    m_beta += probs[g];
}

double MllrTrainer::calculate_alpha(Matrix &Gi, Vector &p, 
                                    Vector &k, double beta)
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

double MllrTrainer::quadratic(Vector &x, Matrix &A, Vector &y)
{
  double sum = 0;
  
  for (int i = 0; i < (int)A.rows(); i++)
  {
    for (int j = 0; j < (int)A.cols(); j++)
    {
      sum += A(i,j) * x(i) * y(j);
    }
  }
  return sum;
}
