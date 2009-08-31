#include "MllrTrainer.hh"
#include "float.h"

void
MllrTrainer::MllTrainerComponent::merge(PDFGroupModule *pgm)
{
  MllTrainerComponent *mtr = dynamic_cast<MllTrainerComponent*> (pgm);
  for (unsigned int i = 0; i < m_k.size(); ++i) {
    Blas_Add_Mult(m_k[i], 1.0, mtr->m_k[i]);
    Blas_Add_Mat_Mult(m_G[i], 1.0, mtr->m_G[i]);
  }
  m_beta += mtr->m_beta;
}

void
MllrTrainer::collect_data(double prior, HmmState *state, const FeatureVec &f)
{
  Mixture *mixture = m_model->get_emission_pdf(state->emission_pdf);
  Gaussian *gaussian;

  // make the extended feature vector [1 f(1) .. f(n)]
  Vector feature(m_model->dim() + 1);
  feature(0) = 1;
  for (int i = 0; i < m_model->dim(); ++i)
    feature(i + 1) = f[i];

  // make the matrix feature * t(feature)
  Matrix feature_feature_t(m_model->dim() + 1, m_model->dim() + 1);
  feature_feature_t = 0;
  Blas_R1_Update(feature_feature_t, feature, feature);

  // calculate the probability of each gaussian (total prob = prior)
  double probs[mixture->size()];
  double probsum = 0;
  for (int g = 0; g < mixture->size(); ++g) {
    gaussian = dynamic_cast<Gaussian*> (mixture->get_base_pdf(g));
    probs[g] = gaussian->compute_likelihood(*f.get_vector());
    probsum += probs[g];
  }

  for (int g = 0; g < mixture->size(); ++g)
    probs[g] = prior * probs[g] / probsum;

  Vector mean, covar;
  for (int g = 0; g < mixture->size(); ++g) {
    gaussian = dynamic_cast<Gaussian*> (mixture->get_base_pdf(g));
    gaussian->get_mean(mean);
    gaussian->get_covariance(covar);
    get_comp(mixture->get_base_pdf_index(g))->collect_data(probs[g], mean,
        covar, feature, feature_feature_t);
  }
}

void
MllrTrainer::calculate_transform(ConstrainedMllr *cm, double min_frames, int info)
{
  m_comp_map.merge_modules(min_frames);
  std::map<std::vector<std::string>, MllTrainerComponent*> components;
  std::map<std::vector<std::string>, MllTrainerComponent*>::iterator it;

  components = m_comp_map.get_comps_to_t_map();

  double total_frames = 0;
  for (it = components.begin(); it != components.end(); ++it) {
    total_frames += it->second->get_frame_count();
    cm->add_transformation_couple(it->first,
        it->second->calculate_transform());
  }

  if(info > 0) {
    std::cout << total_frames << " frames, " << components.size() << " transform matrices" << std::endl;
  }

  switch(m_comp_map.get_unit_mode()) {
  case RegClassTree::UNIT_GAUSSIAN:
    cm->set_unit_mode(ConstrainedMllr::UNIT_GAUSSIAN);
    break;
  case RegClassTree::UNIT_MIX:
    cm->set_unit_mode(ConstrainedMllr::UNIT_MIX);
    break;
  case RegClassTree::UNIT_PHONE:
    cm->set_unit_mode(ConstrainedMllr::UNIT_PHONE);
    break;
  case RegClassTree::UNIT_NO:
    cm->set_unit_mode(ConstrainedMllr::UNIT_NO);
    break;
  }
}

void
MllrTrainer::calculate_transform(LinTransformModule *ltm)
{
  m_comp_map.merge_modules(DBL_MAX);
  std::map<std::vector<std::string>, MllTrainerComponent*> components;

  Matrix W = m_comp_map.get_comps_to_t_map().begin()->second->calculate_transform();

  int dim = W.size(0);

  Matrix A = W(LaIndex(0, dim - 1), LaIndex(1, dim));
  Vector b(dim);
  b.ref(W.col(0));

  if(ltm->is_defined()) {
  // calculate A x + b =  A2 (A1 x + b1) + b2
    const std::vector<float> &old_bias = *(ltm->get_transformation_bias());
    const std::vector<float> &old_matrix = *(ltm->get_transformation_matrix());

    Matrix old_A(dim, dim);
    Vector old_b(dim);

    for(int i=0; i < dim; ++i) {
      old_b(i) = old_bias[i];
      for(int j=0; j < dim; ++j) {
        old_A(i,j) = old_matrix[i*dim+j];
      }
    }

    Blas_Mat_Vec_Mult(old_A, b, b);
    Blas_Add_Mult(b, 1.0, old_b);
    Blas_Mat_Mat_Mult(A, old_A, A, 1.0, 0.0);
  }



  std::vector<float> bf(dim);
  std::vector<float> Af(dim*dim);

  for(int i=0; i < dim; ++i) {
    bf[i] = b(i);
    for(int j=0; j < dim; ++j) {
      Af[i*dim+j] = A(i,j);
    }
  }
  ltm->set_transformation_bias(bf);
  ltm->set_transformation_matrix(Af);
}

void
MllrTrainer::MllTrainerComponent::collect_data(double prob,
    const Vector &mean, const Vector &covar, const Vector &feature,
    const Matrix &feature_feature_t)
{
  for (unsigned int i = 0; i < m_k.size(); ++i) {
    Blas_Add_Mult(m_k[i], mean(i) / covar(i) * prob, feature);
    Blas_Add_Mat_Mult(m_G[i], 1 / covar(i) * prob, feature_feature_t);
  }
  m_beta += prob;
}

Matrix
MllrTrainer::MllTrainerComponent::calculate_transform()
{
  int dim = m_k[0].size() - 1;

  Matrix A(dim, dim);
  Matrix trans(dim, dim + 1);

  Vector p(dim + 1);
  Vector w(dim + 1);

  double detA, alpha;
  int i = 0, j = 0;

  LaVectorLongInt pivots(dim + 1);
  Vector work(dim + 1);

  trans = 0;
  for (i = 0; i < dim; ++i)
    trans(i, i + 1) = 1;


  // calculate inverses of G
  for (i = 0; i < dim; ++i) {
    LUFactorizeIP(m_G[i], pivots);
    LaLUInverseIP(m_G[i], pivots, work);
  }

  pivots.resize(dim, 1);

  int row;
  for (int round = 0; round < 20 * dim; round++) {
    row = round % dim;

    for (i = 0; i < dim; ++i)
      for (j = 0; j < dim; ++j)
        A(j, i) = trans(i, j + 1);

    LUFactorizeIP(A, pivots);

    detA = 1;
    for (i = 0; i < dim; ++i)
      detA *= A(i, i);

    LaLUInverseIP(A, pivots, work);
    Blas_Scale(detA, A);

    p(0) = 0;
    for (i = 0; i < dim; ++i)
      p(i + 1) = A(row, i);

    alpha = calculate_alpha(m_G[row], p, m_k[row], m_beta, work);

    Blas_Scale(alpha, p);
    Blas_Add_Mult(p, 1.0, m_k[row]);
    Blas_Mat_Trans_Vec_Mult(m_G[row], p, w);

    for (i = 0; i < dim + 1; ++i)
      trans(row, i) = w(i);

  }
  return trans;
}

double
MllrTrainer::MllTrainerComponent::calculate_alpha(const Matrix &Gi,
    const Vector &p, const Vector &k, double beta, Vector &work)
{
  double c2 = get_product(p, Gi, p, work);
  double c1 = get_product(p, Gi, k, work);

  // solve quadratic equation c1 x^2 + c2 x - beta

  double a1 = (-c1 + sqrt(c1 * c1 + 4* c2 * beta)) / (2* c2 );
  double a2 = (-c1 - sqrt(c1 * c1 + 4* c2 * beta)) / (2* c2 );

  double m1 = beta * log(fabs(a1 * c2 + c1)) - (c2 / 2) * a1 * a1;
  double m2 = beta * log(fabs(a2 * c2 + c1)) - (c2 / 2) * a2 * a2;

  // select the maximizing value
  if (m1 > m2) return a1;

  return a2;

}

double
MllrTrainer::MllTrainerComponent::get_product(const Vector &x,
    const Matrix &A, const Vector &y, Vector &work)
{
  Blas_Mat_Vec_Mult(A, y, work);
  return Blas_Dot_Prod(x, work);
}
