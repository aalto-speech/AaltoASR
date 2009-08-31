#include "ModelModules.hh"
#include "RegClassTree.hh"
#include "str.hh"
#include "LinearAlgebra.hh"

#include <iostream>
#include <vector>

ModelModule*
ModelTransformer::get_new_module(const std::string &name, HmmSet *model)
{
  ModelModule *mod = NULL;
  if (name == ConstrainedMllr::get_name()) mod = new ConstrainedMllr(model);
  return mod;
}

ModelModule*
ModelTransformer::module(const std::string &name)
{
  std::map<std::string, ModelModule*>::iterator it = m_module_map.find(name);
  ModelModule *mod;
  if (it != m_module_map.end()) {
    mod = it->second;
  }
  else {
    mod = ModelTransformer::get_new_module(name, m_model);

    if (mod != NULL) {
      m_module_map[name] = mod;
      m_module_list.push_back(mod);
    }
    else
      throw std::string("unknown model module requested: ") + name;
  }

  return mod;
}

void
ModelTransformer::load_transforms()
{
  if (m_module_list.size() > 0) m_model->reset_cache();
  for (unsigned int i = 0; i < m_module_list.size(); ++i)
    m_module_list[i]->load_transform();
  m_is_reset = false;
}

void
ModelTransformer::reset_transforms()
{
  if (!m_is_reset && m_module_list.size() > 0) {
    m_model->reset_cache();
    for (int i = (m_module_list.size() - 1); i >= 0; i--)
      m_module_list[i]->reset_transform();
    m_is_reset = true;
  }
}

void
ConstrainedMllr::set_parameters(const ModuleConfig &params)
{
  m_trans_matrices.clear();
  std::string unit_mode;
  params.get("unitmode", unit_mode);
  if (unit_mode == "UNIT_NO") m_um = UNIT_NO;
  if (unit_mode == "UNIT_GAUSSIAN") m_um = UNIT_GAUSSIAN;
  if (unit_mode == "UNIT_MIX") m_um = UNIT_MIX;
  if (unit_mode == "UNIT_PHONE") m_um = UNIT_PHONE;

  unsigned int matrix_dim = m_model->dim() * (m_model->dim() + 1);
  int unit_elements;

  std::vector<std::string> parts;
  int i = 1;
  unsigned int required_size = m_um == UNIT_NO ? matrix_dim : matrix_dim + 1;

  while (params.get(concatenate("w", i), parts)) {
    if (parts.size() < required_size)
      throw std::string("ERROR: not enough elements for matrix " + concatenate("w", i));

    unit_elements = parts.size() - matrix_dim;
    std::vector<std::string> elements(parts.begin(), parts.begin() + unit_elements);

    std::vector<double> d;
    convert(std::vector<std::string>(parts.begin() + unit_elements, parts.end()), d);

    create_matrix(d, m_trans_matrices[elements], m_model->dim(), m_model->dim() + 1);
    ++i;
  }

  if(m_um == UNIT_NO && m_trans_matrices.size() > 1)
    throw std::string("ERROR: speaker can only contain one transform when UNIT_NO (global transform) is set");

}

std::string
ConstrainedMllr::concatenate(std::string const &name, int i) const
{
  std::stringstream s;
  s << name << i;
  return s.str();
}

void
ConstrainedMllr::convert(std::vector<std::string> const &str, std::vector<
    double> &d) const
{
  d.resize(str.size());
  bool ok = true;
  for (unsigned int i = 0; i < str.size(); ++i) {
    d[i] = str::str2float(&str[i], &ok);
    if (!ok) throw std::string("invalid value: ") + str[i];
  }
}

void
ConstrainedMllr::create_matrix(const std::vector<double> &d, Matrix &mat,
    const int &n, const int &m)
{
  mat.resize(n, m);
  for (int ni = 0; ni < n; ++ni)
    for (int mi = 0; mi < m; ++mi)
      mat(ni, mi) = d[ni * m + mi];
}

void
ConstrainedMllr::get_parameters(ModuleConfig &params)
{

  std::map<std::vector<std::string>, Matrix>::iterator it;
  int i = 1;

  for (it = m_trans_matrices.begin(); it != m_trans_matrices.end(); ++it) {
    std::vector<std::string> line(it->first);

    for (int n = 0; n < m_model->dim(); ++n)
      for (int m = 0; m < (m_model->dim() + 1); ++m)
        line.push_back(str::fmt(64, "%g", it->second(n, m)));

    params.set(concatenate("w", i), line);
    ++i;
  }

  switch (m_um) {
  case UNIT_GAUSSIAN:
    params.set("unitmode", "UNIT_GAUSSIAN");
    break;
  case UNIT_MIX:
    params.set("unitmode", "UNIT_MIX");
    break;
  case UNIT_PHONE:
    params.set("unitmode", "UNIT_PHONE");
    break;
  case UNIT_NO:
    params.set("unitmode", "UNIT_NO");
    break;
  }

}

void
ConstrainedMllr::load_transform()
{
  if (m_is_loaded || m_disabled) return;
  PDFPool *pool = m_model->get_pool();
  m_orig_pdfs.resize(pool->size());

  for (int i = 0; i < pool->size(); ++i) {
    Gaussian *g = dynamic_cast<Gaussian *> (pool->get_pdf(i));
    m_orig_pdfs[i] = g;
  }

  std::map<std::vector<std::string>, Matrix>::iterator it;

  std::set<int> gaussians;

  if(pool->use_clustering()) {
    std::vector<PDF*> cluster_pdfs = pool->get_cluster_centers();
    m_orig_cluster_pdfs.clear();
    m_orig_cluster_pdfs.insert(m_orig_cluster_pdfs.begin(), cluster_pdfs.begin(), cluster_pdfs.end());
  }

  for (it = m_trans_matrices.begin(); it != m_trans_matrices.end(); ++it) {
    switch (m_um) {
    case UNIT_PHONE:
      RegClassTree::UnitPhoneme::get_gaussians(m_model, it->first, gaussians);
      break;
    case UNIT_MIX:
      RegClassTree::UnitMixture::get_gaussians(m_model, it->first, gaussians);
      break;
    case UNIT_GAUSSIAN:
      RegClassTree::UnitGaussian::get_gaussians(m_model, it->first, gaussians);
      break;
    case UNIT_NO:
      RegClassTree::UnitGlobal::get_gaussians(m_model, gaussians);
      break;
    }


    Matrix &W = it->second;
    Matrix A(W(LaIndex(0, W.size(0) - 1), LaIndex(1, W.size(0))).copy());

    Vector b(W.col(0).copy());

    AdaptedFeatureVector *fv = new AdaptedFeatureVector(A, b);
    m_feature_vecs.insert(fv);
    m_model->register_reset_cache_object(fv);

    if(pool->use_clustering()) {
      std::vector<PDF*> &cluster_pdfs = pool->get_cluster_centers();
      std::vector<std::vector<int> > &cluster_to_gaussians = pool->get_cluster_to_gaussians();

      for(unsigned int i = 0; i < cluster_to_gaussians.size(); ++i) {
        if(cluster_to_gaussians[i].size() > 0 && gaussians.find(cluster_to_gaussians[i][0]) != gaussians.end()) {
          cluster_pdfs[i] = new AdaptedGaussian(dynamic_cast<Gaussian*> (cluster_pdfs[i]), *fv);
        }
      }

    }

    std::set<int>::iterator itg;
    for (itg = gaussians.begin(); itg != gaussians.end(); ++itg) {
      pool->set_pdf(*itg, new AdaptedGaussian(m_orig_pdfs[*itg], *fv));
    }
  }

  m_is_loaded = true;

}

void
ConstrainedMllr::calculate_trans_matrices(const Matrix &W, Matrix &Aa,
    Vector &ba) const
{
  Matrix A = W(LaIndex(0, W.size(0) - 1), LaIndex(1, W.size(0)));
  Vector b(W.size(0));
  b.ref(W.col(0));

  LaVectorLongInt pivots(W.size(0));

  Aa.copy(A);
  ba = 0;

  LUFactorizeIP(Aa, pivots);
  LaLUInverseIP(Aa, pivots);

  Blas_Mat_Vec_Mult(A, b, ba);
}

void
ConstrainedMllr::reset_transform()
{
  if (!m_is_loaded) return;
  PDFPool *pool = m_model->get_pool();
  for (unsigned int i = 0; i < m_orig_pdfs.size(); ++i) {
    AdaptedGaussian *a;
    a = dynamic_cast<AdaptedGaussian*> (pool->get_pdf(i));
    if (a != NULL) delete a;
    pool->set_pdf(i, m_orig_pdfs[i]);
  }


  if(pool->use_clustering()) {
    std::vector<PDF*> &cluster_pdfs = pool->get_cluster_centers();

    for(unsigned int i = 0; i < cluster_pdfs.size(); ++i) {
      AdaptedGaussian *a;
      a = dynamic_cast<AdaptedGaussian*> (cluster_pdfs[i]);
      if (a != NULL) delete a;
    }
    cluster_pdfs.clear();
    cluster_pdfs.insert(cluster_pdfs.begin(), m_orig_cluster_pdfs.begin(), m_orig_cluster_pdfs.end());
  }

  for (std::set<AdaptedFeatureVector*>::iterator it = m_feature_vecs.begin(); it != m_feature_vecs.end(); ++it ) {
    m_model->unregister_reset_cache_object(*it);
    delete *it;
  }

  m_feature_vecs.clear();
  m_orig_pdfs.clear();
  m_is_loaded = false;
}
