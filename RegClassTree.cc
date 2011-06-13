#include "RegClassTree.hh"

#include "LinearAlgebra.hh"
#include "blas1pp.h"
#include "blas2pp.h"
#include "blas3pp.h"
#include "str.hh"
#include <set>


namespace aku {

void
RegClassTree::initialize_root_node(HmmSet *model)
{
  m_dim = model->dim();
  m_root = new Node(1, m_dim);

  switch (m_unit_mode) {
  case UNIT_NO:
    m_root->m_components = UnitGlobal::get_all_components(model);
    break;
  case UNIT_PHONE:
    m_root->m_components = UnitPhoneme::get_all_components(model);
    break;
  case UNIT_MIX:
    m_root->m_components = UnitMixture::get_all_components(model);
    break;
  case UNIT_GAUSSIAN:
    m_root->m_components = UnitGaussian::get_all_components(model);
    break;
  }
}

void
RegClassTree::build_tree(int n_terminals)
{
  m_root->update_score_mean();
  std::priority_queue<Node*, std::vector<Node*>, Node::RegClassCmp> tnodes;

  tnodes.push(m_root);

  int n = 1;
  Node *cur_node;

  while (!tnodes.empty() && (n < n_terminals)) {
    cur_node = tnodes.top();
    tnodes.pop();

    if (cur_node->m_components.size() > 1) {
      split_node(cur_node);
      tnodes.push(cur_node->m_c1);
      tnodes.push(cur_node->m_c2);
      ++n;
    }
  }
}

void
RegClassTree::split_node(RegClassTree::Node *node, double iter_treshold,
    double perturbation)
{
  node->m_c1 = new Node(node->m_index * 2, m_dim);
  node->m_c2 = new Node(node->m_index * 2 + 1, m_dim);

  node->update_covar();
  node->m_c1->m_mean.copy(node->m_mean);
  node->m_c2->m_mean.copy(node->m_mean);

  Matrix cholesky;
  Vector perturbations(node->m_mean.size());

  LinearAlgebra::cholesky_factor(node->m_covar, cholesky);
  for (int i = 0; i < perturbations.size(); i++)
    perturbations(i) = perturbation;

  Blas_Mat_Vec_Mult(cholesky, perturbations, node->m_c1->m_mean, -1.0, 1.0);
  Blas_Mat_Vec_Mult(cholesky, perturbations, node->m_c2->m_mean, 1.0, 1.0);

  double old_score, new_score;
  new_score = node->m_score;
  do {

    old_score = new_score;
    node->m_c1->m_components.clear();
    node->m_c2->m_components.clear();
    for (unsigned int i = 0; i < node->m_components.size(); ++i) {
      if (node->m_c1->get_distance(node->m_components[i])
          < node->m_c2->get_distance(node->m_components[i]))
        node->m_c1->m_components.push_back(node->m_components[i]);
      else
        node->m_c2->m_components.push_back(node->m_components[i]);
    }

    node->m_c1->update_score_mean();
    node->m_c2->update_score_mean();

    new_score = node->m_c1->m_score + node->m_c2->m_score;
  } while (old_score - new_score > iter_treshold);

  node->m_components.clear();
  node->m_terminal_node = false;

}

void
RegClassTree::get_terminal_nodes(std::vector<Node*> &v)
{
  assert(m_root->m_index == 1);
  m_root->get_terminal_child_nodes(v);
}

void
RegClassTree::write(std::ostream *out) const
{
  switch (m_unit_mode) {
  case UNIT_PHONE:
    *out << "UNIT_PHONE";
    break;
  case UNIT_MIX:
    *out << "UNIT_MIX";
    break;
  case UNIT_GAUSSIAN:
    *out << "UNIT_GAUSSIAN";
    break;
  case UNIT_NO:
    *out << "UNIT_NO";
    break;
  }
  *out << " " << m_dim << std::endl;
  m_root->write(out);
}

void
RegClassTree::read(std::istream *in, HmmSet *model)
{
  std::string str;
  std::getline(*in, str, ' ');

  *in >> m_dim;
  m_root = new Node(1, m_dim);

  if (str.compare("UNIT_PHONE") == 0) {
    m_unit_mode = UNIT_PHONE;
  }
  else if (str.compare("UNIT_MIX") == 0) {
    m_unit_mode = UNIT_MIX;
  }
  else if (str.compare("UNIT_GAUSSIAN") == 0) {
    m_unit_mode = UNIT_GAUSSIAN;
  }
  else if (str.compare("UNIT_NO") == 0) {
    m_unit_mode = UNIT_NO;
    m_root->m_components.push_back(new UnitGlobal());
    return;
  }
  else {
    throw std::string("Not a valid regression tree file");
  }


  int index;
  unsigned int count;

  Node* node;
  Unit* component;
  std::vector<std::string> components;

  while (true) {
    *in >> index;
    if (in->eof() || in->fail()) {
      break;
    }
    node = get_node(index);

    *in >> count;
    if (count > 0) {
      std::getline(*in, str);
      str::chomp(&str);
      str::clean(&str, " ");
      str::split(&str, " ", true, &components, count);
      if (components.size() != count)
        throw std::string("Error: Regression tree file; number of components does not match count");
    }

    for (unsigned int i = 0; i < count; ++i) {
      switch (m_unit_mode) {
      case UNIT_PHONE:
        component = new UnitPhoneme(components[i]);
        break;
      case UNIT_MIX:
        component = new UnitMixture(components[i]);
        break;
      case UNIT_GAUSSIAN:
        component = new UnitGaussian(components[i]);
        break;
      case UNIT_NO:
        throw std::string("Programming error, check RegClassTree.cc RegClassTree::read()");
        break;
      }
      component->initialize_from_model(model);
      node->m_components.push_back(component);
    };
  }
}

RegClassTree::~RegClassTree()
{
  delete m_root;
}

void
RegClassTree::Node::update_score_mean()
{
  Vector cur_mean;

  m_mean = 0;
  m_score = 0;
  m_total_occ = 0;

  for (unsigned int i = 0; i < m_components.size(); i++) {
    m_components[i]->get_mean(cur_mean);
    Blas_Add_Mult(m_mean, m_components[i]->get_occ(), cur_mean);
    m_total_occ += m_components[i]->get_occ();
  }

  Blas_Scale(1 / m_total_occ, m_mean);

  for (unsigned int i = 0; i < m_components.size(); i++) {
    m_score += m_components[i]->get_occ() * get_distance(m_components[i]);
  }

}

void
RegClassTree::Node::update_covar()
{
  Matrix cur_covar;
  Vector cur_mean;

  m_covar = Matrix::zeros(m_covar.size(1), m_covar.size(2));

  for (unsigned int i = 0; i < m_components.size(); i++) {
    m_components[i]->get_mean(cur_mean);
    m_components[i]->get_covar(cur_covar);
    Blas_R1_Update(cur_covar, cur_mean, cur_mean);

    Blas_Add_Mat_Mult(m_covar, m_components[i]->get_occ(), cur_covar);
  }
  Blas_Scale(1 / m_total_occ, m_covar);
  Blas_R1_Update(m_covar, m_mean, m_mean, -1.0);

}

double
RegClassTree::Node::get_distance(RegClassTree::Unit* rcu) const
{
  Vector cur_mean;
  rcu->get_mean(cur_mean);
  Blas_Add_Mult(cur_mean, -1, m_mean);
  return Blas_Norm2(cur_mean);
}

std::vector<RegClassTree::Unit*>
RegClassTree::UnitPhoneme::get_all_components(HmmSet *model)
{
  std::map<std::string, UnitPhoneme*> phone_map;
  Hmm hmm;

  for (int h = 0; h < model->num_hmms(); h++) {
    hmm = model->hmm(h);
    std::string cphone = hmm.get_center_phone();
    if (phone_map[cphone] == NULL) {
      phone_map[cphone] = new UnitPhoneme(cphone);
    }
    phone_map[cphone]->m_hmms.push_back(hmm);
  }

  std::vector<Unit*> v;
  for (std::map<std::string, UnitPhoneme*>::iterator it = phone_map.begin(); it
      != phone_map.end(); ++it) {
    (*it).second->calculate_statistics(model);
    v.push_back(it->second);
  }
  return v;
}

void
RegClassTree::UnitPhoneme::initialize_from_model(HmmSet *model)
{
  for (int i = 0; i < model->num_hmms(); ++i)
    if (model->hmm(i).get_center_phone() == m_phone)
      m_hmms.push_back(model->hmm(i));
}

void
RegClassTree::UnitPhoneme::get_gaussians(HmmSet *model, const std::vector<
    std::string> &phones, std::set<int> &gaussians)
{
  gaussians.clear();
  Hmm hmm;

  std::vector<int> ints;
  std::vector<double> doubles;

  for (int h = 0; h < model->num_hmms(); h++) {
    hmm = model->hmm(h);
    std::string cphone = hmm.get_center_phone();
    if (std::find(phones.begin(), phones.end(), cphone) == phones.end()) continue;

    for (int s = 0; s < hmm.num_states(); ++s) {
      Mixture *mix = model->get_emission_pdf(hmm.state(s));
      mix->get_components(ints, doubles);
      gaussians.insert(ints.begin(), ints.end());
    }
  }
}

void
RegClassTree::UnitPhoneme::gather_pdf_indices(HmmSet *model)
{

  std::vector<int> pointers;
  std::vector<double> weights;
  for (unsigned int i = 0; i < m_hmms.size(); ++i) {
    for (int j = 0; j < m_hmms[i].num_states(); ++j) {
      Mixture *mix = model->get_emission_pdf(
          model->state(m_hmms[i].state(j)).emission_pdf);

      mix->get_components(pointers, weights);
      m_pdf_indices.insert(pointers.begin(), pointers.end());
    }
  }

}

std::vector<RegClassTree::Unit*>
RegClassTree::UnitMixture::get_all_components(HmmSet *model)
{
  std::vector<RegClassTree::Unit*> mixture_comps;
  mixture_comps.reserve(model->num_emission_pdfs());
  for (int i = 0; i < model->num_emission_pdfs(); ++i) {
    mixture_comps.push_back(new UnitMixture(i, model->get_emission_pdf(i)));
  }

  for (unsigned int i = 0; i < mixture_comps.size(); ++i) {
    mixture_comps[i]->calculate_statistics(model);
  }

  return mixture_comps;

}

void
RegClassTree::UnitMixture::initialize_from_model(HmmSet *model)
{
  bool ok;
  m_mixture = model->get_emission_pdf((int) str::str2long(&m_id, &ok));

}

void
RegClassTree::UnitMixture::get_gaussians(HmmSet *model, const std::vector<
    std::string> &mixtures, std::set<int> &gaussians)
{
  gaussians.clear();
  std::vector<int> ints;
  std::vector<double> doubles;
  bool ok;
  for (unsigned int i = 0; i < mixtures.size(); ++i) {
    ok = true;
    int mixid = (int) str::str2long(&mixtures[i], &ok);
    if (!ok) continue;

    Mixture *mix = model->get_emission_pdf(mixid);
    mix->get_components(ints, doubles);
    gaussians.insert(ints.begin(), ints.end());
  }

}

void
RegClassTree::UnitMixture::gather_pdf_indices(HmmSet *model)
{

  std::vector<int> pointers;
  std::vector<double> weights;

  m_mixture->get_components(pointers, weights);
  m_pdf_indices.insert(pointers.begin(), pointers.end());

}

std::vector<RegClassTree::Unit*>
RegClassTree::UnitGaussian::get_all_components(HmmSet *model)
{
  std::vector<RegClassTree::UnitGaussian*> gaussian_comps;

  gaussian_comps.reserve(model->get_pool()->size());

  for (int i = 0; i < model->get_pool()->size(); ++i) {
    Gaussian *g = dynamic_cast <Gaussian*> (model->get_pool_pdf(i));
    gaussian_comps.push_back(new UnitGaussian(i, g ));
  }

  // collect all occupancy statistics (is the easiast here)
  std::vector<double> occs(gaussian_comps.size(), 0.0);

  for (int m = 0; m < model->num_emission_pdfs(); ++m) {
    Mixture *mix = model->get_emission_pdf(m);
    for (int g = 0; g < mix->size(); ++g) {
      occs[mix->get_base_pdf_index(g)] = mix->get_accumulated_gamma(
          Mixture::ML_BUF, g);
    }
  }

  for (unsigned int i = 0; i < gaussian_comps.size(); ++i) {
    gaussian_comps[i]->m_occ = occs[i];
    gaussian_comps[i]->calculate_statistics(model);
  }

  std::vector<RegClassTree::Unit*> unit_comps;
  unit_comps.reserve(gaussian_comps.size());
  for (unsigned int i = 0; i < gaussian_comps.size(); ++i) {
    unit_comps.push_back(gaussian_comps[i]);
  }
  return unit_comps;
}

void
RegClassTree::UnitGaussian::initialize_from_model(HmmSet *model)
{
  bool ok;
  m_g
      = dynamic_cast<Gaussian *> (model->get_pool_pdf((int) str::str2long(&m_id, &ok)));
}

void
RegClassTree::UnitGaussian::get_gaussians(HmmSet *model, const std::vector<
    std::string> &mixtures, std::set<int> &gaussians)
{
  gaussians.clear();

  bool ok;
  for (unsigned int i = 0; i < mixtures.size(); ++i) {
    gaussians.insert((int) str::str2long(&mixtures[i], &ok));
  }

}

void
RegClassTree::UnitGaussian::gather_pdf_indices(HmmSet *model)
{
  bool ok;
  m_pdf_indices.insert((int) str::str2long(&m_id, &ok));
}


std::vector<RegClassTree::Unit*>
RegClassTree::UnitGlobal::get_all_components(HmmSet *model)
{
  UnitGlobal *unit = new UnitGlobal();
  std::vector<RegClassTree::Unit*> unit_comps;
  unit_comps.push_back(unit);
  return unit_comps;
}

void
RegClassTree::UnitGlobal::get_gaussians(HmmSet *model, std::set<int> &gaussians)
{
  gaussians.clear();
  for(int i = 0; i < model->get_pool()->size(); ++i)
    gaussians.insert(i);
}

void
RegClassTree::UnitGlobal::gather_pdf_indices(HmmSet *model)
{
  get_gaussians(model, m_pdf_indices);
}

void
RegClassTree::Node::get_terminal_child_nodes(std::vector<Node*> &v)
{
  if (m_terminal_node) {
    v.push_back(this);
  }
  else {
    assert(m_c1->m_index == m_index*2);
    assert(m_c2->m_index == m_index*2+1);
    assert(m_c1 != NULL);
    assert(m_c2 != NULL);
    m_c1->get_terminal_child_nodes(v);
    m_c2->get_terminal_child_nodes(v);
  }
}

void
RegClassTree::Node::get_pdf_indices(HmmSet *model, std::set<int> &v)
{
  if (m_terminal_node) {
    for (unsigned int i = 0; i < m_components.size(); i++) {
      m_components[i]->get_pdf_indices(model, v);
    }
  }
  else {
    m_c1->get_pdf_indices(model, v);
    m_c2->get_pdf_indices(model, v);
  }
}

void
RegClassTree::Node::write(std::ostream *out, bool print_empty_nodes) const
{
  if (m_components.size() != 0 || print_empty_nodes) {
    *out << m_index << " " << m_components.size();
    for (unsigned int i = 0; i < m_components.size(); i++) {
      *out << " " << m_components[i]->get_identifier();
    }
    *out << std::endl;
  }
  if (!m_terminal_node) {
    m_c1->write(out);
    m_c2->write(out);
  }
}

void
RegClassTree::UnitPhoneme::calculate_statistics(HmmSet* model)
{
  m_mean = (Vector) Matrix::zeros(model->dim(), 1);
  m_covar = Matrix::zeros(model->dim(), model->dim());

  m_occ = 0.0;

  //gather occ (accumulated gamma) for every gaussian
  std::map<int, double> gaussian_gamma;

  Mixture* mix;
  //for every hmm
  for (unsigned int h = 0; h < m_hmms.size(); ++h) {
    //for every state
    for (int m = 0; m < m_hmms[h].num_states(); ++m) {

      //get the emission_pdf
      mix = model->get_emission_pdf(model->emission_pdf_index(
          m_hmms[h].state(m)));

      //for every gaussian, take the gamma
      for (int g = 0; g < mix->size(); ++g) {
        gaussian_gamma[mix->get_base_pdf_index(g)]
            = mix->get_accumulated_gamma(Mixture::ML_BUF, g);
      }
    }
  }

  Vector cur_mean;
  Matrix cur_covar;
  std::map<int, double>::iterator it;
  Gaussian *gaussian;
  for (it = gaussian_gamma.begin(); it != gaussian_gamma.end(); it++) {
    gaussian = dynamic_cast<Gaussian*> (model->get_pool_pdf(it->first));
    gaussian->get_mean(cur_mean);
    Blas_Add_Mult(m_mean, it->second, cur_mean);

    gaussian->get_covariance(cur_covar);
    Blas_R1_Update(cur_covar, cur_mean, cur_mean);

    Blas_Add_Mat_Mult(m_covar, it->second, cur_covar);
    m_occ += it->second;
  }

  Blas_Scale(1.0 / m_occ, m_mean);
  Blas_Scale(1.0 / m_occ, m_covar);
  Blas_R1_Update(m_covar, m_mean, m_mean, -1.0);

}

void
RegClassTree::UnitMixture::calculate_statistics(HmmSet* model)
{
  m_mean = (Vector) Matrix::zeros(model->dim(), 1);
  m_covar = Matrix::zeros(model->dim(), model->dim());

  m_occ = 0.0;

  std::map<int, double> gaussian_gamma;

  for (int g = 0; g < m_mixture->size(); ++g) {
    gaussian_gamma[m_mixture->get_base_pdf_index(g)]
        = m_mixture->get_accumulated_gamma(Mixture::ML_BUF, g);
  }

  Vector cur_mean;
  Matrix cur_covar;
  std::map<int, double>::iterator it;
  Gaussian *gaussian;
  for (it = gaussian_gamma.begin(); it != gaussian_gamma.end(); it++) {
    gaussian = dynamic_cast<Gaussian*> (model->get_pool_pdf(it->first));
    gaussian->get_mean(cur_mean);
    Blas_Add_Mult(m_mean, it->second, cur_mean);

    gaussian->get_covariance(cur_covar);
    Blas_R1_Update(cur_covar, cur_mean, cur_mean);

    Blas_Add_Mat_Mult(m_covar, it->second, cur_covar);
    m_occ += it->second;
  }

  Blas_Scale(1.0 / m_occ, m_mean);
  Blas_Scale(1.0 / m_occ, m_covar);
  Blas_R1_Update(m_covar, m_mean, m_mean, -1.0);

}

void
RegClassTree::UnitGaussian::calculate_statistics(HmmSet* model)
{
  m_g->get_mean(m_mean);
  m_g->get_covariance(m_covar);
}

RegClassTree::Node*
RegClassTree::RegClassTree::get_node(int index)
{

  int targetlevel = 1;
  //binary tree, so highest bit index is level of the requested node
  while (index >> targetlevel)
    targetlevel++;

  int curlevel = 1;
  // root must exist!
  Node* cur_node = m_root;

  //walk through the tree to the node, and create between nodes on the way
  while (curlevel != targetlevel) {
    curlevel++;
    if (cur_node->m_terminal_node) {
      cur_node->m_c1 = new Node(cur_node->m_index * 2, m_dim);
      cur_node->m_c2 = new Node(cur_node->m_index * 2 + 1, m_dim);
      cur_node->m_terminal_node = false;
    }

    //choose to go right or left based on the bit in the current level
    if (index & (1 << (targetlevel - curlevel)))
      cur_node = cur_node->m_c2;
    else
      cur_node = cur_node->m_c1;
  }

  return cur_node;
}

}
