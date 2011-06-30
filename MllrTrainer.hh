#ifndef MODELMLLRTRAINER_HH_
#define MODELMLLRTRAINER_HH_

#include "RegClassTree.hh"
#include "LinearAlgebra.hh"
#include "ModelModules.hh"
#include "FeatureModules.hh"


namespace aku {

class MllrTrainer {

public:
  void collect_data(double prior, HmmState *state, const FeatureVec &f);
  void calculate_transform(ConstrainedMllr *cm, double min_frames, int info = 0);
  void calculate_transform(LinTransformModule *ltm);


  class MllTrainerComponent : public PDFGroupModule  {

  private:
    double calculate_alpha(const Matrix &Gi, const Vector &p, const Vector &k, double beta, Vector &work);
    double get_product(const Vector &x, const Matrix &A, const Vector &y, Vector &work);

  public:
    double m_beta;
    std::vector<Matrix> m_G;
    std::vector<Vector> m_k;

    MllTrainerComponent(HmmSet *model) : m_beta(0.0)
    {
      Matrix zero_m = LaGenMatDouble::zeros(model->dim() + 1, model->dim() + 1);
      Vector zero_v(model->dim() + 1);
      zero_v = 0;

      m_G.resize(model->dim(), zero_m);
      m_k.resize(model->dim(), zero_v);
    }
    ~MllTrainerComponent() { }

    virtual void merge(PDFGroupModule *pgm);
    virtual double get_frame_count() { return m_beta; }
    void collect_data(double prob, const Vector &mean, const Vector &covar, const Vector &feature, const Matrix &feature_feature_t);
    Matrix calculate_transform();
  };


private:
  TreeToModuleMap<MllTrainerComponent> m_comp_map;
  HmmSet *m_model;

  MllTrainerComponent* get_comp(int pdf_index) { return m_comp_map.get_module(pdf_index); }

public:
  MllrTrainer(RegClassTree *rtree, HmmSet *model) :
    m_comp_map(rtree, model), m_model(model)  { }
  ~MllrTrainer() { }
};

}

#endif /* MLLRTRAINER_HH */
