#ifndef MODELMODULE_HH_
#define MODELMODULE_HH_

#include "ModuleConfig.hh"
#include "HmmSet.hh"
#include "LinearAlgebra.hh"
#include "Distributions.hh"
#include <iostream>


namespace aku {

/**
 * A module which performs an action on a model. This module are initiated by the SpeakerConfig like a FeatureModule
 */
class ModelModule {
  friend class ModelTransformer;

protected:
  bool m_disabled;

  virtual void load_transform() = 0;
  virtual void reset_transform() = 0;

public:
  ModelModule() {}
  virtual ~ModelModule() {};
  virtual void set_parameters(const ModuleConfig &params) = 0;
  virtual void get_parameters(ModuleConfig &params) = 0;

  virtual void enable_loading() { m_disabled = false; }
  virtual void disable_loading() {
    m_disabled = true;
    reset_transform();
  }

};

/**
 * Class responsible for loading multiple modules and loading/resetting them. Gateway to SpeakerConfig
 */
class ModelTransformer {
private:
  HmmSet *m_model;
  std::map<std::string, ModelModule*> m_module_map;
  std::vector<ModelModule*> m_module_list;
  bool m_is_reset;

  static ModelModule* get_new_module(const std::string &name, HmmSet *model);

public:
  ModelTransformer() : m_model(NULL), m_is_reset(true) {}
  ModelTransformer(HmmSet *model) : m_model(model), m_is_reset(true) {}
  ModelModule* module(const std::string &name);
  void set_model(HmmSet *model) { m_model = model; }
  bool is_reset() { return m_is_reset; }
  void reset_transforms();
  void load_transforms();

  ~ModelTransformer() {
    if(!m_is_reset) reset_transforms();
    for(unsigned int i = 0; i < m_module_list.size(); ++i) {
      if(m_module_list[i] != NULL) delete m_module_list[i];
    }
   }

};

/**
 * Constrained Mllr module. This module is responsible for applying CMLLR transform on the model.
 */
class ConstrainedMllr : public ModelModule {
  friend class MllrTrainer;

public:
  enum UnitMode { UNIT_PHONE = 0, UNIT_MIX = 1, UNIT_GAUSSIAN = 2, UNIT_NO = 3 };

private:
  HmmSet *m_model;

  UnitMode m_um;
  bool m_is_loaded;

  std::map< std::vector<std::string> , Matrix> m_trans_matrices;
  std::vector<Gaussian*> m_orig_pdfs;
  std::vector<PDF*> m_orig_cluster_pdfs;


public:
  static std::string get_name() { return "cmllr"; };

  virtual void set_parameters(const ModuleConfig &params);
  virtual void get_parameters(ModuleConfig &params);

  ConstrainedMllr(HmmSet *model) :
    m_model(model), m_um(UNIT_NO), m_is_loaded(false)
  {
    m_disabled = false;
  }

  virtual ~ConstrainedMllr()
  {
    reset_transform();
  }

  virtual void set_unit_mode(UnitMode um)
  {
    m_um = um;
  }
protected:
  virtual void load_transform();
  virtual void reset_transform();

  void add_transformation_couple(const std::vector<std::string> &comps, Matrix w)
  {
    m_trans_matrices[comps] = w;
  }

private:
  void calculate_trans_matrices(const Matrix &W, Matrix &Aa, Vector &ba) const;

  std::string concatenate(std::string const &name, int i) const;
  void convert(std::vector<std::string> const &str, std::vector<double> &fl) const;
  void create_matrix(const std::vector<double> &d, Matrix &mat, const int &n, const int &m);

public:

  class AdaptedFeatureVector : public ResetCacheInterface {
  private:
    const Matrix m_A;
    const Vector m_b;
    bool m_calculated;
    Vector m_o_adap;

  public:
    const double determinant_A;
    const double log_determinant_A;

  public:
    AdaptedFeatureVector(const Matrix &A, const Vector &b)
    : m_A(A), m_b(b), m_calculated(false), m_o_adap(b.size()),
      determinant_A(util::abs<double>(LinearAlgebra::full_matrix_determinant(A))),
      log_determinant_A(LinearAlgebra::full_matrix_log_determinant(A))
      { }

    virtual ~AdaptedFeatureVector() { }

    inline Vector& get_adapted_vector(const Vector &v)
    {
      if(!m_calculated) calculate_new_ada_vector(v);
      return m_o_adap;
    }

    inline virtual void reset_cache() { m_calculated = false; }

  private:
    void calculate_new_ada_vector(const Vector &v);

  };

  class AdaptedGaussian : public Gaussian {
  private:
    Gaussian *m_g;
    AdaptedFeatureVector &m_fv;

  public:

    AdaptedGaussian(Gaussian *g, AdaptedFeatureVector &fv) :
      m_g(g), m_fv(fv) { }
    virtual ~AdaptedGaussian() {}

    virtual double compute_likelihood(const Vector &f) const { return m_g->compute_likelihood(m_fv.get_adapted_vector(f))*m_fv.determinant_A; }
    virtual double compute_log_likelihood(const Vector &f) const { return m_g->compute_log_likelihood(m_fv.get_adapted_vector(f)) + m_fv.log_determinant_A ; }
    virtual bool is_diagonal_covariance(void) const { return m_g->is_diagonal_covariance(); }
    virtual void write(std::ostream &os) const {m_g->write(os);}
    virtual void read(std::istream &is) { m_g->read(is); }
    virtual void reset(int dim) { m_g->reset(dim); }
    virtual void start_accumulating(StatisticsMode mode) { m_g->start_accumulating(mode); }
    virtual void accumulate(double prior, const Vector &f, int accum_pos = 0) {m_g->accumulate(prior, m_fv.get_adapted_vector(f), accum_pos); }
    virtual void get_mean(Vector &mean) const { m_g->get_mean(mean); }
    virtual void get_covariance(Matrix &covariance) const { m_g->get_covariance(covariance); }
    virtual void set_mean(const Vector &mean) { m_g->set_mean(mean);}
    virtual void set_covariance(const Matrix &covariance, bool finish_statistics = true) { m_g->set_covariance(covariance, finish_statistics); }
    virtual Gaussian* copy_gaussian(void)  { return m_g->copy_gaussian(); }
    virtual double compute_likelihood_exponential(const Vector &exponential_feature) const  { throw std::string("Not implemented!"); return m_g->compute_likelihood_exponential(exponential_feature); }
    virtual double compute_log_likelihood_exponential(const Vector &exponential_feature) const { throw std::string("Not implemented!"); return m_g->compute_log_likelihood_exponential(exponential_feature); }
    virtual void accumulate_aux_gamma(double gamma, int accum_pos) { m_g->accumulate_aux_gamma(gamma, accum_pos);}
    virtual void copy_gamma_to_aux_gamma(int source, int target) { m_g->copy_gamma_to_aux_gamma(source, target); }
    virtual void dump_statistics(std::ostream &os) const { m_g->dump_statistics(os); }
    virtual void accumulate_from_dump(std::istream &is, StatisticsMode mode) { m_g->accumulate_from_dump(is, mode); }
    virtual void stop_accumulating() { m_g->stop_accumulating(); }
    virtual bool accumulated(int accum_pos) const {return m_g->accumulated(accum_pos);}
    virtual void ismooth_statistics(int source, int target, double smoothing) { m_g->ismooth_statistics(source, target, smoothing); }
    virtual void estimate_parameters(EstimationMode mode, double minvar, double covsmooth, double c1, double c2, double tau, bool ml_stats_target) { m_g->estimate_parameters(mode, minvar, covsmooth, c1, c2, tau, ml_stats_target); }
    virtual void split(Gaussian &g1, Gaussian &g2, double perturbation) const { throw std::string("Not implemented!"); }
    virtual void split(Gaussian &g2, double perturbation) { throw std::string("Not implemented!");}
    virtual void merge(double weight1, const Gaussian &m1, double weight2, const Gaussian &m2, bool finish_statistics) { throw std::string("Not implemented!"); }
    virtual void merge(const std::vector<double> &weights, const std::vector<const Gaussian*> &gaussians, bool finish_statistics) {throw std::string("Not implemented!"); }
    virtual void draw_sample(Vector &sample) {throw std::string("Not implemented!");}
    virtual bool full_stats_accumulated(int accum_pos) {return m_g->full_stats_accumulated(accum_pos);}
    virtual void set_covariance(const Vector &covariance, bool finish_statistics) {return m_g->set_covariance(covariance, finish_statistics);}
  };

  private:
    std::set<AdaptedFeatureVector*> m_feature_vecs;
};

inline void ConstrainedMllr::AdaptedFeatureVector::calculate_new_ada_vector(const Vector &v) {
  m_o_adap.copy(m_b);
  Blas_Mat_Vec_Mult(m_A, v, m_o_adap, 1.0, 1.0 );
  m_calculated = true;
}

}

#endif /* MODELMODULE_HH_ */
