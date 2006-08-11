#ifndef TRIPHONESET_HH
#define TRIPHONESET_HH

#include <vector>
#include <map>

#include "FeatureBuffer.hh"

// Matrix template library stuff
#include "mtl/mtl_config.h"
#include "mtl/mtl.h"
#include "mtl/matrix.h"
#include "mtl/lu.h"

typedef mtl::matrix<double, mtl::rectangle<>, mtl::dense<>, 
		    mtl::row_major>::type MatrixD;
typedef mtl::external_vec<double> ExtVectorD;
typedef mtl::dense1D<double> VectorD;

class TriphoneSet {
public:

  struct Triphone;
  
  struct TriphoneState {
    Triphone *tri;
    int state;
    double *mean;
    MatrixD *cov;
    int count;
  };

  struct Triphone {
    bool long_phoneme;
    std::string center;
    std::string left;
    std::string right;
    std::vector<TriphoneState*> states;
  };

  struct ContextStateCluster {
    std::vector<TriphoneState*> states;
    double cov_det;
    int count;
  };
  
  struct TriphoneCluster {
    std::string center;
    // For each state (the outermost vector) a set of clustered contexts
    std::vector< std::vector<ContextStateCluster> > state_clusters;
  };
    

  struct DecisionRule {
    std::string rule_name;
    enum {PHONEME_LENGTH, LEFT_CONTEXT, RIGHT_CONTEXT} rule_type;
    std::vector<std::string> phoneme_list;
  };

public:
  TriphoneSet();
  ~TriphoneSet();
  
  void set_dimension(int dim) { m_dim = dim; }
  void set_info(int info) { m_info = info; }
  void set_min_count(int min_count) { m_min_count = min_count; }
  void set_min_likelihood_gain(double lhg) { m_min_likelihood_gain = lhg; }
  void set_length_award(double award) { m_length_award = award; }
  void set_ignore_length(bool il) { m_ignore_length = il; }
  void set_ignore_context_length(bool icl) { m_ignore_context_length = icl; }

  void add_feature(const FeatureVec &f, const std::string &left,
                   const std::string &center,
                   const std::string &right, int state_index);
  void finish_triphone_statistics(void);
  void fill_missing_contexts(bool boundary);
  void add_missing_triphone(std::string &left, std::string &center,
                            std::string &right);
  void tie_triphones(void);
  void load_rule_set(const std::string &filename);
  int save_to_basebind(const std::string &filename, int initial_statenum);

private:
  std::string get_triphone_label(const std::string &left,
                                 const std::string &center,
                                 const std::string &right) {
    return left + "-" + center + "+" + right; }
  int triphone_index(const std::string &left,
                     const std::string &center,
                     const std::string &right);
  void tie_states(std::vector<ContextStateCluster> &cncl);
  int find_best_split_rule(ContextStateCluster &states,
                           double *likelihood_gain);
  void apply_rule_to_group(ContextStateCluster &cl, int rule,
                           bool in, ContextStateCluster &out);
  void fill_context_cluster_statistics(ContextStateCluster &ccl);
  double cov_determinant(MatrixD *m);

private:
  int m_dim;
  int m_info;

  std::map<std::string, int> m_tri_map;
  std::vector<Triphone*> m_triphones;

  std::map<std::string, int> m_tri_center_map;
  std::vector<TriphoneCluster> m_tri_centers;

  std::vector<DecisionRule> m_rule_set;

  bool m_ignore_length;
  bool m_ignore_context_length;

  int m_min_count;
  double m_min_likelihood_gain;
  double m_length_award;

  int count_limit_count;
  int lh_limit_count;
};



#endif // TRIPHONESET_HH
