#ifndef PHONEPOOL_HH
#define PHONEPOOL_HH

#include <set>
#include <map>
#include <vector>
#include <stdio.h>
#include "HmmSet.hh"
#include "Distributions.hh"


namespace aku {

/** A class for tying the states of context dependent phones.
 */
class PhonePool {
private:

  typedef std::set<std::string> PhoneLabelSet;

  struct DecisionRule {
    std::string rule_name;
    enum {CONTEXT} rule_type;
    PhoneLabelSet phone_set;
  };

  struct AppliedDecisionRule {
    AppliedDecisionRule(const DecisionRule *r, int c, bool a) :
      rule(r), context(c), answer(a) { }
    const DecisionRule *rule;
    int context;
    bool answer;
  };

  /** Class for a state of a context dependent phone
   */
  class ContextPhone {
  public:
    ContextPhone(const std::string &label, PhonePool *pool,
                 bool persistent = true);
    bool rule_answer(const DecisionRule *rule, int context_index);

    const std::string& label(void) const { return m_label; }

    double occupancy_count(void) const { return m_occupancy; }
    int num_left_contexts(void) const { return (int)m_left_contexts.size(); }
    int num_right_contexts(void) const { return (int)m_right_contexts.size(); }
    const FullCovarianceGaussian* statistics(void) const { return &m_stats; }

    inline void add_feature(double prior, const FeatureVec &f);
    void finish_statistics(void);
    
  private:
    std::string m_label;
    std::vector<std::string> m_left_contexts;
    std::vector<std::string> m_right_contexts;
    double m_occupancy;

    PhonePool *m_pool;

    /// Full covariance statistics for the state
    FullCovarianceGaussian m_stats;
  };
  
  typedef std::map<std::string, ContextPhone*> ContextPhoneMap;
  typedef std::set<ContextPhone*> ContextPhoneSet;
  
  /** Class for a cluster of context dependent phone states
   */
  class ContextPhoneCluster {
  public:
    ContextPhoneCluster(int dim) : m_sum_occupancy(0), m_sum_stats(dim) { }
    void fill_cluster(ContextPhoneSet &context_phones);
    void compute_statistics(void);
    double occupancy_count(void) const { return m_sum_occupancy; }
    int num_context_phones(void) const { return (int)m_contexts.size(); }
    FullCovarianceGaussian* statistics(void) {return &m_sum_stats;}
    const FullCovarianceGaussian* statistics(void) const {return &m_sum_stats;}

    double compute_new_cluster_occupancy(DecisionRule *rule, int context_index,
                                         bool answer,
                                         int *num_context_phones = NULL);
    void fill_new_cluster_context_phones(DecisionRule *rule,
                                         int context_index, bool answer,
                                         ContextPhoneSet &new_set);
    void remove_from_cluster(const ContextPhoneCluster &cl);
    void merge_clusters(const ContextPhoneCluster *cl1,
                        const ContextPhoneCluster *cl2);
    void add_rule(AppliedDecisionRule &rule);
    int num_applied_rule_sets(void) const { return (int)m_applied_rules.size(); }
    const std::vector<AppliedDecisionRule>& applied_rules(int set_index) const { return m_applied_rules[set_index]; }

    void set_state_index(int state_index) { m_state_index = state_index; }
    int state_index(void) const { return m_state_index; }

    void add_final_gaussian(double weight, int gauss_index);
    int num_final_gaussians(void) { return (int)m_gauss_index.size(); }
    int final_gauss_index(int i) { return m_gauss_index[i]; }
    double final_gauss_weight(int i) { return m_gauss_weight[i]; }

    const ContextPhoneSet& get_context_phones(void) { return m_contexts; }
    
  private:
    /** A collection of ordered vector of rules for this cluster.
     * During splitting, the first (and only) rule vector is the ordered
     * set of rules applied so far. During merging, several rule sets
     * can be defined for the same cluster.
     */
    std::vector< std::vector<AppliedDecisionRule> > m_applied_rules;

    /// Context phones defining the statistics of the cluster
    ContextPhoneSet m_contexts;

    /// Sum of occupancy counts
    double m_sum_occupancy;

    /// Summed statistics of the context phones
    FullCovarianceGaussian m_sum_stats;

    /// State index allocated for this cluster
    int m_state_index;

    std::vector<int> m_gauss_index;
    std::vector<double> m_gauss_weight;
  };


  /** Class for a collection of context dependent phones.
   */
  class Phone {
  public:
    Phone(std::string &center_label, PhonePool *pool);
    ~Phone();
    ContextPhone* get_context_phone(const std::string &label, int state);

    /** Finishes the statistics for this phone.
     * \return The number of different context dependent phone states
     */
    int finish_statistics(void);

    int num_states(void) { return (int)m_cp_states.size(); }
    int max_left_contexts(void) { return m_max_left_contexts; }
    int max_right_contexts(void) { return m_max_right_contexts; }

    ContextPhoneCluster* get_initial_clustered_state(int state);
    void add_final_cluster(int state, ContextPhoneCluster *cl);
    void merge_clusters(int state, int cl1_index, int cl2_index);
    std::vector<ContextPhoneCluster*>& get_state_clusters(int state) { return m_cluster_states[state]; }

    std::string& label(void) { return m_center_phone; }
    
  private:
    std::string m_center_phone;

    /// Handles allocation and deletion of ContextPhone objects!
    std::vector< ContextPhoneMap > m_cp_states;

    /** The final clusters. The outer vector corresponds to different
     * states, which are itself collections of \ref ContextPhoneCluster
     * objects.
     */
    std::vector< std::vector<ContextPhoneCluster*> > m_cluster_states;
    PhonePool *m_pool;

    /// Maximum left context index in this phone
    int m_max_left_contexts;
    /// Maximum right context index in this phone
    int m_max_right_contexts;
  };
  
  typedef std::map<std::string, Phone*> PhoneMap;

  /** Baseclass for defining callback functions for iterating through
   * all the context phones with their labels and state indices.
   */
  class ContextPhoneCallback {
  public:
    virtual ~ContextPhoneCallback() { }
    virtual void add_label(std::string &label, int num_states) { }
    virtual void add_state(int state_index) { }
  };

  /** Callback class for saving the basebind file
   */
  class SaveToBasebind : public ContextPhoneCallback {
  public:
    SaveToBasebind(FILE *fp, int init_state_index) : m_fp(fp), m_initial_state_index(init_state_index) { m_state_counter = -1; }
    virtual ~SaveToBasebind() { }
    virtual void add_label(std::string &label, int num_states);
    virtual void add_state(int state_index);
  private:
    FILE *m_fp;
    int m_initial_state_index;
    int m_state_counter;
  };


  /** Callback class for making the model to \ref HmmSet
   */
  class MakeHmmModel : public ContextPhoneCallback {
  public:
    MakeHmmModel(HmmSet &model) : m_model(model) { }
    virtual ~MakeHmmModel() { }
    virtual void add_label(std::string &label, int num_states);
    virtual void add_state(int state_index);
  private:
    HmmSet &m_model;
    Hmm *m_cur_hmm;
    int m_state_count;
  };

public:

  /** Interface class for \ref get_context_phone(). It is used to pass
   * the features to the correct ContextPhone object
   */
  class ContextPhoneContainer {
  private:
    ContextPhone *cp;
  public:
    ContextPhoneContainer(ContextPhone *cp_) : cp(cp_) { }

    /** Adds a feature to the ContextPhone object this container points to.
     * \param prior Prior probability of the state (1 for Viterbi)
     * \param f     The feature
     */
    void add_feature(double prior, const FeatureVec &f) { cp->add_feature(prior, f); }
  };

public:
  // Static methods for phone label handling
  static std::string center_phone(const std::string &label);
  static void fill_left_contexts(const std::string &label,
                                 std::vector<std::string> &contexts);
  static void fill_right_contexts(const std::string &label,
                                  std::vector<std::string> &contexts);

public:

  // PhonePool interface

  PhonePool(void);
  ~PhonePool();

  /** Set the clustering parameters
   * \param min_occupancy Minimum occupancy (feature) count for the states
   * \param min_split_ll_gain Minimum loglikelihood gain for a cluster split
   * \param max_merge_ll_loss Maximum loglikelihood loss for a cluster merge
   */
  inline void set_clustering_parameters(double min_occupancy,
                                        double min_split_ll_gain,
                                        double max_merge_ll_loss);
  /** Sets feature dimension
   * \param dim Feature dimension
   */
  void set_dimension(int dim) { m_dim = dim; }
  
  /// \return Feature dimension
  int dimension(void) { return m_dim; }

  /** Sets verbosity
   * \param info Level of verbosity: 0 = none, 1 = some, >1 = lots of info
   */
  void set_info(int info) { m_info = info; }

  /** Loads the rule set for the decision tree
   * \param fp Pointer to FILE object
   */
  void load_decision_tree_rules(FILE *fp);

  /** Returns a \ref ContextPhoneContainer object for adding features to
   * a certain context phone
   * \param label Context phone label
   * \param state State number of the context phone
   * \return \ref ContextPhoneContainer object
   */
  ContextPhoneContainer get_context_phone(const std::string &label, int state);

  /** Finishes the collection of statistics.
   * \note Must be called before \ref decision_tree_cluster_context_phones!
   */
  void finish_statistics(void);

  /** Forms tied states of context phones by splitting the context phone
   * clusters according to the decision tree
   * \param max_context_index Maximum context index (left and right) to
   *                          be tested in cluster splitting
   */
  void decision_tree_cluster_context_phones(int max_context_index);

  /** Merges the clusters created by \ref decision_tree_cluster_context_phones
   * if the loss in likelihood is less than the threshold set by
   * \ref set_clustering_parameters
   */
  void merge_context_phones(void);

  void soft_kmeans_clustering(double mixture_threshold, bool diagonal);

  /** Saves the result of context phone tying to a HMM model.
   * The model will have one full covariance Gaussian for each state
   * \param base              Base filename for the model
   * \param max_context_index Maximum context index (left and right)
   */
  void save_model(const std::string &base, int max_context_index);

  /** Saves the result of context phone tying to a basebind file
   * \param fp                  Pointer to FILE object
   * \param initial_state_index First state index to be allocated
   * \param max_context_index   Maximum context index (left and right)
   */
  void save_to_basebind(FILE *fp, int initial_state_index,
                        int max_context_index);

  /// Internal function for adding the labels of different contexts
  inline void add_context(const std::string &context);
  
private:
  void apply_best_splitting_rule(ContextPhoneCluster *cl,
                                 int min_context_index, int max_context_index,
                                 ContextPhoneCluster **new_cl);
  double compute_log_likelihood_gain(ContextPhoneCluster &parent,
                                     ContextPhoneCluster &child1,
                                     ContextPhoneCluster &child2);

  void iterate_context_phones(ContextPhoneCallback &c, int max_context_index);

  int add_final_gaussian(Gaussian *g);

private:
  double m_min_occupancy;
  double m_min_split_ll_gain;
  double m_max_merge_ll_loss;

  /// Handles allocation and deletion of Phone objects!
  PhoneMap m_phones;

  PhoneLabelSet m_contexts;

  std::vector<Gaussian*> m_final_gaussians;

  int m_dim; //!< Feature dimension
  int m_info; //!< Verbosity

  std::vector< DecisionRule > m_rules;
};


void
PhonePool::ContextPhone::add_feature(double prior, const FeatureVec &f)
{
  m_stats.accumulate(prior, *f.get_vector());
  m_occupancy += prior;
}

void
PhonePool::set_clustering_parameters(double min_occupancy,
                                     double min_split_ll_gain,
                                     double max_merge_ll_loss)
{
  m_min_occupancy = min_occupancy;
  m_min_split_ll_gain = min_split_ll_gain;
  m_max_merge_ll_loss = max_merge_ll_loss;
}

void
PhonePool::add_context(const std::string &context)
{
  m_contexts.insert(m_contexts.begin(), context);
}

}

#endif // PHONEPOOL_HH
