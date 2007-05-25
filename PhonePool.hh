#ifndef PHONEPOOL_HH
#define PHONEPOOL_HH

#include <set>
#include <map>
#include <vector>
#include <stdio.h>
#include "Distributions.hh"

class PhonePool {
public:

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

    std::string& label(void) { return m_label; }

    double occupancy_count(void) { return m_occupancy; }
    int num_left_contexts(void) { return (int)m_left_contexts.size(); }
    int num_right_contexts(void) { return (int)m_right_contexts.size(); }
    FullCovarianceGaussian* statistics(void) { return &m_stats; }

    inline void add_feature(double prior, const FeatureVec &f);
    void finish_statistics(void);
    
  private:
    std::string m_label;
    std::vector<std::string> m_left_contexts;
    std::vector<std::string> m_right_contexts;
    double m_occupancy;

    PhonePool *m_pool;
    
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
    double occupancy_count(void) { return m_sum_occupancy; }
    int num_context_phones(void) { return (int)m_contexts.size(); }
    FullCovarianceGaussian* statistics(void) { return &m_sum_stats; }

    double compute_new_cluster_occupancy(DecisionRule *rule, int context_index,
                                         bool answer,
                                         int *num_context_phones = NULL);
    void fill_new_cluster_context_phones(DecisionRule *rule,
                                         int context_index, bool answer,
                                         ContextPhoneSet &new_set);
    void remove_from_cluster(const ContextPhoneCluster &cl);
    void add_rule(AppliedDecisionRule &rule) { m_applied_rules.push_back(rule); }
    const std::vector<AppliedDecisionRule>& applied_rules(void) { return m_applied_rules; }

    void set_state_number(int state_number) { m_state_number = state_number; }
    int state_number(void) { return m_state_number; }
    
  private:
    /// Ordered vector of rules applied so far
    std::vector<AppliedDecisionRule> m_applied_rules;

    /// Context phones defining the statistics of the cluster
    ContextPhoneSet m_contexts;

    /// Sum of occupancy counts
    double m_sum_occupancy;

    /// Summed statistics of the context phones
    FullCovarianceGaussian m_sum_stats;

    /// State number allocated for this cluster
    int m_state_number;
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
    std::vector<ContextPhoneCluster*>& get_state_cluster(int state) { return m_cluster_states[state]; }

    std::string& label(void) { return m_center_phone; }
    
  private:
    std::string m_center_phone;

    /// Handles allocation and deletion of ContextPhone objects!
    std::vector< ContextPhoneMap > m_cp_states;

    std::vector< std::vector<ContextPhoneCluster*> > m_cluster_states;
    PhonePool *m_pool;
    int m_max_left_contexts, m_max_right_contexts;
  };
  
  typedef std::map<std::string, Phone*> PhoneMap;

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
  
  inline void set_clustering_parameters(double min_occupancy,
                                        double min_ll_gain);
  void set_dimension(int dim) { m_dim = dim; }
  int dimension(void) { return m_dim; }
  void set_info(int info) { m_info = info; }
  void load_decision_tree_rules(FILE *fp);
  
  ContextPhone* get_context_phone(const std::string &label, int state);
  void finish_statistics(void);
  void decision_tree_cluster_context_phones(int max_context_index);

  void save_to_basebind(FILE *fp, int initial_statenum, int max_context_index);

  /// Internal function for adding the labels of different contexts
  inline void add_context(const std::string &context);
  
private:
  void apply_best_splitting_rule(ContextPhoneCluster *cl,
                                 int min_context_index, int max_context_index,
                                 ContextPhoneCluster **new_cl);
  double compute_log_likelihood_gain(ContextPhoneCluster &parent,
                                     ContextPhoneCluster &child1,
                                     ContextPhoneCluster &child2);

private:
  double m_min_occupancy;
  double m_min_ll_gain;
  
  PhoneMap m_phones;

  PhoneLabelSet m_contexts;

  int m_dim; //!< Feature dimension
  int m_info; //!< Verbosity

  std::vector< DecisionRule > m_rules;
};


void
PhonePool::ContextPhone::add_feature(double prior, const FeatureVec &f)
{
  m_stats.accumulate(prior, f);
  m_occupancy += prior;
}

void
PhonePool::set_clustering_parameters(double min_occupancy, double min_ll_gain)
{
  m_min_occupancy = min_occupancy;
  m_min_ll_gain = min_ll_gain;
}

void
PhonePool::add_context(const std::string &context)
{
  m_contexts.insert(m_contexts.begin(), context);
}



#endif // PHONEPOOL_HH
