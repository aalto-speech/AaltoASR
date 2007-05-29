#include <algorithm>

#include "PhonePool.hh"
#include "LinearAlgebra.hh"
#include "str.hh"


int safe_tolower(int c)
{
  if (c == '�')
    return '�';
  else if (c == '�')
    return '�';
  else if (c == '�')
    return '�';
  return tolower(c);
}


PhonePool::ContextPhone::ContextPhone(const std::string &label,
                                      PhonePool *pool, bool persistent) :
  m_label(label), m_pool(pool), m_stats(pool->dimension())
{
  PhonePool::fill_left_contexts(label, m_left_contexts);
  PhonePool::fill_right_contexts(label, m_right_contexts);
  m_occupancy = 0;

  if (persistent)
  {
    m_stats.set_estimation_mode(PDF::ML);
    m_stats.start_accumulating();

    // Store contexts to the pool
    for (int i = 0; i < (int)m_left_contexts.size(); i++)
      m_pool->add_context(m_left_contexts[i]);
    for (int i = 0; i < (int)m_right_contexts.size(); i++)
      m_pool->add_context(m_right_contexts[i]);
  }
}

bool
PhonePool::ContextPhone::rule_answer(const DecisionRule *rule,
                                     int context_index)
{
  if (rule->rule_type == DecisionRule::CONTEXT)
  {
    if (context_index < 0)
    {
      if (context_index < -num_left_contexts())
        return false;
      if (rule->phone_set.find(m_left_contexts[-context_index-1]) !=
          rule->phone_set.end())
        return true;
    }
    else if (context_index > 0)
    {
      if (context_index > num_right_contexts())
        return false;
      if (rule->phone_set.find(m_right_contexts[context_index-1]) !=
          rule->phone_set.end())
        return true;
    }
    else
      throw std::string("PhonePool::ContextPhone::rule_answer: Invalid context index 0");
  }
  return false;
}

void
PhonePool::ContextPhone::finish_statistics(void)
{
  m_stats.estimate_parameters();
  m_stats.stop_accumulating();
}


void
PhonePool::ContextPhoneCluster::fill_cluster(ContextPhoneSet &context_phones)
{
  m_contexts = context_phones;
  compute_statistics();
}

void
PhonePool::ContextPhoneCluster::compute_statistics(void)
{
  std::vector<double> weights;
  std::vector<const Gaussian*> gaussians;

  m_sum_occupancy = 0;
  for (ContextPhoneSet::iterator it = m_contexts.begin();
       it != m_contexts.end(); it++)
  {
    weights.push_back((*it)->occupancy_count());
    gaussians.push_back((*it)->statistics());
    m_sum_occupancy += (*it)->occupancy_count();
  }
  m_sum_stats.merge(weights, gaussians);
}


double
PhonePool::ContextPhoneCluster::compute_new_cluster_occupancy(
  DecisionRule *rule, int context_index, bool answer, int *num_context_phones)
{
  double occupancy = 0;
  int count = 0;
  for (ContextPhoneSet::iterator it = m_contexts.begin();
       it != m_contexts.end(); it++)
  {
    if ((*it)->rule_answer(rule, context_index) == answer)
    {
      occupancy += (*it)->occupancy_count();
      count++;
    }
  }
  if (num_context_phones != NULL)
    *num_context_phones = count;
  return occupancy;
}

void
PhonePool::ContextPhoneCluster::fill_new_cluster_context_phones(
  DecisionRule *rule, int context_index, bool answer,
  ContextPhoneSet &new_set)
{
  for (ContextPhoneSet::iterator it = m_contexts.begin();
       it != m_contexts.end(); it++)
  {
    if ((*it)->rule_answer(rule, context_index) == answer)
      new_set.insert(*it);
  }
}

void
PhonePool::ContextPhoneCluster::remove_from_cluster(
  const ContextPhoneCluster &cl)
{
  ContextPhoneSet new_set;

  set_difference(m_contexts.begin(), m_contexts.end(),
                 cl.m_contexts.begin(), cl.m_contexts.end(),
                 inserter(new_set, new_set.begin()));
  m_contexts = new_set;
  compute_statistics();
}


PhonePool::Phone::Phone(std::string &center_label, PhonePool *pool) :
  m_center_phone(center_label),
  m_pool(pool),
  m_max_left_contexts(0),
  m_max_right_contexts(0)
{
}

PhonePool::Phone::~Phone()
{
  // Delete the ContextPhone objects
  for (int i = 0; i < (int)m_cp_states.size(); i++)
  {
    for (ContextPhoneMap::iterator it = m_cp_states[i].begin();
         it != m_cp_states[i].end(); it++)
    {
      delete (*it).second;
    }
  }

  // Delete the ContextPhoneCluster objects
  for (int i = 0; i < (int)m_cluster_states.size(); i++)
  {
    for (int j = 0; j < (int)m_cluster_states[i].size(); j++)
      delete m_cluster_states[i][j];
  }
}

PhonePool::ContextPhone*
PhonePool::Phone::get_context_phone(const std::string &label, int state)
{
  if (state < 0)
    throw str::fmt(128,"PhonePool::Phone::get_context_phone: Invalid state %i",
                   state);
  if (state >= (int)m_cp_states.size())
  {
    // Resize the state vector to hold enough states
    m_cp_states.resize(state+1);
  }
  ContextPhoneMap::iterator it = m_cp_states[state].find(label);
  if (it == m_cp_states[state].end())
  {
    return (*((m_cp_states[state].insert(
                 ContextPhoneMap::value_type(label,
                                             new ContextPhone(label,m_pool)))
                ).first)).second;
  }
  return (*it).second;
}

int
PhonePool::Phone::finish_statistics(void)
{
  int num_context_phones = 0;
  m_max_left_contexts = 0;
  m_max_right_contexts = 0;
  for (int i = 0; i < (int)m_cp_states.size(); i++)
  {
    num_context_phones += (int)m_cp_states[i].size();
    for (ContextPhoneMap::iterator it = m_cp_states[i].begin();
         it != m_cp_states[i].end(); it++)
    {
      (*it).second->finish_statistics();
      if ((*it).second->num_left_contexts() > m_max_left_contexts)
        m_max_left_contexts = (*it).second->num_left_contexts();
      if ((*it).second->num_right_contexts() > m_max_right_contexts)
        m_max_right_contexts = (*it).second->num_right_contexts();
    }
  }
  return num_context_phones;
}

PhonePool::ContextPhoneCluster*
PhonePool::Phone::get_initial_clustered_state(int state)
{
  if (state < 0 || state > num_states())
    throw str::fmt(
      128,"PhonePool::Phone::get_initial_clustered_state: Invalid state %i",
      state);
  
  ContextPhoneCluster *cl = new ContextPhoneCluster(m_pool->dimension());
  ContextPhoneSet context_phones;
  for (ContextPhoneMap::iterator it = m_cp_states[state].begin();
       it != m_cp_states[state].end(); it++)
    context_phones.insert(((*it).second));
  cl->fill_cluster(context_phones);
  return cl;
}

void
PhonePool::Phone::add_final_cluster(int state, ContextPhoneCluster *cl)
{
  if ((int)m_cluster_states.size() <= state)
    m_cluster_states.resize(state+1);
  m_cluster_states[state].push_back(cl);
}


PhonePool::PhonePool(void) :
  m_min_occupancy(0),
  m_min_ll_gain(0),
  m_dim(0),
  m_info(0)
{
}


PhonePool::~PhonePool()
{
  // Delete the Phone objects
  for (PhoneMap::iterator it = m_phones.begin(); it != m_phones.end(); it++)
  {
    delete (*it).second;
  }
}


std::string
PhonePool::center_phone(const std::string &label)
{
  int pos1 = label.find_last_of('-');
  int pos2 = label.find_first_of('+');
  std::string temp = "";
  if (pos1 >= 0 && pos2 >= 0)
  {
    if (pos2 > pos1+1)
      temp = label.substr(pos1+1, pos2-pos1-1);
  }
  else if (pos1 >= 0)
  {
    temp = label.substr(pos1);
  }
  else if (pos2 >= 0)
  {
    temp = label.substr(0, pos2);
  }
  else
    temp = label;
  if ((int)temp.size() <= 0)
    throw std::string("PhonePool: Invalid phone label") + label;
  return temp;
}

void
PhonePool::fill_left_contexts(const std::string &label,
                              std::vector<std::string> &contexts)
{
  int cur_pos = 0;
  int next_delim_pos;
  std::vector<std::string> reversed_contexts;

  while ((next_delim_pos = label.find('-', cur_pos+1)) >= cur_pos)
  {
    reversed_contexts.push_back(label.substr(cur_pos, next_delim_pos-cur_pos));
    cur_pos = next_delim_pos+1;
  }
  for (int i = reversed_contexts.size()-1; i >= 0; i--)
    contexts.push_back(reversed_contexts[i]);
}

void
PhonePool::fill_right_contexts(const std::string &label,
                               std::vector<std::string> &contexts)
{
  int next_delim_pos;
  int cur_pos = label.find('+');
  if (cur_pos > 0)
  {
    cur_pos++;
    while ((next_delim_pos = label.find('+', cur_pos+1)) >= cur_pos)
    {
      contexts.push_back(label.substr(cur_pos, next_delim_pos-cur_pos));
      cur_pos = next_delim_pos+1;
    }
    contexts.push_back(label.substr(cur_pos));
  }
}


void
PhonePool::load_decision_tree_rules(FILE *fp)
{
  std::string line;
  std::vector<std::string> fields;
  while (str::read_line(&line, fp, true))
  {
    fields.clear();
    str::split(&line, " \t", true, &fields, 3);
    if ((int)fields.size() > 0)
    {
      if ((int)fields.size() < 2)
        throw std::string("PhonePool::load_decision_tree_rules: Invalid rule line:\n") + line;
      
      std::transform(fields[1].begin(), fields[1].end(),
                     fields[1].begin(), safe_tolower);
      if (fields[1] == "context")
      {
        DecisionRule r;
        std::vector<std::string> phones;
        r.rule_name = fields[0];
        r.rule_type = DecisionRule::CONTEXT;
        str::split(&(fields[2]), ", ", true, &phones);
        if ((int)phones.size() < 1)
          throw std::string("PhonePool::load_decision_tree_rules: No phones in the context rule:\n") + line;
        for (int i = 0; i < (int)phones.size(); i++)
          r.phone_set.insert(phones[i]);
        m_rules.push_back(r);
      }
      else
      {
        throw std::string("PhonePool::load_decision_tree_rules: Invalid rule type ") + fields[1];
      }
    }
  }
}


PhonePool::ContextPhone*
PhonePool::get_context_phone(const std::string &label, int state)
{
  std::string center_label(center_phone(label));
  PhoneMap::iterator it = m_phones.find(center_label);
  if (it == m_phones.end())
  {
    if (m_info > 1)
      fprintf(stderr, "New phone %s\n", center_label.c_str());

    return (*((m_phones.insert(PhoneMap::value_type(center_label, new Phone(center_label, this)))).first)).second->get_context_phone(label, state);
  }
  return (*it).second->get_context_phone(label, state);
}


void
PhonePool::finish_statistics(void)
{
  int num_states = 0;
  for (PhoneMap::iterator it = m_phones.begin(); it != m_phones.end(); it++)
  {
    num_states += (*it).second->finish_statistics();
  }
  if (m_info > 0)
    fprintf(stderr,"%i context dependent phone states in total\n",num_states);
}


void
PhonePool::decision_tree_cluster_context_phones(int max_context_index)
{
  int context_start, context_end;
  int total_clusters = 0;
  for (PhoneMap::iterator it = m_phones.begin(); it != m_phones.end(); it++)
  {
    for (int s = 0; s < (*it).second->num_states(); s++)
    {
      if (m_info > 0)
        fprintf(stderr, "Processing phone %s, state %i\n",
                (*it).second->label().c_str(), s);
      
      std::vector<ContextPhoneCluster*> clusters;
      clusters.push_back((*it).second->get_initial_clustered_state(s));

      // Determine the context range
      if (max_context_index > 0)
      {
        context_start = -std::min((*it).second->max_left_contexts(),
                                  max_context_index);
        context_end = std::min((*it).second->max_right_contexts(),
                               max_context_index);
      }
      else
      {
        context_start = -(*it).second->max_left_contexts();
        context_end = (*it).second->max_right_contexts();
      }
      if (m_info > 1)
        fprintf(stderr, "Checking context indices %i through %i\n",
                context_start, context_end);

      // Split the clusters until no more clusters can be split
      for (int c = 0; c < (int)clusters.size(); c++)
      {
        // Find the best rule to split the cluster
        ContextPhoneCluster *new_cluster;
        apply_best_splitting_rule(clusters[c], context_start, context_end,
                                  &new_cluster);
        if (new_cluster != NULL)
        {
          // The cluster was split, add the new cluster to the vector
          clusters.push_back(new_cluster);
          c--; // Reconsider the split cluster for splitting again
        }
      }

      // Save the result
      for (int c = 0; c < (int)clusters.size(); c++)
        (*it).second->add_final_cluster(s, clusters[c]);
      if (m_info > 0)
        fprintf(stderr, "%i clusters generated\n", (int)clusters.size());
      total_clusters += (int)clusters.size();
    }
  }
  if (m_info > 0)
    fprintf(stderr, "Total: %i clusters generated\n", total_clusters);
}


void
PhonePool::apply_best_splitting_rule(
  ContextPhoneCluster *cl, int min_context_index, int max_context_index,
  ContextPhoneCluster **new_cl)
{
  double c1, c2; // Temporary cluster occupancy counts
  bool cur_first_answer;
  std::vector< ContextPhoneSet > applied_sets;
  int num_new_context_phones;
  
  // Initialize the best applied rule to invalid values (context 0)
  AppliedDecisionRule best_applied_rule(&(m_rules[0]), 0, false);
  ContextPhoneCluster best_cl1(m_dim), best_cl2(m_dim);
  double best_ll_gain = -1;
  int s;
  
  // Iterate through rules
  for (int r = 0; r < (int)m_rules.size(); r++)
  {
    for (int i=min_context_index; i <= max_context_index; i++)
    {
      if (i == 0) // Zero is an invalid context index
        continue;
      
      c1 = cl->compute_new_cluster_occupancy(&(m_rules[r]), i, true,
                                             &num_new_context_phones);
      c2 = cl->occupancy_count() - c1;

      if (c1 < m_min_occupancy || c2 < m_min_occupancy)
        continue;

      ContextPhoneCluster cl1(m_dim), cl2(m_dim);
      ContextPhoneSet new_context_phones;

      // Use the smaller set to check whether we have a new set of phones
      if (num_new_context_phones <= cl->num_context_phones()/2)
        cur_first_answer = true;
      else
        cur_first_answer = false;
      
      cl->fill_new_cluster_context_phones(&(m_rules[r]), i, cur_first_answer,
                                          new_context_phones);

      // Check we haven't tested this set before
      for (s = 0; s < (int)applied_sets.size(); s++)
      {
        ContextPhoneSet temp_set;
        set_symmetric_difference(applied_sets[s].begin(),
                                 applied_sets[s].end(),
                                 new_context_phones.begin(),
                                 new_context_phones.end(),
                                 inserter(temp_set, temp_set.begin()));
        if (temp_set.size() == 0)
          break;
      }
      if (s < (int)applied_sets.size()) // Found the same set
        continue;
      
      cl1.fill_cluster(new_context_phones);
      cl2 = *cl;
      cl2.remove_from_cluster(cl1);

      applied_sets.push_back(new_context_phones);

      double gain = compute_log_likelihood_gain(*cl, cl1, cl2);
      if (gain > best_ll_gain && gain > m_min_ll_gain)
      {
        best_cl1 = cl1;
        best_cl2 = cl2;
        best_applied_rule = AppliedDecisionRule(&(m_rules[r]), i,
                                                cur_first_answer);
        best_ll_gain = gain;
      }
      
      if (m_rules[r].rule_type != DecisionRule::CONTEXT)
        break; // Context index has no meaning, no need to iterate
    }                                
  }
  if (best_applied_rule.context != 0)
  {
    // Found an applicable rule
    *cl = best_cl1;
    *new_cl = new ContextPhoneCluster(m_dim);
    **new_cl = best_cl2;
    cl->add_rule(best_applied_rule);
    best_applied_rule.answer = !best_applied_rule.answer;
    (*new_cl)->add_rule(best_applied_rule);

    if (m_info > 1)
    {
      fprintf(stderr, "Applying rule %s:\n",
              best_applied_rule.rule->rule_name.c_str());
      fprintf(stderr, "   context index:   %i\n",best_applied_rule.context);
      fprintf(stderr, "   likelihood gain: %.2f\n", best_ll_gain);
      fprintf(stderr, "   cluster counts:  %i + %i\n",
              (int)best_cl1.occupancy_count(),(int)best_cl2.occupancy_count());
    }
  }
  else
    *new_cl = NULL;
}


double
PhonePool::compute_log_likelihood_gain(ContextPhoneCluster &parent,
                                       ContextPhoneCluster &child1,
                                       ContextPhoneCluster &child2)
{
  Matrix parent_cov, child1_cov, child2_cov;

  parent.statistics()->get_covariance(parent_cov);
  child1.statistics()->get_covariance(child1_cov);
  child2.statistics()->get_covariance(child2_cov);
  return
    (LinearAlgebra::log_determinant(parent_cov)*parent.occupancy_count() -
     LinearAlgebra::log_determinant(child1_cov)*child1.occupancy_count() -
     LinearAlgebra::log_determinant(child2_cov)*child2.occupancy_count())/2;
}


void
PhonePool::save_to_basebind(FILE *fp, int initial_statenum,
                            int max_context_index)
{
  int state_num = initial_statenum;
    
  for (PhoneMap::iterator it = m_phones.begin(); it != m_phones.end(); it++)
  {
    // Allocate state numbers
    for (int s = 0; s < (*it).second->num_states(); s++)
    {
      std::vector<ContextPhoneCluster*> &clusters =
        (*it).second->get_state_cluster(s);
      for (int i = 0; i < (int)clusters.size(); i++)
        clusters[i]->set_state_number(state_num++);
    }

    if ((*it).second->label()[0] != '_')
    {
      // Iterate through all context phones (with this center phone) and
      // write them to basebind
      if (max_context_index > 0)
      {
        PhoneLabelSet::iterator *context_it =
          new PhoneLabelSet::iterator[max_context_index*2];
        for (int i = 0; i < max_context_index*2; i++)
          context_it[i] = m_contexts.begin();
        
        while (context_it[0] != m_contexts.end())
        {
          std::string label = "";
          // Construct the label
          for (int i = 0; i < max_context_index; i++)
            label += *(context_it[i]) + "-";
          label += (*it).second->label();
          for (int i = max_context_index; i < 2*max_context_index; i++)
            label += std::string("+") + *(context_it[i]);

          fprintf(fp, "%s %d", label.c_str(), (*it).second->num_states());
        
          // Construct a ContextPhone object for rule testing
          ContextPhone cur_phone(label, this, false);

          // Iterate through states
          for (int s = 0; s < (*it).second->num_states(); s++)
          {
            std::vector<ContextPhoneCluster*> &clusters =
              (*it).second->get_state_cluster(s);
            int cluster_index = -1;
            
            // Find the cluster this context phone belongs to
            for (int i = 0; i < (int)clusters.size(); i++)
            {
              const std::vector<AppliedDecisionRule> &rules =
                clusters[i]->applied_rules();
              int j;
              for (j = 0; j < (int)rules.size(); j++)
                if (cur_phone.rule_answer(rules[j].rule, rules[j].context) !=
                    rules[j].answer)
                  break;
              if (j == (int)rules.size()) // Found a match
              {
                cluster_index = i;
                break;
              }
            }
            assert( cluster_index >= 0 );

            fprintf(fp, " %d", clusters[cluster_index]->state_number());
          }
          fprintf(fp, "\n");
        
          // Update the iterators
          for (int i = max_context_index*2-1;
               i >= 0 && ++context_it[i] == m_contexts.end(); i--)
          {
            if (i > 0)
              context_it[i] = m_contexts.begin();
          }
        }
      }
    }
    else
    {
      fprintf(fp, "%s %d", (*it).second->label().c_str(),
              (*it).second->num_states());
      for (int s = 0; s < (*it).second->num_states(); s++)
      {
        std::vector<ContextPhoneCluster*> &clusters =
          (*it).second->get_state_cluster(s);    
        fprintf(fp, " %d", clusters[0]->state_number());
      }
      fprintf(fp, "\n");
    }
  }
}