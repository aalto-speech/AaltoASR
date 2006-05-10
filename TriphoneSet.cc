#include <stdio.h>
#include <math.h>
#include <algorithm>

#include "TriphoneSet.hh"


int safe_tolower(int c)
{
  if (c == 'Å')
    return 'å';
  else if (c == 'Ä')
    return 'ä';
  else if (c == 'Ö')
    return 'ö';
  return tolower(c);
}


int safe_toupper(int c)
{
  if (c == 'å')
    return 'Å';
  else if (c == 'ä')
    return 'Ä';
  else if (c == 'ö')
    return 'Ö';
  return toupper(c);
}


TriphoneSet::TriphoneSet()
  : m_dim(0),
    m_info(0),
    m_ignore_length(false),
    m_min_count(300),
    m_min_likelihood_gain(1000),
    m_length_award(0)
{
}

TriphoneSet::~TriphoneSet()
{
  int i,j;
  for (i = 0; i < (int)m_triphones.size(); i++)
  {
    for (j = 0; j < (int)m_triphones[i]->states.size(); j++)
    {
      delete m_triphones[i]->states[j]->cov;
      delete m_triphones[i]->states[j]->mean;
      delete m_triphones[i]->states[j];
    }
  }
}


int
TriphoneSet::triphone_index(const std::string &left,
                             const std::string &center,
                             const std::string &right)
{
  std::map<std::string,int>::const_iterator it =
    m_tri_map.find(get_triphone_label(left, center, right));
  if (it == m_tri_map.end())
    return -1;
  return (*it).second;
}


void TriphoneSet::add_feature(const FeatureVec &f, const std::string &left,
                              const std::string &center,
                              const std::string &right, int state_index)
{
  int index = triphone_index(left, center, right);
  int center_index;
  int i,j;
  std::string lowered_center;

  lowered_center = center;
  if (!m_ignore_length)
    std::transform(lowered_center.begin(), lowered_center.end(),
                   lowered_center.begin(), safe_tolower);
  
  if (index == -1)
  {
    // New triphone
    std::string label = get_triphone_label(left, center, right);
    index = (int)m_triphones.size();
    m_tri_map[label] = index;
    m_triphones.push_back(new Triphone);
    m_triphones[index]->center = center;
    m_triphones[index]->left = left;
    m_triphones[index]->right = right;

    if (lowered_center == center || m_ignore_length)
      m_triphones[index]->long_phoneme = false;
    else
      m_triphones[index]->long_phoneme = true;

    // Make initial triphone cluster
    std::map<std::string,int>::const_iterator it =
      m_tri_center_map.find(lowered_center);
    if (it == m_tri_center_map.end())
    {
      // New phoneme
      m_tri_center_map[lowered_center] = (int)m_tri_centers.size();
      m_tri_centers.resize(m_tri_centers.size()+1);
      m_tri_centers.back().center = lowered_center;
    }
  }
  if ((int)m_triphones[index]->states.size() < state_index+1)
  {
    int old_size = (int)m_triphones[index]->states.size();
    m_triphones[index]->states.resize(state_index+1);
    for (i = old_size; i <= state_index; i++)
    {
      m_triphones[index]->states[i] = NULL;
    }
  }
  
  TriphoneState *state = m_triphones[index]->states[state_index];
  if (state == NULL)
  {
    // New state, initialize
    state = new TriphoneState;
    m_triphones[index]->states[state_index] = state;
    state->tri = m_triphones[index];
    state->state = state_index;
    state->mean = new double[m_dim];
    for (i = 0; i < m_dim; i++)
      state->mean[i] = 0;
    state->cov = new MatrixD(m_dim,m_dim);
    state->count = 0;

    // Add to the initial cluster
    std::map<std::string,int>::const_iterator it =
      m_tri_center_map.find(lowered_center);

    assert( it != m_tri_center_map.end());

    center_index = (*it).second;
    if ((int)m_tri_centers[center_index].state_clusters.size() <= state_index)
    {
      int old_size = (int)m_tri_centers[center_index].state_clusters.size();
      m_tri_centers[center_index].state_clusters.resize(state_index+1);
      for (i = old_size; i <= state_index; i++)
      {
        m_tri_centers[center_index].state_clusters[i].resize(1);
      }
    }
    m_tri_centers[center_index].state_clusters[state_index].front().states.push_back(state);
  }

  for (i = 0; i < m_dim; i++)
  {
    state->mean[i] += f[i];
  }
  for (i = 0; i < m_dim; i++)
  {
    for (j = 0; j < m_dim; j++)
    {
      (*state->cov)(i,j) = (*state->cov)(i,j) + f[i]*f[j];
    }
  }
  state->count++;
}

void
TriphoneSet::finish_triphone_statistics(void)
{
  int t, i, j, k;

  for (t = 0; t < (int)m_triphones.size(); t++)
  {
    for (i = 0; i < (int)m_triphones[t]->states.size(); i++)
    {
      TriphoneState *state = m_triphones[t]->states[i];
      for (j = 0; j < m_dim; j++)
      {
        state->mean[j] /= m_triphones[t]->states[i]->count;
      }
      for (j = 0; j < m_dim; j++)
      {
        for (k = 0; k < m_dim; k++)
        {
          (*state->cov)(j,k) = (*state->cov)(j,k)/state->count -
            state->mean[j]*state->mean[k];
        }
      }
    }
  }
  for (i = 0; i < (int)m_tri_centers.size(); i++)
  {
    for (j = 0; j < (int)m_tri_centers[i].state_clusters.size(); j++)
    {
      fill_context_cluster_statistics(m_tri_centers[i].state_clusters[j].front());
    }
  }
}


void
TriphoneSet::fill_missing_contexts(bool boundary)
{
  std::vector<std::string> context;
  std::map<std::string, int> context_map;
  int i, j, k;
  
  // Find all contexts
  for (i = 0; i < (int)m_triphones.size(); i++)
  {
    std::map<std::string,int>::const_iterator it1 =
      context_map.find(m_triphones[i]->left);
    if (it1 == context_map.end())
    {
      context_map[m_triphones[i]->left] = (int)context.size();
      context.push_back(m_triphones[i]->left);
    }
    std::map<std::string,int>::const_iterator it2 =
      context_map.find(m_triphones[i]->right);
    if (it2 == context_map.end())
    {
      context_map[m_triphones[i]->right] = (int)context.size();
      context.push_back(m_triphones[i]->right);
    }
  }
  if (m_info > 0)
  {
    fprintf(stderr, "%d contexts, %d phonemes\n", (int)context.size(),
            m_tri_centers.size());
    fprintf(stderr, "Triphones before filling: %d\n", (int)m_triphones.size());
  }
  // Fill the missing ones
  for (i = 0; i < (int)m_tri_centers.size(); i++)
  {
    for (j = 0; j < (int)context.size(); j++)
    {
      for (k = 0; k < (int)context.size(); k++)
      {
        if (boundary && context[j]!="=" && context[k] !="=")
          continue;
        if (triphone_index(context[j], m_tri_centers[i].center, context[k])
            == -1)
        {
          add_missing_triphone(context[j], m_tri_centers[i].center,
                               context[k]);
        }
        if (!m_ignore_length)
        {
          std::string uppered_center;
          uppered_center = m_tri_centers[i].center;
          std::transform(uppered_center.begin(), uppered_center.end(),
                         uppered_center.begin(), safe_toupper);
          if (triphone_index(context[j], uppered_center, context[k]) == -1)
          {
            add_missing_triphone(context[j], uppered_center,
                                 context[k]);
          }
        }
      }
    }
  }
  if (m_info > 0)
  {
    fprintf(stderr, "Triphones after filling: %d\n", (int)m_triphones.size());
  }
}


void
TriphoneSet::add_missing_triphone(std::string &left, std::string &center,
                                  std::string &right)
{
  std::string lowered_center;
  int center_index, i, j;

  lowered_center = center;
  if (!m_ignore_length)
    std::transform(lowered_center.begin(), lowered_center.end(),
                   lowered_center.begin(), safe_tolower);
  // Add the triphone
  std::string label = get_triphone_label(left, center, right);
  int index = (int)m_triphones.size();
  m_tri_map[label] = index;
  m_triphones.push_back(new Triphone);
  m_triphones[index]->center = center;
  m_triphones[index]->left = left;
  m_triphones[index]->right = right;
  if (lowered_center == center || m_ignore_length)
    m_triphones[index]->long_phoneme = false;
  else
    m_triphones[index]->long_phoneme = true;

  // Add to the initial cluster
  std::map<std::string,int>::const_iterator it =
    m_tri_center_map.find(lowered_center);

  assert( it != m_tri_center_map.end() );
  center_index = (*it).second;

  // Add states
  for (i = 0; i < (int)m_tri_centers[center_index].state_clusters.size(); i++)
  {
    TriphoneState *state = new TriphoneState;
    state->tri = m_triphones[index];
    state->state = i;
    state->mean = new double[m_dim];
    for (j = 0; j < m_dim; j++)
      state->mean[j] = 0;
    state->cov = new MatrixD(m_dim,m_dim);
    state->count = 0;
    m_triphones[index]->states.push_back(state);
    m_tri_centers[center_index].state_clusters[i].front().states.push_back(state);
  }
}


void
TriphoneSet::tie_triphones(void)
{
  int c, s;

  if (m_info > 0)
  {
    fprintf(stderr, "%d triphones in %d phonemes\n", (int)m_triphones.size(),
            (int)m_tri_centers.size());
  }

  count_limit_count = 0;
  lh_limit_count = 0;
  
  for (c = 0; c < (int)m_tri_centers.size(); c++)
  {
    for (s = 0; s < (int)m_tri_centers[c].state_clusters.size(); s++)
    {

      tie_states(m_tri_centers[c].state_clusters[s]);
      if (m_info > 0)
      {
        fprintf(stderr, "State %d of phoneme %s was split to %d clusters.\n",
                s+1, m_tri_centers[c].center.c_str(),
                (int)m_tri_centers[c].state_clusters[s].size());
      }
    }
  }

  if (m_info > 0)
  {
    fprintf(stderr, "Count limit: %d\nLikelihood limit: %d\n",
            count_limit_count, lh_limit_count);
  }
}


void
TriphoneSet::tie_states(std::vector<ContextStateCluster> &cncl)
{
  int cl;
  int rule, best_rule;
  int best_rule_cl;
  double lhg, best_lhg = -1;

  while (1)
  {
    best_rule = -1;
    best_lhg = -1;
    best_rule_cl = -1;
    for (cl = 0; cl < (int)cncl.size(); cl++)
    {
      rule = find_best_split_rule(cncl[cl], &lhg);
      if (rule != -1)
      {
        if (lhg > best_lhg)
        {
          best_rule = rule;
          best_rule_cl = cl;
          best_lhg = lhg;
        }
      }
    }
    if (best_rule == -1)
      break; // No more rules can be applied

    if (m_info > 1)
    {
      fprintf(stderr, "Applying rule %s (%s) to cluster %d of phoneme %s, count = %d\n",
              m_rule_set[best_rule].rule_name.c_str(),
              (m_rule_set[best_rule].rule_type == DecisionRule::LEFT_CONTEXT?
               "left":
               (m_rule_set[best_rule].rule_type == DecisionRule::RIGHT_CONTEXT?
                "right":"length")), best_rule_cl,
              cncl[best_rule_cl].states.front()->tri->center.c_str(),
              cncl[best_rule_cl].count);
    }

    // Split the node
    ContextStateCluster temp = cncl[best_rule_cl];
    //cncl[best_rule_cl].resize(0);
    cncl.resize((int)cncl.size()+1);
    cncl[best_rule_cl].states.clear();
    apply_rule_to_group(temp, best_rule, true, cncl[best_rule_cl]);
    apply_rule_to_group(temp, best_rule, false, cncl[(int)cncl.size()-1]);
    fill_context_cluster_statistics(cncl[best_rule_cl]);
    fill_context_cluster_statistics(cncl[(int)cncl.size()-1]);
  }
}


int
TriphoneSet::find_best_split_rule(ContextStateCluster &states,
                                  double *likelihood_gain)
{
  int rule, best_rule = -1;
  ContextStateCluster leaf[2];
  double lg, best_lg = -1;

  for (rule = 0; rule < (int)m_rule_set.size(); rule++)
  {
    leaf[0].states.clear();
    leaf[1].states.clear();
    apply_rule_to_group(states, rule, true, leaf[0]);
    apply_rule_to_group(states, rule, false, leaf[1]);

    // Verify minimum count and evaluate the likelihood gain
    fill_context_cluster_statistics(leaf[0]);
    if (leaf[0].count >= m_min_count)
    {
      fill_context_cluster_statistics(leaf[1]);
      if (leaf[1].count >= m_min_count)
      {
        lg = 0.5*(-log(leaf[0].cov_det)*leaf[0].count -
                  log(leaf[1].cov_det)*leaf[1].count +
                  log(states.cov_det)*states.count);
        // We measure the likelihood only from acoustical similarity
        // of the features, so a likelihood award is given when the cluster
        // is divided with a phoneme length rule.
        if (m_rule_set[rule].rule_type == DecisionRule::PHONEME_LENGTH)
          lg += m_length_award;
        if (lg > m_min_likelihood_gain)
        {
          if (lg > best_lg)
          {
            best_lg = lg;
            best_rule = rule;
          }
        }
        else
          lh_limit_count++;
      }
      else
        count_limit_count++;
    }
    else
      count_limit_count++;
  }
  return best_rule;
}


void
TriphoneSet::apply_rule_to_group(ContextStateCluster &cl, int rule,
                                 bool in, ContextStateCluster &out)
{
  int i;

  for (i = 0; i < (int)cl.states.size(); i++)
  {
    std::vector<std::string>::iterator f;
    if (m_rule_set[rule].rule_type == DecisionRule::PHONEME_LENGTH)
    {
      if ((in && cl.states[i]->tri->long_phoneme) ||
          (!in && !cl.states[i]->tri->long_phoneme))
      {
        out.states.push_back(cl.states[i]);
      }
    }
    else
    {
      if (m_rule_set[rule].rule_type == DecisionRule::LEFT_CONTEXT)
      {
        f = find(m_rule_set[rule].phoneme_list.begin(),
                 m_rule_set[rule].phoneme_list.end(),
                 cl.states[i]->tri->left);
      }
      else if (m_rule_set[rule].rule_type == DecisionRule::RIGHT_CONTEXT)
      {
        f = find(m_rule_set[rule].phoneme_list.begin(),
                 m_rule_set[rule].phoneme_list.end(),
                 cl.states[i]->tri->right);
      }
      if ((in && f != m_rule_set[rule].phoneme_list.end()) ||
          (!in && f == m_rule_set[rule].phoneme_list.end()))
      {
        out.states.push_back(cl.states[i]);
      }
    }
  }
}


void
TriphoneSet::fill_context_cluster_statistics(ContextStateCluster &ccl)
{
  int i, j;
  MatrixD cov(m_dim, m_dim);
  VectorD tot_mean(m_dim);

  ccl.count = 0;
  for (i = 0; i < (int)ccl.states.size(); i++)
  {
    if (ccl.states[i]->count > 0)
    {
      ExtVectorD mean_vec(ccl.states[i]->mean, m_dim);
      ccl.count += ccl.states[i]->count;
      add(scaled(*(ccl.states[i]->cov), ccl.states[i]->count), cov);
      for (j = 0; j < m_dim; j++)
      {
        add(rows(cov)[j],
            scaled(mean_vec, ccl.states[i]->count*mean_vec[j]),
            rows(cov)[j]);
      }
      add(scaled(mean_vec, ccl.states[i]->count), tot_mean);
    }
  }
  if (ccl.count > 0)
  {
    scale(cov, 1.0/(double)ccl.count);
    scale(tot_mean, 1.0/(double)ccl.count);
    for (j = 0; j < m_dim; j++)
    {
      add(rows(cov)[j], scaled(tot_mean, -tot_mean[j]), rows(cov)[j]);
    }
    ccl.cov_det = cov_determinant(&cov);
  }
  else
  {
    ccl.cov_det = 0;
  }
}


double
TriphoneSet::cov_determinant(MatrixD *m)
{
  MatrixD temp(m_dim, m_dim);
  mtl::dense1D<int> pivots(m_dim, 0);
  double det;
  copy(*m, temp);
  lu_factor(temp, pivots);
  det = 1;
  for (int i = 0; i < m_dim; i++)
    det *= temp(i,i);
  det = fabs(det);
  return det;
}

void
TriphoneSet::load_rule_set(const std::string &filename)
{
  FILE *fp;
  char line[200], *token;
  DecisionRule rule;
  std::string tmp_string;

  fp = fopen(filename.c_str(), "r");
  if (fp == NULL)
  {
    fprintf(stderr, "Could not open file %s for reading\n", filename.c_str());
    exit(1);
  }
  while (fgets(line, 200, fp) != NULL)
  {
    token = strtok(line," \t\n");
    if (token == NULL)
    {
      fprintf(stderr,"TriphoneSet::load_rule_set: Invalid file format\n");
      exit(2);
    }
    rule.rule_name = token;
    rule.phoneme_list.clear();
    token = strtok(NULL, " \t\n");
    if (token == NULL)
    {
      fprintf(stderr, "TriphoneSet::load_rule_set: Invalid file format\n");
      exit(2);
    }
    tmp_string = token;
    if (tmp_string == "context")
    {
      token = strtok(NULL, " ,\n");
      if (token == NULL)
      {
        fprintf(stderr, "TriphoneSet::load_rule_set: Invalid file format\n");
        exit(2);
      }
      while (token != NULL)
      {
        tmp_string = token;
        rule.phoneme_list.push_back(tmp_string);
        token = strtok(NULL, " ,\n");
      }
      rule.rule_type = DecisionRule::LEFT_CONTEXT;
      m_rule_set.push_back(rule);
      rule.rule_type = DecisionRule::RIGHT_CONTEXT;
      m_rule_set.push_back(rule);
    }
    else if (tmp_string == "length")
    {
      rule.rule_type = DecisionRule::PHONEME_LENGTH;
      m_rule_set.push_back(rule);
    }
  }
  fclose(fp);
}


int
TriphoneSet::save_to_basebind(const std::string &filename,int initial_statenum)
{
  FILE *fp;
  int i, j, k, l;
  int statenum = initial_statenum;

  fp = fopen(filename.c_str(), "a");
  if (fp == NULL)
  {
    fprintf(stderr, "Could not open file %s for writing\n", filename.c_str());
    exit(1);
  }
  for (i = 0; i < (int)m_triphones.size(); i++)
  {
    for (j = 0; j < (int)m_triphones[i]->states.size(); j++)
      m_triphones[i]->states[j]->count = -1;
  }
  // Allocate state number
  for (i = 0; i < (int)m_tri_centers.size(); i++)
  {
    for (j = 0; j < (int)m_tri_centers[i].state_clusters.size(); j++)
    {
      for (k = 0; k < (int)m_tri_centers[i].state_clusters[j].size(); k++)
      {
        for (l = 0; l<(int)m_tri_centers[i].state_clusters[j][k].states.size();
             l++)
        {
          m_tri_centers[i].state_clusters[j][k].states[l]->count = statenum;
        }
        statenum++;
      }
    }
  }
  for (i = 0; i < (int)m_triphones.size(); i++)
  {
    fprintf(fp, "%s %d", get_triphone_label(m_triphones[i]->left,
                                         m_triphones[i]->center,
                                         m_triphones[i]->right).c_str(),
            (int)m_triphones[i]->states.size());
    for (j = 0; j < (int)m_triphones[i]->states.size(); j++)
    {
      assert( m_triphones[i]->states[j]->count > -1 );
      fprintf(fp, " %d", m_triphones[i]->states[j]->count);
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
  return statenum;
}

