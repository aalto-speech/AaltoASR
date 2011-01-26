#include <string.h>
#include "HmmNetBaumWelch.hh"
#include "str.hh"


HmmNetBaumWelch::LLType HmmNetBaumWelch::loglikelihoods;


HmmNetBaumWelch::HmmNetBaumWelch(FeatureGenerator &fea_gen, HmmSet &model)
  : m_fea_gen(fea_gen), m_model(model)
{
  m_initial_node_id = -1;
  m_final_node_id = -1;
  m_epsilon_string = ",";
  m_first_frame = m_last_frame = 0;
  m_eof_flag = false;
  m_cur_buffer = 0;
  m_collect_transitions = 0;
  m_acoustic_scale = 1;

  // Default mode
  m_mode = MODE_BAUM_WELCH;

  m_custom_data_callback = NULL; // Not in use by default

  // Default pruning thresholds
  m_forward_beam = 15;
  m_backward_beam = 100;

  // Total statistics
  m_sum_total_loglikelihood = loglikelihoods.zero();
  m_sum_total_custom_score = 0;
}

HmmNetBaumWelch::~HmmNetBaumWelch()
{
  close();
}

void
HmmNetBaumWelch::open(std::string ref_file)
{
  FILE *fp = fopen(ref_file.c_str(), "r");
  if (fp == NULL)
    throw std::string("Could not open file ")+ref_file;
  read_fst(fp);
  fclose(fp);
}

void 
HmmNetBaumWelch::read_fst(FILE *file)
{
  close();
  m_initial_node_id = -1;
  m_final_node_id = -1;

  std::string line;
  std::vector<std::string> fields;
  bool ok = true;

  while (str::read_line(&line, file, true)) {
    str::split(&line, " \t", true, &fields);
    if (fields[0] == "#FSTBinary")
      throw std::string("FSTBinary format not supported");

    // Initial node
    // 
    if (fields[0] == "I") {
      if (fields.size() != 2)
        ok = false;
      else
        m_initial_node_id = str::str2long(&fields[1], &ok);
      if (!ok || m_initial_node_id < 0)
        throw std::string("invalid initial node specification: ") + line;
    }
    // Final node
    //
    else if (fields[0] == "F") {
      if (m_final_node_id != -1)
        throw std::string("final node redefined: ") + line;

      if (fields.size() != 2 && fields.size() != 3)
        ok = false;
      else {
        m_final_node_id = str::str2long(&fields[1], &ok);
      }

      if (!ok || m_final_node_id < 0)
        throw std::string("invalid final node specification: ") + line;

      while ((int)m_nodes.size() <= m_final_node_id)
        m_nodes.push_back(Node(m_nodes.size()));
    }

    // Transition
    //
    else if (fields[0] == "T") {
      int source = -1;
      int target = -1;
      int in = -1;
      std::string label = "";
      double score = loglikelihoods.one();

      if (fields.size() < 3 || fields.size() > 6)
        ok = false;
      else {
        source = str::str2long(&fields[1], &ok);
        target = str::str2long(&fields[2], &ok);
        if (fields.size() > 3) {
          if (fields[3] == m_epsilon_string)
            in = EPSILON;
          else
          {
            if (fields[3].substr(0, 1) == "#")
            {
              // Special epsilon arc with triphone label
              in = EPSILON;
              label = fields[3].substr(1);
            }
            else
            {
              int n = fields[3].find_first_of('-');
              if (n > 0 && n < (int)fields[3].size())
              {
                std::string num = fields[3].substr(0, n);
                in = str::str2long(&num, &ok);
                label = fields[3].substr(n+1);
              }
              else
                throw std::string("HmmNetBaumWelch: Invalid input symbol ") + fields[3];
            }
          }
        }
//         if (fields.size() > 4) {
//           if (fields[4] == m_epsilon_string)
//             out = "";
//           else
//             out = fields[4];
//         }
        if (fields.size() > 5)
          score = str::str2float(&fields[5], &ok);
      }

      if (!ok || source < 0 || target < 0)
        throw std::string("HmmNetBaumWelch: Invalid transition specification: ") + line;

      while ((int)m_nodes.size() <= source || (int)m_nodes.size() <= target)
        m_nodes.push_back(Node(m_nodes.size()));

      Node &src_node = m_nodes.at(source);
      Node &target_node = m_nodes.at(target);
      int arc_index = m_arcs.size();
      m_arcs.push_back(Arc(arc_index, source, target, in, label, score));
      src_node.out_arcs.push_back(arc_index);
      target_node.in_arcs.push_back(arc_index);
    }
  }

  if (m_initial_node_id < 0)
    throw std::string("initial node not specified");
  if (m_final_node_id < 0)
    throw std::string("final node not specified");

  check_network_structure();
}


void
HmmNetBaumWelch::close(void)
{
  clear_bw_scores();
  
  m_transition_prob.clear();
  m_active_transition_table.clear();
  m_pdf_prob.clear();
  m_active_pdf_table.clear();
  m_active_arcs.clear();
  m_active_node_table[0].clear();
  m_active_node_table[1].clear();
  m_nodes.clear();
  m_arcs.clear();
}


void
HmmNetBaumWelch::set_frame_limits(int first_frame, int last_frame) 
{
  m_first_frame = first_frame;
  m_last_frame = last_frame;
}


void
HmmNetBaumWelch::set_pruning_thresholds(double backward, double forward)
{
  if (backward > 0)
    m_backward_beam = backward;
  if (forward > 0)
    m_forward_beam = forward;
}


void
HmmNetBaumWelch::check_network_structure(void)
{
  for (int i = 0; i < (int)m_nodes.size(); i++)
  {
    bool epsilon_transitions = false;
    int non_eps_in_transitions = 0;
    bool self_transition = false;
    
    // Check the in arcs
    for (int j = 0; j < (int)m_nodes[i].in_arcs.size(); j++)
    {
      int arc_id = m_nodes[i].in_arcs[j];
      if (m_arcs[arc_id].epsilon())
      {
        if (m_arcs[arc_id].target == m_arcs[arc_id].source)
          throw std::string("HmmNetBaumWelch: Epsilon self loops are not allowed");
        if (non_eps_in_transitions > 0)
          throw std::string("HmmNetBaumWelch: Both epsilon and normal transitions to a node (detected at ") + m_arcs[arc_id].label + std::string(")");
        epsilon_transitions = true;
      }
      else if (m_arcs[arc_id].target != m_arcs[arc_id].source)
      {
        if (epsilon_transitions)
          throw std::string("HmmNetBaumWelch: Both epsilon and normal transitions to a node (detected at ") + m_arcs[arc_id].label + std::string(")");
        non_eps_in_transitions++;
      }
      else
        self_transition = true;
    }

    int eps_out_transitions = 0;
    int non_eps_out_transitions = 0;

    // Check the out arcs
    for (int j = 0; j < (int)m_nodes[i].out_arcs.size(); j++)
    {
      int arc_id = m_nodes[i].out_arcs[j];
      if (m_arcs[arc_id].epsilon())
      {
        if (m_arcs[arc_id].target == m_arcs[arc_id].source)
          throw std::string("HmmNetBaumWelch: Epsilon self loops are not allowed");
        if (non_eps_out_transitions > 0)
          throw std::string("HmmNetBaumWelch: Both epsilon and normal transitions from a node (detected at ") + m_arcs[arc_id].label + std::string(")");
        eps_out_transitions++;
      }
      else if (m_arcs[arc_id].target != m_arcs[arc_id].source)
      {
        if (eps_out_transitions > 0)
          throw std::string("HmmNetBaumWelch: Both epsilon and normal transitions from a node (detected at ") + m_arcs[arc_id].label + std::string(")");
        non_eps_out_transitions++;
      }
      else
        self_transition = true;
    }

    if (non_eps_out_transitions > 1)
      throw std::string("HmmNetBaumWelch: Only one non epsilon transition from a node is allowed");
    if (non_eps_in_transitions>1 && (!eps_out_transitions || self_transition))
      throw std::string("HmmNetBaumWelch: Multiple non epsilon transitions to a node are allowed only when the node has only epsilon out transitions");

    if (eps_out_transitions > 1 && self_transition)
      throw std::string("HmmNetBaumWelch: A node may not have a self transition if it has multiple (epsilon) out transitions");

    if (eps_out_transitions > 0)
    {
      assert( non_eps_out_transitions == 0 ); // This is checked above
      m_nodes[i].num_epsilon_out = eps_out_transitions;
    }
  }

  if (m_nodes[m_initial_node_id].num_epsilon_out == 0)
    throw std::string("HmmNetBaumWelch: Initial node must have only epsilon out arcs!");
  if (m_nodes[m_initial_node_id].in_arcs.size() > 0)
    throw std::string("HmmNetBaumWelch: Initial node may not have in arcs!");
  if (m_nodes[m_final_node_id].out_arcs.size() > 0)
    throw std::string("HmmNetBaumWelch: Final node may not have out arcs!");
}


void
HmmNetBaumWelch::reset(void)
{
  m_eof_flag = false;
  m_current_frame = -1;
  // Clear the probabilities of the possibly active nodes
  for (int b = 0; b < 2; b++)
  {
    for (int i = 0; i < (int)m_active_node_table[b].size(); i++)
      m_nodes[m_active_node_table[b][i]].log_prob[b] = loglikelihoods.zero();
    m_active_node_table[b].clear();
  }
  m_transition_prob.clear();
  m_active_transition_table.clear();
  m_pdf_prob.clear();
  m_active_pdf_table.clear();
  m_active_arcs.clear();

  m_sum_total_loglikelihood = loglikelihoods.zero();
  m_sum_total_custom_score = 0;
}


bool
HmmNetBaumWelch::init_utterance_segmentation(void)
{
  reset();
  
  // Reserve a table for pdf occupancy probabilities
  m_pdf_prob.resize(m_model.num_emission_pdfs());

  if (m_collect_transitions)
    m_transition_prob.resize(m_model.num_transitions());

  // Fill the backward probabilities
  if (!fill_backward_probabilities())
    return false; // Backward beam should be increased

  // Compute the total likelihood and check the backward phase got
  // proper probabilities for the initial nodes
  compute_total_bw_scores();
  if (m_sum_total_loglikelihood <= loglikelihoods.zero())
    return false; // Backward beam should be increased
  
  return true;
}


bool
HmmNetBaumWelch::fill_backward_probabilities(void)
{
  int target_buffer = 0;
  double best_log_prob;
  //double prev_max_log_prob = loglikelihoods.one();
  
  clear_bw_scores();
  if (m_last_frame > 0)
    m_current_frame = m_last_frame + 1;
  else
    m_current_frame = m_fea_gen.last_frame() + 1;

  // Initialize the final node
  m_nodes[m_final_node_id].log_prob[target_buffer] = loglikelihoods.one();
  m_nodes[m_final_node_id].custom_score[target_buffer] = 0;
  assert( m_active_node_table[0].empty() );
  assert( m_active_node_table[1].empty() );
  m_active_node_table[target_buffer].push_back(m_final_node_id);

  // Propagate the epsilon arcs leading to the final node
  for (int i = 0; i < (int)m_active_node_table[target_buffer].size(); i++)
  {
    FeatureVec empty_fea_vec;
    backward_propagate_node_arcs(
      m_active_node_table[target_buffer][i],
      m_nodes[m_active_node_table[target_buffer][i]].log_prob[target_buffer],
      m_nodes[m_active_node_table[target_buffer][i]].custom_score[target_buffer],
      target_buffer, true, empty_fea_vec, loglikelihoods.zero());
  }

  while (m_current_frame > m_first_frame)
  {
    FeatureVec fea_vec = m_fea_gen.generate(--m_current_frame);
    int source_buffer = target_buffer;
    target_buffer ^= 1;

    best_log_prob = loglikelihoods.zero();

    m_model.reset_cache();

    // Iterate through active nodes and propagate the normal transitions.
    // Fill the backward probabilities for the arcs.
    for (int i = 0; i < (int)m_active_node_table[source_buffer].size(); i++)
    {
      if (m_active_node_table[source_buffer][i] != -1)
      {
        double cur_node_score =
          m_nodes[m_active_node_table[source_buffer][i]].log_prob[source_buffer];
        double temp = backward_propagate_node_arcs(
          m_active_node_table[source_buffer][i], cur_node_score,
          m_nodes[m_active_node_table[source_buffer][i]].custom_score[source_buffer],
          target_buffer, false, fea_vec, best_log_prob);
        
        if (temp > best_log_prob)
          best_log_prob = temp;

        // Reset the probability
        m_nodes[m_active_node_table[source_buffer][i]].log_prob[source_buffer]=
          loglikelihoods.zero();
      }
    }

    // Clear the old active nodes
    m_active_node_table[source_buffer].clear();

    assert( !m_active_node_table[target_buffer].empty() );
    
    // Iterate through active nodes and propagate the epsilon transitions.
    // The new active nodes are appended to the end of m_active_node_table.
    // The previous probabilities are taken from the target buffer. Also
    // fills the backward probabilities of epsilon arcs.
    // Identifies the best active node which will be moved to the first
    // position.
    double temp_max_log_prob = loglikelihoods.zero();
    int best_log_prob_node_index = -1;
    for (int i = 0; i < (int)m_active_node_table[target_buffer].size(); i++)
    {
      double cur_node_score =
        m_nodes[m_active_node_table[target_buffer][i]].log_prob[target_buffer];
      // Propagate the node only if its loglikelihood is within the beam
      if (loglikelihoods.divide(cur_node_score, best_log_prob) >
          -m_backward_beam)
      {
        // No pruning during these propagations!
        backward_propagate_node_arcs(
          m_active_node_table[target_buffer][i], cur_node_score,
          m_nodes[m_active_node_table[target_buffer][i]].custom_score[target_buffer],
          target_buffer, true, fea_vec, loglikelihoods.zero());

        if (cur_node_score > temp_max_log_prob &&
            (int)m_nodes[m_active_node_table[target_buffer][i]].out_arcs.size() >
            m_nodes[m_active_node_table[target_buffer][i]].num_epsilon_out)
        {
          // "The best node" is required to have non-epsilon arcs
          temp_max_log_prob = cur_node_score;
          best_log_prob_node_index = i;
        }
      }
      else
      {
        // Reset the probability
        m_nodes[m_active_node_table[target_buffer][i]].log_prob[target_buffer]=
          loglikelihoods.zero();
        // Do not propagate in the next frame
        m_active_node_table[target_buffer][i] = -1;
      }
    }

    // Move the best node to the beginning of the active node table to
    // enhance pruning
    if (best_log_prob_node_index != -1)
    {
      int temp_active_node = m_active_node_table[target_buffer][0];
      m_active_node_table[target_buffer][0] =
        m_active_node_table[target_buffer][best_log_prob_node_index];
      m_active_node_table[target_buffer][best_log_prob_node_index] =
        temp_active_node;
    }

  }

  // Clear the active nodes
  for (int i = 0; i < (int)m_active_node_table[target_buffer].size(); i++)
  {
    if (m_active_node_table[target_buffer][i] != -1)
      m_nodes[m_active_node_table[target_buffer][i]].log_prob[target_buffer] =
        loglikelihoods.zero();
  }
  m_active_node_table[target_buffer].clear();
  
  // Reset the current frame for the forward phase
  m_current_frame = -1;
  m_eof_flag = false;
  return true;
}


bool
HmmNetBaumWelch::next_frame(void)
{
  if (m_eof_flag)
    return false;
  
  if (m_current_frame == -1)
  {
    // Initialize forward phase
    m_current_frame = m_first_frame;
    assert( m_active_node_table[0].empty() );
    assert( m_active_node_table[1].empty() );
    m_cur_buffer = 0;
    for (int i = 0; i < (int)m_pdf_prob.size(); i++)
      m_pdf_prob[i] = loglikelihoods.zero();
    if (m_collect_transitions)
    {
      for (int i = 0; i < (int)m_transition_prob.size(); i++)
        m_transition_prob[i] = loglikelihoods.zero();
    }
    m_nodes[m_initial_node_id].log_prob[m_cur_buffer] = loglikelihoods.one();
    m_nodes[m_initial_node_id].custom_score[m_cur_buffer] = 0;
    m_active_node_table[m_cur_buffer].push_back(m_initial_node_id);
  }
  else
  {
    m_current_frame++;
  }

  // Check the frame limit
  if (m_last_frame > 0 && m_current_frame >= m_last_frame)
  {
    m_eof_flag = true;
    return false;
  }
  
  m_cur_buffer ^= 1;
  int source_buffer = m_cur_buffer^1;
  FeatureVec fea_vec = m_fea_gen.generate(m_current_frame);

  if (m_fea_gen.eof()) // Was EOF encountered?
  {
    m_eof_flag = true;
    return false;
  }

  m_active_arcs.clear();
  m_model.reset_cache();

  // Iterate through active nodes and propagate the epsilon transitions.
  // The new active nodes are appended to the end of m_active_node_table.
  for (int i = 0; i < (int)m_active_node_table[source_buffer].size(); i++)
  {
    forward_propagate_node_arcs(
      m_active_node_table[source_buffer][i],
      m_nodes[m_active_node_table[source_buffer][i]].log_prob[source_buffer],
      m_nodes[m_active_node_table[source_buffer][i]].custom_score[source_buffer],
      source_buffer, true, fea_vec);
  }

  // In Viterbi only the last node is active anymore, the others (if many)
  // have only epsilon out transitions which were just propagated.
  if (m_mode == MODE_VITERBI)
  {
    // Clear the other active nodes and leave the last one active
    for (int i = 0; i < (int)m_active_node_table[source_buffer].size()-1; i++)
    {
      m_nodes[m_active_node_table[source_buffer][i]].log_prob[source_buffer] =
        loglikelihoods.zero();
    }
    m_active_node_table[source_buffer].erase(
      m_active_node_table[source_buffer].begin(),
      m_active_node_table[source_buffer].end()-1);
  }

  // Iterate through active nodes and compute the PDF occupancy
  // probabilities
  for (int i = 0; i < (int)m_active_node_table[source_buffer].size(); i++)
  {
    forward_propagate_node_arcs(
      m_active_node_table[source_buffer][i],
      m_nodes[m_active_node_table[source_buffer][i]].log_prob[source_buffer],
      m_nodes[m_active_node_table[source_buffer][i]].custom_score[source_buffer],
      m_cur_buffer, false, fea_vec);
    // Clear the source probability of the active node
    m_nodes[m_active_node_table[source_buffer][i]].log_prob[source_buffer] =
      loglikelihoods.zero();
  }

  if (m_active_node_table[m_cur_buffer].empty())
    throw std::string("Baum-Welch failed during forward phase, try increasing the beam");

  // Clear the old active nodes
  m_active_node_table[source_buffer].clear();

  // Fill PDF and transition probabilities
  double normalizing_score = m_sum_total_loglikelihood;
  
  if (m_mode == MODE_VITERBI)
  {
    assert( m_active_arcs.size() == 1 );
    normalizing_score = m_active_arcs.back().score;
  }

  double cur_best_prob = -1;
  for (int i = 0; i < (int)m_active_arcs.size(); i++)
  {
    HmmTransition &tr = m_model.transition(
      m_arcs[m_active_arcs[i].arc_id].transition_id);
    int pdf_id = m_model.emission_pdf_index(tr.source_index);
    double cur_arc_prob = exp(loglikelihoods.divide(m_active_arcs[i].score,
                                                    normalizing_score));

    // Update the most probable label
    if (cur_best_prob < cur_arc_prob)
    {
      cur_best_prob = cur_arc_prob;
      m_most_probable_label = m_arcs[m_active_arcs[i].arc_id].label;
    }

    if (m_pdf_prob[pdf_id] <= loglikelihoods.zero())
    {
      // PDF didn't have probability for this frame
      m_pdf_prob[pdf_id] = cur_arc_prob;
      m_active_pdf_table.push_back(pdf_id);
    }
    else
    {
      // Add the probability to previous one
      m_pdf_prob[pdf_id] += cur_arc_prob;
    }
    if (m_collect_transitions)
    {
      int tr_id = m_arcs[m_active_arcs[i].arc_id].transition_id;
      if (m_transition_prob[tr_id] <= 0)
      {
        // Transition didn't have probability for this frame
        m_transition_prob[tr_id] = cur_arc_prob;
        m_active_transition_table.push_back(tr_id);
      }
      else
      {
        // Add the probability to previous one
        m_transition_prob[tr_id] += cur_arc_prob;
      }
    }
  }

  // SANITY CHECK: Total likelihood should match
  double total_prob = 0;
  for (int i = 0; i < (int)m_active_pdf_table.size(); i++)
    total_prob += m_pdf_prob[m_active_pdf_table[i]];
  if (fabs(1-total_prob)> 0.02) // Allow small deviation
  {
    fprintf(stderr, "Warning: Sum of PDF probabilities is %g at frame %i\n", total_prob, m_current_frame);
    if (total_prob < 0.01)
      exit(1); // Exit if the probability is completely wrong and small
  }
  // END OF SANITY CHECK
  
  // Fill the PDF probabilities and clear the active PDF probabilities
  m_pdf_prob_pairs.clear();
  for (int i = 0; i < (int)m_active_pdf_table.size(); i++)
  {
    Segmentator::IndexProbPair new_pdf_prob(
      m_active_pdf_table[i], m_pdf_prob[m_active_pdf_table[i]] / total_prob);
    m_pdf_prob_pairs.push_back(new_pdf_prob);
    m_pdf_prob[m_active_pdf_table[i]] = loglikelihoods.zero();
  }
  m_active_pdf_table.clear();

  if (m_collect_transitions)
  {
    m_transition_prob_pairs.clear();
    for (int i = 0; i < (int)m_active_transition_table.size(); i++)
    {
      Segmentator::IndexProbPair new_transition_prob(
        m_active_transition_table[i],
        m_transition_prob[m_active_transition_table[i]] / total_prob);
      m_transition_prob_pairs.push_back(new_transition_prob);
      m_transition_prob[m_active_transition_table[i]] = loglikelihoods.zero();
    }
    m_active_transition_table.clear();
  }

  return true;
}


void
HmmNetBaumWelch::fill_arc_info(std::vector<ArcInfo> &traversed_arcs)
{
  traversed_arcs.clear();
  double normalizing_score = m_sum_total_loglikelihood; 
  if (m_mode == MODE_VITERBI)
  {
    assert( m_active_arcs.size() == 1 );
    normalizing_score = m_active_arcs.back().score;
  }
  traversed_arcs.reserve((int)m_active_arcs.size());
  for (int i = 0; i < (int)m_active_arcs.size(); i++)
  {
    HmmTransition &tr = m_model.transition(
      m_arcs[m_active_arcs[i].arc_id].transition_id);
    int pdf_id = m_model.emission_pdf_index(tr.source_index);
    double cur_arc_prob = exp(loglikelihoods.divide(m_active_arcs[i].score,
                                                    normalizing_score));
    traversed_arcs.push_back(ArcInfo(m_arcs[m_active_arcs[i].arc_id].label,
                                     cur_arc_prob, pdf_id,
                                     m_active_arcs[i].custom_score));
  }
}


double
HmmNetBaumWelch::backward_propagate_node_arcs(int node_id, double cur_score,
                                              double cur_custom_score,
                                              int target_buffer, bool epsilons,
                                              FeatureVec &fea_vec,
                                              double ref_log_prob)
{
  double cur_max_log_prob = loglikelihoods.zero();
  double arc_score;
  double new_score;
  double arc_custom_score;
  
  for (int a = 0; a < (int)m_nodes[node_id].in_arcs.size(); a++)
  {
    int arc_id = m_nodes[node_id].in_arcs[a];
    int next_node_id = m_arcs[arc_id].source;
    if ((!epsilons && m_arcs[arc_id].epsilon()) ||
        (epsilons && !m_arcs[arc_id].epsilon()))
      continue;
    
    fill_arc_scores(arc_id, fea_vec, cur_custom_score, &arc_score,
                    &arc_custom_score);

    new_score = loglikelihoods.times(cur_score, arc_score);

    if (new_score > loglikelihoods.zero() &&
        loglikelihoods.divide(new_score, ref_log_prob) > -m_backward_beam)
    {
      // Fill in the backward probability for the arc
      m_arcs[arc_id].bw_scores.set_new_score(m_current_frame, new_score);
      if (m_custom_data_callback != NULL)
      {
        m_arcs[arc_id].bw_custom_data.set_new_score(m_current_frame,
                                                    arc_custom_score);
      }

      // Propagate the probability of the arc to the next node
      if (m_nodes[next_node_id].log_prob[target_buffer] <=
          loglikelihoods.zero())
      {
        // The node was empty, make it active
        m_active_node_table[target_buffer].push_back(next_node_id);
        m_nodes[next_node_id].log_prob[target_buffer] = new_score;
        m_nodes[next_node_id].custom_score[target_buffer] = arc_custom_score;
      }
      else
      {
        // Node had a probability already
        if (m_mode == MODE_BAUM_WELCH ||
            (m_mode == MODE_EXTENDED_VITERBI &&
             m_nodes[next_node_id].num_epsilon_out > 0))
        {
          // Add the new probability to the previous one
          double prev_log_prob = m_nodes[next_node_id].log_prob[target_buffer];
          m_nodes[next_node_id].log_prob[target_buffer] =
            loglikelihoods.add(prev_log_prob, new_score);
          if (m_custom_data_callback != NULL)
          {
            update_node_custom_score(next_node_id, target_buffer,
                                     prev_log_prob,new_score,arc_custom_score);
          }
        }
        else  // m_mode == MODE_VITERBI ||
              // (m_mode == MODE_EXTENDED_VITERBI &&
              //  m_nodes[next_node_id].num_epsilon_out == 0)
        {
          // Update the probability if it is smaller than the new one
          if (m_nodes[next_node_id].log_prob[target_buffer] < new_score)
          {
            m_nodes[next_node_id].log_prob[target_buffer] = new_score;
            m_nodes[next_node_id].custom_score[target_buffer]=arc_custom_score;
          }
        }
      }
      // Save the maximum node probability for pruning
      if (m_nodes[next_node_id].log_prob[target_buffer] > cur_max_log_prob)
        cur_max_log_prob = m_nodes[next_node_id].log_prob[target_buffer];
    }
  }
  return cur_max_log_prob;
}


void
HmmNetBaumWelch::forward_propagate_node_arcs(int node_id, double cur_score,
                                             double cur_custom_score,
                                             int target_buffer, bool epsilons,
                                             FeatureVec &fea_vec)
{
  double arc_score;
  double new_score;
  double arc_custom_score;
  double viterbi_best_ll = loglikelihoods.zero();
  double viterbi_best_new_score = loglikelihoods.zero();
  double viterbi_best_arc_custom_score = 0;
  double viterbi_best_total_custom_score = 0;
  int viterbi_best_arc_id = -1;
  
  for (int a = 0; a < (int)m_nodes[node_id].out_arcs.size(); a++)
  {
    int arc_id = m_nodes[node_id].out_arcs[a];
    int next_node_id = m_arcs[arc_id].target;
    if ((!epsilons && m_arcs[arc_id].epsilon()) ||
        (epsilons && !m_arcs[arc_id].epsilon()))
      continue;

    double backward_loglikelihood =
      m_arcs[arc_id].bw_scores.get_score(m_current_frame);
    double total_arc_loglikelihood =
      loglikelihoods.times(cur_score, backward_loglikelihood);
    
    if (m_mode == MODE_BAUM_WELCH || m_mode == MODE_EXTENDED_VITERBI)
    {
      // If the total likelihood (forward+backward) of the path
      // is worse than the forward beam compared to the sum of
      // likelihoods of all paths, discard this path.
      // NOTE: This handles also the case when backward score is zero
      if (total_arc_loglikelihood <
          m_sum_total_loglikelihood - m_forward_beam)
        continue;
    }

    fill_arc_scores(arc_id, fea_vec, cur_custom_score, &arc_score,
                    &arc_custom_score);

    new_score = loglikelihoods.times(cur_score, arc_score);

    if (new_score > loglikelihoods.zero())
    {
      if (!epsilons && m_mode == MODE_BAUM_WELCH)
      {
        double total_custom_score = cur_custom_score;
        if (m_custom_data_callback != NULL)
        {
          total_custom_score +=
            m_arcs[arc_id].bw_custom_data.get_score(m_current_frame);
        }

        m_active_arcs.push_back(TraversedArc(arc_id, total_arc_loglikelihood,
                                             total_custom_score));
      }

      if (m_mode == MODE_BAUM_WELCH ||
          (m_mode == MODE_EXTENDED_VITERBI &&
           m_nodes[node_id].num_epsilon_out > 0))
      {
        // Propagate the probability of the arc to the next node
        if (m_nodes[next_node_id].log_prob[target_buffer] <=
            loglikelihoods.zero())
        {
          // The node was empty, make it active
          m_active_node_table[target_buffer].push_back(next_node_id);
          m_nodes[next_node_id].log_prob[target_buffer] = new_score;
          m_nodes[next_node_id].custom_score[target_buffer] = arc_custom_score;
        }
        else
        {
          // Node had a probability already
          // Add the new probability to the previous one
          double prev_log_prob = m_nodes[next_node_id].log_prob[target_buffer];
          m_nodes[next_node_id].log_prob[target_buffer] =
            loglikelihoods.add(prev_log_prob, new_score);
          if (m_custom_data_callback != NULL)
          {
            update_node_custom_score(next_node_id, target_buffer,
                                     prev_log_prob,new_score,arc_custom_score);
          }
        }
      }
      else // m_mode == MODE_VITERBI || (m_mode == MODE_EXTENDED_VITERBI &&
           //                            m_nodes[node_id].num_epsilon_out == 0)
      {        
        if (total_arc_loglikelihood > viterbi_best_ll)
        {
          viterbi_best_ll = total_arc_loglikelihood;
          viterbi_best_new_score = new_score;
          viterbi_best_arc_id = arc_id;
          viterbi_best_arc_custom_score = arc_custom_score;
          viterbi_best_total_custom_score = cur_custom_score;
          if (m_custom_data_callback != NULL)
            viterbi_best_total_custom_score +=
              m_arcs[arc_id].bw_custom_data.get_score(m_current_frame);
        }
      }
    }
  }

  if (viterbi_best_arc_id != -1)
  {
    int next_node_id = m_arcs[viterbi_best_arc_id].target;
    assert( m_mode == MODE_EXTENDED_VITERBI ||
            (m_mode == MODE_VITERBI &&
             m_nodes[next_node_id].log_prob[target_buffer] <=
             loglikelihoods.zero()) );

    if (!m_arcs[viterbi_best_arc_id].epsilon())
    {
      // For MODE_VITERBI, this is the only non epsilon arc traversed at
      // this frame. For MODE_EXTENDED_VITERBI this is the only non epsilon
      // arc traversed at this frame in this particular path or sum
      // of paths (branching occurs only in epsilon arcs!).
      m_active_arcs.push_back(TraversedArc(viterbi_best_arc_id,
                                           viterbi_best_ll,
                                           viterbi_best_total_custom_score));
    }

    if (m_mode == MODE_VITERBI ||
        m_nodes[next_node_id].log_prob[target_buffer] <= loglikelihoods.zero())
    {
      // The node was empty, make it active
      m_active_node_table[target_buffer].push_back(next_node_id);
      m_nodes[next_node_id].log_prob[target_buffer] = viterbi_best_new_score;
      m_nodes[next_node_id].custom_score[target_buffer] =
        viterbi_best_arc_custom_score;
    }
    else
    {
      // Node had a probability already (m_mode == MODE_EXTENDED_VITERBI).
      // Add the new probability to the previous one
      double prev_log_prob = m_nodes[next_node_id].log_prob[target_buffer];
      m_nodes[next_node_id].log_prob[target_buffer] =
        loglikelihoods.add(prev_log_prob, viterbi_best_new_score);
      if (m_custom_data_callback != NULL)
      {
        update_node_custom_score(next_node_id, target_buffer,
                                 prev_log_prob, viterbi_best_new_score,
                                 viterbi_best_arc_custom_score);
      }
    }
  }
}


void
HmmNetBaumWelch::fill_arc_scores(int arc_id, FeatureVec &fea_vec,
                                 double cur_custom_score,
                                 double *arc_score, double *arc_custom_score)
{
  *arc_custom_score = cur_custom_score;
  if (m_arcs[arc_id].epsilon())
    *arc_score = m_arcs[arc_id].score;
  else
  {
    HmmTransition &tr = m_model.transition(m_arcs[arc_id].transition_id);
    double model_score = m_model.state_likelihood(tr.source_index,
                                                  fea_vec);
    *arc_score = loglikelihoods.times(
      m_acoustic_scale*(util::safe_log(model_score * tr.prob)),
      m_arcs[arc_id].score);
    if (m_custom_data_callback != NULL)
    {
      *arc_custom_score +=
        m_custom_data_callback->custom_data_value(m_current_frame,
                                                  m_arcs[arc_id]);
    }
  }
}


void
HmmNetBaumWelch::update_node_custom_score(int node_id, int target_buffer,
                                          double old_log_prob,
                                          double new_log_prob,
                                          double new_custom_score)
{
  double p1 =
    exp(old_log_prob - m_nodes[node_id].log_prob[target_buffer]);
  double p2 =
    exp(new_log_prob - m_nodes[node_id].log_prob[target_buffer]);
  m_nodes[node_id].custom_score[target_buffer] =
    m_nodes[node_id].custom_score[target_buffer] * p1 +
    new_custom_score * p2;
}


void
HmmNetBaumWelch::compute_total_bw_scores(void)
{
  double ll_sum = loglikelihoods.zero();
  double custom_sum = 0;
  int node_id = m_initial_node_id;
  int frame = m_first_frame;

  assert( m_nodes[node_id].num_epsilon_out > 0 );
  
  for (int i = 0; i < (int)m_nodes[node_id].out_arcs.size(); i++)
  {
    int arc_id = m_nodes[node_id].out_arcs[i];
    double cur_addition = m_arcs[arc_id].bw_scores.get_score(frame);
    double cur_custom_score = m_arcs[arc_id].bw_custom_data.get_score(frame);
    if (cur_addition > loglikelihoods.zero())
    {
      if (m_mode == MODE_BAUM_WELCH || m_mode == MODE_EXTENDED_VITERBI)
      {
        double prev_sum = ll_sum;
        ll_sum = loglikelihoods.add(ll_sum, cur_addition);
        double p1 = exp(prev_sum - ll_sum);
        double p2 = exp(cur_addition - ll_sum);
        custom_sum = custom_sum * p1 + cur_custom_score * p2;
      }
      else if (m_mode == MODE_VITERBI)
      {
        if (cur_addition > ll_sum)
        {
          ll_sum = cur_addition;
          custom_sum = cur_custom_score;
        }
      }
    }
  }

  m_sum_total_loglikelihood = ll_sum;
  m_sum_total_custom_score = custom_sum;
}


void
HmmNetBaumWelch::clear_bw_scores(void)
{
  for (int i = 0; i < (int)m_arcs.size(); i++) {
    m_arcs[i].bw_scores.clear();
    m_arcs[i].bw_custom_data.clear();
  }
}


void
HmmNetBaumWelch::FrameScores::set_score(int frame, double score)
{
  if (!frame_blocks.empty() && frame_blocks.back().start == frame)
    score_table[num_scores-1] = score;
  else
    set_new_score(frame, score);
}

void
HmmNetBaumWelch::FrameScores::set_new_score(int frame, double score)
{
  if (num_scores == score_table_size)
  {
    // Increase the size of the score table
    int new_size = std::max(score_table_size*2, score_table_size+4);
    double *new_table = new double[new_size];
    if (score_table_size > 0)
    {
      memcpy(new_table, score_table, score_table_size*sizeof(double));
      delete [] score_table;
    }
    score_table = new_table;
    score_table_size = new_size;
  }
  if (!frame_blocks.empty() && frame_blocks.back().start == frame+1)
  {
    // Add to previous block
    frame_blocks.back().start--;
    score_table[num_scores++] = score;
  }
  else
  {
    // Create a new block
    FrameBlock b(frame, frame, num_scores);
    frame_blocks.push_back(b);
    score_table[num_scores++] = score;
  }
}

double
HmmNetBaumWelch::FrameScores::get_score(int frame)
{
  for (int i = 0; i < (int)frame_blocks.size() && frame <= frame_blocks[i].end;
       i++)
  {
    if (frame >= frame_blocks[i].start && frame <= frame_blocks[i].end)
    {
      // Found the block
      return score_table[frame_blocks[i].buf_start +
                         frame_blocks[i].end-frame];
    }
  }
  // There is no score for this frame
  return HmmNetBaumWelch::loglikelihoods.zero();
}

void
HmmNetBaumWelch::FrameScores::clear(void)
{
  if (score_table_size > 0)
    delete [] score_table;
  score_table_size = 0;
  num_scores = 0;
  frame_blocks.clear();
}
