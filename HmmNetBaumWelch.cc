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

  // Default pruning thresholds
  m_forward_beam = 15;
  m_backward_beam = 100;
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
      std::string out = "";
      double score = 0;

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
            in = str::str2long(&fields[3], &ok);
          }
        }
        if (fields.size() > 4) {
          if (fields[4] == m_epsilon_string)
            out = "";
          else
            out = fields[4];
        }
        if (fields.size() > 5)
          score = str::str2float(&fields[5], &ok);
      }

      if (!ok || source < 0 || target < 0)
        throw std::string("invalid transition specification: ") + line;

      while ((int)m_nodes.size() <= source || (int)m_nodes.size() <= target)
        m_nodes.push_back(Node(m_nodes.size()));

      Node &src_node = m_nodes.at(source);
      Node &target_node = m_nodes.at(target);
      int arc_index = m_arcs.size();
      m_arcs.push_back(Arc(arc_index, source, target, in, out, score));
      src_node.out_arcs.push_back(arc_index);
      target_node.in_arcs.push_back(arc_index);
    }
  }

  if (m_initial_node_id < 0)
    throw std::string("initial node not specified");
  if (m_final_node_id < 0)
    throw std::string("final node not specified");
}


void
HmmNetBaumWelch::close(void)
{
  clear_bw_scores();
  m_pdf_prob.clear();
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
HmmNetBaumWelch::reset(void)
{
  m_eof_flag = false;
  m_current_frame = -1;
  // Clear the probabilities of possible active nodes
  for (int i = 0; i < (int)m_active_node_table[m_cur_buffer].size(); i++)
    m_nodes[m_active_node_table[m_cur_buffer][i]].prob[m_cur_buffer] =
      loglikelihoods.zero();
  m_active_node_table[m_cur_buffer].clear();
}


void
HmmNetBaumWelch::init_utterance_segmentation(void)
{
  // Reserve a table for pdf occupancy probabilities
  m_pdf_prob.resize(m_model.num_emission_pdfs());

  if (m_collect_transitions)
    m_transition_prob.resize(m_model.num_transitions());

  // Fill the backward probabilities
  fill_backward_probabilities();
}


void
HmmNetBaumWelch::fill_backward_probabilities(void)
{
  int target_buffer = 0;
  double max_prob;
  double prev_max_prob = 0;
  
  clear_bw_scores();
  if (m_last_frame > 0)
    m_current_frame = m_last_frame + 1;
  else
    m_current_frame = m_fea_gen.last_frame() + 1;

  // Initialize the final node
  m_nodes[m_final_node_id].prob[target_buffer] = loglikelihoods.one();
  assert( m_active_node_table[0].empty() );
  assert( m_active_node_table[1].empty() );
  m_active_node_table[target_buffer].push_back(m_final_node_id);

  while (m_current_frame > m_first_frame)
  {
    FeatureVec fea_vec = m_fea_gen.generate(--m_current_frame);
    target_buffer ^= 1;
    int source_buffer = target_buffer^1;

    max_prob = loglikelihoods.zero();

    m_model.reset_cache();

    // Iterate through active nodes and fill the backward probabilities
    // for the arcs. The probability of a node is the log sum of all arcs
    // leaving from that node.
    for (int i = 0; i < (int)m_active_node_table[source_buffer].size(); i++)
    {
      double cur_node_score =
        m_nodes[m_active_node_table[source_buffer][i]].prob[source_buffer];
      if (loglikelihoods.divide(cur_node_score, prev_max_prob) >
          -m_backward_beam)
      {
        double temp = propagate_node_arcs(
          m_active_node_table[source_buffer][i], false, cur_node_score,
          target_buffer, fea_vec);
        if (temp > max_prob)
          max_prob = temp;
      }

      // Reset the probability
      m_nodes[m_active_node_table[source_buffer][i]].prob[source_buffer] =
        loglikelihoods.zero();
    }

    if (m_active_node_table[target_buffer].empty())
    {
      throw std::string("Baum-Welch failed during backward phase, try increasing the beam");
    }

    // Use previous maximum probability for pruning
    prev_max_prob = max_prob;
    
    // Clear the old active nodes
    m_active_node_table[source_buffer].clear();
  }

  // Clear the active nodes
  for (int i = 0; i < (int)m_active_node_table[target_buffer].size(); i++)
    m_nodes[m_active_node_table[target_buffer][i]].prob[target_buffer] =
      loglikelihoods.zero();
  m_active_node_table[target_buffer].clear();
  
  // Reset the current frame for the forward phase
  m_current_frame = -1;
  m_eof_flag = false;
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
      m_pdf_prob[i] = 0;
    if (m_collect_transitions)
    {
      for (int i = 0; i < (int)m_transition_prob.size(); i++)
        m_transition_prob[i] = 0;
    }
    m_nodes[m_initial_node_id].prob[m_cur_buffer] = loglikelihoods.one();
    m_active_node_table[m_cur_buffer].push_back(m_initial_node_id);

    // Compute the total loglikelihood
    m_sum_total_loglikelihood=compute_sum_bw_loglikelihoods(m_initial_node_id);
  }
  else
  {
    m_current_frame++;
  }

  // Check frame limit
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

  m_model.reset_cache();

  // Iterate through active nodes and compute the PDF occupancy
  // probabilities
  for (int i = 0; i < (int)m_active_node_table[source_buffer].size(); i++)
  {
    propagate_node_arcs(
      m_active_node_table[source_buffer][i], true,
      m_nodes[m_active_node_table[source_buffer][i]].prob[source_buffer],
      m_cur_buffer, fea_vec);
    // Clear the source probability of the active node
    m_nodes[m_active_node_table[source_buffer][i]].prob[source_buffer] =
      loglikelihoods.zero();
  }

  if (m_active_node_table[m_cur_buffer].empty())
    throw std::string("Baum-Welch failed during forward phase, try increasing the beam");

  // Clear the old active nodes
  m_active_node_table[source_buffer].clear();

  // SANITY CHECK: Total likelihood must match
  double total_prob = 0;
  for (int i = 0; i < (int)m_active_pdf_table.size(); i++)
    total_prob += m_pdf_prob[m_active_pdf_table[i]];
  if (fabs(1-total_prob)> 0.02) // Allow small deviation
  {
    fprintf(stderr, "Total likelihood does not match, sum of PDF probabilities equals %g\n", total_prob);
    exit(1);
  }
  // END OF SANITY CHECK  
  
  // Fill the PDF probabilities and clear the active PDF probabilities
  m_pdf_prob_pairs.clear();
  for (int i = 0; i < (int)m_active_pdf_table.size(); i++)
  {
    Segmentator::IndexProbPair new_pdf_prob(
      m_active_pdf_table[i], m_pdf_prob[m_active_pdf_table[i]] / total_prob);
    m_pdf_prob_pairs.push_back(new_pdf_prob);
    m_pdf_prob[m_active_pdf_table[i]] = 0;
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
      m_transition_prob[m_active_transition_table[i]] = 0;
    }
    m_active_transition_table.clear();
  }

  return true;
}


double
HmmNetBaumWelch::propagate_node_arcs(int node_id, bool forward,
                                     double cur_score, int target_buffer,
                                     FeatureVec &fea_vec)
{
  double cur_max_prob = loglikelihoods.zero();
  int num_arcs = (int)(forward?m_nodes[node_id].out_arcs.size():
                       m_nodes[node_id].in_arcs.size());

  for (int a = 0; a < num_arcs; a++)
  {
    int arc_id = (forward?m_nodes[node_id].out_arcs[a]:
                  m_nodes[node_id].in_arcs[a]);
    int next_node_id = (forward?m_arcs[arc_id].target:
                        m_arcs[arc_id].source);

    
    if (m_arcs[arc_id].epsilon())
    {
      // Propagate through epsilon arc
      double new_score = loglikelihoods.times(cur_score,m_arcs[arc_id].score);
      double temp = propagate_node_arcs(next_node_id, forward,
                                        new_score, target_buffer, fea_vec);
      if (temp > cur_max_prob)
        cur_max_prob = temp;
    }
    else
    {
      HmmTransition &tr = m_model.transition(m_arcs[arc_id].transition_id);
      double arc_score = loglikelihoods.times(
        m_acoustic_scale*(log(m_model.state_likelihood(tr.source_index,
                                                       fea_vec)) + tr.prob),
        m_arcs[arc_id].score);
      double total_score = loglikelihoods.times(cur_score, arc_score);

      if (total_score > loglikelihoods.zero())
      {
        if (forward)
        {
          int pdf_id = m_model.emission_pdf_index(tr.source_index);
          double backward_likelihood =
            m_arcs[arc_id].bw_scores.get_log_prob(m_current_frame);
          double cur_arc_likelihood =
            loglikelihoods.times(cur_score, backward_likelihood);

          // If the total likelihood (forward+backward) of the path
          // is worse than the forward beam compared to the sum of
          // likelihoods of all paths, discard this path.
          // NOTE: This handles also the case when backward score is zero
          if (cur_arc_likelihood < m_sum_total_loglikelihood - m_forward_beam)
            continue;

          double cur_arc_prob = exp(
            loglikelihoods.divide(cur_arc_likelihood,
                                  m_sum_total_loglikelihood));
          
          if (m_pdf_prob[pdf_id] <= 0)
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
            int tr_id = m_arcs[arc_id].transition_id;
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
        else // !forward
        {
          // Fill in the backward probability for the arc
          m_arcs[arc_id].bw_scores.add_log_prob(m_current_frame,
                                                total_score);
        }

        // Add the probability of the arc to the next node
        if (m_nodes[next_node_id].prob[target_buffer] <=
            loglikelihoods.zero())
        {
          // Node was empty (at next frame), make it active
          m_active_node_table[target_buffer].push_back(next_node_id);
          m_nodes[next_node_id].prob[target_buffer] = total_score;
        }
        else
        {
          // Node had a probability already, add the new probability
          m_nodes[next_node_id].prob[target_buffer] =
            loglikelihoods.add(m_nodes[next_node_id].prob[target_buffer],
                               total_score);
        }
        // Save the maximum node probability for pruning
        if (m_nodes[next_node_id].prob[target_buffer] > cur_max_prob)
          cur_max_prob = m_nodes[next_node_id].prob[target_buffer];
      }
    }
  }
  return cur_max_prob;
}


double
HmmNetBaumWelch::compute_sum_bw_loglikelihoods(int node_id)
{
  double sum = loglikelihoods.zero();

  for (int i = 0; i < (int)m_nodes[node_id].out_arcs.size(); i++)
  {
    int arc_id = m_nodes[node_id].out_arcs[i];
    double cur_addition;
    if (m_arcs[arc_id].epsilon())
    {
      cur_addition = loglikelihoods.times(
        m_arcs[arc_id].score,
        compute_sum_bw_loglikelihoods(m_arcs[arc_id].target));
    }
    else
    {
      cur_addition = m_arcs[arc_id].bw_scores.get_log_prob(m_current_frame);
    }
    sum = loglikelihoods.add(sum, cur_addition);
  }
  return sum;
}


void
HmmNetBaumWelch::clear_bw_scores(void)
{
  for (int i = 0; i < (int)m_arcs.size(); i++) {
    m_arcs[i].bw_scores.clear();
  }
}



void
HmmNetBaumWelch::FrameProbs::add_log_prob(int frame, double prob)
{
  if (num_probs == prob_table_size)
  {
    // Increase the size of the probability table
    int new_size = std::max(prob_table_size*2, prob_table_size+4);
    double *new_table = new double[new_size];
    if (prob_table_size > 0)
    {
      memcpy(new_table, log_prob_table, prob_table_size*sizeof(double));
      delete log_prob_table;
    }
    log_prob_table = new_table;
    prob_table_size = new_size;
  }
  if (!frame_blocks.empty() && frame_blocks.back().start == frame+1)
  {
    // Add to previous block
    frame_blocks.back().start--;
    log_prob_table[num_probs++] = prob;
  }
  else
  {
    // Create a new block
    FrameBlock b(frame, frame, num_probs);
    frame_blocks.push_back(b);
    log_prob_table[num_probs++] = prob;
  }
}

double
HmmNetBaumWelch::FrameProbs::get_log_prob(int frame)
{
  for (int i = 0; i < (int)frame_blocks.size() && frame <= frame_blocks[i].end;
       i++)
  {
    if (frame >= frame_blocks[i].start && frame <= frame_blocks[i].end)
    {
      // Found the block
      return log_prob_table[frame_blocks[i].buf_start +
                            frame_blocks[i].end-frame];
    }
  }
  return HmmNetBaumWelch::loglikelihoods.zero(); // There is no probability for this frame
}

void
HmmNetBaumWelch::FrameProbs::clear(void)
{
  if (prob_table_size > 0)
    delete log_prob_table;
  prob_table_size = 0;
  frame_blocks.clear();
}
