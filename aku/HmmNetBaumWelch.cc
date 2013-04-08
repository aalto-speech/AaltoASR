#include "HmmNetBaumWelch.hh"
#include "str.hh"
#include <list>
#include <set>
#include <stack>
#include <algorithm>
#include <limits.h>
#include <string.h>


namespace aku {


HmmNetBaumWelch::LLType HmmNetBaumWelch::loglikelihoods;


HmmNetBaumWelch::HmmNetBaumWelch(FeatureGenerator &fea_gen, HmmSet &model)
  : m_fea_gen(fea_gen), m_model(model)
{
  m_initial_node_id = -1;
  m_final_node_id = -1;
  m_epsilon_string = ",";
  m_first_frame = m_last_frame = 0;
  m_eof_flag = false;
  m_collect_transitions = false;
  m_acoustic_scale = 1;

  m_features_generated = false;
  m_bw_scores_computed = false;
  m_segmentator_seglat = NULL;
  m_cur_frame = -1;

  m_total_score = loglikelihoods.zero();
  
  // Default modes
  m_segmentation_mode = MODE_BAUM_WELCH;
  m_use_static_scores = true;
  m_use_transition_probabilities = true;
  
  // Default pruning thresholds
  m_forward_beam = 15;
  m_backward_beam = 200;
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
    throw std::string("HmmNetBaumWelch::open: Could not open file `") + ref_file + std::string("'.");
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
  std::vector< LatticeLabel > orig_arc_labels;
  
  while (str::read_line(&line, file, true))
  {
    str::split(&line, " \t", true, &fields);
    if (fields[0] == "#FSTBinary")
      throw std::string("FSTBinary format not supported");

    if (fields[0] == "I") // Initial node
    {
      if (m_initial_node_id != -1)
        throw std::string("Initial node redefined: ") + line;
      if (fields.size() != 2)
        ok = false;
      else
        m_initial_node_id = str::str2long(&fields[1], &ok);
      if (!ok || m_initial_node_id < 0)
        throw std::string("Invalid initial node specification: ") + line;
    }
    else if (fields[0] == "F") // Final node
    {
      if (m_final_node_id != -1)
        throw std::string("Final node redefined: ") + line;
      if (fields.size() != 2)
        ok = false;
      else
        m_final_node_id = str::str2long(&fields[1], &ok);
      if (!ok || m_final_node_id < 0)
        throw std::string("Invalid final node specification: ") + line;
    }
    else if (fields[0] == "T") // Transition
    {
      int source = -1;
      int target = -1;
      int tr_index = EPSILON;
      std::string in_label = "";
      std::string out_label = "";
      double score = loglikelihoods.one();

      if (fields.size() < 3 || fields.size() > 6)
        ok = false;
      else
      {
        source = str::str2long(&fields[1], &ok);
        target = str::str2long(&fields[2], &ok);

        // Create the necessary nodes. NOTE! Node numbers should be continuous
        while ((int)m_nodes.size() <= source || (int)m_nodes.size() <= target)
          m_nodes.push_back(Node(m_nodes.size()));

        if (fields.size() > 3)
        {
          if (fields[3] != m_epsilon_string)
            in_label = fields[3];
          if (fields[3].size() > 0 && fields[3][0] != '#' &&
              fields.size() > 4)
          {
            if (fields[4] != m_epsilon_string)
              out_label = fields[4];
          }
          if (fields.size() > 5)
            score = str::str2float(&fields[5], &ok);
        }

        if (!ok || source < 0 || target < 0)
          throw std::string("HmmNetBaumWelch: Invalid transition specification: ") + line;

        LatticeLabel label(in_label, out_label);
        tr_index = label.get_transition_index();
        Node &src_node = m_nodes.at(source);
        Node &target_node = m_nodes.at(target);
        int arc_index = m_arcs.size();
        m_arcs.push_back(Arc(source, target, -1, tr_index,
                             label.get_label(), score));
        orig_arc_labels.push_back(label);
        src_node.out_arcs.push_back(arc_index);
        target_node.in_arcs.push_back(arc_index);

        if (source == target)
          m_nodes[source].self_transition = true;
      }
    }
  }

  if (m_initial_node_id < 0)
    throw std::string("initial node not specified");
  if (m_final_node_id < 0)
    throw std::string("final node not specified");

  check_network_structure();

  // Process the arcs in topological order and set up the parent logical arcs
  std::stack<int> active_nodes;
  std::vector<int> visit_count;
  typedef std::map< std::string, int > LabelIndexMap;
  std::vector< LabelIndexMap > node_logical_arcs;
  int nodes_processed = 0;
  active_nodes.push(m_initial_node_id);
  visit_count.resize(m_nodes.size(), 0);
  node_logical_arcs.resize(m_nodes.size());
  
  while (!active_nodes.empty())
  {
    int cur_node = active_nodes.top();
    active_nodes.pop();
    nodes_processed++;

    if (nodes_processed > (int)m_nodes.size())
      throw std::string("Error in creating logical arcs for the search network");
    
    // Process the physical arcs from the current node
    for (int i = 0; i < (int)m_nodes[cur_node].out_arcs.size(); i++)
    {
      int cur_arc = m_nodes[cur_node].out_arcs[i];
      int target_node = m_arcs[cur_arc].target;
      assert( m_arcs[cur_arc].source == cur_node );
      if (cur_node != target_node) // Don't count the self-transitions
      {
        visit_count[target_node]++;
        // Activate the target node if all the incoming arcs have
        // been processed.
        if (visit_count[target_node] +
            (m_nodes[target_node].self_transition?1:0) >=
            (int)m_nodes[target_node].in_arcs.size())
          active_nodes.push(target_node);
      }

      if (orig_arc_labels[cur_arc].is_epsilon())
      {
        // Logical arcs do not extend beyond epsilon arcs (?)
        // FIXME: Only true for labeled epsilon arcs?
        continue;
      }
      
      // Set the parent logical arc(s)
      LatticeLabel parent_label =
        orig_arc_labels[cur_arc].higher_level_label();
      int previous_parent = -1;
      // Find out if the required parent logical arc already exists
      if (parent_label.is_valid())
      {
        LabelIndexMap::const_iterator it;
        std::vector< std::string > parent_labels;

        // Look for the parent(s)
        while (parent_label.is_valid())
        {
          parent_labels.push_back(parent_label.get_label());
          it = node_logical_arcs.at(cur_node).find(
            parent_label.get_label());
          if (it == node_logical_arcs[cur_node].end())
            parent_label = parent_label.higher_level_label();
          else
          {
            previous_parent = (*it).second;
            break;
          }
        }
        // Create the parent labels that do not yet exist.
        // Start from the first non-existent parent arc.
        int last_index = parent_labels.size() - 1;
        if (previous_parent != -1)
          last_index--;
        for (int i = last_index; i >= 0; i--)
        {
          // Create the logical arc, parent is previous_parent
          m_logical_arcs.push_back(
            LogicalArc(i+1, previous_parent, parent_labels[i]));
          previous_parent = m_logical_arcs.size()-1;
          
          // Add the logical arcs to the source node
          node_logical_arcs.at(cur_node).insert(
            LabelIndexMap::value_type(parent_labels[i],
                                      previous_parent));
        }
      }

      // Set the parent of the physical arc
      m_arcs[cur_arc].parent_arc = previous_parent;

      if (previous_parent != -1)
      {
        // Fix the parent arc if the target node already has incoming
        // arcs of the same label. If not, propagate the logical
        // arcs to the target node (also the ones that existed already)
        if (cur_node != target_node) // Skip self-transitions
        {
          // Returns the level from which the parent arcs were fixed
          int fix_level = fix_parent_arcs(cur_arc);
          int level = 1;
          // Propagate parent arcs to the target node
          parent_label = orig_arc_labels[cur_arc].higher_level_label();
          LabelIndexMap::const_iterator it;
          while (parent_label.is_valid() &&
                 (fix_level == -1 || level++ < fix_level) &&
                 (it = node_logical_arcs.at(cur_node).find(
                   parent_label.get_label())) !=
                 node_logical_arcs[cur_node].end())
          {
            if (!parent_label.is_last())
              node_logical_arcs.at(target_node).insert(
                LabelIndexMap::value_type(parent_label.get_label(),
                                          (*it).second));
            parent_label = parent_label.higher_level_label();
          }
        }
      }
    }
  }

  if (nodes_processed < (int)m_nodes.size())
  {
    fprintf(stderr, "Creating topological ordering: %i nodes processed, out of %i\n",
            nodes_processed, (int)m_nodes.size());
    throw std::string("Failed to create a topological order of the nodes");
  }
}


int
HmmNetBaumWelch::fix_parent_arcs(int arc_id)
{
  // Check the other incoming arcs of the target node. If parent arcs
  // exists with the same label, replace the parent indices in the
  // incoming branch. Generate a warning if there is a conflict during
  // the replacing.
  int cur_parent_arc = m_arcs[arc_id].parent_arc;
  int target_node = m_arcs[arc_id].target;
  int level = 1;
  // Collect the parent arcs of the incoming arcs of the target node
  std::vector<int> incoming_parent;
  for (int i = 0; i < (int)m_nodes[target_node].in_arcs.size(); i++)
  {
    int id = m_nodes[target_node].in_arcs[i];
    if (id != arc_id && m_arcs[id].parent_arc != -1)
      incoming_parent.push_back(m_arcs[id].parent_arc);
  }

  bool replaced = false;
  while (cur_parent_arc != -1 && incoming_parent.size() > 0)
  {
    for (int i = 0; i < (int)incoming_parent.size(); i++)
    {
      if (incoming_parent[i] != cur_parent_arc &&
          m_logical_arcs[incoming_parent[i]].label ==
          m_logical_arcs[cur_parent_arc].label)
      {
        // Different parent arc but with the same label, propagate the
        // existing parent to the new incoming branch
        std::set<int> processed_arcs;
        if (replace_branch_parent_arc(arc_id, level, incoming_parent[i],
                                      false, processed_arcs) != level)
          throw std::string("Error in parent arc fixing");
        replaced = true;
        break;
      }
      incoming_parent[i] = m_logical_arcs[incoming_parent[i]].parent_arc;
    }
    if (replaced)
      break;

    level++;
    cur_parent_arc = m_logical_arcs[cur_parent_arc].parent_arc;
  }
  if (!replaced)
    return -1;
  return level;
}


int
HmmNetBaumWelch::replace_branch_parent_arc(int arc_id, int parent_level,
                                           int new_parent_id, bool forward,
                                           std::set<int> &processed_arcs)
{
  if (processed_arcs.find(arc_id) != processed_arcs.end())
    return -1; // Processed already
  processed_arcs.insert(arc_id);
  
  bool propagate = false;
  int *cur_parent = &m_arcs[arc_id].parent_arc;
  for (int l = 1; *cur_parent != -1 && l < parent_level; l++)
    cur_parent = &m_logical_arcs[*cur_parent].parent_arc;
  while (*cur_parent != -1)
  {
    // Replace the parent arc of the current arc (if it matches)
    if (*cur_parent != -1 && 
        m_logical_arcs[new_parent_id].label ==
        m_logical_arcs[*cur_parent].label)
    {
      *cur_parent = new_parent_id; // Doesn't matter if it's already the same
      propagate = true;
      break;
    }
    // Check if the replace can be done on the next level
    parent_level++;
    new_parent_id = m_logical_arcs[new_parent_id].parent_arc;
    cur_parent = &m_logical_arcs[*cur_parent].parent_arc;
  }
  if (!propagate)
    return -1;
  
  int node;
  if (forward)
    node = m_arcs[arc_id].target;
  else
    node = m_arcs[arc_id].source;

  // Propagate to the incoming arcs of the node
  for (int i = 0; i < (int)m_nodes[node].in_arcs.size(); i++)
  {
    if (m_nodes[node].in_arcs[i] != arc_id &&
        !m_arcs[m_nodes[node].in_arcs[i]].epsilon()) // Allow epsilons?!?
    {
      // Don't track the success...
      replace_branch_parent_arc(m_nodes[node].in_arcs[i], parent_level,
                                new_parent_id, false, processed_arcs);
    }
  }

  // Propagate to the outgoing arcs of the node
  for (int i = 0; i < (int)m_nodes[node].out_arcs.size(); i++)
  {
    if (m_nodes[node].out_arcs[i] != arc_id &&
        !m_arcs[m_nodes[node].out_arcs[i]].epsilon()) // Allow epsilons?!?
    {
      // Don't track the success...
      replace_branch_parent_arc(m_nodes[node].out_arcs[i], parent_level,
                                new_parent_id, true, processed_arcs);
    }
  }

  // FIXME: Conflict check?
  return parent_level;
}


HmmNetBaumWelch::LatticeLabel::LatticeLabel()
{
  transition_index = -1;
  epsilon = true;
  last = true;
}

HmmNetBaumWelch::LatticeLabel::LatticeLabel(std::string in_str,
                                            std::string out_str)
{
  bool ok = true;
  std::string temp_str = in_str;
  transition_index = EPSILON;
  if (out_str.size() > 0)
    temp_str += ';' + out_str;
  if (temp_str.size() == 0 || temp_str[0] == '#')
  {
    // Epsilon label
    if (temp_str.size() > 0)
    {
      original_label = temp_str;
      label = temp_str.substr(1);
    }
    else
    {
      original_label = "";
      label = "";
    }
    epsilon = true;
    last = true;
  }
  else
  {
    epsilon = false;
    size_t n = temp_str.find_first_of(';');
    if (n == std::string::npos)
    {
      // No logical arcs, just the transition index
      transition_index = str::str2long(&temp_str, &ok);
      if (!ok)
        throw std::string("Invalid arc label ") + in_str;
    }
    else
    {
      std::string transition_str = temp_str.substr(0, n);
      transition_index = str::str2long(&transition_str, &ok);
      if (!ok)
        throw std::string("Invalid arc label ") + temp_str;
    }
    initialize_labels(temp_str);
  }
}

HmmNetBaumWelch::LatticeLabel::LatticeLabel(int tr_id,
                                            std::string raw_label)
{
  transition_index = tr_id;
  initialize_labels(raw_label);
}

void
HmmNetBaumWelch::LatticeLabel::initialize_labels(const std::string &raw_label)
{
  original_label = raw_label;
  label = remove_end_marks(original_label);

  // Find out whether the arc is the final one with the current level label
  size_t pos = original_label.find_first_of("#;",0);
  if (pos != std::string::npos && original_label[pos] == '#')
    last = true;
  else
    last = false;
}


HmmNetBaumWelch::LatticeLabel
HmmNetBaumWelch::LatticeLabel::higher_level_label(void) const
{
  size_t pos = original_label.find_first_of(';');
  if (pos == std::string::npos)
    return LatticeLabel();
  return LatticeLabel(-1, original_label.substr(pos+1));
}


std::string
HmmNetBaumWelch::LatticeLabel::remove_end_marks(const std::string &str) const
{
  std::string s(str);
  size_t pos = 0;
  while ((pos = s.find_first_of('#', pos)) != std::string::npos)
    s.erase(pos, 1);
  return s;
}



void
HmmNetBaumWelch::close(void)
{
  reset();
  m_arcs.clear();
  m_nodes.clear();
}


void
HmmNetBaumWelch::set_frame_limits(int first_frame, int last_frame) 
{
  m_first_frame = first_frame;
  m_last_frame = last_frame;
}


void
HmmNetBaumWelch::set_pruning_thresholds(double forward, double backward)
{
  if (forward > 0)
    m_forward_beam = forward;
  if (backward > 0)
    m_backward_beam = backward;
}


void
HmmNetBaumWelch::check_network_structure(void)
{
  for (int i = 0; i < (int)m_nodes.size(); i++)
  {
    //bool epsilon_transitions = false;
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
//         if (non_eps_in_transitions > 0)
//         {
//           fprintf(stderr, "At node %i:\n", i);
//           throw std::string("HmmNetBaumWelch: Both epsilon and normal transitions to a node (detected at ") + m_arcs[arc_id].label + std::string(")");
//         }
        //epsilon_transitions = true;
      }
      else if (m_arcs[arc_id].target != m_arcs[arc_id].source)
      {
//         if (epsilon_transitions)
//         {
//           fprintf(stderr, "At node %i:\n", i);
//           throw std::string("HmmNetBaumWelch: Both epsilon and normal transitions to a node (detected at ") + m_arcs[arc_id].label + std::string(")");
//         }
        non_eps_in_transitions++;
      }
      else
        self_transition = true;
    }

    size_t eps_out_transitions = 0;
    size_t non_eps_out_transitions = 0;

    // Check the out arcs
    for (int j = 0; j < (int)m_nodes[i].out_arcs.size(); j++)
    {
      int arc_id = m_nodes[i].out_arcs[j];
      if (m_arcs[arc_id].epsilon())
      {
        if (m_arcs[arc_id].target == m_arcs[arc_id].source)
          throw std::string("HmmNetBaumWelch: Epsilon self loops are not allowed");
        if (non_eps_out_transitions > 0)
        {
          // Actually these might be supported now. Try at your own risk!
          // fprintf(stderr, "At node %i:\n", i);
          // throw std::string("HmmNetBaumWelch: Both epsilon and normal transitions from a node (detected at ") + m_arcs[arc_id].label + std::string(")");
        }
        eps_out_transitions++;
      }
      else if (m_arcs[arc_id].target != m_arcs[arc_id].source)
      {
        if (eps_out_transitions > 0)
        {
          // Actually these might be supported now. Try at your own risk!
          // fprintf(stderr, "At node %i:\n", i);
          // throw std::string("HmmNetBaumWelch: Both epsilon and normal transitions from a node (detected at ") + m_arcs[arc_id].label + std::string(")");
        }
        non_eps_out_transitions++;
      }
      else
        self_transition = true;
    }

    // Self transitions with epsilon out paths do not work properly
    if (eps_out_transitions+non_eps_out_transitions > 1 && self_transition)
    {
      fprintf(stderr, "At node %i:\n", i);          
      throw std::string("HmmNetBaumWelch: Self transitions and multiple out transitions can not be combined");
    }
  }

  if (m_nodes[m_initial_node_id].in_arcs.size() > 0)
    throw std::string("HmmNetBaumWelch: Initial node may not have in arcs!");
  if (m_nodes[m_final_node_id].out_arcs.size() > 0)
    throw std::string("HmmNetBaumWelch: Final node may not have out arcs!");
}


void
HmmNetBaumWelch::reset(void)
{
  m_eof_flag = false;
  m_cur_frame = -1;

  // Clear the segmented lattice
  if (m_segmentator_seglat != NULL)
  {
    delete m_segmentator_seglat;
    m_segmentator_seglat = NULL;
  }
  
  // Clear the Segmentator interface probabilities
  m_transition_prob_map.clear();
  m_pdf_prob_map.clear();

  // Clear the features and backward scores
  m_features.clear();
  m_features_generated = false;
  clear_bw_scores();
}


HmmNetBaumWelch::SegmentedLattice*
HmmNetBaumWelch::load_segmented_lattice(std::string &file)
{
  FILE *fp;

  if ((fp = fopen(file.c_str(), "r")) == NULL)
    throw std::string("Could not open file "+file);
  SegmentedLattice *lattice = new SegmentedLattice;
  lattice->load_segmented_lattice(fp, *this);
  fclose(fp);
  return lattice;
}


bool
HmmNetBaumWelch::init_utterance_segmentation(void)
{
  reset();
  
  // Generate lattice segmentation by Forward-Backward-algorithm
  // Fill the backward probabilities
  if (!fill_backward_probabilities())
    return false; // Backward beam should be increased

  m_segmentator_seglat = create_segmented_lattice();
  
  return (m_segmentator_seglat == NULL ? false : true);
}


bool
HmmNetBaumWelch::next_frame(void)
{
  if (m_eof_flag)
    return false;
  
  if (m_segmentator_seglat == NULL)
    throw std::string("Segmentation has not been initialized!");

  // Requires a frame lattice
  assert( m_segmentator_seglat->frame_lattice );
  
  if (m_cur_frame == -1)
  {
    // Initialize segmented lattice traversal
    m_cur_frame = m_first_frame;
    m_active_nodes.clear();
    m_active_nodes.insert(m_segmentator_seglat->initial_node);
  }
  else
    m_cur_frame++;

  // Check the frame limit
  if ((m_last_frame > 0 && m_cur_frame >= m_last_frame) ||
      (m_active_nodes.size() == 1 &&
       m_active_nodes.find(m_segmentator_seglat->final_node) !=
       m_active_nodes.end()))
  {
    m_eof_flag = true;
    return false;
  }

  // Clear the result vectors
  if (m_collect_transitions)
    m_transition_prob_map.clear();
  m_pdf_prob_map.clear();

  // By definition, next_frame() resets the model PDF cache
  m_model.reset_cache();

  std::set<int> target_nodes; // Store target nodes here
  std::set<int>::iterator it = m_active_nodes.begin();
  double prob_sum = 0;
  double highest_prob = 0;
  m_most_probable_label.clear();
  while (it != m_active_nodes.end())
  {
    // Propagate the active node and collect the probabilities
    for (int i = 0;
         i < (int)m_segmentator_seglat->nodes[*it].out_arcs.size(); i++)
    {
      SegmentedArc &arc =
        m_segmentator_seglat->arcs[m_segmentator_seglat->nodes[*it].out_arcs[i]];
      double prob =
        exp(loglikelihoods.divide(arc.total_score,
                                  m_segmentator_seglat->total_score));
      prob_sum += prob;

      HmmTransition &tr = m_model.transition(arc.transition_index);
      int pdf_id = m_model.emission_pdf_index(tr.source_index);

      Segmentator::IndexProbMap::iterator map_it = m_pdf_prob_map.find(pdf_id);
      if (map_it == m_pdf_prob_map.end())
        map_it = m_pdf_prob_map.insert(Segmentator::IndexProbMap::value_type(
                                         pdf_id, prob)).first;
      else
        (*map_it).second += prob;

      if ((*map_it).second > highest_prob)
      {
        m_most_probable_label = arc.label;
        highest_prob = (*map_it).second;
      }

      if (m_collect_transitions)
      {
        map_it = m_transition_prob_map.find(arc.transition_index);
        if (map_it == m_transition_prob_map.end())
          m_transition_prob_map.insert(Segmentator::IndexProbMap::value_type(
                                         arc.transition_index, prob));
        else
          (*map_it).second += prob;
      }

      target_nodes.insert(arc.target_node);
    }
    ++it;
  }

  // Fix the label if it conforms with the hierarchical system.
  // NOTE! Assumes the following label: mixturenum;statenum;(tri-)phone;...
  std::vector<std::string> fields;
  str::split(&m_most_probable_label, ";", false, &fields);
  if (fields.size() >= 3)
    m_most_probable_label = fields[2] + "." + fields[1];

  m_active_nodes.swap(target_nodes);

  // Sanity check for the sum of probabilities
  if (fabs(1-prob_sum)> 0.02) // Allow small deviation
  {
    fprintf(stderr, "Warning: Sum of PDF probabilities is %g at frame %i\n",
            prob_sum, m_cur_frame);
    if (prob_sum < 0.01)
      exit(1); // Exit if the probability is completely wrong and small
  }

  // Normalize the probabilities
  for (Segmentator::IndexProbMap::iterator it = m_pdf_prob_map.begin();
       it != m_pdf_prob_map.end(); ++it)
    (*it).second /= prob_sum;
  for (Segmentator::IndexProbMap::iterator it = m_transition_prob_map.begin();
       it != m_transition_prob_map.end(); ++it)
    (*it).second /= prob_sum;
  
  return true;
}



void
HmmNetBaumWelch::generate_features(void)
{
  // Allocate the buffer. NOTE! May need seeking to determine the buffer size!
  int num_frames;
  if (m_last_frame > 0)
    num_frames = m_last_frame - m_first_frame;
  else
    num_frames = m_fea_gen.last_frame() - m_first_frame + 1;
  assert( num_frames > 0 );
  if (num_frames == 1)
    fprintf(stderr, "HmmNetBaumWelch: Warning: Single frame utterance\n");
  m_features.resize(num_frames, m_fea_gen.dim());

  // Fill the features
  for (int i = 0; i < num_frames; i++)
    m_features[m_first_frame+i].copy(m_fea_gen.generate(m_first_frame + i));

  m_features_generated = true;
}


bool
HmmNetBaumWelch::fill_backward_probabilities(void)
{
  std::vector< BackwardToken > active_tokens;
  NodeTokenMap node_token_map;
  NodeTransitionMap active_transitions; // multimap
  FeatureVec empty_fea_vec;
  int cur_frame;
  
  if (!m_features_generated)
  {
    // Generate the features once.
    generate_features();
  }
  
  clear_bw_scores();
  if (m_last_frame > 0)
    cur_frame = m_last_frame - 1;
  else
    cur_frame = m_first_frame + m_features.num_frames() - 1;

  // Create a token to the final node
  active_tokens.push_back(
    BackwardToken(m_final_node_id, loglikelihoods.one()));
  node_token_map.insert(NodeTokenMap::value_type(m_final_node_id, 0));

  // Propagate the epsilon arcs leading to the final node
  backward_propagate_epsilon_arcs(active_tokens, node_token_map, cur_frame);
  
  cur_frame++;
  while (--cur_frame >= m_first_frame)
  {
    double best_score = loglikelihoods.zero();
    
    m_model.reset_cache();
    active_transitions.clear();

    // Iterate through active nodes and collect active non-epsilon
    // transitions.
    for (int i = 0; i < (int)active_tokens.size(); i++)
    {
      int node_id = active_tokens[i].node_id;
      for (int a = 0; a < (int)m_nodes[node_id].in_arcs.size(); a++)
      {
        int arc_id = m_nodes[node_id].in_arcs[a];
        if (m_arcs[arc_id].epsilon())
          continue;
    
        int next_node_id = m_arcs[arc_id].source;
        double arc_score = get_arc_score(arc_id, get_feature(cur_frame));
        double backward_score =
          loglikelihoods.times(active_tokens[i].score, arc_score);

        // Check the backward score is valid
        if (backward_score > loglikelihoods.zero())
        {
          if (backward_score > best_score)
            best_score = backward_score;
          // Activate the transition
          active_transitions.insert(
            NodeTransitionMap::value_type(
              next_node_id, BackwardTransitionInfo(arc_id,
                                                   backward_score)));
        }
      }
    }

    active_tokens.clear();
    node_token_map.clear();
    // Go through the active transitions and create tokens to the nodes
    for (NodeTransitionMap::iterator it = active_transitions.begin();
         it != active_transitions.end();) // Iterator updated in the code
    {
      double new_node_score = (*it).second.score;
      int best_arc_id = (*it).second.arc_id; // For MODE_VITERBI

      // Pruning
      if (loglikelihoods.divide((*it).second.score, best_score) <
          -m_backward_beam)
      {
        ++it;
        continue;
      }

      // Go through all the transitions to the same node (sequantially
      // ordered) and update the node probabilities according to the
      // segmentation mode. Fill the backward probabilities for the arcs.
      
      // A map of realized arcs for MODE_MULTIPATH_VITERBI
      ParentArcTransitionMap parent_transition_map;
      if (m_segmentation_mode == MODE_MULTIPATH_VITERBI)
      {
        parent_transition_map.insert(
          ParentArcTransitionMap::value_type(
            m_arcs[(*it).second.arc_id].parent_arc, (*it).second));
      }
      else if (m_segmentation_mode == MODE_BAUM_WELCH)
      {
        // Set the backward score
        m_arcs[(*it).second.arc_id].bw_scores.set_new_score(
          cur_frame, (*it).second.score);
      }

      // This iterator goes through all the transitions with the same target
      // (or in fact, as this is the backward phase, the source node).
      NodeTransitionMap::iterator it2(it);
      while (++it2 != active_transitions.end() &&
             (*it2).first == (*it).first)
      {
        // Pruning
        if (loglikelihoods.divide((*it2).second.score, best_score) <
            -m_backward_beam)
          continue;

        if (m_segmentation_mode == MODE_BAUM_WELCH)
        {
          new_node_score = loglikelihoods.plus(new_node_score,
                                               (*it2).second.score);
          // Set the backward score
          m_arcs[(*it2).second.arc_id].bw_scores.set_new_score(
            cur_frame, (*it2).second.score);
        }
        else if (m_segmentation_mode == MODE_MULTIPATH_VITERBI)
        {
          // Pick the best transitions among those sharing the same
          // first-level logical arc
          ParentArcTransitionMap::iterator tr_it = parent_transition_map.find(
            m_arcs[(*it2).second.arc_id].parent_arc);
          if (tr_it != parent_transition_map.end())
          {
            // Update if needed
            if ((*tr_it).second.score < (*it2).second.score)
              (*tr_it).second = (*it2).second; // Update all scores
          }
          else
          {
            // Create a new item to the map of realized arcs
            parent_transition_map.insert(
              ParentArcTransitionMap::value_type(
                m_arcs[(*it2).second.arc_id].parent_arc, (*it2).second));
          }
        }
        else // m_segmentation_mode == MODE_VITERBI
        {
          if (new_node_score < (*it2).second.score)
          {
            // Backward score and best path score are equal in MODE_VITERBI
            new_node_score = (*it2).second.score;
            best_arc_id = (*it2).second.arc_id;
          }
        }
      }
      if (m_segmentation_mode == MODE_MULTIPATH_VITERBI)
      {
        // Update the node probabilities based on the realized arcs
        
        new_node_score = loglikelihoods.zero();
        for (ParentArcTransitionMap::iterator tr_it =
               parent_transition_map.begin();
             tr_it != parent_transition_map.end(); ++tr_it)
        {
          if (new_node_score > loglikelihoods.zero())
            new_node_score = loglikelihoods.plus(new_node_score,
                                                 (*tr_it).second.score);
          else
            new_node_score = (*tr_it).second.score;
          // Set the backward score
          m_arcs[(*tr_it).second.arc_id].bw_scores.set_new_score(
            cur_frame, (*tr_it).second.score);
        }
      }
      else if (m_segmentation_mode == MODE_VITERBI)
      {
        // Set the backward scores
        m_arcs[best_arc_id].bw_scores.set_new_score(
          cur_frame, new_node_score);
      }

      // Create a token to the target node
      node_token_map.insert(
        NodeTokenMap::value_type((*it).first, active_tokens.size()));
      active_tokens.push_back(BackwardToken((*it).first, new_node_score));

      it = it2;
    }

    // Propagate epsilon transitions
    backward_propagate_epsilon_arcs(active_tokens,
                                    node_token_map, cur_frame);
  }

  // Set the total lattice scores
  NodeTokenMap::iterator it = node_token_map.find(m_initial_node_id);
  if (it == node_token_map.end())
    return false; // Did not reach the initial node
  m_total_score = active_tokens[(*it).second].score;

  if (m_total_score <= loglikelihoods.zero())
    return false; // Initial node was not reached

  m_bw_scores_computed = true;
  
  return true;

}


void
HmmNetBaumWelch::backward_propagate_epsilon_arcs(
  std::vector< BackwardToken > &active_tokens,
  NodeTokenMap &node_token_map,
  int cur_frame)
{
  FeatureVec empty_fea_vec;
  for (int i = 0; i < (int)active_tokens.size(); i++)
  {
    int node_id = active_tokens[i].node_id;
    for (int a = 0; a < (int)m_nodes[node_id].in_arcs.size(); a++)
    {
      int arc_id = m_nodes[node_id].in_arcs[a];
      if (!m_arcs[arc_id].epsilon())
        continue;

      int next_node_id = m_arcs[arc_id].source;
      double arc_score = get_arc_score(arc_id, empty_fea_vec);
      double backward_score =
        loglikelihoods.times(active_tokens[i].score, arc_score);

      // Set the backward score
      m_arcs[arc_id].bw_scores.set_new_score(cur_frame, backward_score);

      // Find out whether the next network node has already been activated
      NodeTokenMap::iterator node_token = node_token_map.find(next_node_id);
      if (node_token != node_token_map.end())
      {
        // Update the existing token
        int token_id = (*node_token).second;
        assert( active_tokens[token_id].node_id == next_node_id );
        if (m_segmentation_mode == MODE_VITERBI)
        {
          active_tokens[token_id].score =
            std::max(active_tokens[token_id].score, backward_score);
        }
        else
        {
          active_tokens[token_id].score =
            loglikelihoods.plus(active_tokens[token_id].score,
                                backward_score);
        }
      }
      else
      {
        // Create a new token
        node_token_map.insert(
          NodeTokenMap::value_type(next_node_id, active_tokens.size()));
        active_tokens.push_back(BackwardToken(next_node_id, backward_score));
      }
    }
  }
}


HmmNetBaumWelch::SegmentedLattice*
HmmNetBaumWelch::create_segmented_lattice(void)
{
  if (!m_bw_scores_computed)
  {
    if (!fill_backward_probabilities())
      return NULL;
  }
  
  SegmentedLattice *sl = new SegmentedLattice;
  sl->frame_lattice = true;

  int tbuf = 0; // Target buffer
  int cur_frame = m_first_frame;
  int end_frame;

  // Network traversal is implemented with token passing. Only one token
  // may exist in any single network node. Two token buffers are used for
  // propagating the tokens.
  std::vector< ForwardToken> active_tokens[2];
  NodeTokenMap node_token_map[2];

  // One pending arc represents a traversal through a non-epsilon arc
  // from a certain source node. One pending arc may get connected to
  // multiple target nodes.
  std::vector< PendingArc > pending_arcs;
  
  if (m_last_frame > 0)
    end_frame = m_last_frame;
  else
    end_frame = m_first_frame + m_features.num_frames();
  
  // Create the root of the segmented lattice
  sl->initial_node = 0;
  sl->nodes.push_back(SegmentedNode(cur_frame));
  active_tokens[tbuf].push_back(ForwardToken(m_initial_node_id,
                                             loglikelihoods.one()));
  node_token_map[tbuf].insert(
    NodeTokenMap::value_type(m_initial_node_id,
                             active_tokens[tbuf].size()-1));
  // Initial segmented node
  active_tokens[tbuf].back().source_seg_node = 0;
  
  cur_frame--;
  while (++cur_frame < end_frame)
  {
    int sbuf = tbuf; // Source buffer
    tbuf ^= 1;
    m_model.reset_cache();

    ///////////////////////////////////////////
    // Propagate the epsilon transitions     //
    ///////////////////////////////////////////

    active_tokens[tbuf].clear();
    node_token_map[tbuf].clear();

    // In MODE_VITERBI there should be only one token
    assert( m_segmentation_mode != MODE_VITERBI ||
            active_tokens[sbuf].size() == 1 );
    
    for (int i = 0; i < (int)active_tokens[sbuf].size(); i++)
    {
      int network_node_id = active_tokens[sbuf][i].node_id;

      // For MODE_VITERBI:
      double cur_best_total_score = loglikelihoods.zero();
      double cur_best_forward_score = loglikelihoods.zero();
      double arc_score_in_cur_best = loglikelihoods.one();
      int cur_best_arc_id = -1;
      
      for (int a = 0; a < (int)m_nodes[network_node_id].out_arcs.size(); a++)
      {
        int arc_id = m_nodes[network_node_id].out_arcs[a];
        // Propagate only epsilon arcs, but in MODE_VITERBI, choose the
        // best arc among all the transitions, including non-epsilon ones
        if (m_segmentation_mode != MODE_VITERBI &&
            !m_arcs[arc_id].epsilon())
          continue;

        double arc_total_score = loglikelihoods.times(
          active_tokens[sbuf][i].score,
          m_arcs[arc_id].bw_scores.get_score(cur_frame));

        // Beam pruning
        // NOTE: This handles also the case when backward score is zero
        if (arc_total_score < m_total_score - m_forward_beam)
          continue;

        double arc_score = get_arc_score(arc_id, get_feature(cur_frame));
        double forward_score = loglikelihoods.times(
          active_tokens[sbuf][i].score, arc_score);
        assert( forward_score > loglikelihoods.zero() );

        if (m_segmentation_mode == MODE_VITERBI)
        {
          // In MODE_VITERBI, only the node behind the best arc is
          // activated. Note that there may be further epsilon transitions
          // that need to be propagated after that.
          if (arc_total_score > cur_best_total_score)
          {
            cur_best_total_score = arc_total_score;
            cur_best_forward_score = forward_score;
            cur_best_arc_id = arc_id;
            arc_score_in_cur_best = arc_score;
          }
          continue;
        }

        // Following is not executed for MODE_VITERBI
        
        int next_node_id = m_arcs[arc_id].target;
        int new_token_index;

        // Find out whether the next network node has already been
        // activated during this frame.
        NodeTokenMap::iterator node_token =
          node_token_map[sbuf].find(next_node_id);
        
        if (node_token != node_token_map[sbuf].end())
        {
          // Update the existing token
          assert( active_tokens[sbuf][(*node_token).second].node_id
                  == next_node_id );
          new_token_index = (*node_token).second;
          active_tokens[sbuf][new_token_index].score =
            loglikelihoods.plus(active_tokens[sbuf][new_token_index].score,
                                forward_score);
        }
        else // Create a new token
        {
          active_tokens[sbuf].push_back(
            ForwardToken(next_node_id, forward_score));
          node_token_map[sbuf].insert(
            NodeTokenMap::value_type(next_node_id,
                                     (int)active_tokens[sbuf].size()-1));
          new_token_index = active_tokens[sbuf].size() - 1;
          // source_seg_node is needed for the initial segmented node!
          active_tokens[sbuf][new_token_index].source_seg_node =
            active_tokens[sbuf][i].source_seg_node;
        }

        bool copy_pending_arcs = true;
        if (arc_score != loglikelihoods.one() ||
            m_nodes[network_node_id].out_arcs.size() > 1)
          copy_pending_arcs = false;
            
        // Copy the pending arcs to the target token
        for (std::set<int>::iterator it =
               active_tokens[sbuf][i].pending_arcs.begin();
             it != active_tokens[sbuf][i].pending_arcs.end(); ++it)
        {
          if (copy_pending_arcs)
          {
            // Share the old pending arc
            active_tokens[sbuf][new_token_index].pending_arcs.insert(*it);
          }
          else
          {
            // Create a new pending arc with updated scores
            active_tokens[sbuf][new_token_index].pending_arcs.insert(
              pending_arcs.size());
            double pa_forward_score = loglikelihoods.times(
              pending_arcs[*it].forward_score, arc_score);
            double pa_total_score = loglikelihoods.times(
              pending_arcs[*it].forward_score, // Value before the update!
              m_arcs[arc_id].bw_scores.get_score(cur_frame));
            pending_arcs.push_back(
              PendingArc(pending_arcs[*it].arc_id,
                         pending_arcs[*it].source_seg_node,
                         loglikelihoods.times(pending_arcs[*it].arc_score,
                                              arc_score),
                         pending_arcs[*it].arc_acoustic_score,
                         pa_forward_score, pa_total_score));
          }
        }
      }
      
      if (m_segmentation_mode == MODE_VITERBI && cur_best_arc_id != -1 &&
          m_arcs[cur_best_arc_id].epsilon())
      {
        // Update the current token
        int next_node_id = m_arcs[cur_best_arc_id].target;
        active_tokens[sbuf][i].node_id = next_node_id;
        active_tokens[sbuf][i].score = cur_best_forward_score;

        // Also update the pending arc
        if (active_tokens[sbuf][i].pending_arcs.size() > 0)
        {
          assert( active_tokens[sbuf][i].pending_arcs.size() == 1 );
          assert( (*active_tokens[sbuf][i].pending_arcs.begin()) == 0 );
          pending_arcs[0].arc_score = loglikelihoods.times(
            pending_arcs[0].arc_score, arc_score_in_cur_best);
          pending_arcs[0].forward_score = loglikelihoods.times(
            pending_arcs[0].forward_score, arc_score_in_cur_best);
        }
        
        // Although not really required for MODE_VITERBI, put the updated
        // token to appropriate location in the map
        node_token_map[sbuf].clear();
        node_token_map[sbuf].insert(NodeTokenMap::value_type(next_node_id, i));
        // Only one token in MODE_VITERBI, but reiterate in case there
        // are several epsilon arcs in sequence.
        i--;
      }
    }

    assert( active_tokens[sbuf].size() > 0 );

    /////////////////////////////////////////////////////////////////////
    // Propagate the non-epsilon transitions                           //
    //                                                                 // 
    // Note that during the backward phase the backward probabilities  //
    // have been filled only for the eligible non-epsilon arcs, with   //
    // respect to the segmentation mode.                               //
    /////////////////////////////////////////////////////////////////////

    // Collect the new pending arcs here
    std::vector< PendingArc > temp_pending_arcs;
    
    for (int i = 0; i < (int)active_tokens[sbuf].size(); i++)
    {
      int network_node_id = active_tokens[sbuf][i].node_id;
      bool pending_arcs_created = false;
      
      for (int a = 0; a < (int)m_nodes[network_node_id].out_arcs.size(); a++)
      {
        int arc_id = m_nodes[network_node_id].out_arcs[a];
        int next_node_id = m_arcs[arc_id].target;
        if (m_arcs[arc_id].epsilon())
          continue;
        
        double arc_total_score = loglikelihoods.times(
          active_tokens[sbuf][i].score,
          m_arcs[arc_id].bw_scores.get_score(cur_frame));

        // Beam pruning, based on the best single path
        if (arc_total_score < m_total_score - m_forward_beam)
          continue;

        // The current token is going to be propagated, create
        // the pending arcs now if not done already.
        if (!pending_arcs_created &&
            active_tokens[sbuf][i].pending_arcs.size() > 0)
        {
          // Create the segmented node for the group of pending arcs
          int target_seg_node = sl->nodes.size();
          sl->nodes.push_back(SegmentedNode(cur_frame));

          // Connect the pending arcs
          std::set<int> &pa_set = active_tokens[sbuf][i].pending_arcs;
          for (std::set<int>::iterator pa_it = pa_set.begin();
               pa_it != pa_set.end(); ++pa_it)
          {
            PendingArc &p = pending_arcs[(*pa_it)];
            sl->create_segmented_arc(p.arc_id, m_arcs[p.arc_id].label,
                                     m_arcs[p.arc_id].transition_index,
                                     p.source_seg_node, target_seg_node,
                                     p.arc_score, p.arc_acoustic_score,
                                     p.total_score);
          }
          active_tokens[sbuf][i].source_seg_node = target_seg_node;

          // Share the target segmented node with all the tokens that
          // share this group of pending arcs
          for (int j = i+1; j < (int)active_tokens[sbuf].size(); j++)
          {
            if (active_tokens[sbuf][i].pending_arcs ==
                active_tokens[sbuf][j].pending_arcs)
            {
              active_tokens[sbuf][j].pending_arcs.clear();
              active_tokens[sbuf][j].source_seg_node = target_seg_node;
            }
          }
          active_tokens[sbuf][i].pending_arcs.clear();

          pending_arcs_created = true;

        }

        double arc_score = get_arc_score(arc_id, get_feature(cur_frame));
        double arc_acoustic_score = arc_score;
        if (m_use_static_scores)
           arc_acoustic_score -= m_arcs[arc_id].static_score;
        double forward_score = loglikelihoods.times(
          active_tokens[sbuf][i].score, arc_score);
        assert( forward_score > loglikelihoods.zero() );
        
        ForwardToken &next_token =
          create_or_update_token(active_tokens[tbuf], node_token_map[tbuf],
                                 next_node_id, forward_score);
        // Create the pending arc
        next_token.pending_arcs.insert(temp_pending_arcs.size());
        assert( active_tokens[sbuf][i].source_seg_node != -1 );
        temp_pending_arcs.push_back(
          PendingArc(arc_id, active_tokens[sbuf][i].source_seg_node,
                     (cur_frame == m_first_frame?forward_score:arc_score),
                     arc_acoustic_score,
                     forward_score, arc_total_score));
      }

      // Clear the pending arcs in the source token
      active_tokens[sbuf][i].pending_arcs.clear();
    }

    // Clear the old pending arcs and move the new ones in
    pending_arcs.clear();
    pending_arcs.swap(temp_pending_arcs);
  }

  // Connect the final pending arcs
  double total_score = loglikelihoods.zero();
  int num_end_arcs = 0;
  sl->final_node = sl->nodes.size();
  sl->nodes.push_back(SegmentedNode(cur_frame));
  for (int i = 0; i < (int)active_tokens[tbuf].size(); i++)
  {
    std::set<int> &pa_set = active_tokens[tbuf][i].pending_arcs;
    for (std::set<int>::iterator pa_it = pa_set.begin();
         pa_it != pa_set.end(); ++pa_it)
    {
      PendingArc &p = pending_arcs[(*pa_it)];
      sl->create_segmented_arc(p.arc_id, m_arcs[p.arc_id].label,
                               m_arcs[p.arc_id].transition_index,
                               p.source_seg_node, sl->final_node,
                               p.arc_score, p.arc_acoustic_score, p.total_score);
      num_end_arcs++;
    }

    if (total_score <= loglikelihoods.zero())
      total_score = active_tokens[tbuf][i].score;
    else
      total_score = loglikelihoods.plus(total_score,
                                        active_tokens[tbuf][i].score);
  }
  if (num_end_arcs == 0)
    throw std::string("No paths survived to the end of the network!");

  sl->total_score = total_score;

  return sl;
}


HmmNetBaumWelch::ForwardToken&
HmmNetBaumWelch::create_or_update_token(
  std::vector< ForwardToken > &token_vector, NodeTokenMap &node_token_map,
  int node_id, double forward_score)
{
  // Find out whether the next network node has an active token
  NodeTokenMap::iterator node_token = node_token_map.find(node_id);
  int token_index = -1;
  if (node_token != node_token_map.end())
  {
    // Update the existing tokens
    token_index = (*node_token).second;
    assert( token_vector[token_index].node_id == node_id );
    token_vector[token_index].score =
      loglikelihoods.plus(token_vector[token_index].score, forward_score);
  }
  else
  {
    // Create a new token
    token_index = token_vector.size();
    token_vector.push_back(ForwardToken(node_id, forward_score));
    node_token_map.insert(NodeTokenMap::value_type(node_id, token_index));
  }
  return token_vector[token_index];
}


int
HmmNetBaumWelch::SegmentedLattice::create_segmented_arc(
  int arc_id, std::string &label, int transition_index,
  int source_seg_node, int target_seg_node,
  double arc_score, double acoustic_score, double total_score)
{
  nodes[source_seg_node].out_arcs.push_back(arcs.size());
  nodes[target_seg_node].in_arcs.push_back(arcs.size());
  arcs.push_back(SegmentedArc(arc_id, label, transition_index,
                              source_seg_node, target_seg_node,
                              arc_score, acoustic_score, total_score));
  return arcs.size()-1;
}



void
HmmNetBaumWelch::SegmentedLattice::fill_custom_scores(
  CustomScoreQuery *callback)
{
  for (int a = 0; a < (int)arcs.size(); a++)
    arcs[a].custom_score = callback->custom_score(this, a);
}


void
HmmNetBaumWelch::SegmentedLattice::compute_custom_path_scores(
  CustomScoreQuery *callback, int combination_mode)
{
  std::vector<ScorePair> fw_scores; // Node forward scores
  typedef std::multimap<int, int> FrameNode;
  FrameNode topological_order; // Nodes sorted by frame numbers

  // Fill the custom scores
  if (callback != NULL)
    fill_custom_scores(callback);

  // Construct the topological order of the nodes
  for (int i = 0; i < (int)nodes.size(); i++)
    topological_order.insert(FrameNode::value_type(nodes[i].frame, i));

  // Initial node needs to be the first one in the topological order
  assert( (*topological_order.begin()).second == initial_node );
  // Final node needs to be the last one in the topological order
  assert( (*topological_order.rbegin()).second == final_node );

  fw_scores.resize(nodes.size(), ScorePair(loglikelihoods.zero(), 0));
  fw_scores[initial_node] = ScorePair(loglikelihoods.one(), 0);
  
  for (FrameNode::iterator fn = topological_order.begin();
       fn != topological_order.end(); ++fn)
  {
    if (fw_scores[(*fn).second].score <= loglikelihoods.zero())
      continue; // This node is unreachable (?)
    SegmentedNode &cur_node = nodes[(*fn).second];
    for (int a = 0; a < (int)cur_node.out_arcs.size(); a++)
    {
      int arc_id = cur_node.out_arcs[a];
      if (arcs[arc_id].arc_score > loglikelihoods.zero())
      {
        int target_node_id = arcs[arc_id].target_node;
        double new_score = loglikelihoods.times(fw_scores[(*fn).second].score,
                                                arcs[arc_id].arc_score);
        double new_custom_path_score = fw_scores[(*fn).second].custom_score +
          arcs[arc_id].custom_score;
        if (fw_scores[target_node_id].score <= loglikelihoods.zero())
        {
          // Set the new score to the node
          fw_scores[target_node_id].score = new_score;
          fw_scores[target_node_id].custom_score = new_custom_path_score;
        }
        else
        {
          // Update the node scores
          fw_scores[target_node_id].custom_score =
            combine_custom_scores(new_score, new_custom_path_score,
                                  fw_scores[target_node_id].score,
                                  fw_scores[target_node_id].custom_score,
                                  combination_mode);
          fw_scores[target_node_id].score = loglikelihoods.plus(
            fw_scores[target_node_id].score, new_score);
        }
      }
    }
  }

  assert( fw_scores[final_node].score > loglikelihoods.zero() );
  total_custom_score = fw_scores[final_node].custom_score;

  // Backward phase
  std::vector<ScorePair> bw_scores; // Node backward scores
  bw_scores.resize(nodes.size(), ScorePair(loglikelihoods.zero(), 0));
  bw_scores[final_node] = ScorePair(loglikelihoods.one(), 0);
  for (FrameNode::reverse_iterator fn = topological_order.rbegin();
       fn != topological_order.rend(); ++fn)
  {
    if (bw_scores[(*fn).second].score <= loglikelihoods.zero())
      continue; // This node is unreachable (?)
    SegmentedNode &cur_node = nodes[(*fn).second];
    for (int a = 0; a < (int)cur_node.in_arcs.size(); a++)
    {
      int arc_id = cur_node.in_arcs[a];
      int source_node_id = arcs[arc_id].source_node;
      if (arcs[arc_id].arc_score > loglikelihoods.zero() &&
          fw_scores[source_node_id].score > loglikelihoods.zero())
      {
        double new_score = loglikelihoods.times(bw_scores[(*fn).second].score,
                                                arcs[arc_id].arc_score);
        double new_custom_path_score = bw_scores[(*fn).second].custom_score +
          arcs[arc_id].custom_score;
        
        // Fill the custom path score for this arc
        arcs[arc_id].custom_path_score = new_custom_path_score +
          fw_scores[source_node_id].custom_score;
        
        if (bw_scores[source_node_id].score <= loglikelihoods.zero())
        {
          // Set the new score to the node
          bw_scores[source_node_id].score = new_score;
          bw_scores[source_node_id].custom_score = new_custom_path_score;
        }
        else
        {
          // Update the node scores
          bw_scores[source_node_id].custom_score =
            combine_custom_scores(new_score, new_custom_path_score,
                                  bw_scores[source_node_id].score,
                                  bw_scores[source_node_id].custom_score,
                                  combination_mode);
          bw_scores[source_node_id].score = loglikelihoods.plus(
            bw_scores[source_node_id].score, new_score);
        }
      }
    }
  }

  assert( bw_scores[initial_node].score > loglikelihoods.zero() );
}


double
HmmNetBaumWelch::SegmentedLattice::combine_custom_scores(
  double log_score, double custom_score, double old_log_score,
  double old_custom_score, int combination_mode)
{
  double combined_score;
  if (combination_mode == CUSTOM_AVG)
  {
    double p1 = exp(log_score-old_log_score);
    double p2 = 1;
    combined_score = (p1*custom_score + p2*old_custom_score) / (p1 + p2);
  }
  else if (combination_mode == CUSTOM_SUM)
    combined_score = custom_score + old_custom_score;
  else // combination_mode == CUSTOM_MAX
    combined_score = std::max(custom_score, old_custom_score);
  return combined_score;
}



void
HmmNetBaumWelch::SegmentedLattice::compute_total_scores(void)
{
  std::vector<double> fw_scores; // Node forward scores
  typedef std::multimap<int, int> FrameNode; 
  FrameNode topological_order; // Nodes sorted by frame numbers

  // Construct the topological order of the nodes
  for (int i = 0; i < (int)nodes.size(); i++)
    topological_order.insert(FrameNode::value_type(nodes[i].frame, i));
  
  // Forward phase
  fw_scores.resize(nodes.size(), loglikelihoods.zero());
  fw_scores[initial_node] = loglikelihoods.one();
  for (FrameNode::iterator fn = topological_order.begin();
       fn != topological_order.end(); ++fn)
  {
    if (fw_scores[(*fn).second] <= loglikelihoods.zero())
      continue; // This node is unreachable (?)
    SegmentedNode &cur_node = nodes[(*fn).second];
    for (int a = 0; a < (int)cur_node.out_arcs.size(); a++)
    {
      int arc_id = cur_node.out_arcs[a];
      if (arcs[arc_id].arc_score > loglikelihoods.zero())
      {
        int target_node_id = arcs[arc_id].target_node;
        double new_score = loglikelihoods.times(fw_scores[(*fn).second],
                                                arcs[arc_id].arc_score);
        if (fw_scores[target_node_id] <= loglikelihoods.zero())
          fw_scores[target_node_id] = new_score;
        else
          fw_scores[target_node_id] = loglikelihoods.plus(
            fw_scores[target_node_id], new_score);
      }
      else
        arcs[arc_id].total_score = loglikelihoods.zero();
    }
  }

  assert( fw_scores[final_node] > loglikelihoods.zero() );
  total_score = fw_scores[final_node]; // Total lattice score
  
  // Backward phase
  std::vector<double> bw_scores; // Node backward scores
  bw_scores.resize(nodes.size(), loglikelihoods.zero());
  bw_scores[final_node] = loglikelihoods.one();
  for (FrameNode::reverse_iterator fn = topological_order.rbegin();
       fn != topological_order.rend(); ++fn)
  {
    bool clear_scores = false;
    if (bw_scores[(*fn).second] <= loglikelihoods.zero())
    {
      // This node is unreachable, clear arc total scores
      clear_scores = true;
    }
    SegmentedNode &cur_node = nodes[(*fn).second];
    for (int a = 0; a < (int)cur_node.in_arcs.size(); a++)
    {
      int arc_id = cur_node.in_arcs[a];
      int source_node_id = arcs[arc_id].source_node;
      if (clear_scores)
      {
        arcs[arc_id].total_score = loglikelihoods.zero();
        continue;
      }
      if (arcs[arc_id].arc_score > loglikelihoods.zero() &&
          fw_scores[source_node_id] > loglikelihoods.zero())
      {
        double new_score = loglikelihoods.times(bw_scores[(*fn).second],
                                                arcs[arc_id].arc_score);
        // Fill the total score for this arc
        arcs[arc_id].total_score = loglikelihoods.times(
          fw_scores[source_node_id], new_score);
        
        if (bw_scores[source_node_id] <= loglikelihoods.zero())
          bw_scores[source_node_id] = new_score;
        else
          bw_scores[source_node_id] = loglikelihoods.plus(
            bw_scores[source_node_id], new_score);
      }
      else
        arcs[arc_id].total_score = loglikelihoods.zero();
    }
  }
}


void
HmmNetBaumWelch::SegmentedLattice::propagate_custom_scores_to_frame_segmented_lattice(
  SegmentedLattice *frame_sl, int combination_mode)
{
  if (frame_lattice)
    throw std::string("HmmNetBaumWelch::propagate_custom_scores_to_frame_segmented_lattice: source is a frame-level lattice");
  if (!frame_sl->frame_lattice)
    throw std::string("HmmNetBaumWelch::propagate_custom_scores_to_frame_segmented_lattice: target is not a frame-level lattice");

  assert( arcs.size() == child_arcs.size() );

  std::vector<double> child_arc_scores;
  child_arc_scores.resize(frame_sl->arcs.size(), loglikelihoods.zero());

  for (int i = 0; i < (int)child_arcs.size(); i++)
  {
    for (int j = 0; j < (int)child_arcs[i].size(); j++)
    {
      int child_arc_id = child_arcs[i][j];
      if (child_arc_scores[child_arc_id] <= loglikelihoods.zero())
      {
        // The first custom score to this frame lattice arc
        frame_sl->arcs[child_arc_id].custom_path_score =
          arcs[i].custom_path_score;
        child_arc_scores[child_arc_id] = arcs[i].total_score;
      }
      else
      {
        // Update the custom score in the frame lattice arc
        frame_sl->arcs[child_arc_id].custom_path_score =
          combine_custom_scores(arcs[i].total_score,
                                arcs[i].custom_path_score,
                                child_arc_scores[child_arc_id],
                                frame_sl->arcs[child_arc_id].custom_path_score,
                                combination_mode);
        child_arc_scores[child_arc_id] =
          loglikelihoods.plus(child_arc_scores[child_arc_id],
                              arcs[i].total_score);
      }
    }
  }

  // Copy the total custom score
  frame_sl->total_custom_score = total_custom_score;
}


void
HmmNetBaumWelch::SegmentedLattice::write_segmented_lattice_fst(
  FILE *file, bool arc_total_scores)
{
  fprintf(file, "#FSTBasic MaxPlus\n");
  fprintf(file, "I %d\n", initial_node);
  fprintf(file, "F %d\n", final_node);

  for (int i = 0; i < (int)arcs.size(); i++)
  {
    std::stringstream label;
    label << arcs[i].label;
    // << nodes[arcs[i].source_node].frame << "->"
    // << "->" << nodes[arcs[i].target_node].frame;
    fprintf(file, "T %d %d %s , %.4f\n", arcs[i].source_node,
            arcs[i].target_node, label.str().c_str(),
            arc_total_scores?arcs[i].total_score:arcs[i].arc_score);
            //arcs[i].custom_path_score);
  }
}


void
HmmNetBaumWelch::SegmentedLattice::save_segmented_lattice(FILE *file)
{
  assert( frame_lattice ); // Currently available only for frame lattices
  fprintf(file, "#SegmentedLattice......\n");
  int itemp = nodes.size();
  fwrite(&itemp, sizeof(int), 1, file);
  itemp = arcs.size();
  fwrite(&itemp, sizeof(int), 1, file);
  itemp = initial_node;
  fwrite(&itemp, sizeof(int), 1, file);
  itemp = final_node;
  fwrite(&itemp, sizeof(int), 1, file);
  double dtemp = total_score;
  fwrite(&dtemp, sizeof(double), 1, file);
  dtemp = total_custom_score;
  fwrite(&dtemp, sizeof(double), 1, file);
  for (int i = 0; i < (int)nodes.size(); i++)
  {
    itemp = nodes[i].frame;
    fwrite(&itemp, sizeof(int), 1, file);
  }
  if ((int)nodes.size()%4)
  {
    // Alignment writes
    for (int i = 0; i < (4-((int)nodes.size()%4));i++)
      fwrite(&itemp, sizeof(int), 1, file);
  }
  
  for (int i = 0; i < (int)arcs.size(); i++)
  {
    itemp = arcs[i].net_arc_id;
    fwrite(&itemp, sizeof(int), 1, file);
    itemp = arcs[i].source_node;
    fwrite(&itemp, sizeof(int), 1, file);
    itemp = arcs[i].target_node;
    fwrite(&itemp, sizeof(int), 1, file);
    fwrite(&itemp, sizeof(int), 1, file); // Alignment int
    dtemp = arcs[i].arc_score;
    fwrite(&dtemp, sizeof(double), 1, file);
    dtemp = arcs[i].arc_acoustic_score;
    fwrite(&dtemp, sizeof(double), 1, file);
    dtemp = arcs[i].total_score;
    fwrite(&dtemp, sizeof(double), 1, file);
    dtemp = arcs[i].custom_score;
    fwrite(&dtemp, sizeof(double), 1, file);
    dtemp = arcs[i].custom_path_score;
    fwrite(&dtemp, sizeof(double), 1, file);
  }
}


void
HmmNetBaumWelch::SegmentedLattice::load_segmented_lattice(
  FILE *file, HmmNetBaumWelch &parent)
{
  std::string line;
  std::vector<std::string> fields;
  int num_nodes = -1;
  int num_arcs = -1;
//  bool ok = true;

  assert( nodes.size() == 0 );
  assert( arcs.size() == 0 );
  
  str::read_line(&line, file, true);
  if (line != "#SegmentedLattice......")
    throw std::string("Invalid file type for segmented lattice");

  int itemp;
  if (fread(&num_nodes, sizeof(int), 1, file) != 1 ||
      fread(&num_arcs, sizeof(int), 1, file) != 1)
    throw std::string("Read error");

  frame_lattice = true;

  // Allocate nodes and arcs
  nodes.resize(num_nodes);
  arcs.reserve(num_arcs); // Added dynamically by create_segmented_arc()

  if (fread(&initial_node, sizeof(int), 1, file) != 1 ||
      fread(&final_node, sizeof(int), 1, file) != 1 ||
      fread(&total_score, sizeof(double), 1, file) != 1 ||
      fread(&total_custom_score, sizeof(double), 1, file) != 1)
    throw std::string("Read error");

  for (int i = 0; i < num_nodes; i++)
  {
    if (fread(&itemp, sizeof(int), 1, file) != 1)
      throw std::string("Read error");
    nodes[i].frame = itemp;
  }

  if (num_nodes%4)
  {
    // Alignment reads
    for (int i = 0; i < (4-(num_nodes%4)); i++)
    {
      if (fread(&itemp, sizeof(int), 1, file) != 1)
        throw std::string("Read error");
    }
  }

  for (int i = 0; i < num_arcs; i++)
  {
    int net_arc_id = -1;
    int source_node = -1, target_node = -1;
    int temp;
    double arc_score, arc_acoustic_score;
    double arc_total_score, arc_custom_score, arc_custom_path_score;
    if (fread(&net_arc_id, sizeof(int), 1, file) != 1 ||
        fread(&source_node, sizeof(int), 1, file) != 1 ||
        fread(&target_node, sizeof(int), 1, file) != 1 ||
        fread(&temp, sizeof(int), 1, file) != 1 ||
        fread(&arc_score, sizeof(double), 1, file) != 1 ||
        fread(&arc_acoustic_score, sizeof(double), 1, file) != 1 ||
        fread(&arc_total_score, sizeof(double), 1, file) != 1 ||
        fread(&arc_custom_score, sizeof(double), 1, file) != 1 ||
        fread(&arc_custom_path_score, sizeof(double), 1, file) != 1)
      throw std::string("Read error");

    assert( target_node == temp ); // Alignment int test

    if (net_arc_id >= 0 && net_arc_id < (int)parent.m_arcs.size() &&
        source_node >= 0 && source_node < num_nodes &&
        target_node >= 0 && target_node < num_nodes)
    {
      int tr_index = parent.m_arcs[net_arc_id].transition_index;
      std::string label = parent.m_arcs[net_arc_id].label;
      int new_arc_id =
        create_segmented_arc(net_arc_id, label, tr_index,
                             source_node, target_node,
                             arc_score, arc_acoustic_score, arc_total_score);
      assert( new_arc_id == i );
      arcs[i].custom_score = arc_custom_score;
      arcs[i].custom_path_score = arc_custom_path_score;
    }
    else
    {
      fprintf(stderr, "Invalid transition\n");
      fprintf(stderr, "net_arc_id = %d, max_arc_index = %d\n", net_arc_id,
              (int)parent.m_arcs.size());
      fprintf(stderr, "source_node = %d, target_node = %d, num_nodes = %d\n",
              source_node, target_node, num_nodes);
      exit(1);
    }
  }
}


double
HmmNetBaumWelch::get_arc_score(int arc_id, const FeatureVec &fea_vec)
{
  double score = loglikelihoods.one();
  
  if (m_use_static_scores)
    score = m_arcs[arc_id].static_score;

  if (!m_arcs[arc_id].epsilon())
  {
    double tr_coef = 1;
    HmmTransition &tr = m_model.transition(m_arcs[arc_id].transition_index);
    if (m_use_transition_probabilities)
      tr_coef = tr.prob;
    double model_likelihood = m_model.state_likelihood(tr.source_index,
                                                       fea_vec);
    model_likelihood *= tr_coef;
    if (model_likelihood <= util::tiny_for_log)
      score = loglikelihoods.zero();
    else
    {
      score = loglikelihoods.times(
        score, m_acoustic_scale*(util::safe_log(model_likelihood)));
    }
  }
  return score;
}




struct ESLPendingArc {
  int source_node; // Segmented node in the extracted lattice
  int arc_id; // Logical arc that will be added to the new lattice
  double score; // Logical arc score (combined child arc scores)
  int child_arc_leaf; // Index to child_arc_tree

  ESLPendingArc(int source_node_, int arc_id_, double score_) : source_node(source_node_), arc_id(arc_id_), score(score_) { child_arc_leaf = -1; }
  ESLPendingArc(const ESLPendingArc &source) { source_node = source.source_node; arc_id = source.arc_id; score = source.score; child_arc_leaf = source.child_arc_leaf; }
};

HmmNetBaumWelch::SegmentedLattice*
HmmNetBaumWelch::extract_segmented_lattice(SegmentedLattice *frame_sl,
                                           int level)
{
  if (!frame_sl->frame_lattice)
    throw std::string("HmmNetBaumWelch::extract_segmented_lattice operates on frame-level lattice");
  if (level <= 0)
    throw std::string("HmmNetBaumWelch::extract_segmented_lattice: Invalid hierarchy level");

  SegmentedLattice *sl = new SegmentedLattice;
  sl->frame_lattice = false;

  // One frame lattice node can occupy several pending logical arcs
  typedef std::multimap<int, ESLPendingArc> NodePendingArcs;
  NodePendingArcs active_nodes[2]; // Frame lattice nodes
  std::vector< std::pair<int, int> > child_arc_tree; // (parent index, arc_id)
  int tbuf = 0;
  
  // Initialize
  sl->initial_node = 0;
  sl->final_node = 0; // Set in case the lattice will be empty
  sl->nodes.push_back(
    SegmentedNode(frame_sl->nodes[frame_sl->initial_node].frame));
  active_nodes[tbuf].insert(
    NodePendingArcs::value_type(frame_sl->initial_node, ESLPendingArc(sl->initial_node, -1, loglikelihoods.one())));

  // NOTE! This method operates on a frame-level lattice, so the
  // order of processed nodes is automatically topological.
  
  while (active_nodes[tbuf].size() > 0 &&
         (*active_nodes[tbuf].begin()).first != frame_sl->final_node)
  {
    // Extracted segmented nodes created for a particular frame node
    // (at a certain frame). Map is (frame_sl node) -> (sl node)
    std::map<int, int> new_seg_nodes;
    
    int sbuf = tbuf;
    tbuf ^= 1;
    // Propagate the active nodes
    for (NodePendingArcs::iterator it = active_nodes[sbuf].begin();
         it != active_nodes[sbuf].end(); ++it)
    {
      SegmentedNode &source_frame_node = frame_sl->nodes[(*it).first];
      
      // Indicates if the current pending arc has been connected to this node
      bool logical_connected = false;
      
      for (int a = 0; a < (int)source_frame_node.out_arcs.size(); a++)
      {
        int frame_arc_id = source_frame_node.out_arcs[a];
        if (frame_sl->arcs[frame_arc_id].total_score <= loglikelihoods.zero())
          continue; // Avoid pruned paths
        
        int net_arc_id = frame_sl->arcs[frame_arc_id].net_arc_id;

        // Find the parent network arc
        int logical_arc_id = m_arcs[net_arc_id].parent_arc;
        while (logical_arc_id != -1 &&
               m_logical_arcs[logical_arc_id].level < level)
          logical_arc_id = m_logical_arcs[logical_arc_id].parent_arc;
        if (logical_arc_id == -1 ||
            m_logical_arcs[logical_arc_id].level != level)
          continue;

        int target_frame_node = frame_sl->arcs[frame_arc_id].target_node;
        if ((*it).second.arc_id != logical_arc_id)
        {
          // Logical arc starts/changes here
          int next_seg_node = -1;
          if ((*it).second.arc_id != -1 && !logical_connected)
          {
            // Realize the current pending arc
            std::map<int,int>::iterator nit = new_seg_nodes.find((*it).first);
            if (nit == new_seg_nodes.end())
            {
              nit = new_seg_nodes.insert(
                std::map<int,int>::value_type(
                  (*it).first, sl->nodes.size())).first;
              sl->nodes.push_back(
                SegmentedNode(frame_sl->nodes[(*it).first].frame));
            }
            next_seg_node = (*nit).second;
            int cur_net_arc_id = (*it).second.arc_id;
            int seg_arc_id = 
              sl->create_segmented_arc(cur_net_arc_id,
                                       m_logical_arcs[cur_net_arc_id].label, 
                                       -1, (*it).second.source_node,
                                       next_seg_node, (*it).second.score,
                                       loglikelihoods.zero(), // Not implemented
                                       loglikelihoods.zero());
            sl->child_arcs.resize(seg_arc_id+1);
            esl_fill_child_arcs(sl->child_arcs[seg_arc_id],
                                (*it).second.child_arc_leaf,
                                child_arc_tree);
            logical_connected = true;
          }
          else
          {
            // No need to create logical arcs for the extracted lattice.
            // Fetch the source node for the next logical arc
            next_seg_node = sl->initial_node;
            if ((*it).second.arc_id != -1)
            {
              std::map<int,int>::iterator nit =
                new_seg_nodes.find((*it).first);
              assert( nit != new_seg_nodes.end() );
              next_seg_node = (*nit).second;
            }
          }
          bool exists = false;
          NodePendingArcs::iterator pa_it =
            active_nodes[tbuf].lower_bound(target_frame_node);
          while (pa_it != active_nodes[tbuf].end() &&
                 (*pa_it).first == target_frame_node)
          {
            if ((*pa_it).second.source_node == next_seg_node &&
                (*pa_it).second.arc_id == logical_arc_id)
            {
              // This pending arc already exists
              exists = true;
              break;
            }
            ++pa_it;
          }

          if (!exists)
          {
            // Create a new pending logical arc
            NodePendingArcs::iterator insert_it = 
              active_nodes[tbuf].insert(
                NodePendingArcs::value_type(
                  target_frame_node,
                  ESLPendingArc(next_seg_node, logical_arc_id,
                                frame_sl->arcs[frame_arc_id].arc_score)));
            (*insert_it).second.child_arc_leaf = child_arc_tree.size();
            child_arc_tree.push_back(std::pair<int,int>(-1, frame_arc_id));
          }
        }
        else
        {
          // Logical arc continues, update the pending arc
          bool merged = false;
          ESLPendingArc copy = ESLPendingArc((*it).second);

          // Update the pending arc information
          child_arc_tree.push_back(std::pair<int,int>(copy.child_arc_leaf,
                                                      frame_arc_id));
          copy.child_arc_leaf = child_arc_tree.size() - 1;
          copy.score = loglikelihoods.times(
            copy.score, frame_sl->arcs[frame_arc_id].arc_score);

          // Find other pending arcs in the target node
          NodePendingArcs::iterator pa_it =
            active_nodes[tbuf].lower_bound(target_frame_node);
          while (pa_it != active_nodes[tbuf].end() &&
                 (*pa_it).first == target_frame_node)
          {
            if ((*pa_it).second.source_node == (*it).second.source_node &&
                (*pa_it).second.arc_id == (*it).second.arc_id)
            {
              // These pending arcs have the same endpoints and identity,
              // so they can be merged.
              int new_leaf = esl_merge_child_arcs(
                (*pa_it).second.child_arc_leaf, copy.child_arc_leaf,
                child_arc_tree);
              (*pa_it).second.child_arc_leaf = new_leaf;
              // Add the scores
              (*pa_it).second.score =
                loglikelihoods.plus((*pa_it).second.score, copy.score);
              merged = true;
              break;
            }
            ++pa_it;
          }
          if (!merged)
          {
            // Propagate the pending arc
            active_nodes[tbuf].insert(
              NodePendingArcs::value_type(target_frame_node, copy));
          }
        }
      }
    }
    active_nodes[sbuf].clear();
  }

  // Connect to the final node
  if (active_nodes[tbuf].size() > 0)
  {
    // Create the final node
    sl->final_node = sl->nodes.size();
    sl->nodes.push_back(
      SegmentedNode(frame_sl->nodes[frame_sl->final_node].frame));
    for (NodePendingArcs::iterator it = active_nodes[tbuf].begin();
         it != active_nodes[tbuf].end(); ++it)
    {
      int seg_arc_id = 
        sl->create_segmented_arc((*it).second.arc_id,
                                 m_logical_arcs[(*it).second.arc_id].label, 
                                 -1, (*it).second.source_node,
                                 sl->final_node, (*it).second.score,
                                 loglikelihoods.zero(), // Not implemented
                                 loglikelihoods.zero());
      sl->child_arcs.resize(seg_arc_id+1);
      esl_fill_child_arcs(sl->child_arcs[seg_arc_id],
                          (*it).second.child_arc_leaf,
                          child_arc_tree);
    }
  }

  // Fill total scores for the logical arcs
  sl->compute_total_scores();

  return sl;
}


int
HmmNetBaumWelch::esl_merge_child_arcs(int leaf1, int leaf2,
                                      std::vector< std::pair<int,int> > &tree)
{
  int cur_node1 = leaf1;
  int cur_node2 = leaf2;
  std::vector<int> tree_nodes;
  
  while (cur_node1 != cur_node2)
  {
    if (cur_node2 > cur_node1)
    {
      tree_nodes.push_back(cur_node2);
      cur_node2 = tree[cur_node2].first;
    }
    else
    {
      tree_nodes.push_back(cur_node1);
      cur_node1 = tree[cur_node1].first;
    }
  }

  // Chain the two child arc branches. The last node is already valid.
  // NOTE! It is assumed that the child arc indices in the two branches
  // are unique. If this would not be the case, std::map should be used
  // for tree_nodes instead of std::vector.
  int parent_node = tree_nodes.back();
  for (int i = tree_nodes.size()-2; i >= 0; i--)
  {
    int child_node = tree_nodes[i];
    tree[child_node].first = parent_node;
    parent_node = child_node;
  }

  return parent_node; // Return the new leaf
}


void
HmmNetBaumWelch::esl_fill_child_arcs(std::vector<int> &child_arcs,
                                     int leaf_index,
                                     std::vector< std::pair<int,int> > &tree)
{
  int cur_node = leaf_index;
  std::vector<int> temp;
  while (cur_node != -1)
  {
    temp.push_back(tree[cur_node].second);
    cur_node = tree[cur_node].first;
  }
  child_arcs.resize(temp.size());
  std::copy(temp.rbegin(), temp.rend(), child_arcs.begin());
}


void
HmmNetBaumWelch::rescore_segmented_lattice(SegmentedLattice *frame_sl)
{
  assert( frame_sl->frame_lattice );

  int prev_frame = -1;
  for (int i = 0; i < (int)frame_sl->nodes.size(); i++)
  {
    SegmentedNode &cur_node = frame_sl->nodes[i];
    if (prev_frame != cur_node.frame)
    {
      m_model.reset_cache();
      prev_frame = cur_node.frame;
    }
    for (int a = 0; a < (int)cur_node.out_arcs.size(); a++)
    {
      int segmented_arc_id = cur_node.out_arcs[a];
      int net_arc_id = frame_sl->arcs[segmented_arc_id].net_arc_id;
      double new_acoustic_score =
        get_arc_score(net_arc_id, get_feature(cur_node.frame));
      if (new_acoustic_score <= loglikelihoods.zero())
      {
        frame_sl->arcs[segmented_arc_id].arc_score = loglikelihoods.zero();
        frame_sl->arcs[segmented_arc_id].arc_acoustic_score =
          loglikelihoods.zero();
      }
      else
      {
        if (m_use_static_scores)
          new_acoustic_score -= m_arcs[net_arc_id].static_score;
        frame_sl->arcs[segmented_arc_id].arc_score +=
          (new_acoustic_score -
           frame_sl->arcs[segmented_arc_id].arc_acoustic_score);
        frame_sl->arcs[segmented_arc_id].arc_acoustic_score =
          new_acoustic_score;
      }
    }
  }

  // Compute total scores
  frame_sl->compute_total_scores();
}


void
HmmNetBaumWelch::clear_bw_scores(void)
{
  for (int i = 0; i < (int)m_arcs.size(); i++) {
    m_arcs[i].bw_scores.clear();
  }
  m_bw_scores_computed = false;
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

}
