#include <string>
#include <string.h>
#include <algorithm>
#include <list>

#include "SegErrorEvaluator.hh"


namespace aku {

SegErrorEvaluator::RefIterator::RefIterator(SegErrorEvaluator &parent,
                                            std::vector<int> &arcs_in_frame,
                                            std::vector<int> &sorted_arcs,
                                            int from_frame) :
  m_parent(parent),
  m_arcs_in_frame(arcs_in_frame),
  m_sorted_arcs(sorted_arcs),
  m_frame(from_frame)
{
  if (from_frame >= 0)
  {
    // Note! Assumes (reasonably) that there are arcs that occupy
    // the given frame, e.g. arcs_in_frame is not empty
    assert( m_arcs_in_frame.size() > 0 );
    m_in_frame = true;
    m_index = 0;
  }
  else
  {
    // End iterator
    m_index = -1;
  }
}


// SegErrorEvaluator::RefIterator(const RefIterator& i) :
//   m_ref_lattice(i.m_ref_lattice)
// {
// }


std::string
SegErrorEvaluator::extract_center_phone(const std::string &label)
{
  int pos1 = label.find_last_of('-');
  int pos2 = label.find_first_of('+');
  std::string temp = "";
  if (pos2 < 0)
    pos2 = label.find_first_of(";", std::max(pos1, 0));
  if (pos1 >= 0 && pos2 >= 0 && pos2 > pos1+1)
    temp = label.substr(pos1+1, pos2-pos1-1);
  else if (pos2 >= 0)
    temp = label.substr(0, pos2);
  else if (pos1 >= 0)
    temp = label.substr(pos1+1);
  else
    temp = label;
  if ((int)temp.size() > 0)
    return temp;
  return label;
}


std::string
SegErrorEvaluator::extract_sublabel(const std::string &label, int count)
{
  int pos = 0;
  for (int i = 0; i <= count; i++)
  {
    int new_pos = label.find_first_of(';', pos);
    if (new_pos >= 0)
    {
      if (count == i)
        return label.substr(pos, new_pos-pos);
      pos = new_pos + 1;
    }
    else if (count == i)
      return label.substr(pos);
    else
      break;
  }
  return std::string("");
}


std::string
SegErrorEvaluator::extract_word(const std::string &label)
{
  int pos = label.find_last_of(';');
  if (pos >= 0)
    return label.substr(pos+1);
  return label;

}


double
SegErrorEvaluator::custom_score(HmmNetBaumWelch::SegmentedLattice const *sl,
                                int arc_index)
{
  double result = -1e6; // Uninitialized
  HmmNetBaumWelch::SegmentedArc const &cur_arc = sl->arcs[arc_index];
  int start_frame = sl->nodes[cur_arc.source_node].frame;
  int end_frame = sl->nodes[cur_arc.target_node].frame;
  std::string center_phone_label;
  std::string first_sublabel;
  std::string phone_sublabel;
  int cur_arc_pdf = -1;

  assert( sl->frame_lattice || (m_error_mode!=MPFE_MONOPHONE_LABEL &&
                                m_error_mode!=MPFE_MONOPHONE_STATE &&
                                m_error_mode!=MPFE_CONTEXT_LABEL &&
                                m_error_mode!=MPFE_PDF &&
                                m_error_mode!=MPFE_CONTEXT_PHONE_STATE &&
                                m_error_mode!=MPFE_HYP_CONTEXT_PHONE_STATE) );
  
  // Precompute labels for the current arc
  if (m_error_mode == MPE)
    center_phone_label = extract_center_phone(cur_arc.label);
  else if (m_error_mode == MPFE_PDF ||
           m_error_mode == MPFE_CONTEXT_PHONE_STATE)
    first_sublabel = extract_sublabel(cur_arc.label, 0);
  else if (m_error_mode == MPFE_HYP_CONTEXT_PHONE_STATE)
    phone_sublabel = extract_sublabel(cur_arc.label, 2);

  if (m_error_mode == MPFE_CONTEXT_PHONE_STATE)
    cur_arc_pdf = atoi(first_sublabel.c_str());

  if (m_ignore_silence)
  {
    std::string word_label = extract_word(cur_arc.label);
    if (!word_label.compare("_")) // Silence label
      return 0;
  }

  // Go through all the reference arcs that overlap this arc
  RefIterator ref_it = reference_iterator(start_frame);
  while (ref_it != reference_iterator_end() &&
         m_ref_lattice->nodes[m_ref_lattice->arcs[*ref_it].source_node].frame
         < end_frame)
  {
    HmmNetBaumWelch::SegmentedArc const &ref_arc =
      m_ref_lattice->arcs[*ref_it];
    int ref_start_frame = m_ref_lattice->nodes[ref_arc.source_node].frame;
    int ref_end_frame = m_ref_lattice->nodes[ref_arc.target_node].frame;
    double e = std::min(end_frame, ref_end_frame) - std::max(start_frame, ref_start_frame);
    assert( e > 0 );
    e /= (ref_end_frame - ref_start_frame);
    
    if (m_error_mode == MWE)
    {
      double new_custom = 0;
      if (cur_arc.label == ref_arc.label)
        new_custom = -1 + 2*e;
      else
        new_custom = -1 + e;
      if (new_custom > result)
        result = new_custom;
    }
    else if (m_error_mode == MPE)
    {
      double new_custom = 0;
      std::string ref_label = extract_center_phone(ref_arc.label);
      if (center_phone_label == ref_label)
        new_custom = -1 + 2*e;
      else
        new_custom = -1 + e;
      if (new_custom > result)
        result = new_custom;
    }
    else if (m_error_mode == MPFE_PDF)
    {
      std::string ref_first_sublabel = extract_sublabel(ref_arc.label, 0);
      HmmTransition &ref_tr =
        m_model->transition(atoi(ref_first_sublabel.c_str()));
      HmmTransition &tr = m_model->transition(atoi(first_sublabel.c_str()));
      if (m_model->emission_pdf_index(ref_tr.source_index) ==
          m_model->emission_pdf_index(tr.source_index))
        result = 1;
      else
        result = std::max(result, 0.0);
    }
    else if (m_error_mode == MPFE_CONTEXT_PHONE_STATE)
    {
      std::string ref_phone_label = extract_sublabel(ref_arc.label, 2);
      Hmm &hmm = m_model->hmm(ref_phone_label);
      double temp_result = 0;
      for (int s = 0; s < hmm.num_states(); s++)
      {
        if (hmm.state(s) == cur_arc_pdf)
          temp_result = 1;
      }
      result = std::max(temp_result, result);
    }
    else if (m_error_mode == MPFE_HYP_CONTEXT_PHONE_STATE)
    {
      std::string ref_trans_label = extract_sublabel(ref_arc.label, 0);
      HmmTransition &ref_tr =
        m_model->transition(atoi(ref_trans_label.c_str()));
      Hmm &hmm = m_model->hmm(phone_sublabel);      
      double temp_result = 0;
      for (int s = 0; s < hmm.num_states(); s++)
      {
        if (hmm.state(s) == ref_tr.source_index)
          temp_result = 1;
      }
      result = std::max(temp_result, result);
    }
    else
      throw std::string("Requested error mode not implemented");
    ++ref_it;
  }
  return result;
}


void
SegErrorEvaluator::reset(void)
{
  m_first_frame = -1;
  m_arcs_in_frame.clear();
  m_sorted_arcs.clear();
  m_non_silence_frames = 0;
  m_non_silence_occupancy = 0;
}


void
SegErrorEvaluator::initialize_reference(
  HmmNetBaumWelch::SegmentedLattice const *ref_lattice)
{
  m_ref_lattice = ref_lattice;

  // Fill and sort m_sorted_arcs
  m_sorted_arcs.resize(m_ref_lattice->arcs.size());
  for (int i = 0; i < (int)m_sorted_arcs.size(); i++)
    m_sorted_arcs[i] = i;
  std::sort(m_sorted_arcs.begin(), m_sorted_arcs.end(),
            ref_arc_frame_compare(*this));

  // Fill m_arcs_in_frame
  m_first_frame = m_ref_lattice->nodes[m_ref_lattice->initial_node].frame;
  m_last_frame = m_ref_lattice->nodes[m_ref_lattice->final_node].frame;
  m_arcs_in_frame.resize(m_last_frame - m_first_frame);
  std::list<int> arcs_in_frame;
  int cur_frame = m_first_frame;
  int arc_index = 0;
  int num_active = 0;
  for (int i = 0; i < (int)m_arcs_in_frame.size(); i++, cur_frame++)
  {
    // Remove arcs no longer active
    std::list<int>::iterator it = arcs_in_frame.begin();
    while (it != arcs_in_frame.end())
    {
      std::list<int>::iterator next_it = it;
      ++next_it;
      if (m_ref_lattice->nodes[m_ref_lattice->arcs[*it].target_node].frame
          <= cur_frame)
      {
        // Remove
        arcs_in_frame.erase(it);
        num_active--;
      }
      it = next_it;
    }

    // Activate arcs that begin from this frame
    const std::vector<HmmNetBaumWelch::SegmentedNode> &nodes = m_ref_lattice->nodes;
    const std::vector<HmmNetBaumWelch::SegmentedArc> &arcs = m_ref_lattice->arcs;
    while (arc_index < (int)m_sorted_arcs.size() &&
           nodes[arcs[m_sorted_arcs[arc_index]].source_node].frame ==
           cur_frame)
    {
      arcs_in_frame.insert(arcs_in_frame.begin(), m_sorted_arcs[arc_index]);
      num_active++;
      arc_index++;
    }

    // Copy the indices of the active arcs to the vector
    m_arcs_in_frame[i].resize(num_active);
    std::copy(arcs_in_frame.begin(), arcs_in_frame.end(),
              m_arcs_in_frame[i].begin());
  }
}

}
