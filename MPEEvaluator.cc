#include <string>
#include <string.h>

#include "MPEEvaluator.hh"

std::string
MPEEvaluator::extract_center_phone(const std::string &label)
{
  int pos1 = label.find_last_of('-');
  int pos2 = label.find_first_of('+');
  std::string temp = "";
  if (pos2 < 0)
    pos2 = label.find_first_of(".");
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
MPEEvaluator::extract_context_phone(const std::string &label)
{
  int pos = label.find_last_of('.'); // Remove the state number
  if (pos > 0)
    return label.substr(0, pos);
  return label;
}


int
MPEEvaluator::extract_state(const std::string &label)
{
  int pos = label.find_last_of('.'); // Find the state number
  if (pos >= 0)
  {
    std::string temp;
    temp = label.substr(pos+1);
    return atoi(temp.c_str());
  }
  return -1;
}

double
MPEEvaluator::custom_data_value(int frame, HmmNetBaumWelch::Arc &arc,
                                int extra)
{
  int internal_frame = frame - m_first_frame;
  if (internal_frame < 0 || internal_frame >= (int)m_ref_segmentation.size())
    return 0;

  if (m_phone_error)
  {
    if (extra >= 0 && arc.label != "#")
    {
      std::string label;
      assert( arc.epsilon() );
      // Add insertion penalty
      label = extract_center_phone(arc.label);
      assert( label.size() > 0 );
      int beg_frame = std::max(std::min(frame, extra), m_first_frame);
      int end_frame = std::max(frame, extra);
      double max_ratio = 0;
      double subs_penalty = 0;
      double max_correct_penalty = 0;
      int prev_frame = end_frame;
      int cur_frame =
        (*m_ref_segmentation[end_frame-m_first_frame])[0].last_epsilon_frame;
      assert( cur_frame <= end_frame );
      while (cur_frame >= beg_frame)
      {                
        int cur_internal_frame = cur_frame - m_first_frame;
        assert( cur_internal_frame >= 0 );
        double ratio = (prev_frame - cur_frame + 1)/
          (*m_ref_segmentation[cur_internal_frame])[0].custom_score;
        assert( ratio <= 1 );
        if (ratio > max_ratio)
          max_ratio = ratio;
        if ((*m_ref_segmentation[cur_internal_frame])[0].label == label)
        {
          if (ratio > max_correct_penalty)
            max_correct_penalty = ratio;
        }
        subs_penalty += ratio;
        prev_frame = cur_frame-1;
        if (cur_frame == beg_frame)
          break;
        cur_frame =
          (*m_ref_segmentation[cur_internal_frame-1])[0].last_epsilon_frame;
      }
      if (cur_frame < beg_frame)
      {
        double ratio = (prev_frame - beg_frame + 1) /
          (*m_ref_segmentation[beg_frame-m_first_frame])[0].custom_score;
        assert( ratio <= 1 );
        if (ratio > max_ratio)
          max_ratio = ratio;
        if ((*m_ref_segmentation[beg_frame-m_first_frame])[0].label == label)
        {
          if (ratio > max_correct_penalty)
            max_correct_penalty = ratio;
        }
        subs_penalty += ratio;
      }
      subs_penalty = subs_penalty - max_correct_penalty + (1-max_ratio);
      return subs_penalty;
    }
  }
  else
  {
    if (arc.transition_id == HmmNetBaumWelch::EPSILON)
    {
      if (arc.label.size() > 0 && extract_state(arc.label) == -1)
        return m_mpfe_insertion_penalty;
      return 0;
    }

    if (m_ignore_silence)
    {
      // Ignore silence nodes
      if (arc.label.find('-') == std::string::npos &&
          arc.label.find('+') == std::string::npos &&
          arc.label[0] == '_') // Silence node
        return 0;
    }
  
    // Check the label against the correct one

    std::string label;
    int state = -1;
    if (m_mode == MPEM_MONOPHONE_LABEL || m_mode == MPEM_MONOPHONE_STATE)
      label = extract_center_phone(arc.label);
    else if (m_mode == MPEM_CONTEXT_LABEL ||
             m_mode == MPEM_HYP_CONTEXT_PHONE_STATE)
      label = extract_context_phone(arc.label);
    else if (m_mode == MPEM_STATE || m_mode == MPEM_CONTEXT_PHONE_STATE)
    {
      HmmTransition &tr = m_model->transition(arc.transition_id);
      state = m_model->emission_pdf_index(tr.source_index);
    }
    if (m_mode == MPEM_MONOPHONE_STATE)
      state = extract_state(arc.label);

    double correct_prob = 0;
  
    for (int i = 0; i < (int)m_ref_segmentation[internal_frame]->size(); i++)
    {
      if (m_mode == MPEM_MONOPHONE_LABEL || m_mode == MPEM_CONTEXT_LABEL)
      {
        if ((*m_ref_segmentation[internal_frame])[i].label == label)
          correct_prob = std::max(correct_prob,
                                  (*m_ref_segmentation[internal_frame])[i].prob);
      }
      else if (m_mode == MPEM_MONOPHONE_STATE)
      {
        if ((*m_ref_segmentation[internal_frame])[i].label == label &&
            (*m_ref_segmentation[internal_frame])[i].pdf_index == state)
          correct_prob = std::max(correct_prob,
                                  (*m_ref_segmentation[internal_frame])[i].prob);
      }
      else if (m_mode == MPEM_STATE)
      {
        if ((*m_ref_segmentation[internal_frame])[i].pdf_index == state)
          correct_prob = std::max(correct_prob,
                                  (*m_ref_segmentation[internal_frame])[i].prob);
      }
      else if (m_mode == MPEM_CONTEXT_PHONE_STATE)
      {
        Hmm &hmm = m_model->hmm((*m_ref_segmentation[internal_frame])[i].label);
        for (int s = 0; s < hmm.num_states(); s++)
          if (hmm.state(s) == state)
          {
            correct_prob=std::max(correct_prob,
                                  (*m_ref_segmentation[internal_frame])[i].prob);
          }
      }
      else if (m_mode == MPEM_HYP_CONTEXT_PHONE_STATE)
      {
        Hmm &hmm = m_model->hmm(label);
        for (int s = 0; s < hmm.num_states(); s++)
          if ((*m_ref_segmentation[internal_frame])[i].pdf_index==hmm.state(s))
          {
            correct_prob=std::max(correct_prob,
                                  (*m_ref_segmentation[internal_frame])[i].prob);
          }
      }
    }
  
    return (m_binary_mpfe ? (correct_prob > 0 ? 1 : 0) : correct_prob);
  }
  return 0;
}

void
MPEEvaluator::reset(void)
{
  m_first_frame = -1;
  // Clear the old reference segmentation
  for (int i = 0; i < (int)m_ref_segmentation.size(); i++)
    delete m_ref_segmentation[i];
  m_ref_segmentation.clear();
  m_non_silence_frames = 0;
  m_non_silence_occupancy = 0;
}

void
MPEEvaluator::fetch_frame_info(HmmNetBaumWelch *seg)
{
  int seg_frame = seg->current_frame();
  if (m_first_frame == -1)
    m_first_frame = seg_frame;
  else if (m_cur_frame+1 != seg_frame)
    throw std::string("Non-continuous numerator segmentation");
  m_cur_frame = seg_frame;

  m_ref_segmentation.push_back(
    new std::vector<HmmNetBaumWelch::ArcInfo> );
  seg->fill_arc_info(*(m_ref_segmentation.back()));

  if (m_phone_error)
  {
    assert( (*m_ref_segmentation.back()).size() == 1 );
    std::string &label = (*m_ref_segmentation.back())[0].label;
    label = extract_center_phone(label);
    if ((*m_ref_segmentation.back())[0].next_arc_is_epsilon)
    {
      // Set the phone length for all phone frames
      int l = seg_frame-(*m_ref_segmentation.back())[0].last_epsilon_frame+1;
      int f = (*m_ref_segmentation.back())[0].last_epsilon_frame-m_first_frame;
      for (; f < (int)m_ref_segmentation.size(); f++)
        (*m_ref_segmentation[f])[0].custom_score = l;
    }
  }
  else
  {
    bool silence = false;
    double silence_prob = 0;
  
    // Process the labels to speed up the custom data query
    for (int i = 0; i < (int)m_ref_segmentation.back()->size(); i++)
    {
      std::string &label = (*m_ref_segmentation.back())[i].label;
      if (label.find('-') == std::string::npos &&
          label.find('+') == std::string::npos &&
          label[0] == '_') // Silence node
      {
        silence = true;
        silence_prob += (*m_ref_segmentation.back())[i].prob;
      }
      if (m_mode == MPEM_MONOPHONE_STATE)
        (*m_ref_segmentation.back())[i].pdf_index = extract_state(label);
      if (m_mode == MPEM_MONOPHONE_LABEL || m_mode == MPEM_MONOPHONE_STATE)
        label = extract_center_phone(label);
      else if (m_mode == MPEM_CONTEXT_LABEL ||
               m_mode == MPEM_CONTEXT_PHONE_STATE)
        label = extract_context_phone(label);
    }
    if (!silence)
      m_non_silence_frames++;
    m_non_silence_occupancy += 1-silence_prob;
  }
}
