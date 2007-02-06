#include <iostream>
#include <stdio.h>
#include <math.h>

#include "Viterbi.hh"
#include "util.hh"


Viterbi::Viterbi(HmmSet &model, FeatureGenerator &fea_gen, 
		 PhnReader *phn_reader)
  : m_model(model),
    m_fea_gen(fea_gen),
    m_phn_reader(phn_reader),
    m_lattice(),
    m_current_frame(0),
    m_feature_frame(0),
    m_best_log_prob(-INF),
    m_best_position(-1),
    m_prob_beam(INF),
    m_state_beam(INT_MAX),
    m_force_end(false),
    m_last_window(false),
    m_print_all_states(false)
{
}

void
Viterbi::reset()
{
  m_lattice.reset();
  m_current_frame = 0;
  m_feature_frame = 0;
  m_last_frame = m_lattice.frames();
  m_last_position = m_lattice.positions();
  m_transcription.clear();
  m_speakers.clear();
  m_best_log_prob = -INF;
  m_best_position = -1;
  m_force_end = false;
  m_last_window = false;
  m_state_prob.resize(m_lattice.positions());
  m_accumulated_log_prob = 0;
  m_final_log_prob = 0;
}

void
Viterbi::resize(int frames, int positions, int block)
{
  m_lattice.resize(frames, positions, block);
  m_best_path.resize(frames);
  m_best_path[0].position = 0; // Initialize

  m_transcription.resize(positions);
  m_transcription.clear();
  
  m_last_frame = frames;
  m_last_position = positions;

  m_current_frame = 0;
}


void
Viterbi::fill_transcription()
{
  static PhnReader::Phn phn;
  static bool eof;

  if (m_phn_reader != NULL)
  {
    // FIXME!!!
    if (m_transcription.size() == 0)
      eof = !m_phn_reader->next(phn);
    while (!eof && (int)m_transcription.size() < m_last_position) {
      // Read the next label and get the hmm of the phoneme
      std::string label = phn.label[0];
      phn.label.erase(phn.label.begin()); // Remove the first HMM label
      add_hmm_to_transcription(m_model.hmm_index(label),phn.comment,
                               phn.label, phn.speaker);
      eof = !m_phn_reader->next(phn);
    }

    if (m_last_position > (int)m_transcription.size())
      m_last_position = m_transcription.size();
  }
  else
  {
    fprintf(stderr, "Viterbi::fill_transcription: m_phn_reader is undefined!\n");
    exit(1);
  }
}

void
Viterbi::add_hmm_to_transcription(int hmm_index,std::string &comment,
                                  std::vector<std::string> &additional_hmms,
                                  std::string &speaker)
{
  std::string state_label;
  Hmm &hmm = m_model.hmm(hmm_index);
  register int t;
  register int s;
  int transition_index;
  char temp[10];
  for (s = 0; s < hmm.num_states(); s++) {

    // Only add comment to the first state
    state_label = hmm.label;
    if (m_print_all_states)
    {
      sprintf(temp, ".%d", s);
      state_label += temp;
    }

    if (s == 0) {
      TranscriptionState state(hmm.state(s), state_label, comment,
                               hmm_index, s);
      state.printed = false;
      state.additional_labels = additional_hmms;

      m_transcription.push_back(state);
    }
    else
    {
      //m_transcription.push_back(TranscriptionState(hmm.state(s), "", ""));
      TranscriptionState state(hmm.state(s), m_print_all_states?state_label:
                               "", "", hmm_index, s);
      state.printed = (m_print_all_states?false:true);
      m_transcription.push_back(state);
    }
    
    // Set speaker
    m_speakers.push_back(speaker);
    
    // Precompute transitions for the given transcription.  In the
    // transcription want the transition indices to be relative to
    // the position in the transcription.
    TranscriptionState &state = m_transcription.back();
    std::vector<int> transitions = hmm.transitions(s);
    for (t = 0; t < (int)transitions.size(); t++) {
      transition_index = transitions[t];
      HmmTransition &orig = m_model.transition(transition_index);
      TranscriptionTransition copy;

      assert(orig.target != -1);

      copy.transition_index = transition_index;
      copy.log_prob = util::safe_log(orig.prob); 
      if (orig.target == -2)
        copy.target = hmm.num_states() - s;
      else
        copy.target = orig.target - s;

      state.transitions.push_back(copy);
    }
  }
}

// ASSUMES:
//  - cells in m_current_frame are unused
//  - m_frames is set correctly if not enough data
//  - Frame (current_frame-1) has been computed
//  - Range is correct for frame (current_frame-1)
//  - m_best_log_prob and m_best_position set correctly
void
Viterbi::fill_transition_probs()
{
  register int p;
  Lattice::Range &range = m_lattice.range(m_current_frame - 1);
  assert(range.end > range.start);

  // Prune at beginning (both prob_beam and state_beam)
  for (p = range.start; ; p++) {
    assert(p < range.end);
    
    Lattice::Cell &source_cell = m_lattice.at(m_current_frame - 1, p);
    if (source_cell.unused() ||
        source_cell.log_prob + m_prob_beam < m_best_log_prob 
	|| p + m_state_beam < m_best_position) {
      m_lattice.reset_frame(m_current_frame - 1, range.start + 1, range.end);
    }
    else
      break;
  }

  // Prune at end (both prob_beam and state_beam)
  for (p = range.end - 1; ; p--) {
    assert(p >= range.start);

    Lattice::Cell &source_cell = m_lattice.at(m_current_frame - 1, p);
    if (source_cell.unused() ||
        source_cell.log_prob + m_prob_beam < m_best_log_prob
	|| p - m_state_beam > m_best_position) {
      m_lattice.reset_frame(m_current_frame - 1, range.start, range.end - 1);
    }
    else
      break;
  }

  // Fill range in this frame
  Lattice::Range &target_range = m_lattice.range(m_current_frame);
  register int t;
  float new_prob;
  int target;
  for (p = range.start; p < range.end; p++) {
    Lattice::Cell &source_cell = m_lattice.at(m_current_frame - 1, p);

    // Iterate through all transitions leaving from the state.
    for (t = 0; t < (int)m_transcription[p].transitions.size(); t++) {

      // Transitions in the transcription have relative targets (for
      // example, -1 means the previous position)
      TranscriptionTransition &transition = m_transcription[p].transitions[t];
      target = p + transition.target;

      // Skip transitions which lead outside the lattice.
      // if (transition.target < 0 || transition.target >= m_last_position) 
      if (target < 0 || target >= m_last_position) 
	continue; // FIXME!!  Is this correct?!

      // Update lattice

      // Update ranges too
      if (target_range.end <= target)
	m_lattice.reset_frame(m_current_frame, target_range.start, target + 1);
      
      if (target_range.start == -1 || target_range.start > target)
	m_lattice.reset_frame(m_current_frame, target, target_range.end);

      Lattice::Cell &target_cell = m_lattice.at(m_current_frame, target);
      new_prob = source_cell.log_prob + transition.log_prob;
      if (target_cell.unused() || new_prob > target_cell.log_prob) {
	target_cell.from = p;
	target_cell.log_prob = new_prob;
	target_cell.transition_index = transition.transition_index;
      }
      assert(target_range.start >= 0 && target_range.start < target_range.end);
    }
  }
}

void
Viterbi::fill_observation_probs(const FeatureVec &fea_vec)
{
  m_model.precompute(fea_vec);
  Lattice::Range &range = m_lattice.range(m_current_frame);
  assert(range.end > range.start);

  // Compute normalized probabilities.  FIXME: think about this?
  float best_prob = -1;
  register int p;
  for (p = range.start; p < range.end; p++) {
    m_state_prob[p] = m_model.state_prob(m_transcription[p].state, fea_vec);
    if (m_state_prob[p] > best_prob)
      best_prob = m_state_prob[p];
  }
  assert(best_prob > 0);

  float best_log_prob = util::safe_log(best_prob);
  for (p = range.start; p < range.end; p++) 
    m_state_prob[p] = util::safe_log(m_state_prob[p]) - best_log_prob;

  m_accumulated_log_prob += best_log_prob;
  
  // Add observation probabilities to lattice.
  m_best_position = -1;
  m_best_log_prob = -INF;
  for (p = range.start; p < range.end; p++) {
    Lattice::Cell &cell = m_lattice.at(m_current_frame, p);
    
    // REMOVE
    if (cell.from < 0) {
      printf("m_current_frame = %i\n", m_current_frame);
      printf("m_last_frame = %i\n", m_last_frame);
      printf("range: %i (%i-%i)\n", p, range.start, range.end);
      assert(false);
    }

    if (!cell.unused())
    {
      cell.log_prob += m_state_prob[p];
      if (cell.log_prob > m_best_log_prob) {
        m_best_log_prob = cell.log_prob;
        m_best_position = p;
      }

      // Floor values to -INF.
      if (cell.log_prob < -INF)
        cell.log_prob = -INF;
    }
  }
  assert(m_best_position >= 0);
}


// NOTES:
// - the new path should match with the old best path in the first frame,
//   otherwise there can be holes in segmentation, throws if not
void
Viterbi::compute_best_path()
{
  int frame = m_current_frame - 1;
  int position;

  // If the end must be forced, we have to start at upper right
  // corner.  Otherwise we start at the best last position.
  if (m_force_end && m_last_window) {
    position = m_last_position - 1;
    if (position < (int)m_transcription.size() - 1)
      std::cerr << "WARNING: Viterbi::compute_best_path: transcription end out of window" << std::endl;
    
    else if (!m_lattice.range(frame).has(position))
      std::cerr << "WARNING: Viterbi::compute_best_path: transcription end out of range" << std::endl;
  }
  else
    position = m_best_position;
  
  // FIXME: We ignore cells with weird from-pointers.  This happens
  // for some reason, if transcription or models are not very good at
  // the end of transcription.  Should not happen elsewhere.
  // --combined with too tight pruning limits -> end gets pruned (mavarjok)
  
  if (!m_lattice.range(frame).has(position)) {
    int new_pos = m_lattice.range(frame).end - 1;
    fprintf(stderr, "%d last transcription states were lost because of bad transcription or models combined with too tight pruning.\nsolution: increase state and/or probability beams\n", position-new_pos);
    position = new_pos;
  }

  // Save the log-likelihood of the final path
  m_final_log_prob = at(frame, position).log_prob;

  // In the end we want to check that the path is continuous
  int check_position = m_best_path[0].position;

  // Backtrack the best path.
  m_best_path[frame].position = position;
  m_best_path[frame].transition_index = -1; // No transition for last state
  while (frame > 0) {
    Lattice::Cell &cell = at(frame, position);
    position = cell.from;

    // Sanity check of ranges, REMOVE
    //Lattice::Range &range = m_lattice.range(frame - 1);
    //assert(cell.from >= range.start);
    //assert(cell.from < range.end);

    // Update best path
    m_best_path[frame - 1].position = position;
    m_best_path[frame - 1].transition_index = cell.transition_index;
    frame--;
  }

  if (m_best_path[0].position != check_position)
    std::cerr 
      << "WARNING (Viterbi): discontinuous path due to too small a window, difference: "
      << m_best_path[0].position - check_position << std::endl;  
}

void Viterbi::fill()
{
  fill_transcription();
  m_model.reset_state_probs();
  if (m_current_frame == 0) {
    m_lattice.reset_frame(0, 0, 1);
    // FIXME: Ok? We do not use real probabilities anyway.    
    m_lattice.at(0, 0).log_prob = 0;

    // Fill the correct log-prob for reporting (the probability of
    // the first state does not change the viterbi path)
    const FeatureVec feavec = m_fea_gen.generate(m_feature_frame);
    m_accumulated_log_prob = util::safe_log(
      m_model.state_prob(m_transcription[0].state, feavec));
    
    m_feature_frame++;
    m_current_frame++;
  }
  // fills the frames
  while (m_current_frame < m_last_frame) {
    const FeatureVec feavec = m_fea_gen.generate(m_feature_frame);
    if (m_fea_gen.eof())
    {
      m_last_frame = m_current_frame;
      m_last_window = true;
      break;
    }
    m_model.reset_state_probs();
    fill_transition_probs();
    fill_observation_probs(feavec);  
    m_feature_frame++;
    m_current_frame++;
    //    sanity_check(); // enable for debugging
  }
  compute_best_path();
}



// NOTES:
// - cell.log_prob's are not consistent after a move, because move
//   normalizes frame m_current_frame-1
// INVARIANTS:
// - all cells after (and including) m_current_frame should be cleared
void
Viterbi::move(int frame, int position)
{
  m_speakers.erase(m_speakers.begin(), m_speakers.begin() + position);
  m_transcription.erase(m_transcription.begin(), 
			m_transcription.begin() + position);
  m_lattice.move(frame, position);
  m_current_frame -= frame;
  m_best_position -= position;
  
  for (int f = frame; f < m_last_frame; f++) {
    m_best_path[f - frame] = m_best_path[f]; 
    m_best_path[f - frame].position -= position;
  }

  // With big audio files, the log_prob grows very much and we lose
  // precicion in float.  Thus, we subtract the best log_prob at the
  // last frame.
  int f = m_current_frame - 1;
  float log_prob = at(f, m_best_path[f].position).log_prob;
  m_accumulated_log_prob += log_prob;
  m_final_log_prob = 0;
  Lattice::Range &range = m_lattice.range(f);
  for (int p = range.start; p < range.end; p++)
    at(f, p).log_prob -= log_prob;
}

void
Viterbi::sanity_check()
{
  for (int f = 0; f < m_lattice.frames(); f++) {
    // Check ranges
    Lattice::Range &range = m_lattice.range(f);
    assert((range.start == -1 && range.end == -1) ||
	   (range.start < range.end && range.start >= 0));

    // Check frame cells
    for (int p = range.start; p < range.end; p++) {
      Lattice::Cell &cell = at(f, p);
      // Check cells
      assert(cell.log_prob >= -INF); 
      
      if (f == 0) {
	assert(cell.from < 0);
	assert(cell.transition_index < 0);
      }      
      else {
	assert(cell.from >= 0);
	assert(cell.transition_index >= 0 && cell.transition_index < 100000);
      }
    }
  }
}



void
Viterbi::debug_show_lattice()
{
  // Show transcription
  for (int p = 0; p < (int)m_transcription.size(); p++) {
    TranscriptionState &state = m_transcription[p];
    if (state.label.length() > 0)
      std::cerr << state.label[0];
    else
      std::cerr << ".";
  }
  std::cerr << std::endl;

  // Show lattice
  for (int f = 0; f < m_current_frame; f++) {
    Lattice::Range &range = m_lattice.range(f);

    // Inactive bottom part
    for (int p = 0; p < range.start; p++) {
      std::cerr << ".";
    }

    // Active part
    for (int p = range.start; p < range.end; p++) {
      std::cerr << "*";
    }

    // Inactive top part
    for (int p = range.end; p < m_lattice.positions(); p++) {
      std::cerr << ".";
    }
    std::cerr << std::endl;
  }
}
