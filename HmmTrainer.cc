#include <stdio.h>
#include <ctype.h>
#include <fstream>
#include <algorithm>

#include "HmmTrainer.hh"
#include "io.hh"


#define MAX_MLLT_ITER 7
#define MAX_MLLT_A_ITER 80
#define KERNEL_MINWEIGHT 0.005f
#define MAX_DURATION_COUNT 100


inline double safe_log(double x)
{
  if (x < 1e-30)
  {
    return -69; // about log(1e-30)
  }
  else
    return log(x);
}


int my_tolower(int c)
{
  if (c == 'Å')
    return 'å';
  else if (c == 'Ä')
    return 'ä';
  else if (c == 'Ö')
    return 'ö';
  return tolower(c);
}

std::string transform_context(const std::string &context, bool ignore_context_length)
{
  if (context[0] == '_')
    return "_";
  if (ignore_context_length)
    return context;
  std::string temp = context;
  std::transform(temp.begin(), temp.end(), temp.begin(), my_tolower);
  return temp;
}

std::string extract_left_context(const std::string &tri)
{
  return tri.substr(0, tri.rfind('-'));
}

std::string extract_right_context(const std::string &tri)
{
  return tri.substr(tri.find('+')+1);
}

std::string extract_center_pho(const std::string &tri)
{
  std::string temp = tri.substr(tri.rfind('-')+1);
  return temp.substr(0, temp.find('+'));
}

HmmTrainer::HmmTrainer(FeatureGenerator &fea_gen)
  : m_fea_gen(fea_gen),
    m_info(0),
    m_transform_module(NULL),
    m_source_dim(0),
    m_mllt(false),
    m_hlda(false),
    m_set_speakers(false),
    m_min_var(0.1),
    m_win_size(1000),
    m_overlap(0.6),
    m_cov_update(false),
    m_durstat(false),
    m_triphone_tying(false),
    m_no_force_end(false),
    m_print_segment(false),
    m_log_likelihood(0),
    m_fill_missing_contexts(false),
    m_tying_min_count(0),
    m_tying_min_lhg(0),
    m_tying_length_award(0),
    m_skip_short_silence_context(false),
    m_triphone_phn(false),
    m_num_dur_models(0),
    m_print_speakered(false),
    cov_m(NULL),
    m_transform_matrix(NULL),
    m_speaker_config(fea_gen)
{
}


HmmTrainer::~HmmTrainer()
{
  int i;
  
  // Free memory
  if (m_durstat)
  {
    // Be nice and free the allocations
    for (int i = 0; i < m_num_dur_models; i++)
    {
      delete [] dur_table[i];
    }
    delete [] dur_table;
  }
  if (cov_m != NULL)
  {
    for (i = 0; i < model_tmp.num_kernels(); i++) {
      delete cov_m[i];
    }
    delete [] cov_m;
  }
  if (m_hlda)
  {
    delete [] kernel_means;
    delete global_cov;
  }
  if (m_transform_matrix != NULL)
    delete m_transform_matrix;
}


void HmmTrainer::init(HmmSet &model, std::string adafile)
{
  int i, j;
  
  // Initialize accu structure and reset values to 0
  gk_norm.clear();
  gk_norm.resize(model.num_kernels(), 0.0);
  model_tmp=model;
  model_tmp.reset();

  if (m_durstat)
  {
    // Allocate buffers for collection duration statistics
    dur_table = new int*[model.num_states()];
    m_num_dur_models = model.num_states();
    for (int i = 0; i < model.num_states(); i++)
    {
      dur_table[i] = new int[MAX_DURATION_COUNT];
      for (int j = 0; j < MAX_DURATION_COUNT; j++)
      {
        dur_table[i][j] = 0;
      }
    }
  }

  if (m_transform_module != NULL)
  {
    // Allocate transform matrix (for MLLT or HLDA)
    m_source_dim = m_transform_module->sources().front()->dim();
    m_transform_matrix = new Matrix(m_source_dim, m_source_dim);
    Matrix temp_m(m_source_dim, m_source_dim);
    mtl::dense1D<int> pivots(m_source_dim, 0);

    const std::vector<float> *tr =
      m_transform_module->get_transformation_matrix();
    if ((int)tr->size() != m_source_dim*m_source_dim)
      throw std::string("MLLT transform must be square");
    
    for (i = 0; i < m_source_dim; i++)
      for (j = 0; j < m_source_dim; j++)
        (*m_transform_matrix)(i,j) = (*tr)[i*m_source_dim + j];
    copy(*m_transform_matrix, temp_m);
    lu_factor(temp_m, pivots);
    mllt_determinant = 1;
    for (i = 0; i < m_source_dim; i++)
      mllt_determinant *= temp_m(i,i);
    mllt_determinant = fabs(mllt_determinant);
    if (m_info > 0)
      fprintf(stderr, "Transformation determinant %f\n", mllt_determinant);
  }
  else
    mllt_determinant = 1;

  if (m_cov_update)
  {
    // Allocate covariance matrix accumulators
    cov_m = new Matrix*[model.num_kernels()];
    for (i = 0; i < model.num_kernels(); i++) {
      if (m_hlda)
      {
        assert( m_source_dim > 0 );
        cov_m[i] = new Matrix(m_source_dim, m_source_dim);
      }
      else
        cov_m[i] = new Matrix(m_fea_gen.dim(), m_fea_gen.dim());
    }
  }
      
  // Allocate HLDA buffers
  if (m_hlda)
  {
    kernel_means = new std::vector<float>[model.num_kernels()];
    for (i = 0; i < model.num_kernels(); i++) {
      kernel_means[i].resize(m_source_dim);
    }
    global_mean.resize(m_source_dim);
    global_cov = new Matrix(m_source_dim, m_source_dim);
    global_count = 0;
  }

  m_em_norm_warning_count = 0;
  
  // Load speaker configurations
  if (adafile.size() > 0)
  {
    m_speaker_config.read_speaker_file(io::Stream(adafile));
    m_set_speakers = true;
  }
}


void HmmTrainer::viterbi_train(int start_frame, int end_frame,
                               HmmSet &model,
                               Viterbi &viterbi,
                               FILE *phn_out, std::string speaker,
                               std::string utterance)
{
  // Compute window borders
  int window_start_frame = start_frame;
  int window_end_frame = 0;
  int target_frame;
 
  viterbi.reset();
  viterbi.set_feature_frame(window_start_frame);
  viterbi.set_force_end(!m_no_force_end);

  if (m_triphone_tying)
  {
    // Initialize triphone tying
    triphone_set.set_dimension(m_fea_gen.dim());
    triphone_set.set_info(m_info);
    if (m_tying_min_count > 0)
      triphone_set.set_min_count(m_tying_min_count);
    if (m_tying_min_lhg > 0)
      triphone_set.set_min_likelihood_gain(m_tying_min_lhg);
    triphone_set.set_length_award(m_tying_length_award);
    triphone_set.set_ignore_length(m_ignore_tying_length);
    triphone_set.set_ignore_context_length(m_ignore_tying_context_length);

    cur_tri_stat_hmm_index = -1;
    cur_tri_stat_state = -1;
    cur_tri_stat_right = "_";
    cur_tri_stat_center = "_";
  }

  if (m_set_speakers && speaker.size() > 0)
  {
    m_speaker_config.set_speaker(speaker);
    if (utterance.size() > 0)
      m_speaker_config.set_utterance(utterance);
  }

  bool last_window = false;
  int print_start = -1;
  std::string print_label;
  std::string print_speaker;
  std::string print_comment;

  m_log_likelihood = 0;

  // Process the file window by window
  while (1)
  {
    // Compute window borders
    window_end_frame = window_start_frame + m_win_size;
    if (end_frame > 0) {
      if (window_start_frame >= end_frame)
        break;
      if (window_end_frame >= end_frame) {
        window_end_frame = end_frame;
        last_window = true;
      }
    }
    
    // Fill lattice
    int old_current_frame = viterbi.current_frame();
    viterbi.set_last_window(last_window);
    viterbi.set_last_frame(window_end_frame - window_start_frame);
    viterbi.fill();
    if (m_fea_gen.eof())
    {
      // Viterbi encountered eof and stopped
      last_window = true;
      window_end_frame = window_start_frame + viterbi.last_frame();
    }

    assert( viterbi.feature_frame() == window_end_frame );

    // Print debug info
    if (m_info > 0 && old_current_frame < viterbi.current_frame()) {
      int start_frame = old_current_frame;
      int end_frame = viterbi.current_frame();
      int start_pos = viterbi.best_position(start_frame);
      int end_pos = viterbi.best_position(end_frame-1);
      float average_log_prob = 
        ((viterbi.at(end_frame-1, end_pos).log_prob - 
          viterbi.at(start_frame, start_pos).log_prob) 
         / (end_frame - start_frame));

      fprintf(stderr, "filled frames %d-%d (%f)\n",
              start_frame + window_start_frame,
              end_frame + window_start_frame,
              average_log_prob);
    }

    // The beginning part of the lattice is used for teaching.
    // Compute the frame dividing the lattice in two parts.  NOTE:
    // if the end of speech is in the window, we use the whole
    // window and do not continue further.
    target_frame = (int)(m_win_size * m_overlap);
    if (last_window)
      target_frame = window_end_frame - window_start_frame;
    if (window_start_frame + target_frame > window_end_frame)
      target_frame = window_end_frame - window_start_frame;

    // Print progress info
    if (m_info > 1) {
      fprintf(stderr, "teaching frames %d-%d\n",
              window_start_frame, window_start_frame + target_frame);
    }

    // Update parameters 
    if (m_durstat)
    {
      // Just collect the duration statistics
      update_duration_statistics(model, viterbi, target_frame);
    }
    else if (m_triphone_tying)
    {
      update_triphone_stat(viterbi, window_start_frame, 
                           window_start_frame + target_frame,
                           model);
    }
    else if (m_hlda)
      update_hlda_tmp_parameters(model, model_tmp, gk_norm,
                                 viterbi, window_start_frame, 
                                 window_start_frame + target_frame);
    else
    {
      int untreated_frames = 

	update_tmp_parameters(model, model_tmp,
			      gk_norm, viterbi, window_start_frame, 
			      window_start_frame + target_frame);
      target_frame = target_frame - untreated_frames;

    }
    // Print best path
    if (m_print_segment) {
      int f = 0;
      for (f = 0; f < target_frame; f++) {

        int pos = viterbi.best_position(f);
        const Viterbi::TranscriptionState &state = 
          viterbi.transcription(pos);

        if (!state.printed) {
          // Print pending line
          print_line(phn_out, m_fea_gen.frame_rate(), print_start,
                     f + window_start_frame, print_label, print_speaker,
                     print_comment);

          // Prepare the next print
          print_start = f + window_start_frame;
          print_label = state.label;
          print_comment = state.comment;

	  // Speaker ID
          print_speaker = m_speaker_config.get_cur_speaker();
          state.printed = true;
        }
      }
    }

    // Check if we have done the job; if not, move to next window

    window_start_frame += target_frame;
      
    if (last_window && window_start_frame >= end_frame)
      break;

    int position = viterbi.best_position(target_frame);

    // We used to leave some magic space under the best path, but we
    // noticed that we do not need it because the path must match with
    // the start of the old path.  This is problem if we have long
    // silence which is longer than the window.  In the previous
    // window, the silence ends up in the last state, and in the next
    // window the best path gets back to first state if we have space
    // below the window.

    //      position -= (int)(viterbi.last_position() * 0.10);
    //      if (position < 0)
    //        position = 0;

    viterbi.move(target_frame, position);
  } // Process the next window

  if (m_print_segment) {
    // FIXME: The end point window_start_frame+1 assumes 50% frame overlap

    print_line(phn_out, m_fea_gen.frame_rate(), print_start,
               window_start_frame + 1, print_label, print_speaker,
               print_comment);
  }
}


void HmmTrainer::finish_train(HmmSet &model)
{
  // Finish updating parameters
  if (m_durstat)
  {
    write_duration_statistics(model);
  }
  else if (m_triphone_tying)
  {
    triphone_set.finish_triphone_statistics();
    if (m_fill_missing_contexts)
      triphone_set.fill_missing_contexts(false);
    triphone_set.tie_triphones();
  }
  else if (m_mllt)
    update_mllt_parameters(model, model_tmp, gk_norm,
                           *m_transform_matrix);
  else if (m_hlda)
    update_hlda_parameters(model, model_tmp, gk_norm,
                           *m_transform_matrix);
  else
    update_parameters(model, model_tmp, gk_norm);

  if (!m_durstat && !m_triphone_tying)
  {
    // Replace the old models with new ones
    model = model_tmp;
  }
}


void
HmmTrainer::print_line(FILE *f, float fr, 
		       int start, int end, 
                       const std::string &label,
		       const std::string &speaker,
		       const std::string &comment)
{
  int frame_mult = (int)(16000/fr); // NOTE: phn files assume 16kHz sample rate
    
  if (start < 0)
    return;

  if (!m_print_speakered)
  {
    // normal phns
    fprintf(f, "%d %d %s %s\n", start * frame_mult, end * frame_mult, 
            label.c_str(), comment.c_str());
  }
  else
  {
    // speakered phns: speaker ID printed between label & comments
    fprintf(f, "%d %d %s %s %s\n", start * frame_mult, end * frame_mult,
            label.c_str(), speaker.c_str(), comment.c_str());
  }
}


void
HmmTrainer::update_triphone_stat(Viterbi &viterbi,
                                 int start_frame, 
                                 int end_frame, HmmSet &model)
{
  int frames = end_frame - start_frame;
  bool out = false;

  for (int f = 0; f < frames; f++)
  {
    const Viterbi::TranscriptionState &tr_state =
      viterbi.transcription(viterbi.best_position(f));

    if (!m_triphone_phn)
    {
      if (tr_state.state != cur_tri_stat_state)
      {
        if (tr_state.hmm_state_index == 0 ||
            cur_tri_stat_hmm_index != tr_state.hmm_index)
        {
          if (!m_skip_short_silence_context ||
              cur_tri_stat_center != "_")
            cur_tri_stat_left = cur_tri_stat_center;
          if (!m_skip_short_silence_context ||
              tr_state.label != "_")
            cur_tri_stat_center = cur_tri_stat_right;
          else
            cur_tri_stat_center = tr_state.label; // Short silence
        
          if (cur_tri_stat_state == -1)
            cur_tri_stat_center = tr_state.label;

          if (out)
            exit(2);
          if (cur_tri_stat_center != tr_state.label)
          {
            printf("HmmTrainer::update_triphone_state: Error:\n");
            printf("cur_tri_stat_center = %s\n", cur_tri_stat_center.c_str());
            printf("tr_state.label = %s\n", tr_state.label.c_str());
            printf("cur_tri_stat_state = %i\n", cur_tri_stat_state);
            printf("tr_state.state = %i\n", tr_state.state);
            printf("At frame %d\n", start_frame + f);
            out = true;
          }
          if (!m_skip_short_silence_context || tr_state.label != "_")
            cur_tri_stat_right = tr_state.next_label;
          cur_tri_stat_hmm_index = tr_state.hmm_index;
        }
        cur_tri_stat_state_index = tr_state.hmm_state_index;
        cur_tri_stat_state = tr_state.state;
      }

      if (cur_tri_stat_center[0] != '_')
      {
        std::string left=transform_context(cur_tri_stat_left,
                                           m_ignore_tying_context_length);
        std::string right=transform_context(cur_tri_stat_right,
                                            m_ignore_tying_context_length);

        triphone_set.add_feature(m_fea_gen.generate(start_frame+f),left,
                                 cur_tri_stat_center, right,
                                 cur_tri_stat_state_index);
      }
    }
    else
    {
      if (tr_state.state != cur_tri_stat_state &&
          tr_state.hmm_state_index == 0)
      {
        cur_tri_stat_left=extract_left_context(tr_state.label);
        cur_tri_stat_right=extract_right_context(tr_state.label);
        cur_tri_stat_center=extract_center_pho(tr_state.label);
        cur_tri_stat_state = tr_state.state;
      }

      if (cur_tri_stat_center[0] != '_')
      {
        triphone_set.add_feature(m_fea_gen.generate(start_frame+f),
                                 cur_tri_stat_left, cur_tri_stat_center,
                                 cur_tri_stat_right,
                                 tr_state.hmm_state_index);
      }
    }
  }
}


void
HmmTrainer::update_parameters(HmmSet &model, HmmSet &model_tmp, 
                              const std::vector<float> &gk_norm)
{
  float sum;
//  static float *tmpcov = NULL;
  int i,j;

  for (int k = 0; k < model.num_kernels(); k++)
  {
    // Normalize and update mean and covariance
    if (gk_norm[k] > 0)
    {
      for (i = 0; i < model.dim(); i++)
      {
        model_tmp.kernel(k).center[i] /= gk_norm[k];
      }

      if (m_cov_update)
      {
        scale(*cov_m[k], 1.0/gk_norm[k]);
        
        // Subtract mean squared from the covariance
        ExtVector mean_vec(&model_tmp.kernel(k).center[0], model.dim());
        for (i = 0; i < model.dim(); i++)
        {
          add(rows(*cov_m[k])[i],scaled(mean_vec,-(mean_vec)[i]),
              rows(*cov_m[k])[i]);
        }
      
        if (model_tmp.kernel(k).cov.type() == HmmCovariance::SINGLE)
        {
          model_tmp.kernel(k).cov.var() = 0;
          for (i = 0; i < model.dim(); i++) {
            model_tmp.kernel(k).cov.var() += (*cov_m[k])(i,i);
          }
          model_tmp.kernel(k).cov.var() /= model.dim();
          model_tmp.kernel(k).cov.var() = 
	    std::max(model_tmp.kernel(k).cov.var(), m_min_var);
        }
        else if (model_tmp.kernel(k).cov.type() == HmmCovariance::DIAGONAL)
        {
          // Extract diagonal
          // The worst case scenario for the determinant is when all
          // the diagonal values are at their minimum. Thus there must
          // be a limit which ensures that the square root of the
          // determinant, used in likelihood calculations, is inside
          // floating point range. sqrt(0.002**26) = 8.192e-36, which
          // is just enough (1e-38 is the lower limit of a single
          // precision float).  On the other hand, too small a
          // variance is pretty much useless, as it usually means the
          // gaussian is modeling only a few points, and thus is not
          // affecting the real performance.
          for (i = 0; i < model.dim(); i++) {
            model_tmp.kernel(k).cov.diag(i) = std::max((*cov_m[k])(i,i), 
						       m_min_var);
          }
        }
        else if (model_tmp.kernel(k).cov.type() == HmmCovariance::FULL
		 || model_tmp.kernel(k).cov.type() == HmmCovariance::PCGMM) 
	{
          for (i = 0; i < model.dim(); i++)
	    for (j = 0; j < model.dim(); j++)
	      if (i==j)
		model_tmp.kernel(k).cov.full(i,j) = std::max((*cov_m[k])(i,i), m_min_var);
	      else
		model_tmp.kernel(k).cov.full(i,j) = (*cov_m[k])(i,i); 
	}
      }
      else
      {
        if (model_tmp.kernel(k).cov.type() == HmmCovariance::SINGLE)
          model_tmp.kernel(k).cov.var() = model.kernel(k).cov.var();
        else if (model_tmp.kernel(k).cov.type() == HmmCovariance::DIAGONAL)
        {
          for (i = 0; i < model.dim(); i++)
          {
            model_tmp.kernel(k).cov.diag(i) = model.kernel(k).cov.diag(i);
          }
        }
        else if (model_tmp.kernel(k).cov.type() == HmmCovariance::FULL
		 || model_tmp.kernel(k).cov.type() == HmmCovariance::PCGMM)
	  {
          for (i = 0; i < model.dim(); i++)
	    for (j = 0; j < model.dim(); j++)
	      model_tmp.kernel(k).cov.full(i,j) = model.kernel(k).cov.full(i,j);
	}
      }
    }
    else
    {
      // No data for this gaussian, copy the old values
      for (i = 0; i < model.dim(); i++)
      {
        model_tmp.kernel(k).center[i] = model.kernel(k).center[i];
      }
      if (model_tmp.kernel(k).cov.type() == HmmCovariance::SINGLE)
        model_tmp.kernel(k).cov.var() = model.kernel(k).cov.var();
      else if (model_tmp.kernel(k).cov.type() == HmmCovariance::DIAGONAL)
      {
        for (i = 0; i < model.dim(); i++)
        {
          model_tmp.kernel(k).cov.diag(i) = model.kernel(k).cov.diag(i);
        }
      }
      else if (model_tmp.kernel(k).cov.type() == HmmCovariance::FULL 
	       || model_tmp.kernel(k).cov.type() == HmmCovariance::PCGMM)
      {
	  for (i = 0; i < model.dim(); i++)
	    for (j = 0; i < model.dim(); j++)
	      model_tmp.kernel(k).cov.full(i,j) = model.kernel(k).cov.full(i,j);
      }

    }
  }

  if (m_cov_update)
    model_tmp.compute_covariance_determinants();

  for (int s = 0; s < model.num_states(); s++) {
    HmmState &tmp_state = model_tmp.state(s);

    // Find normalization;
    sum=0.0;
    for (int k = 0; k < (int)tmp_state.weights.size(); k++)
      sum += tmp_state.weights[k].weight;

    // Normalize
    if (sum > 0) {
      for (int k = 0; k < (int)tmp_state.weights.size(); k++)
	tmp_state.weights[k].weight /= sum;
    }
    else
    {
      // Copy the old values
      HmmState &old_state = model.state(s);
      for (int k = 0; k < (int)tmp_state.weights.size(); k++)
	tmp_state.weights[k].weight = old_state.weights[k].weight;
    }
  }
  update_transition_probabilities(model, model_tmp);
}


void
HmmTrainer::update_transition_probabilities(HmmSet &model, HmmSet &model_tmp)
{
  float sum;
  
  /* transition probabilities */
  for (int h = 0; h < (int)model.num_hmms(); h++) {
    for (int s = 0; s < (int)model.hmm(h).num_states(); s++) {
      sum = 0.0;

      std::vector<int> &hmm_transitions = model_tmp.hmm(h).transitions(s);
      for (int t = 0; t < (int)hmm_transitions.size(); t++) 
	sum += model_tmp.transition(hmm_transitions[t]).prob;
      if (sum == 0.0)
      {
        // No data, copy the old values
        for (int t = 0; t < (int)hmm_transitions.size(); t++) {
          model_tmp.transition(hmm_transitions[t]).prob =
            model.transition(hmm_transitions[t]).prob;
        }
      }
      else
      {
        for (int t = 0; t < (int)hmm_transitions.size(); t++) {
          model_tmp.transition(hmm_transitions[t]).prob = 
            model_tmp.transition(hmm_transitions[t]).prob/sum;
          if (model_tmp.transition(hmm_transitions[t]).prob < .001)
            model_tmp.transition(hmm_transitions[t]).prob = .001;
        }
      }
    }
  }
}

int
HmmTrainer::update_tmp_parameters(HmmSet &model, HmmSet &model_tmp,
                                  std::vector<float> &gk_norm,
                                  Viterbi &viterbi,
                                  int start_frame, 
                                  int end_frame)
{
  int dim = m_fea_gen.dim();
  int frames = end_frame - start_frame;

  for (int f = 0; f < frames; f++) {
    FeatureVec feature = m_fea_gen.generate(start_frame + f);
    HmmState &state = model.state(viterbi.best_state(f));
    HmmState &state_accu= model_tmp.state(viterbi.best_state(f));

    update_state_kernels(model, model_tmp, state, state_accu,
                         feature, dim, true, gk_norm);

    /* Update transition counts.  Note, that the last state does not
       have transition (marked with -1) */
    int transition = viterbi.best_transition(f);
    if (transition >= 0)
    {
      model_tmp.transition(transition).prob++;
    }

    if (m_set_speakers && viterbi.current_speaker(f).size() > 0 &&
        viterbi.current_speaker(f) != m_speaker_config.get_cur_speaker())
    {
      std::string utterance = m_speaker_config.get_cur_utterance();
      m_speaker_config.set_speaker(viterbi.current_speaker(f));
      m_speaker_config.set_utterance(utterance); // Preserve utterance ID
      
      // if speaker change occurred while processing a window,
      // the number of frames left untreated is returned
      if (f != 0)
        return frames - f;
    }
  }

  return 0;

}

void
HmmTrainer::update_state_kernels(HmmSet &model, HmmSet &model_tmp,
                                 HmmState &state, HmmState &state_accu,
                                 const FeatureVec &feature, int dim,
                                 bool update_ll, std::vector<float> &gk_norm)
                                 
{
  double gamma_norm;
  int i;
  ExtVectorConst fea_vec(&feature[0], dim);
  
  gam.reserve(state.weights.size());
  gamma_norm = 0;
    
  // Calculate the probabilities
  for (int state_k = 0; state_k < (int)state.weights.size(); state_k++) {
    int hmm_k = state.weights[state_k].kernel;
    gam[state_k] = model.compute_kernel_likelihood(hmm_k, feature)*
      state.weights[state_k].weight;
    gamma_norm += gam[state_k];
  }

  if (update_ll) m_log_likelihood += safe_log(gamma_norm) + safe_log(mllt_determinant);

  if (gamma_norm == 0)
  {
    // If the feature vector is too far from every Gaussian, share its
    // influence evenly
    m_em_norm_warning_count++;
    for (int state_k = 0; state_k < (int)state.weights.size(); state_k++) {
      gam[state_k] = 1.0/state.weights.size();
    }
    gamma_norm = 1;
  }

  // Normalize the probabilities and add to the accumulators
  for (int state_k = 0; state_k < (int)state.weights.size(); state_k++)
  {
    int hmm_k = state.weights[state_k].kernel;
    HmmKernel &accuker=model_tmp.kernel(hmm_k);
      
    gam[state_k] /= gamma_norm;
    state_accu.weights[state_k].weight += gam[state_k];

    // Because of the possibly shared kernels, we have to accumulate
    // gamma values also per kernel to normalize the mean and covariance
    gk_norm[hmm_k] += gam[state_k];

    // Add mean
    for (i = 0; i < dim; i++)
      accuker.center[i] += feature[i]*gam[state_k];

    if (m_cov_update)
    {
      // Add covariance
      for (i = 0; i < (int)cov_m[hmm_k]->nrows(); i++)
      {
        add(rows(*cov_m[hmm_k])[i], scaled(fea_vec, gam[state_k]*fea_vec[i]),
            rows(*cov_m[hmm_k])[i]);
      }
    }
  }
}

void
HmmTrainer::update_mllt_parameters(HmmSet &model, HmmSet &model_tmp,
                                   std::vector<float> &gk_norm,
                                   Matrix &A)
{
  float sum;
  int i,j,k;
  Matrix **G;
  Matrix temp_m(model.dim(), model.dim());
  Vector temp_v(model.dim());
  mtl::dense1D<int> pivots(model.dim(), 0);
  int iter1, iter2;
  double temp, Adet;
  double beta;
  Matrix oldA(model.dim(), model.dim());

  copy(A, oldA);
  for (i = 0; i < model.dim(); i++)
  {
    for (j = 0; j < model.dim(); j++)
      A(i,j) = 0;
    A(i,i) = 1;
  }

  // Allocate G matrices
  G = new Matrix*[model.dim()];
  for (i = 0; i < model.dim(); i++)
  {
    G[i] = new Matrix(model.dim(), model.dim());
  }

  beta = 0;
  for (i = 0; i < model.num_kernels(); i++)
  {
    // Normalize and update the mean and covariances and compute the
    // sum of probabilities
    if (gk_norm[i] < TINY)
    {
      gk_norm[i] = 0;

      // No data for this gaussian, copy the old mean and reset variance
      for (j = 0; j < model.dim(); j++)
        model_tmp.kernel(i).center[j] = model.kernel(i).center[j];

      // Copy the old variance values too.
      if (model_tmp.kernel(i).cov.type() == HmmCovariance::SINGLE)
        model_tmp.kernel(i).cov.var() = model.kernel(i).cov.var();
      else if (model_tmp.kernel(i).cov.type() == HmmCovariance::DIAGONAL)
      {
        for (j = 0; j < model.dim(); j++)
        {
          model_tmp.kernel(i).cov.diag(j) = model.kernel(i).cov.diag(j);
        }
      }
    }
    else
    {
      for (j = 0; j < model.dim(); j++)
      {
        model_tmp.kernel(i).center[j] /= gk_norm[i];
      }
      scale(*cov_m[i], 1.0/gk_norm[i]);
      // Subtract mean squared from the covariance
      ExtVector mean_vec(&model_tmp.kernel(i).center[0], model.dim());
      for (j = 0; j < model.dim(); j++)
      {
        add(rows(*cov_m[i])[j],scaled(mean_vec,-(mean_vec)[j]),
            rows(*cov_m[i])[j]);
      }
      
      beta += gk_norm[i];
    }
  }
  
  for (iter1 = 0; iter1 < MAX_MLLT_ITER; iter1++)
  {
    // Estimate the diagonal variances
    for (k = 0; k < model.num_kernels(); k++)
    {
      if (gk_norm[k] > 0)
      {
        for (i = 0; i < model.dim(); i++)
          for (j = 0; j < model.dim(); j++)
            temp_m(i,j) = 0;
        mult(A, *cov_m[k], temp_m);
        for (i = 0; i < model.dim(); i++)
        {
          // See update_parameters for comments on the minimum value
          model_tmp.kernel(k).cov.diag(i) =
            std::max(dot(rows(temp_m)[i], rows(A)[i]), m_min_var);
        }
      }
    }

    // Estimate the transform matrix

    // Calculate the auxiliary matrix G
    for (i = 0; i < model.dim(); i++)
    {
       for (j = 0; j < model.dim(); j++)
         for (k = 0; k < model.dim(); k++)
            temp_m(j,k) = 0;
 
       for (k = 0; k < model.num_kernels(); k++)
       {
         if (gk_norm[k] > 0)
           add(scaled(*cov_m[k], gk_norm[k]/model_tmp.kernel(k).cov.diag(i)),
               temp_m);
       }
       // Invert
       lu_factor(temp_m, pivots);
       lu_inverse(temp_m, pivots, *G[i]);
    }
    // Iterate A
    for (iter2 = 0; iter2 < MAX_MLLT_A_ITER; iter2++)
    {
      // Get the cofactors of A
      transpose(A);
      lu_factor(A, pivots);
      lu_inverse(A, pivots, temp_m);
      temp = 1;
      for (i = 0; i < model.dim(); i++)
        temp *= A(i,i);
      // Note, the sign of det(A) is assumed to be positive.. Is it REALLY?
      Adet = fabs(temp);
      scale(temp_m, Adet); // Scale with determinant
      // Estimate the rows of A
      for (i = 0; i < model.dim(); i++)
      {
        mult(trans(*G[i]), rows(temp_m)[i], rows(A)[i]);
        temp = sqrt(beta/dot(rows(A)[i], rows(temp_m)[i]));
        scale(rows(A)[i], temp);
      }
    }

    // Normalize A
    copy(A, temp_m);
    lu_factor(temp_m, pivots);
    temp = 1;
    for (i = 0; i < model.dim(); i++)
      temp *= temp_m(i,i);
    Adet = fabs(temp);
    double scale = pow(Adet, 1/(double)model.dim());
    for (i = 0; i < model.dim(); i++)
      for (j = 0; j < model.dim(); j++)
        A(i,j) = A(i,j)/scale;
  }

  // Transform mean vectors
  for (k = 0; k < model.num_kernels(); k++)
  {
    ExtVector mean(&model_tmp.kernel(k).center[0], model.dim());
    
    copy(mean, temp_v);
    mult(A, temp_v, mean);
  }

  // Set weights
  for (int s = 0; s < model.num_states(); s++) {
    HmmState &tmp_state = model_tmp.state(s);

    // Find normalization;
    sum=0.0;
    for (int k = 0; k < (int)tmp_state.weights.size(); k++)
    {
      sum += tmp_state.weights[k].weight;
    }

    // Normalize
    if (sum > 0) {
      temp = 0;
      for (int k = 0; k < (int)tmp_state.weights.size(); k++)
      {
	tmp_state.weights[k].weight /= sum;
        if (tmp_state.weights[k].weight < KERNEL_MINWEIGHT)
        {
          tmp_state.weights[k].weight = KERNEL_MINWEIGHT;
        }
        temp += tmp_state.weights[k].weight;
      }
      // And renormalize...
      for (int k = 0; k < (int)tmp_state.weights.size(); k++)
      {
	tmp_state.weights[k].weight /= temp;
      }
    }
    else
    {
      // Copy the old values
      HmmState &old_state = model.state(s);
      for (int k = 0; k < (int)tmp_state.weights.size(); k++)
	tmp_state.weights[k].weight = old_state.weights[k].weight;
    }
  }

  update_transition_probabilities(model, model_tmp);

  // Once more, estimate the diagonal covariances
  for (k = 0; k < model.num_kernels(); k++)
  {
    if (gk_norm[k] > 0)
    {
      for (i = 0; i < model.dim(); i++)
        for (j = 0; j < model.dim(); j++)
          temp_m(i,j) = 0;
      mult(A, *cov_m[k], temp_m);
      for (i = 0; i < model.dim(); i++)
      {
        // See update_parameters for comments on the minimum value
        model_tmp.kernel(k).cov.diag(i) = 
	  std::max(dot(rows(temp_m)[i], rows(A)[i]), m_min_var);
      }
    }
  }

  // Set transformation
  for (i = 0; i < model.dim(); i++)
    for (j = 0; j < model.dim(); j++)
      temp_m(i,j) = 0;
  mult(A, oldA, temp_m);
  std::vector<float> tr;
  tr.resize(model.dim()*model.dim());
  for (i = 0; i < model.dim(); i++)
    for (j = 0; j < model.dim(); j++)
      tr[i*model.dim() + j] = temp_m(i,j);
  m_transform_module->set_transformation_matrix(tr);
  model_tmp.compute_covariance_determinants();
  
  // Free allocated memory
  for (i = 0; i < model.dim(); i++)
    delete G[i];
  delete [] G;
}


void
HmmTrainer::update_hlda_parameters(HmmSet &model, HmmSet &model_tmp,
                                   std::vector<float> &gk_norm,
                                   Matrix &A)
{
  float sum;
  int i,j,k;
  Matrix **G;
  int dim = m_source_dim;
  Matrix temp_m(dim, dim);
  mtl::dense1D<int> pivots(dim, 0);
  int iter1, iter2;
  double temp, Adet;
  double beta;
  std::vector<float> *kernel_covar;

  // Allocate G matrices
  G = new Matrix*[dim];
  for (i = 0; i < dim; i++)
  {
    G[i] = new Matrix(dim, dim);
  }

  // Allocate kernel covariances
  kernel_covar = new std::vector<float>[model.num_kernels()];
  for (i = 0; i < model.num_kernels(); i++)
  {
    kernel_covar[i].resize(dim);
  }

  beta = 0;
  for (i = 0; i < model.num_kernels(); i++)
  {
    // Normalize the mean and covariances and compute the sum of probabilities
    if (gk_norm[i] < TINY)
    {
      gk_norm[i] = 0;

      // No data for this gaussian, copy the old mean and reset variance
      for (j = 0; j < model.dim(); j++)
        model_tmp.kernel(i).center[j] = model.kernel(i).center[j];
      
      if (model_tmp.kernel(i).cov.type() == HmmCovariance::SINGLE)
        model_tmp.kernel(i).cov.var() = 1;
      else if (model_tmp.kernel(i).cov.type() == HmmCovariance::DIAGONAL)
      {
        for (j = 0; j < model.dim(); j++)
        {
          model_tmp.kernel(i).cov.diag(j) = 1;
        }
      }
    }
    else
    {
      for (j = 0; j < dim; j++)
      {
        kernel_means[i][j] /= gk_norm[i];
      }
      scale(*cov_m[i], 1.0/gk_norm[i]);
      // Subtract mean squared from the covariance
      ExtVector mean_vec(&kernel_means[i][0], dim);
      for (j = 0; j < dim; j++)
      {
        add(rows(*cov_m[i])[j],scaled(mean_vec,-(mean_vec)[j]),
            rows(*cov_m[i])[j]);
      }
      
      beta += gk_norm[i];
    }
  }

  // Subtract global mean from the global covariance
  for (i = 0; i < dim; i++)
  {
    global_mean[i] /= (float)global_count;
  }
  ExtVector glob_mean_vec(&global_mean[0], dim);
  scale(*global_cov, 1.0/global_count);
  for (i = 0; i < dim; i++)
  {
    add(rows(*global_cov)[i],scaled(glob_mean_vec,-(glob_mean_vec)[i]),
        rows(*global_cov)[i]);
  }

  for (iter1 = 0; iter1 < MAX_MLLT_ITER; iter1++)
  {
    // Estimate the diagonal variances
    for (k = 0; k < model.num_kernels(); k++)
    {
      if (gk_norm[k] > 0)
      {
        for (i = 0; i < dim; i++)
          for (j = 0; j < dim; j++)
            temp_m(i,j) = 0;
        mult(A, *cov_m[k], temp_m);
        for (i = 0; i < model.dim(); i++)
        {
          // See update_parameters for comments on the minimum value
          kernel_covar[k][i] = std::max(dot(rows(temp_m)[i], rows(A)[i]), 
					m_min_var);
        }
        for (i = 0; i < dim; i++)
          for (j = 0; j < dim; j++)
            temp_m(i,j) = 0;
        mult(A, *global_cov, temp_m);
        for (i = model.dim(); i < dim; i++)
        {
          // See update_parameters for comments on the minimum value
          kernel_covar[k][i] = std::max(dot(rows(temp_m)[i], rows(A)[i]), 
					m_min_var);
        }
      }
      else
      {
        // Reset variance
        for (i = 0; i < dim; i++)
        {
          kernel_covar[k][i] = 1;
        }
      }
    }

    // Estimate the transform matrix

    // Calculate the auxiliary matrix G
    for (i = 0; i < model.dim(); i++)
    {
       for (j = 0; j < dim; j++)
         for (k = 0; k < dim; k++)
            temp_m(j,k) = 0;
 
       for (k = 0; k < model.num_kernels(); k++)
       {
         if (gk_norm[k] > 0)
           add(scaled(*cov_m[k], gk_norm[k]/kernel_covar[k][i]),
               temp_m);
       }
       // Invert
       lu_factor(temp_m, pivots);
       lu_inverse(temp_m, pivots, *G[i]);
    }
    for (i = model.dim(); i < dim; i++)
    {
       for (j = 0; j < dim; j++)
         for (k = 0; k < dim; k++)
            temp_m(j,k) = 0;

       temp = 0;
       for (k = 0; k < model.num_kernels(); k++)
       {
         temp += gk_norm[k]/kernel_covar[k][i];
       }
       assert( temp > 0);
       add(scaled(*global_cov, temp), temp_m);
       // Invert
       lu_factor(temp_m, pivots);
       lu_inverse(temp_m, pivots, *G[i]);
    }

    // Iterate A
    for (iter2 = 0; iter2 < MAX_MLLT_A_ITER; iter2++)
    {
      // Get the cofactors of A
      transpose(A);
      lu_factor(A, pivots);
      lu_inverse(A, pivots, temp_m);
      temp = 1;
      for (i = 0; i < dim; i++)
        temp *= A(i,i);
      // Note, the sign of det(A) is assumed to be positive.. Is it REALLY?
      Adet = fabs(temp);
      scale(temp_m, Adet); // Scale with determinant
      // Estimate the rows of A
      for (i = 0; i < dim; i++)
      {
        mult(trans(*G[i]), rows(temp_m)[i], rows(A)[i]);
        temp = sqrt(beta/dot(rows(A)[i], rows(temp_m)[i]));
        scale(rows(A)[i], temp);
      }
    }
  }

  typedef Matrix::submatrix_type SubMatrix;
  SubMatrix Ap;
  
  Ap = A.sub_matrix(0, model.dim(), 0, dim);
  // Transform mean vectors
  for (k = 0; k < model.num_kernels(); k++)
  {
    if (gk_norm[k] > 0)
    {
      ExtVector mean(&model_tmp.kernel(k).center[0], model.dim());
      ExtVector source(&kernel_means[k][0], dim);

      mult(Ap, source, mean);
    }
  }

  // Once more, estimate the diagonal covariances
  for (k = 0; k < model.num_kernels(); k++)
  {
    if (gk_norm[k] > 0)
    {
      for (i = 0; i < dim; i++)
        for (j = 0; j < dim; j++)
          temp_m(i,j) = 0;
      mult(A, *cov_m[k], temp_m);
      for (i = 0; i < model.dim(); i++)
      {
        // See update_parameters for comments on the minimum value
        model_tmp.kernel(k).cov.diag(i) = 
	  std::max(dot(rows(temp_m)[i], rows(A)[i]), m_min_var);
      }
    }
  }

  for (int s = 0; s < model.num_states(); s++) {
    HmmState &tmp_state = model_tmp.state(s);

    // Find normalization;
    sum=0.0;
    for (int k = 0; k < (int)tmp_state.weights.size(); k++)
      sum += tmp_state.weights[k].weight;

    // Normalize
    if (sum > 0) {
      temp = 0;
      for (int k = 0; k < (int)tmp_state.weights.size(); k++)
      {
	tmp_state.weights[k].weight /= sum;
        if (tmp_state.weights[k].weight < KERNEL_MINWEIGHT)
        {
          tmp_state.weights[k].weight = KERNEL_MINWEIGHT;
        }
        temp += tmp_state.weights[k].weight;
      }
      // And renormalize...
      for (int k = 0; k < (int)tmp_state.weights.size(); k++)
      {
	tmp_state.weights[k].weight /= temp;
      }
    }
    else
    {
      fprintf(stderr, "Warning: No data for state %d\n", s);
      for (int k = 0; k < (int)tmp_state.weights.size(); k++)
        tmp_state.weights[k].weight = 1.0/tmp_state.weights.size();
    }
  }

  /* transition probabilities */
  for (int h = 0; h < model.num_hmms(); h++) {
    for (int s = 0; s < model.hmm(h).num_states(); s++) {
      sum = 0.0;

      std::vector<int> &hmm_transitions = model_tmp.hmm(h).transitions(s);
      for (int t = 0; t < (int)hmm_transitions.size(); t++) 
	sum += model_tmp.transition(hmm_transitions[t]).prob;
      if (sum == 0.0)
	sum = 1.0;
      for (int t = 0; t < (int)hmm_transitions.size(); t++) {
	model_tmp.transition(hmm_transitions[t]).prob = 
	  model_tmp.transition(hmm_transitions[t]).prob/sum;
	if (model_tmp.transition(hmm_transitions[t]).prob < .001)
	  model_tmp.transition(hmm_transitions[t]).prob = .001;
      }
    }
  }

  // Set transformation
  std::vector<float> tr;
  tr.resize(dim*dim);
  for (i = 0; i < dim; i++)
    for (j = 0; j < dim; j++)
      tr[i*dim + j] = A(i,j);
  m_transform_module->set_transformation_matrix(tr);
  model_tmp.compute_covariance_determinants();
  
  // Free allocated memory
  for (i = 0; i < dim; i++)
  {
    delete G[i];
  }
  delete [] G;
  delete [] kernel_covar;
}


void
HmmTrainer::update_hlda_tmp_parameters(HmmSet &model, HmmSet &model_tmp,
                                       std::vector<float> &gk_norm,
                                       Viterbi &viterbi,
                                       int start_frame, 
                                       int end_frame)
{
  int dim = m_source_dim;
  int frames = end_frame - start_frame;
  double gamma_norm;
  int i;
  
  for (int f = 0; f < frames; f++) {
    FeatureVec feavec = m_fea_gen.generate(start_frame + f);
    FeatureVec untransformed_fea =
      m_transform_module->sources().front()->at(start_frame + f);
    ExtVectorConst untransformed_fea_vec(&untransformed_fea[0], dim);
    HmmState &state = model.state(viterbi.best_state(f));
    HmmState &state_accu= model_tmp.state(viterbi.best_state(f));

    gam.reserve(state.weights.size());
    gamma_norm = 0;

    // Calculate the probabilities
    for (int state_k = 0; state_k < (int)state.weights.size(); state_k++) {
      int hmm_k = state.weights[state_k].kernel;
      gam[state_k] = model.compute_kernel_likelihood(hmm_k, feavec)*
        state.weights[state_k].weight;
      gamma_norm += gam[state_k];
    }

    m_log_likelihood +=  safe_log(gamma_norm);

    // Global mean and covariance
    for (i = 0; i < dim; i++)
    {
      global_mean[i] += untransformed_fea[i];
      add(rows(*global_cov)[i], 
	  scaled(untransformed_fea_vec, untransformed_fea_vec[i]), 
	  rows(*global_cov)[i]);
    }
    global_count++;

    if (gamma_norm > 0)
    {
      // Normalize the probabilities and add to the accumulators
      for (int state_k = 0; state_k < (int)state.weights.size(); state_k++) {
        int hmm_k = state.weights[state_k].kernel;
      
        gam[state_k] /= gamma_norm;
        state_accu.weights[state_k].weight += gam[state_k];

        // Because of the possibly shared kernels, we have to accumulate
        // gamma values also per kernel to normalize the mean and covariance
        gk_norm[hmm_k] += gam[state_k];

        // Add mean
        for (i = 0; i < dim; i++)
          kernel_means[hmm_k][i] += untransformed_fea[i]*gam[state_k];

        // Add covariance
        for (i = 0; i < (int)cov_m[hmm_k]->nrows(); i++)
        {
          add(rows(*cov_m[hmm_k])[i], 
	      scaled(untransformed_fea_vec, 
		     gam[state_k]*untransformed_fea_vec[i]), 
	      rows(*cov_m[hmm_k])[i]);
        }
      }
    }
    
    /* Update transition counts.  Note, that the last state does not
       have transition (marked with -1) */
    int transition = viterbi.best_transition(f);
    if (transition >= 0)
      model_tmp.transition(transition).prob++;
  }
}


void
HmmTrainer::update_duration_statistics(HmmSet &model, Viterbi &viterbi,
                                       int frames)
{
  static int prev_state = -1;
  static int state_count = 0;

  for (int f = 0; f < frames; f++) {

    if (prev_state != viterbi.best_state(f))
    {
      if (prev_state != -1)
      {
        if (state_count >= MAX_DURATION_COUNT)
          state_count = MAX_DURATION_COUNT-1;
        dur_table[prev_state][state_count]++;
      }
      prev_state = viterbi.best_state(f);
      state_count = 0;
    }
    else
    {
      state_count++;
    }
  }
}


void
HmmTrainer::update_boundary_duration_statistics(HmmSet &model,Viterbi *viterbi,
                                                int frames)
{
  int prev_state = -1;
  int state_count = 0;

  for (int f = 0; f < frames; f++) {

    if (prev_state != viterbi->best_state(f))
    {
      if (prev_state != -1)
      {
        if (state_count >= MAX_DURATION_COUNT)
          state_count = MAX_DURATION_COUNT-1;
        dur_table[prev_state][state_count]++;
      }
      prev_state = viterbi->best_state(f);
      state_count = 0;
    }
    else
    {
      state_count++;
    }
  }
  if (prev_state != -1)
  {
    if (state_count >= MAX_DURATION_COUNT)
      state_count = MAX_DURATION_COUNT-1;
    dur_table[prev_state][state_count]++;
  }
}



void
HmmTrainer::write_duration_statistics(HmmSet &model)
{
  char filename[30];
  for (int i = 0; i < model.num_states(); i++)
  {
    sprintf(filename, "duration_%i", i);
    std::ofstream out(filename);

    for (int j = 0; j < MAX_DURATION_COUNT; j++)
    {
      out << dur_table[i][j] << "\n";
    }
  }
}


void
HmmTrainer::save_tying(const std::string &filename)
{
  FILE *fp;
  int state_num;

  // Save silence models
  if ((fp = fopen(filename.c_str(), "w")) == NULL)
  {
    fprintf(stderr, "Could not open file %s for writing.\n", filename.c_str());
    exit(1);
  }
  fprintf(fp, "_ 1 0\n__ 3 1 2 3\n");
  fclose(fp);
  
  state_num = triphone_set.save_to_basebind(filename, 4);
}
