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


HmmTrainer::HmmTrainer(FeatureGenerator &fea_gen)
  : m_fea_gen(fea_gen),
    m_info(0),
    m_transform_module(NULL),
    m_source_dim(0),
    m_mllt(false),
    m_hlda(false),
    m_set_speakers(false),
    m_min_var(0.1),
    m_cov_update(false),
    m_durstat(false),
    m_log_likelihood(0),
    m_num_dur_models(0),
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

    if (!m_hlda)
    {
      int dim = m_fea_gen.dim();
      // For normal covariance, use tight triangular representation
      tri_cov_m = new std::vector<float>*[model.num_kernels()];
      for (i = 0; i < model.num_kernels(); i++)
      {
        tri_cov_m[i] = new std::vector<float>;
        tri_cov_m[i]->resize(dim*(dim+1)/2);
      }
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


void HmmTrainer::train(PhnReader &phn_reader,
                       int start_frame, int end_frame,
                       HmmSet &model,
                       std::string speaker, std::string utterance)
{
  Hmm hmm;
  PhnReader::Phn cur_phn, next_phn;
  int state_index;
  int phn_start_frame, phn_end_frame;
  bool cur_not_eof, next_not_eof;

  if (m_set_speakers && speaker.size() > 0)
  {
    m_speaker_config.set_speaker(speaker);
    if (utterance.size() > 0)
      m_speaker_config.set_utterance(utterance);
  }

  cur_not_eof = phn_reader.next(cur_phn);

  m_log_likelihood = 0;
  
  while (cur_not_eof)
  {
    if (cur_phn.state < 0)
      throw std::string("Training requires state segmented phn files!");

    next_not_eof = phn_reader.next(next_phn);

    phn_start_frame=(int)((double)cur_phn.start/16000.0*m_fea_gen.frame_rate()
                          +0.5);
    phn_end_frame=(int)((double)cur_phn.end/16000.0*m_fea_gen.frame_rate()
                        +0.5);
    if (phn_start_frame < start_frame)
    {
      assert( phn_end_frame > start_frame );
      phn_start_frame = start_frame;
    }
    if (end_frame != 0 && phn_end_frame > end_frame)
    {
      assert( phn_start_frame < end_frame );
      phn_end_frame = end_frame;
    }

    if (m_set_speakers && cur_phn.speaker.size() > 0 &&
        cur_phn.speaker != m_speaker_config.get_cur_speaker())
    {
      m_speaker_config.set_speaker(speaker);
      if (utterance.size() > 0)
        m_speaker_config.set_utterance(utterance);
    }

    hmm = model.hmm(model.hmm_index(cur_phn.label[0]));
    state_index = hmm.state(cur_phn.state);

    // Find transitions
    std::vector<int> &trans = hmm.transitions(cur_phn.state);
    int self_transition = -1;
    int out_transition = -1;
    for (int i = 0; i < (int)trans.size(); i++)
    {
      if (model.transition(trans[i]).target == cur_phn.state)
      {
        self_transition = trans[i];
        break;
      }
    }
    assert( self_transition >= 0 );
    if (next_not_eof)
    {
      int target = -2;
      if (next_phn.label[0] == cur_phn.label[0] &&
          next_phn.state > cur_phn.state) // NOTE: Left to right HMMs
        target = next_phn.state;
      for (int i = 0; i < (int)trans.size(); i++)
      {
        if (model.transition(trans[i]).target == target)
        {
          out_transition = trans[i];
          break;
        }
      }
    }

    // Update parameters 
    if (m_durstat)
    {
      // Just collect the duration statistics
      update_duration_statistics(state_index,
                                 phn_end_frame - phn_start_frame - 1);
    }
    else if (m_hlda)
    {
      if (!update_hlda_tmp_parameters(model, model_tmp, state_index, gk_norm,
                                      self_transition, out_transition,
                                      phn_start_frame, phn_end_frame))
        break;
    }
    else
    {
      if (!update_tmp_parameters(model, model_tmp, state_index, gk_norm,
                                 self_transition, out_transition,
                                 phn_start_frame, phn_end_frame))
        break;
    }

    cur_phn = next_phn;
    cur_not_eof = next_not_eof;
  }
}


void HmmTrainer::finish_train(HmmSet &model)
{
  // Finish updating parameters
  if (m_durstat)
  {
    write_duration_statistics(model);
  }
  else
  {
    if (m_cov_update && !m_hlda)
    {
      // Expand triangular accumulators to symmetric covariances
      int k, i, r, c;
      for (k = 0; k < model.num_kernels(); k++)
      {
        for (r = i = 0; r < model.dim(); r++)
        {
          (*cov_m[k])[r][r] = (*tri_cov_m[k])[i++];
          for (c = r+1; c < model.dim(); c++)
          {
            (*cov_m[k])[r][c] = (*tri_cov_m[k])[i];
            (*cov_m[k])[c][r] = (*tri_cov_m[k])[i++];
          }
        }
      }
    }
    if (m_mllt)
      update_mllt_parameters(model, model_tmp, gk_norm,
                             *m_transform_matrix);
    else if (m_hlda)
      update_hlda_parameters(model, model_tmp, gk_norm,
                             *m_transform_matrix);
    else
      update_parameters(model, model_tmp, gk_norm);

    // Replace the old models with new ones
    model = model_tmp;
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

bool
HmmTrainer::update_tmp_parameters(HmmSet &model, HmmSet &model_tmp,
                                  int state_index, std::vector<float> &gk_norm,
                                  int self_transition, int out_transition,
                                  int start_frame, int end_frame)
{
  int dim = m_fea_gen.dim();
  int frames = end_frame - start_frame;
  HmmState &state = model.state(state_index);
  HmmState &state_accu = model_tmp.state(state_index);

  
  for (int f = 0; f < frames; f++) {
    FeatureVec feature = m_fea_gen.generate(start_frame + f);
    if (m_fea_gen.eof())
      return false;

    update_state_kernels(model, model_tmp, state, state_accu,
                         feature, dim, gk_norm);
  }

  /* Update transition counts.  Note, that the last state does not
     have transition (marked with -1) */
  if (self_transition != -1)
  {
    int num_frames = std::max(end_frame - start_frame - 1, 0);
    model_tmp.transition(self_transition).prob += num_frames;
    m_log_likelihood += num_frames*
      util::safe_log(model.transition(self_transition).prob);
  }
  if (out_transition != -1)
  {
    model_tmp.transition(out_transition).prob++;
    m_log_likelihood += util::safe_log(model.transition(out_transition).prob);
  }
  
  return true;
}

void
HmmTrainer::update_state_kernels(HmmSet &model, HmmSet &model_tmp,
                                 HmmState &state, HmmState &state_accu,
                                 const FeatureVec &feature, int dim,
                                 std::vector<float> &gk_norm)
                                 
{
  double gamma_norm;
  int i;
  ExtVectorConst fea_vec(&feature[0], dim);
  Vector temp_fea_vec(dim);
  int r,c;
  
  gam.reserve(state.weights.size());
  gamma_norm = 0;
    
  // Calculate the probabilities
  for (int state_k = 0; state_k < (int)state.weights.size(); state_k++) {
    int hmm_k = state.weights[state_k].kernel;
    gam[state_k] = model.compute_kernel_likelihood(hmm_k, feature)*
      state.weights[state_k].weight;
    gamma_norm += gam[state_k];
  }

  m_log_likelihood += util::safe_log(gamma_norm) +
    util::safe_log(mllt_determinant);

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

  // Precompute the outer product of the feature vector to a tight
  // triangular matrix
  std::vector<float> outer_prod;
  outer_prod.resize((dim+1)*dim/2);
  for (r = i = 0; r < dim; r++)
  {
    outer_prod[i++] = feature[r]*feature[r];
    for (c = r+1; c < dim; c++)
      outer_prod[i++] = feature[c]*feature[r];
  }
  
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
    {
      accuker.center[i] += feature[i]*gam[state_k];
    }

    if (m_cov_update)
    {
      // Add covariance
      r = (dim+1)*dim/2;
      for (i = 0; i < r; i++)
        (*tri_cov_m[hmm_k])[i] += gam[state_k]*outer_prod[i];
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


bool
HmmTrainer::update_hlda_tmp_parameters(HmmSet &model, HmmSet &model_tmp,
                                       int state_index,
                                       std::vector<float> &gk_norm,
                                       int self_transition, int out_transition,
                                       int start_frame, int end_frame)
{
  int dim = m_source_dim;
  int frames = end_frame - start_frame;
  double gamma_norm;
  int i;
  HmmState &state = model.state(state_index);
  HmmState &state_accu= model_tmp.state(state_index);

  
  for (int f = 0; f < frames; f++) {
    FeatureVec feavec = m_fea_gen.generate(start_frame + f);
    if (m_fea_gen.eof())
      return false;
    
    FeatureVec untransformed_fea =
      m_transform_module->sources().front()->at(start_frame + f);
    ExtVectorConst untransformed_fea_vec(&untransformed_fea[0], dim);

    gam.reserve(state.weights.size());
    gamma_norm = 0;

    // Calculate the probabilities
    for (int state_k = 0; state_k < (int)state.weights.size(); state_k++) {
      int hmm_k = state.weights[state_k].kernel;
      gam[state_k] = model.compute_kernel_likelihood(hmm_k, feavec)*
        state.weights[state_k].weight;
      gamma_norm += gam[state_k];
    }

    m_log_likelihood += util::safe_log(gamma_norm);

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
  }
  /* Update transition counts.  Note, that the last state does not
     have transition (marked with -1) */
  if (self_transition != -1)
  {
    int num_frames = std::max(end_frame - start_frame - 1, 0);
    model_tmp.transition(self_transition).prob += num_frames;
    m_log_likelihood += num_frames*
      util::safe_log(model.transition(self_transition).prob);
  }
  if (out_transition != -1)
  {
    model_tmp.transition(out_transition).prob++;
    m_log_likelihood += util::safe_log(model.transition(out_transition).prob);
  }
  return true;
}


void
HmmTrainer::update_duration_statistics(int state_index, int num_frames)
{
  if (num_frames >= MAX_DURATION_COUNT)
    num_frames = MAX_DURATION_COUNT-1;
  dur_table[state_index][num_frames]++;  
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

