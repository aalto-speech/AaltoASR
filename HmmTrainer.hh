#ifndef HMMTRAINER_HH
#define HMMTRAINER_HH

#include "FeatureModules.hh"
#include "HmmSet.hh"
#include "PhnReader.hh"
#include "SpeakerConfig.hh"

// Matrix template library stuff
#include "mtl/mtl_config.h"
#include "mtl/mtl.h"
#include "mtl/matrix.h"
#include "mtl/lu.h"

#define TINY 1e-16

typedef mtl::matrix<float, mtl::rectangle<>, mtl::dense<>, 
		    mtl::row_major>::type Matrix;
typedef mtl::matrix<double, mtl::rectangle<>, mtl::dense<>, 
		    mtl::row_major>::type MatrixD;
typedef mtl::dense1D<float> Vector;
typedef mtl::dense1D<double> VectorD;
typedef mtl::external_vec<float> ExtVector;
typedef mtl::external_vec<const float> ExtVectorConst;


class HmmTrainer {
public:
  HmmTrainer(FeatureGenerator &fea_gen);
  ~HmmTrainer();

  void init(HmmSet &model, std::string adafile = "");
  void train(PhnReader &phn_reader,
             int start_frame, int end_frame,
             HmmSet &model,
             std::string speaker = "", std::string utterance = "");
  void finish_train(HmmSet &model);

  int num_unused_features(void) { return m_em_norm_warning_count; }
  double get_log_likelihood(void) { return m_log_likelihood; }

private:

  void update_parameters(HmmSet &model, HmmSet &model_tmp, 
                         const std::vector<float> &gk_norm);
  void update_transition_probabilities(HmmSet &model, HmmSet &model_tmp);
 
  // update_tmp_parameters returns the number of frames left untreated
  bool update_tmp_parameters(HmmSet &model, HmmSet &model_tmp,
                             int state_index, std::vector<float> &gk_norm,
                             int self_transition, int out_transition,
                             int start_frame, int end_frame);
  void update_state_kernels(HmmSet &model, HmmSet &model_tmp,
                            HmmState &state, HmmState &state_accu,
                            const FeatureVec &feature, int dim,
                            std::vector<float> &gk_norm);
                                 
  void update_mllt_parameters(HmmSet &model, HmmSet &model_tmp,
                              std::vector<float> &gk_norm, Matrix &A);
  void update_hlda_parameters(HmmSet &model, HmmSet &model_tmp,
                              std::vector<float> &gk_norm, Matrix &A);
  bool update_hlda_tmp_parameters(HmmSet &model, HmmSet &model_tmp,
                                  int state_index,
                                  std::vector<float> &gk_norm,
                                  int self_transition, int out_transition,
                                  int start_frame, int end_frame);
  void update_duration_statistics(int state_index, int num_frames);
  void write_duration_statistics(HmmSet &model);

public:
  void set_info(int info) { m_info = info; }
  void set_transform_module(LinTransformModule *mod) { m_transform_module = mod; }
  void set_mllt(bool mllt_flag) { m_mllt = mllt_flag; }
  void set_hlda(bool hlda_flag) { m_hlda = hlda_flag; }
  void set_min_var(float min_var) { m_min_var = min_var; }
  void set_cov_update(bool cov_update) { m_cov_update = cov_update; }
  void set_duration_statistics(bool durstat) { m_durstat = durstat; }
  
private:
  FeatureGenerator &m_fea_gen;

  int m_info;
  LinTransformModule *m_transform_module;
  int m_source_dim;
  bool m_mllt;
  bool m_hlda;
  bool m_set_speakers; // Do we need to change speakers?
  float m_min_var;
  bool m_cov_update;
  bool m_durstat;
  double m_log_likelihood;
  
  int **dur_table;
  int m_num_dur_models;

  // MLLT stuff
  std::vector<float> gam;
  Matrix **cov_m;
  Matrix *m_transform_matrix;
  double mllt_determinant;

  std::vector<float> **tri_cov_m;
  
  // HLDA stuff
  std::vector<float> *kernel_means;
  std::vector<float> global_mean;
  Matrix *global_cov;
  int global_count;
  int m_em_norm_warning_count;

  SpeakerConfig m_speaker_config; // Reads and handles speaker configurations

  HmmSet model_tmp;
  std::vector<float> gk_norm;
};


#endif // HMMTRAINER_HH
