#ifndef FEATUREMODULES_HH
#define FEATUREMODULES_HH

#include <vector>

#include "ModuleConfig.hh"
#include "FeatureModule.hh"
#include "BaseFeaModule.hh"
#include "AudioFileModule.hh"
#include "FFTModule.hh"


namespace aku {

class FeatureGenerator;

//////////////////////////////////////////////////////////////////
// Feature module implementations
//////////////////////////////////////////////////////////////////


class PreModule : public BaseFeaModule {
public:
  PreModule();
  static const char *type_str() { return "pre"; }

  virtual void set_fname(const char *fname);
  virtual void set_file(FILE *fp, bool stream=false);
  virtual void discard_file(void);
  virtual bool eof(int frame);
  virtual int sample_rate(void) { return m_sample_rate; }
  virtual float frame_rate(void) { return m_frame_rate; }
  virtual int last_frame(void);
  
private:
  virtual void get_module_config(ModuleConfig &config);
  virtual void set_module_config(const ModuleConfig &config);
  virtual void reset_module();
  virtual void generate(int frame);

private:
  int m_sample_rate;
  float m_frame_rate;
  int m_eof_frame;
  int m_legacy_file; //!< If nonzero, the dimension in the file is a byte
  int m_file_offset;
  int m_cur_pre_frame;
  bool m_close_file;

  FILE *m_fp;

  std::vector<double> m_first_feature; //!< Feature returned for negative frames
  std::vector<double> m_last_feature; //!< Feature returned after EOF
  int m_last_feature_frame; //!< The frame of the feature returned after EOF

  std::vector<float> m_temp_fea_buf; //!< Need a float buffer for reading
};


class MelModule : public FeatureModule {
public:
  MelModule(FeatureGenerator *fea_gen);
  static const char *type_str() { return "mel"; }
private:
  virtual void get_module_config(ModuleConfig &config);
  virtual void set_module_config(const ModuleConfig &config);
  virtual void generate(int frame);

  void create_mel_bins(void);

private:
  FeatureGenerator *m_fea_gen;

  int m_bins;
  int m_root; //!< If nonzero, take 10th root of the output instead of logarithm
  std::vector<float> m_bin_edges;
};


class PowerModule : public FeatureModule {
public:
  PowerModule();
  static const char *type_str() { return "power"; }
private:
  virtual void get_module_config(ModuleConfig &config);
  virtual void set_module_config(const ModuleConfig &config);
  virtual void generate(int frame);
};


class MelPowerModule : public FeatureModule {
public:
  MelPowerModule();
  static const char *type_str() { return "mel_power"; }
private:
  virtual void get_module_config(ModuleConfig &config);
  virtual void set_module_config(const ModuleConfig &config);
  virtual void generate(int frame);
};


class DCTModule : public FeatureModule {
public:
  DCTModule();
  static const char *type_str() { return "dct"; }
private:
  virtual void get_module_config(ModuleConfig &config);
  virtual void set_module_config(const ModuleConfig &config);
  virtual void generate(int frame);
private:
  int m_zeroth_comp; //!< If nonzero, output includes zeroth component
};


class DeltaModule : public FeatureModule {
public:
  DeltaModule();
  static const char *type_str() { return "delta"; }
private:
  virtual void get_module_config(ModuleConfig &config);
  virtual void set_module_config(const ModuleConfig &config);
  virtual void generate(int frame);
private:
  int m_delta_width;
  float m_delta_norm;
};


class NormalizationModule : public FeatureModule {
public:
  NormalizationModule();
  static const char *type_str() { return "normalization"; }
  void set_normalization(const std::vector<float> &mean,
                         const std::vector<float> &scale);
private:
  virtual void get_module_config(ModuleConfig &config);
  virtual void set_module_config(const ModuleConfig &config);
  virtual void set_parameters(const ModuleConfig &config);
  virtual void get_parameters(ModuleConfig &config);
  virtual void generate(int frame);
private:
  std::vector<float> m_mean;
  std::vector<float> m_scale;
};


class LinTransformModule : public FeatureModule {
public:
  LinTransformModule();
  static const char *type_str() { return "lin_transform"; }
  const std::vector<float> *get_transformation_matrix(void) { return &m_transform; }
  const std::vector<float> *get_transformation_bias(void) { return &m_bias; }
  void set_transformation_matrix(std::vector<float> &t);
  void set_transformation_bias(std::vector<float> &b);
  virtual void set_parameters(const ModuleConfig &config);
  virtual void get_parameters(ModuleConfig &config);
private:
  virtual void get_module_config(ModuleConfig &config);
  virtual void set_module_config(const ModuleConfig &config);
  virtual void generate(int frame);
  void check_transform_parameters(void);
private:
  std::vector<float> m_transform;
  std::vector<float> m_bias;
  std::vector<float> m_original_transform;
  std::vector<float> m_original_bias;
  bool m_matrix_defined, m_bias_defined;
  int m_src_dim;
public:
  virtual bool is_defined() { return m_matrix_defined && m_bias_defined; }
};


class MergerModule : public FeatureModule {
public:
  MergerModule();
  static const char *type_str() { return "merge"; }
  virtual void add_source(FeatureModule *source);
private:
  virtual void get_module_config(ModuleConfig &config);
  virtual void set_module_config(const ModuleConfig &config);
  virtual void generate(int frame);
};


class MeanSubtractorModule : public FeatureModule {
public:
  MeanSubtractorModule();
  static const char *type_str() { return "mean_subtractor"; }
private:
  virtual void get_module_config(ModuleConfig &config);
  virtual void set_module_config(const ModuleConfig &config);
  virtual void reset_module();
  virtual void generate(int frame);
private:
  std::vector<double> m_cur_mean;
  int m_cur_frame;
  int m_width;
};


class ConcatModule : public FeatureModule {
public:
  ConcatModule();
  static const char *type_str() { return "concat"; }
private:
  virtual void get_module_config(ModuleConfig &config);
  virtual void set_module_config(const ModuleConfig &config);
  virtual void generate(int frame);
private:
  int left, right;
};


class VtlnModule : public FeatureModule {
public:
  VtlnModule();
  static const char *type_str() { return "vtln"; }

  void set_warp_factor(float factor);
  void set_slapt_warp(std::vector<float> &params);
  float get_warp_factor(void) { return m_warp_factor; }
  virtual void set_parameters(const ModuleConfig &config);
  virtual void get_parameters(ModuleConfig &config);
  
private:
  virtual void get_module_config(ModuleConfig &config);
  virtual void set_module_config(const ModuleConfig &config);
  virtual void generate(int frame);

  void create_pwlin_bins(void);
  void create_blin_bins(void);
  void create_slapt_bins(void);
  void create_sinc_coef_table(void);
  void create_all_pass_blin_transform(void);
  void create_all_pass_slapt_transform(void);
  void set_all_pass_transform(Matrix &trmat);

private:
  int m_use_pwlin;
  float m_pwlin_turn_point;
  int m_use_slapt;
  int m_sinc_interpolation_rad;
  int m_all_pass;
  bool m_lanczos_window;
  std::vector<float> m_vtln_bins;
  std::vector< std::vector<float> > m_sinc_coef;
  std::vector<int> m_sinc_coef_start;
  float m_warp_factor;
  std::vector<float> m_slapt_params;
};


class SRNormModule : public FeatureModule {
public:
  SRNormModule();
  static const char *type_str() { return "sr_norm"; }

  void set_speech_rate(float factor);
  float get_speech_rate(void) { return m_speech_rate; }
  virtual void set_parameters(const ModuleConfig &config);
  virtual void get_parameters(ModuleConfig &config);
  
private:
  virtual void get_module_config(ModuleConfig &config);
  virtual void set_module_config(const ModuleConfig &config);
  virtual void generate(int frame);

private:
  int m_in_frames;
  int m_out_frames;
  int m_frame_dim;
  int m_lanczos_order;
  float m_speech_rate;
  std::vector< std::vector<float> > m_coef;
  std::vector<int> m_interpolation_start;
};

class QuantEqModule : public FeatureModule {
public:
  QuantEqModule();
  static const char *type_str() { return "quanteq"; }

  void set_alpha(std::vector<float> &alpha);
  void set_gamma(std::vector<float> &gamma);
  void set_quant_max(std::vector<float> &quant_max);
  std::vector<float> get_quant_train(void) { return m_quant_train; }
  virtual void set_parameters(const ModuleConfig &config);
  virtual void get_parameters(ModuleConfig &config);
  
private:
  virtual void get_module_config(ModuleConfig &config);
  virtual void set_module_config(const ModuleConfig &config);
  virtual void generate(int frame);

private:
  std::vector<float> m_quant_train;
  std::vector<float> m_alpha;
  std::vector<float> m_gamma;
  std::vector<float> m_quant_max;
};

}

#endif /* FEATUREMODULES_HH */
