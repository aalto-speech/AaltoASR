#ifndef FEATUREMODULES_HH
#define FEATUREMODULES_HH

#ifdef KISS_FFT
#include "kiss_fftr.h"
#else // Use FFTW
#include <fftw3.h>
#endif
#include <vector>

#include <errno.h>
#include <string.h>
#include "FeatureBuffer.hh"
#include "AudioReader.hh"
#include "ModuleConfig.hh"
#include "LinearAlgebra.hh"


namespace aku {

class FeatureGenerator;

/** A base class of a module that computes features from other
 * features, audio file or some other file.  For the caller, the
 * FeatureModule provides interface for accesing the features in a
 * buffered manner.  With \ref FeatureGenerator class these modules
 * can be combined to perform complex feature extraction.
 *
 * \section semantics Some semantics about the module structure
 *
 * A FeatureModule is initialized by first linking it to its sources
 * with the add_source() method and then calling set_config() with the
 * desired settings.  The set_config() calls module's virtual private
 * set_module_config(), which must check that source dimensions match
 * with the given settings.  After set_module_config(), set_config()
 * calls set_buffer() for its each source module.
 * 
 * When a feature is requested from the module via at() and the
 * requested frame is not in the buffer, the module updates its buffer
 * so that the new frame will be the last one in the buffer. The
 * buffer is recomputed from left to right, possibly reusing the
 * values already in the buffer.
 *
 * The computation of one feature is done in generate(), which will be
 * called by at() when necessary.
 *
 * In addition to module configuration, modules may also have parameters
 * which are used to change the feature computations on-line. These
 * are used for e.g. speaker adaptation. The methods used to handle
 * the parameters, set_parameters() and get_parameters(), use the same
 * \ref ModuleConfig class for passing the parameters as does the
 * set_config() and get_config() methods. The on-line parameters may not
 * change the feature dimension or buffering behaviour.
 *
 */ 
class FeatureModule {
public:
  FeatureModule();
  virtual ~FeatureModule();

  /** Set the name of the module.  Should be used only by
   * FeatureGenerator. */
  void set_name(const std::string &name) { m_name = name; }

  /** Return the name of the module. */
  std::string name() const { return m_name; }

  /** Return the type of the module. */
  std::string type_str() const { return m_type_str; }

  /** Request buffering in addition to the central frame.  Buffering
   * is requested recursively from the source modules if necessary.
   * \note Should be called only through set_config()!
   *
   * \param left = number of frames to left of the central frame
   * \param right = number of frames to right of the central frame
   */
  void set_buffer(int left, int right);

  /** Add a source module.  The default implementation allows only one
   * source, but many derived classes allow several sources.  \note
   * All sources must be added before calling set_config().
   */
  virtual void add_source(FeatureModule *source);

  /** Configure the module using the possible settings in \c config.
   * \note All sources must be added before calling set_config(), so
   * that the module can check the input dimensions in the
   * configuration.
   */
  void set_config(const ModuleConfig &config);

  /** Write all essential configuration in \c config class.
   * Configuring a newly created class with \c config should result in
   * an identical configuration. */
  void get_config(ModuleConfig &config);

  /** Reset the internal state of the module.  FeatureGenerator calls
   * this method for all modules, when it opens a new audio file.
   * \note Derived classes should implement the virtual method
   * reset_module() if resetting is desired. */
  void reset();

  /** Update buffer offsets required for the initial buffer filling.
   *  Used by FeatureGenerator to ensure large enough buffering so that when
   *  feature buffers are initially filled and there are branching feature
   *  module streams the features are always generated from left to right
   *  in all modules.
   *
   * \param target = target module from which the offsets are fetched.
   */
  void update_init_offsets(const FeatureModule &target);

  /** Set the module's parameters. This is used for e.g. speaker adaptation
      to change the module's behaviour on-line. */
  virtual void set_parameters(const ModuleConfig &params) { }

  /** Get the current module parameters. */
  virtual void get_parameters(ModuleConfig &params) { }

  /** Access features computed by the module. */
  const FeatureVec at(int frame);

  /** The dimension of the feature. \note Valid only after the module
   * has been configured with set_config(). */
  int dim(void) { return m_dim; }

  /** Access the source modules. */
  const std::vector<FeatureModule*> &sources() const { return m_sources; }

  /** Print module info in DOT node format. */
  void print_dot_node(FILE *file);
  
private:
  virtual void set_module_config(const ModuleConfig &config) = 0;
  virtual void get_module_config(ModuleConfig &config) = 0;

  /** Virtual method for resetting the internal states of the derived
   * modules. */
  virtual void reset_module() { }
  
  virtual void generate(int frame) = 0;
  
protected:
  std::string m_name; //!< The name of the module given by FeatureGenerator

  /** The type of the module.  Should be equal to type_str(). */
  std::string m_type_str; 
  int m_own_offset_left;  //!< Buffer offsets for own computations
  int m_own_offset_right;
  int m_req_offset_left;  //!< Required buffer offsets by calling modules
  int m_req_offset_right;
  int m_init_offset_left; //!< Buffer offsets used in initial buffer filling
  int m_init_offset_right;
  int m_buffer_size;
  int m_buffer_last_pos;  //!< The last frame number in the buffer
  int m_buffer_first_pos; //!< The first frame number in the buffer
  FeatureBuffer m_buffer;

  int m_dim;
  
  std::vector<FeatureModule*> m_sources;
};


// Interface for modules that generate features from a file
class BaseFeaModule : public FeatureModule {
public:
  virtual void set_fname(const char *fname) = 0;
  virtual void set_file(FILE *fp, bool stream=false) = 0;
  virtual void discard_file(void) = 0;

  // Note: eof() may return false for a frame that is past the real EOF
  // if at() for that frame has not yet been called.
  virtual bool eof(int frame) = 0;
  
  virtual int sample_rate(void) = 0;
  virtual float frame_rate(void) = 0;

  virtual int last_frame(void) = 0;
  
  virtual void add_source(FeatureModule *source) 
  { throw std::string("base module FFT can not have sources"); }
};


//////////////////////////////////////////////////////////////////
// Feature module implementations
//////////////////////////////////////////////////////////////////


class AudioFileModule : public BaseFeaModule {
public:
  AudioFileModule(FeatureGenerator *fea_gen);
  virtual ~AudioFileModule();
  static const char *type_str() { return "audiofile"; }

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
  FeatureGenerator *m_fea_gen;

  AudioReader m_reader;
  int m_sample_rate;
  float m_frame_rate;
  float m_window_advance;
  int m_window_width;

  float m_emph_coef; //!< Pre-emphasis filter coefficient
  
  int m_eof_frame;

  int m_endian; // RAW-file endianess: 0=default, 1=little, 2=big
  bool m_raw; // File mode enforced to RAW

  /** Should we copy border frames when negative or after-eof frames
   * are requested?  Otherwise, we assume that AudioReader gives zero
   * samples outside the file. */
  int m_copy_borders;
  std::vector<double> m_first_feature; //!< Feature returned for negative frames
  std::vector<double> m_last_feature; //!< Feature returned after EOF
  int m_last_feature_frame; //!< The frame of the feature returned after EOF
};


class FFTModule : public FeatureModule {
public:
  FFTModule();
  virtual ~FFTModule();
  static const char *type_str() { return "fft"; }
  
private:
  virtual void get_module_config(ModuleConfig &config);
  virtual void set_module_config(const ModuleConfig &config);
  virtual void generate(int frame);

private:

  int m_magnitude; //!< If nonzero, compute magnitude spectrum instead of power
  int m_log; //!< If nonzero, take logarithms of the output

  std::vector<float> m_hamming_window;
#ifdef KISS_FFT
  kiss_fftr_cfg m_coeffs;
  kiss_fft_scalar *m_kiss_fft_datain;
  kiss_fft_cpx *m_kiss_fft_dataout;
#else
  fftw_plan m_coeffs;
  std::vector<double> m_fftw_datain;
  std::vector<double> m_fftw_dataout;
#endif
};


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
