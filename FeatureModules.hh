#ifndef FEATUREMODULES_HH
#define FEATUREMODULES_HH

#include <fftw3.h>
#include <vector>
#include "ModuleConfig.hh"
#include "FeatureBuffer.hh"
#include "AudioReader.hh"
#include "ModuleConfig.hh"


class FeatureGenerator;

/*
  Some semantics about the module structure:

  The module is initialized by first linking it to its sources (by link())
  and then calling set_config() with the loaded settings. set_config()
  calls module's private set_module_config(), which must check that source
  dimensions match with the given settings. After set_module_config()
  set_config() calls source modules' set_buffer().
  
  When a feature is requested from the module via at() and the feature
  does not exist for the requsted frame, the module updates its buffer
  so that the new frame will be the last one in the buffer. The buffer
  is recomputed from left to right, possibly reusing the values already
  in the buffer.

  The computation of one feature is done in generate(), which will be
  called by at() when necessary.
*/
class FeatureModule {
public:
  FeatureModule();
  virtual ~FeatureModule();

  void set_name(const std::string &name) { m_name = name; }
  std::string name() const { return m_name; }

  // Do not call set_buffer() before configuration has been finished!
  void set_buffer(int left, int right);

  // Called before set_config(). The default implementation allows
  // only one source.
  virtual void link(FeatureModule *source);

  // Linking has been completed before set_config(), so the module
  // can check the dimensions in the configuration.
  void set_config(const ModuleConfig &config);
  void get_config(ModuleConfig &config);
  
  const FeatureVec at(int frame);
  int dim(void) { return m_dim; } // Valid only after configuration

  const std::vector<FeatureModule*> sources() const { return m_sources; }
  
private:
  virtual void set_module_config(const ModuleConfig &config) = 0;
  virtual void get_module_config(ModuleConfig &config) = 0;
  virtual void generate(int frame) = 0;
  
protected:
  std::string m_name;
  std::string m_type_str;
  int m_own_offset_left;  // Buffer offsets for own computations
  int m_own_offset_right;
  int m_req_offset_left;  // Required buffer offsets by calling modules
  int m_req_offset_right;
  int m_buffer_size;
  int m_buffer_last_pos;  // The last frame number in the buffer
  FeatureBuffer m_buffer;

  int m_dim;
  
  std::vector<FeatureModule*> m_sources;
};


// Interface for modules that generate features from a file
class BaseFeaModule : public FeatureModule {
public:
  virtual void set_file(FILE *fp) = 0;
  virtual void discard_file(void) = 0;

  // Note: eof() may return false for a frame that is past the real EOF
  // if at() for that frame has not yet been called.
  virtual bool eof(int frame) = 0;
  
  virtual int sample_rate(void) = 0;
  virtual int frame_rate(void) = 0;
  
  virtual void link(FeatureModule *source) { assert( 0 ); }
};


//////////////////////////////////////////////////////////////////
// Feature module implementations
//////////////////////////////////////////////////////////////////


class FFTModule : public BaseFeaModule {
public:
  FFTModule(FeatureGenerator *fea_gen);
  virtual ~FFTModule();
  static const char *type_str() { return "fft"; }
  
  virtual void set_file(FILE *fp);
  virtual void discard_file(void);
  virtual bool eof(int frame);
  virtual int sample_rate(void) { return m_sample_rate; }
  virtual int frame_rate(void) { return m_frame_rate; }
  
private:
  virtual void get_module_config(ModuleConfig &config);
  virtual void set_module_config(const ModuleConfig &config);
  virtual void generate(int frame);

private:
  FeatureGenerator *m_fea_gen;

  AudioReader m_reader;
  int m_sample_rate;
  int m_frame_rate;
  int m_eof_frame;

  int m_window_advance;
  int m_window_width;
  std::vector<float> m_hamming_window;
  fftw_plan m_coeffs;
  std::vector<double> m_fftw_datain;
  std::vector<double> m_fftw_dataout;
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


class DCTModule : public FeatureModule {
public:
  DCTModule();
  static const char *type_str() { return "dct"; }
private:
  virtual void get_module_config(ModuleConfig &config);
  virtual void set_module_config(const ModuleConfig &config);
  virtual void generate(int frame);
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
private:
  NormalizationModule();
  static const char *type_str() { return "normalization"; }
  virtual void get_module_config(ModuleConfig &config);
  virtual void set_module_config(const ModuleConfig &config);
  virtual void generate(int frame);
private:
  std::vector<float> m_mean;
  std::vector<float> m_scale;
};


class TransformationModule : public FeatureModule {
private:
  TransformationModule();
  static const char *type_str() { return "transform"; }
  virtual void get_module_config(ModuleConfig &config);
  virtual void set_module_config(const ModuleConfig &config);
  virtual void generate(int frame);
private:
  std::vector<float> m_transform;
  int m_src_dim;
};


class MergerModule : public FeatureModule {
public:
  MergerModule();
  static const char *type_str() { return "merge"; }
  virtual void link(FeatureModule *source);
private:
  virtual void get_module_config(ModuleConfig &config);
  virtual void set_module_config(const ModuleConfig &config);
  virtual void generate(int frame);
};



#endif /* FEATUREMODULES_HH */
