#ifndef FEATUREMODULES_HH
#define FEATUREMODULES_HH

#include <fftw3.h>
#include "FeatureBuffer.hh"
#include "AudioReader.hh"


class FeatureGenerator;


struct ConfigPair {
  std::string name;
  std::string value;
};


/*
  Some semantics about the module structure:

  The module is initialized by first linking it to its sources (by link())
  and then calling configure() with the loaded settings. configure()
  calls module's private configure_module(), which must check that source
  dimensions match with the given settings. After configure_module()
  configure() calls source modules' set_buffer().
  
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

  // Do not call set_buffer() before configuration has been finished!
  void set_buffer(int left, int right);

  // Called before configure(). The default implementation allows
  // only one source.
  virtual void link(FeatureModule *source);

  // Linking has been completed before configure(), so the module
  // can check the dimensions in the configuration.
  void configure(std::vector<struct ConfigPair> &config);
  
  ConstFeatureVec at(int frame);
  int dim(void) { return m_dim; } // Valid only after configuration
  
private:
  virtual void configure_module(std::vector<struct ConfigPair> &config) = 0;
  virtual void generate(int frame) = 0;
  
protected:
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
  
  virtual void set_file(FILE *fp);
  virtual void discard_file(void);
  virtual bool eof(int frame);
  virtual int sample_rate(void) { return m_sample_rate; }
  virtual int frame_rate(void) { return m_frame_rate; }
  
private:
  virtual void configure_module(std::vector<struct ConfigPair> &config);
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
private:
  virtual void configure_module(std::vector<struct ConfigPair> &config);
  virtual void generate(int frame);

  void create_mel_bins(void);

private:
  FeatureGenerator *m_fea_gen;

  int m_bins;
  std::vector<float> m_bin_edges;
};


class PowerModule : public FeatureModule {
private:
  virtual void configure_module(std::vector<struct ConfigPair> &config);
  virtual void generate(int frame);
};


class DCTModule : public FeatureModule {
private:
  virtual void configure_module(std::vector<struct ConfigPair> &config);
  virtual void generate(int frame);
};


class DeltaModule : public FeatureModule {
private:
  virtual void configure_module(std::vector<struct ConfigPair> &config);
  virtual void generate(int frame);
private:
  int m_delta_width;
  int m_delta_norm;
};


class MergerModule : public FeatureModule {
public:
  virtual void link(FeatureModule *source);
private:
  virtual void configure_module(std::vector<struct ConfigPair> &config);
  virtual void generate(int frame);
};

#endif /* FEATUREMODULES_HH */
