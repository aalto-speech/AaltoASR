#ifndef AUDIOFILEMODULE_HH
#define AUDIOFILEMODULE_HH

#include <vector>

#include "BaseFeaModule.hh"
#include "AudioReader.hh"
#include "ModuleConfig.hh"


namespace aku {

class FeatureGenerator;

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

}

#endif
