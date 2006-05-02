#ifndef FEATUREGENERATOR_HH
#define FEATUREGENERATOR_HH

#include "FeatureModules.hh"


class FeatureGenerator {
public:
  FeatureGenerator();
  
  void open(const std::string &filename, int raw_sample_rate = 0);
  void close();
  void load_configuration(const std::string &filename);
  void write_configuration(const std::string &filename);
  
  inline ConstFeatureVec generate(int frame);
  inline bool eof(void) { return m_eof_on_last_frame; }

  inline int sample_rate(void);
  inline int frame_rate(void);
  inline int dim(void);

  typedef enum {AF_RAW, AF_WAV, AF_HTK, AF_PRE} AudioFormat;
  
  AudioFormat audio_format() { return m_audio_format; }

private:
  std::vector<FeatureModule*> m_modules;
  BaseFeaModule *m_base_module;
  FeatureModule *m_last_module;
  AudioFormat m_audio_format;

  FILE *m_file;

  bool m_eof_on_last_frame;
};


ConstFeatureVec
FeatureGenerator::generate(int frame)
{
  assert( m_last_module != NULL );
  ConstFeatureVec temp = m_last_module->at(frame);
  if (m_base_module->eof(frame))
    m_eof_on_last_frame = true;
  return temp;
}

int
FeatureGenerator::sample_rate(void)
{
  assert( m_base_module != NULL );
  return m_base_module->sample_rate();
}

int
FeatureGenerator::frame_rate(void)
{
  assert( m_base_module != NULL );
  return m_base_module->frame_rate();
}

int
FeatureGenerator::dim(void)
{
  assert( m_last_module != NULL );
  return m_last_module->dim();
}



#endif /* FEATUREGENERATOR_HH */
