#ifndef FEATUREGENERATOR_HH
#define FEATUREGENERATOR_HH

#include <vector>
#include <map>
#include <string>
#include "FeatureModules.hh"

/** A class for generating feature vectors from an audio file. */
class FeatureGenerator {
public:
  /** Format of the audio file. */
  typedef enum {AF_RAW, AF_AUTO} AudioFormat;

  FeatureGenerator();
  ~FeatureGenerator();

  /** Open an audio file closing the possible previously opened file.
   *
   * \param filename = The name of the audio file.  
   * \param raw_audio = If true, the file assumed to contain raw audio
   * samples.  Otherwise, automatic file format is used.
   */
  void open(const std::string &filename, bool raw_audio);

  /** Close the previously opened file. */
  void close();

  /** Load the configuration of feature modules from a file. */
  void load_configuration(FILE *file);

  /** Write the configuation of feature modules. */
  void write_configuration(FILE *file);

  /** Fetch a module by name. */
  FeatureModule *module(const std::string &name);
  
  /** Generates a feature vector for the requested \c frame.
   * Generating frames sequentially is guaranteed to use buffers of
   * the feature moduls efficiently.  If you need more random access,
   * you might want to buffer frames yourself.  
   *
   * \note Some feature configurations may allow generating features
   * after the physical file end.  Use \ref eof() after generate()
   * call to check if the end of file was reached during the generated
   * frame.
   **/
  inline const FeatureVec generate(int frame);

  /** Returns true if the frame requested from generate() contained
   * end of file. */
  inline bool eof(void) { return m_eof_on_last_frame; }

  /** Return the sample rate (samples per second). */
  inline int sample_rate(void);

  /** Return the frame rate (frames per second). */
  inline int frame_rate(void);

  /** Return the dimension of the feature fector. */
  inline int dim(void);

  /** Return the format of the audio file. */
  AudioFormat audio_format() { return m_audio_format; }

private:
  typedef std::map<std::string, FeatureModule*> ModuleMap;

  std::vector<FeatureModule*> m_modules; //!< The feature modules
  ModuleMap m_module_map; //!< Mapping names of the modules to modules

  /** The base feature module that is responsible for reading the
   * physical file. */
  BaseFeaModule *m_base_module;

  /** The last module in the module chain, which generates the final
   * features. */
  FeatureModule *m_last_module;

  /** The format of the audio file. */
  AudioFormat m_audio_format;

  /** The audio file. */
  FILE *m_file;

  /** Was end of file reached on the frame requested from generate(). */
  bool m_eof_on_last_frame;
};


const FeatureVec
FeatureGenerator::generate(int frame)
{
  assert( m_last_module != NULL );
  const FeatureVec temp = m_last_module->at(frame);
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
