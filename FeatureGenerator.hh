#ifndef FEATUREGENERATOR_HH
#define FEATUREGENERATOR_HH

#include <vector>
#include <map>
#include <string>
#include "FeatureModules.hh"

/** A class for generating feature vectors from an audio file.  The
 * FeatureGenerator and the feature modules are configured in a text
 * file that contains block for each module in the feature module
 * chain (see doc/ directory).
 */
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

  /** Open an audio file closing the possible previously opened file.
   *
   * \param file = The file pointer of the audio file.  
   * \param raw_audio = If true, the file assumed to contain raw audio
   * samples.  Otherwise, automatic file format is used.
   * \param dont_fclose = If true, the file is not closed by FeatureGenerator
   */
  void open(FILE *file, bool raw_audio, bool dont_fclose);

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
   * the feature modules efficiently.  If you need more random access,
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
  inline float frame_rate(void);

  /** Return the dimension of the feature fector. */
  inline int dim(void);

  /** Return the format of the audio file. */
  AudioFormat audio_format() { return m_audio_format; }

  /** Print the module structure in DOT format. */
  void print_dot_graph(FILE *file);
      

private:

  /** Compute buffer offsets for modules so that duplicate computation
   * is avoided in module branches. */
  void compute_init_buffers();

  /** Check module structure and warn about anomalities. */
  void check_model_structure();

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

  /** Should we call fclose when closing the file. */
  bool m_dont_fclose;

  /** Was end of file reached on the frame requested from generate(). */
  bool m_eof_on_last_frame;
};


const FeatureVec
FeatureGenerator::generate(int frame)
{
  assert( m_last_module != NULL );
  const FeatureVec temp = m_last_module->at(frame);
  m_eof_on_last_frame = m_base_module->eof(frame);
  return temp;
}

int
FeatureGenerator::sample_rate(void)
{
  assert( m_base_module != NULL );
  return m_base_module->sample_rate();
}

float
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
