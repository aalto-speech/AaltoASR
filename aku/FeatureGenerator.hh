#ifndef FEATUREGENERATOR_HH
#define FEATUREGENERATOR_HH

#include <vector>
#include <map>
#include <string>

#include "FeatureBuffer.hh"


namespace aku {

// Defined in FeatureModules.hh. Don't include FeatureModules.hh to avoid
// unnecessary inclusion of the FFT library header.
class FeatureModule;
class BaseFeaModule;

/** A class for generating feature vectors from an audio file.  The
 * FeatureGenerator and the feature modules are configured in a text
 * file that contains block for each module in the feature module
 * chain (see doc/ directory).
 */
class FeatureGenerator {
public:
  FeatureGenerator();
  ~FeatureGenerator();

  /** Open an audio file closing the possible previously opened file.
   *
   * \param filename = The name of the audio file.
   * samples.  Otherwise, automatic file format is used.
   * \exception string If cannot open file.
   */
  void open(const std::string &filename);
  void open_fd(const int fd);

  /** Open an audio file closing the possible previously opened file.
   *
   * \param file = The file pointer of the audio file.
   * samples.  Otherwise, automatic file format is used.
   * \param dont_fclose = If true, the file is not closed by FeatureGenerator
   */
  void open(FILE *file, bool dont_fclose, bool stream = false);

  /** Close the previously opened file. */
  void close();

  /** Load the configuration of feature modules from a file. */
  void load_configuration(FILE *file);

  /** Write the configuration of feature modules. */
  void write_configuration(FILE *file);

  /** Close the configuration of feature modules and clear things. */
  void close_configuration();

  /** Fetch a module by name. */
  FeatureModule *module(const std::string &name);

  /** Generates a feature vector for the requested \c frame.
   * Generating frames sequentially (either forward or backward)
   * is guaranteed to use buffers of the feature modules efficiently.
   * If you need more random access, you might want to buffer frames
   * yourself.
   *
   * \note Some feature configurations may allow generating features
   * after the physical file end.  Use \ref eof() after generate()
   * call to check if the end of file was reached during the generated
   * frame.
   **/
  const FeatureVec generate(int frame);

  /** Returns the last frame that does not generate eof */
  int last_frame();

  /** Returns true if the frame requested from generate() contained
   * end of file. */
  bool eof() { return m_eof_on_last_frame; }

  /** Return the sample rate (samples per second). */
  int sample_rate();

  /** Return the frame rate (frames per second). */
  float frame_rate();

  /** Return the dimension of the feature fector. */
  int dim();

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

  /** The audio file. */
  FILE *m_file;

  /** Should we call fclose when closing the file. */
  bool m_dont_fclose;

  /** Was end of file reached on the frame requested from generate(). */
  bool m_eof_on_last_frame;
};

}

#endif /* FEATUREGENERATOR_HH */
