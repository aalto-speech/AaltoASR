#ifndef FEATUREMODULE_HH
#define FEATUREMODULE_HH

#include <string>

#include "FeatureBuffer.hh"
#include "ModuleConfig.hh"


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

}

#endif
