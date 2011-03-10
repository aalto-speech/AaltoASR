#ifndef SPEAKERCONFIG_HH
#define SPEAKERCONFIG_HH

#include <vector>
#include <set>
#include <map>
#include <string>

#include "FeatureGenerator.hh"
#include "ModelModules.hh"


namespace aku {

/** A class for handling speaker and utterance adaptations.
 * Speaker ID is the primary method for changing feature configurations.
 * If another level of configurations is needed, utterance IDs can
 * be used. Note that utterance ID is reset when speaker ID changes.
 */
class SpeakerConfig {
public:
  SpeakerConfig(FeatureGenerator &fea_gen, HmmSet *model = NULL);
  
  void read_speaker_file(FILE *file);
  void write_speaker_file(FILE *file, std::set<std::string> *speakers = NULL,
                          std::set<std::string> *utterances = NULL);

  void set_speaker(const std::string &speaker_id);
  const std::string& get_cur_speaker(void) { return m_cur_speaker; }

  void set_utterance(const std::string &utterance_id);
  const std::string& get_cur_utterance(void) { return m_cur_utterance; }

  ModelTransformer& get_model_transformer() { return m_model_trans; }

private:
  typedef std::map<std::string, ModuleConfig> ModuleMap;
  typedef std::map<std::string,  ModuleMap> SpeakerMap;
  
  void retrieve_speaker_config(const std::string &speaker_id);
  void retrieve_utterance_config(const std::string &utterance_id);
  void set_modules(const ModuleMap &modules, bool load_new_model_trans = false);
  
private:
  FeatureGenerator &m_fea_gen;
  ModelTransformer m_model_trans;
  HmmSet *m_model;
  SpeakerMap m_speaker_config;
  SpeakerMap m_utterance_config;

  ModuleMap m_default_speaker_config;
  ModuleMap m_default_utterance_config;
  bool m_default_speaker_set;
  bool m_default_utterance_set;

  std::string m_cur_speaker;
  std::string m_cur_utterance;
};

}

#endif // SPEAKERCONFIG_HH
