#ifndef SPEAKERCONFIG_HH
#define SPEAKERCONFIG_HH

#include <vector>
#include <map>
#include <string>

#include "FeatureGenerator.hh"

class SpeakerConfig {
public:
  SpeakerConfig(FeatureGenerator &fea_gen);
  
  void read_speaker_file(FILE *file);
  void write_speaker_file(FILE *file);

  void set_speaker(const std::string &speaker_id);
  const std::string& get_cur_speaker(void) { return m_cur_speaker; }

private:
  void retrieve_speaker_config(const std::string &speaker_id);
  
private:
  FeatureGenerator &m_fea_gen;

  typedef std::map<std::string, ModuleConfig> ModuleMap;
  typedef std::map<std::string,  ModuleMap> SpeakerMap;
  SpeakerMap m_speaker_config;

  ModuleMap m_default_config;
  bool m_default_set;

  std::string m_cur_speaker;
};

#endif // SPEAKERCONFIG_HH
