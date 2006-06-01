#include "SpeakerConfig.hh"
#include "str.hh"


SpeakerConfig::SpeakerConfig(FeatureGenerator &fea_gen) :
  m_fea_gen(fea_gen)
{
  m_cur_speaker = "";
  m_default_set = false;
}


void SpeakerConfig::read_speaker_file(FILE *file)
{
  std::string line;
  int lineno = 0;
  bool fetch_default;
  std::string speaker_id;
  SpeakerMap::iterator speaker_it;
  
  while (str::read_line(&line, file, true))
  {
    fetch_default = false;
    lineno++;
    str::clean(&line, " \t");
    if (line.empty())
      continue;
    if (line == "default")
    {
      if (m_default_set)
        throw str::fmt(256, "SpeakerConfig: Default speaker configuration already defined, redefinition on line %d: ", lineno) + line;
      fetch_default = true;
      m_default_set = true;
    }
    else
    {
      std::vector<std::string> fields;
      str::split(&line, " \t", true, &fields);
      if (fields.size() != 2 || fields[0] != "speaker")
        throw str::fmt(256, "SpeakerConfig: Expected keyword 'speaker' and a speaker ID on line %d: ", lineno) + line;

      ModuleMap dummy;
      SpeakerMap::value_type v(fields[1], dummy);
      speaker_it = m_speaker_config.insert(v).first;
    }

    while (str::read_line(&line, file, true))
    {
      lineno++;
      str::clean(&line, " \t");
      if (line.empty())
        continue;
      
      if (line != "{")
        throw std::string("'{' expected in speaker config file: ") + line;
      break;
    }

    // Read the modules
    while (str::read_line(&line, file, true))
    {
      lineno++;
      str::clean(&line, " \t");
      if (line.empty())
        continue;
      if (line == "}")
      {
        // End of speaker configuration
        break;
      }
      // line should now contain the module name
      try {
        m_fea_gen.module(line);
      } catch (std::string &str) {
        throw str::fmt(256, "SpeakerConfig: error on line %d: ", lineno) + str;
      }

      // Read module config
      ModuleConfig config;
      try { 
        config.read(file);
      }
      catch (std::string &str) {
        lineno += config.num_lines_read();
        throw str::fmt(256, "SpeakerConfig: Failed reading module parameters around line %d: ",
                       lineno) + str;
      }

      ModuleMap::value_type v(line, config);
      if (fetch_default)
        m_default_config.insert(v);
      else
        (*speaker_it).second.insert(v);
      
      lineno += config.num_lines_read();
    }
  }
}


void SpeakerConfig::write_speaker_file(FILE *file)
{
  if (m_cur_speaker.size() > 0)
    retrieve_speaker_config(m_cur_speaker);

  // Write default configuration
  if (m_default_set)
  {
    fputs("default\n{\n", file);
    for (ModuleMap::const_iterator module_it = m_default_config.begin();
       module_it != m_default_config.end(); module_it++)
    {
      fprintf(file, "  %s\n", (*module_it).first.c_str());
      (*module_it).second.write(file, 2);
      fputs("\n", file);
    }
    fputs("}\n\n",file);
  }

  // Write speaker configurations
  for (SpeakerMap::const_iterator speaker_it = m_speaker_config.begin();
       speaker_it != m_speaker_config.end(); speaker_it++)
  {
    fprintf(file, "speaker %s\n{\n", (*speaker_it).first.c_str());
    for (ModuleMap::const_iterator module_it = (*speaker_it).second.begin();
       module_it != (*speaker_it).second.end(); module_it++)
    {
      fprintf(file, "  %s\n", (*module_it).first.c_str());
      (*module_it).second.write(file, 2);
      fputs("\n", file);
    }
    fputs("}\n\n",file);
  }
}


void SpeakerConfig::set_speaker(const std::string &speaker_id)
{
  if (m_cur_speaker.size() > 0)
    retrieve_speaker_config(m_cur_speaker);

  if (speaker_id.size() == 0)
  {
    // Use the default speaker
    if (!m_default_set)
      throw std::string("SpeakerConfig: No speaker defined, needs a default speaker.");
    for (ModuleMap::const_iterator module_it = m_default_config.begin();
         module_it != m_default_config.end(); module_it++)
    {
      m_fea_gen.module((*module_it).first)->set_parameters(
        (*module_it).second);
    }
  }
  else
  {
    SpeakerMap::const_iterator speaker_it =
      m_speaker_config.find(speaker_id);
    if (speaker_it == m_speaker_config.end())
    {
      if (!m_default_set)
        throw std::string("SpeakerConfig: Unknown speaker ") + speaker_id +
          ", and default speaker settings are missing.";
      // Insert a new speaker with default configuration
      SpeakerMap::value_type v(speaker_id, m_default_config);
      speaker_it = m_speaker_config.insert(v).first;
    }
  
    for (ModuleMap::const_iterator module_it = (*speaker_it).second.begin();
         module_it != (*speaker_it).second.end(); module_it++)
    {
      m_fea_gen.module((*module_it).first)->set_parameters(
        (*module_it).second);
    }
  }
  
  m_cur_speaker = speaker_id;
}


void SpeakerConfig::retrieve_speaker_config(const std::string &speaker_id)
{
  SpeakerMap::iterator speaker_it =
    m_speaker_config.find(speaker_id);
  if (speaker_it == m_speaker_config.end())
    throw std::string("SpeakerConfig: Unknown speaker ") + speaker_id;

  for (ModuleMap::iterator module_it = (*speaker_it).second.begin();
       module_it != (*speaker_it).second.end(); module_it++)
  {
    m_fea_gen.module((*module_it).first)->get_parameters(
      (*module_it).second);
  }
}
