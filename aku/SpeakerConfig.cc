#include "SpeakerConfig.hh"
#include "FeatureModules.hh"
#include "str.hh"


namespace aku {

SpeakerConfig::SpeakerConfig(FeatureGenerator &fea_gen, HmmSet *model) :
  m_fea_gen(fea_gen), m_model(model)
{
  m_cur_speaker = "";
  m_cur_utterance = "";
  m_default_speaker_set = false;
  m_default_utterance_set = false;
  if(model != NULL) m_model_trans.set_model(model);
}


void
SpeakerConfig::read_speaker_file(FILE *file)
{
  std::string line;
  int lineno = 0;
  bool fetch_default;
  std::string speaker_id;
  SpeakerMap::iterator speaker_it;
  bool fetch_speaker;
  
  while (str::read_line(&line, file, true))
  {
    fetch_default = false;
    lineno++;
    str::clean(&line, " \t");
    if (line.empty())
      continue;

    std::vector<std::string> fields;
    str::split(&line, " \t", true, &fields);
    if (fields.size() != 2 ||
        (fields[0] != "speaker" && fields[0] != "utterance"))
      throw str::fmt(256, "SpeakerConfig: Syntax error on line %d: ", lineno) + line;

    if (fields[1] == "default")
      fetch_default = true;
    
    if (fields[0] == "speaker")
    {
      if (fetch_default && m_default_speaker_set)
        throw str::fmt(256, "SpeakerConfig: Default speaker configuration already defined, redefinition on line %d: ", lineno) + line;
      fetch_speaker = true;
      if (!fetch_default)
      {
        ModuleMap dummy;
        SpeakerMap::value_type v(fields[1], dummy);
        speaker_it = m_speaker_config.insert(v).first;
      }
      else
        m_default_speaker_set = true;
    }
    else // "utterance"
    {
      if (fetch_default && m_default_utterance_set)
        throw str::fmt(256, "SpeakerConfig: Default utterance configuration already defined, redefinition on line %d: ", lineno) + line;
      fetch_speaker = false;
      if (!fetch_default)
      {
        ModuleMap dummy;
        SpeakerMap::value_type v(fields[1], dummy);
        speaker_it = m_utterance_config.insert(v).first;
      }
      else
        m_default_utterance_set = true;
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
        // End of speaker/utterance configuration
        break;
      }
      // line should now contain the module name
      // if line does not start with model, it's assumed to be a feature module
      std::vector<std::string> parts;
      str::split(&line, " \t", true, &parts, 2);

      if(parts.size() < 2) {
        line = "feature "+line;
        str::split(&line, " \t", true, &parts, 2);
        //throw str::fmt(256, "Warning: feature namespace assumed at line %d", lineno);
      }
      else {
        if(parts[0] != "model" && parts[0] != "feature")
          throw str::fmt(256, "SpeakerConfig: Unknown module namespace at line %d",
                                 lineno);
      }

      assert(parts.size() == 2);

      try {
        if(parts[0] == "feature")
          m_fea_gen.module(parts[1]);

        if(parts[0] == "model")
          m_model_trans.module(parts[1]);

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
      {
        if (fetch_speaker)
          m_default_speaker_config.insert(v);
        else
          m_default_utterance_config.insert(v);
      }
      else
        (*speaker_it).second.insert(v);
      
      lineno += config.num_lines_read();
    }
  }
}


void
SpeakerConfig::write_speaker_file(FILE *file,
                                       std::set<std::string> *speakers,
                                       std::set<std::string> *utterances)
{
  if (m_cur_speaker.size() > 0)
    retrieve_speaker_config(m_cur_speaker);

  if (m_cur_utterance.size() > 0)
    retrieve_utterance_config(m_cur_utterance);

  // Write default speaker configuration
  if (m_default_speaker_set && (speakers == NULL ||
                                speakers->find("default") != speakers->end()))
  {
    fputs("speaker default\n{\n", file);
    for (ModuleMap::const_iterator module_it=m_default_speaker_config.begin();
       module_it != m_default_speaker_config.end(); module_it++)
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
    if (speakers == NULL ||
        speakers->find((*speaker_it).first) != speakers->end())
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

  // Write default utterance configuration
  if (m_default_utterance_set &&
      (utterances == NULL || utterances->find("default") != utterances->end()))
  {
    fputs("utterance default\n{\n", file);
    for (ModuleMap::const_iterator module_it =
           m_default_utterance_config.begin();
         module_it != m_default_utterance_config.end(); module_it++)
    {
      fprintf(file, "  %s\n", (*module_it).first.c_str());
      (*module_it).second.write(file, 2);
      fputs("\n", file);
    }
    fputs("}\n\n",file);
  }

  // Write utterance configurations
  for (SpeakerMap::const_iterator ut_it = m_utterance_config.begin();
       ut_it != m_utterance_config.end(); ut_it++)
  {
    if (utterances == NULL ||
        utterances->find((*ut_it).first) != utterances->end())
    {
      fprintf(file, "utterance %s\n{\n", (*ut_it).first.c_str());
      for (ModuleMap::const_iterator module_it = (*ut_it).second.begin();
           module_it != (*ut_it).second.end(); module_it++)
      {
        fprintf(file, "  %s\n", (*module_it).first.c_str());
        (*module_it).second.write(file, 2);
        fputs("\n", file);
      }
      fputs("}\n\n",file);
    }
  }
}


void
SpeakerConfig::set_speaker(const std::string &speaker_id)
{
  if (m_cur_speaker.size() > 0)
    retrieve_speaker_config(m_cur_speaker);

  // Reset the utterance
  if (m_cur_utterance.size() > 0)
    set_utterance("");

  bool load_new_model_trans = false;

  if(speaker_id != m_cur_speaker) {
    // Reset model transformations
    m_model_trans.reset_transforms();
  }

  if(m_model_trans.is_reset()) load_new_model_trans = true;



  if (speaker_id.size() == 0)
  {
    // Use the default speaker
    if (!m_default_speaker_set)
      throw std::string("SpeakerConfig: No speaker defined, needs a default speaker.");
    set_modules(m_default_speaker_config, load_new_model_trans);
  }
  else
  {
    SpeakerMap::const_iterator speaker_it =
      m_speaker_config.find(speaker_id);
    if (speaker_it == m_speaker_config.end())
    {
      if (!m_default_speaker_set)
        throw std::string("SpeakerConfig: Unknown speaker ") + speaker_id +
          ", and default speaker settings are missing.";
      // Insert a new speaker with default configuration
      SpeakerMap::value_type v(speaker_id, m_default_speaker_config);
      speaker_it = m_speaker_config.insert(v).first;
    }

    set_modules((*speaker_it).second, load_new_model_trans);
  }
  
  m_cur_speaker = speaker_id;
  if(load_new_model_trans) m_model_trans.load_transforms();
}

void
SpeakerConfig::set_utterance(const std::string &utterance_id)
{
  if (m_cur_utterance.size() > 0)
    retrieve_utterance_config(m_cur_utterance);

  if (utterance_id.size() == 0)
  {
    // Use the default utterance
    if (!m_default_utterance_set)
      throw std::string("SpeakerConfig: Default utterance is required.");
    set_modules(m_default_utterance_config);
  }
  else
  {
    SpeakerMap::const_iterator ut_it = m_utterance_config.find(utterance_id);
    if (ut_it == m_utterance_config.end())
    {
      if (!m_default_utterance_set)
        throw std::string("SpeakerConfig: Unknown utterance ") + utterance_id +
          ", and default utterance settings are missing.";
      // Insert a new utterance with default configuration
      SpeakerMap::value_type v(utterance_id, m_default_utterance_config);
      ut_it = m_utterance_config.insert(v).first;
    }

    set_modules((*ut_it).second);
  }
  
  m_cur_utterance = utterance_id;
}


void
SpeakerConfig::retrieve_speaker_config(const std::string &speaker_id)
{
  SpeakerMap::iterator speaker_it =
    m_speaker_config.find(speaker_id);
  if (speaker_it == m_speaker_config.end())
    throw std::string("SpeakerConfig: Unknown speaker ") + speaker_id;

  for (ModuleMap::iterator module_it = (*speaker_it).second.begin();
       module_it != (*speaker_it).second.end(); module_it++)
  {
    std::vector<std::string> parts;
    str::split(&(*module_it).first, " \t", true, &parts, 2);
    if(parts[0] == "feature")
      m_fea_gen.module(parts[1])->get_parameters((*module_it).second);

    if(parts[0] == "model")
      m_model_trans.module(parts[1])->get_parameters((*module_it).second);

  }
}


void
SpeakerConfig::retrieve_utterance_config(const std::string &utterance_id)
{
  SpeakerMap::iterator utterance_it =
    m_utterance_config.find(utterance_id);
  if (utterance_it == m_utterance_config.end())
    throw std::string("SpeakerConfig: Unknown utterance ") + utterance_id;

  for (ModuleMap::iterator module_it = (*utterance_it).second.begin();
       module_it != (*utterance_it).second.end(); module_it++)
  {
    std::vector<std::string> parts;
    str::split(&(*module_it).first, " \t", true, &parts, 2);
    if(parts[0] == "feature")
      m_fea_gen.module(parts[1])->set_parameters((*module_it).second);
    if(parts[0] == "model")
      m_model_trans.module(parts[1])->set_parameters((*module_it).second);

  }
}

void
SpeakerConfig::set_modules(const ModuleMap &modules, bool load_new_model_trans)
{
  for (ModuleMap::const_iterator module_it=modules.begin();
         module_it != modules.end(); module_it++)
  {
    std::vector<std::string> parts;
    str::split(&(*module_it).first, " \t", true, &parts, 2);
    if(parts[0] == "feature")
      m_fea_gen.module(parts[1])->set_parameters((*module_it).second);
    if(parts[0] == "model")
      m_model_trans.module(parts[1])->set_parameters((*module_it).second);


  }
}

}
