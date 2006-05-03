#include "ModuleConfig.hh"
#include "str.hh"
#include "FeatureGenerator.hh"

FeatureGenerator::FeatureGenerator(void) :
  m_base_module(NULL),
  m_last_module(NULL),
  m_eof_on_last_frame(false)
{
}

FeatureGenerator::~FeatureGenerator()
{
  for (int i = 0; i < (int)m_modules.size(); i++)
    delete m_modules[i];
}

void
FeatureGenerator::open(const std::string &filename, int raw_sample_rate)
{
  if (raw_sample_rate > 0)
  {
    m_audio_format = AF_RAW;
  }
  else if (filename.find(".htk") != std::string::npos)
  {
    m_audio_format = AF_HTK;
  } 
  else if (filename.find(".pre") != std::string::npos)
  {
    m_audio_format = AF_PRE;
  }
  else
  {
    m_audio_format = AF_WAV;
  }

  if ((m_file = fopen(filename.c_str(), "rb")) == NULL)
  {
    throw std::string("Could not open file ")+filename;
  }
  
  assert( m_base_module != NULL );
  m_base_module->set_file(m_file);
}


void
FeatureGenerator::close(void)
{
  if (m_file != NULL)
  {
    m_base_module->discard_file();
  }
}


void
FeatureGenerator::load_configuration(FILE *file)
{
  assert(m_modules.empty());
  std::string line;
  std::vector<std::string> fields;
  int lineno = 0;
  while (str::read_line(&line, file, true)) {
    lineno++;
    str::clean(&line, " \t");
    if (line.empty())
      continue;
    if (line != "module")
      throw str::fmt(256, "expected keyword 'module' on line %d: ", lineno) +
	line;

    // Read module config
    //
    ModuleConfig config;
    try { 
      config.read(file);
    }
    catch (std::string &str) {
      lineno += config.num_lines_read();
      throw str::fmt(256, "failed reading feature module around line %d: ",
		     lineno) + str;
    }
    lineno += config.num_lines_read();


    // Create module
    //
    std::string type;
    std::string name;
    if (!config.get("type", type))
      throw str::fmt(256, "type not defined for module ending on line %d",
		     lineno);
    if (!config.get("name", name))
      throw str::fmt(256, "name not defined for module ending on line %d",
		     lineno);
    assert(!name.empty());
  
    FeatureModule *module = NULL;
    if (type == FFTModule::type_str()) 
      module = new FFTModule(this);
    else if (type == MelModule::type_str())
      module = new MelModule(this);
    else if (type == PowerModule::type_str())
      module = new PowerModule();
    else if (type == DCTModule::type_str())
      module = new DCTModule();
    else if (type == DeltaModule::type_str())
      module = new DeltaModule();
    else if (type == MergerModule::type_str())
      module = new MergerModule();
    module->set_name(name);

    // Insert module in module structures
    //
    if (m_modules.empty()) {
      m_base_module = dynamic_cast<BaseFeaModule*>(module);
      if (m_base_module == NULL)
	throw std::string("first module should be a base module");
    }
    m_last_module = module;
    m_modules.push_back(module);
    if (m_module_map.find(name) != m_module_map.end())
      throw std::string("multiple definitions of module name: ") + name;
    m_module_map[name] = module;

    // Create source links
    //
    bool has_sources = config.exists("sources");
    if (m_base_module == module && has_sources)
      throw std::string("can not define sources for the first module");
    if (m_base_module != module && !has_sources)
      throw std::string("sources not defined for module: ") + name;
    
    std::vector<std::string> sources;
    config.get("sources", sources);
    assert(!sources.empty());
    for (int i = 0; i < (int)sources.size(); i++) {
      ModuleMap::iterator it = m_module_map.find(sources[i]);
      if (it == m_module_map.end())
	throw std::string("unknown source module: ") + sources[i];
      module->link(it->second);
    }
    
    module->set_config(config);
  }
}


void
FeatureGenerator::write_configuration(FILE *file)
{
  assert(!m_modules.empty());
  for (int i = 0; i < (int)m_modules.size(); i++) {
    FeatureModule *module = m_modules[i];

    ModuleConfig config;
    module->get_config(config);
    std::vector<std::string> sources;
    for (int i = 0; i < (int)module->sources().size(); i++)
      sources.push_back(module->sources().at(i)->name());
    config.set("sources", sources);

    fputs("module\n{\n", file);
    config.write(file);
    fputs("}\n\n", file);
  }
}

