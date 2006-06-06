#include <set>
#include <errno.h>
#include <string.h>
#include "ModuleConfig.hh"
#include "str.hh"
#include "FeatureGenerator.hh"

FeatureGenerator::FeatureGenerator(void) :
  m_base_module(NULL),
  m_last_module(NULL),
  m_file(NULL),
  m_eof_on_last_frame(false)
{
}

FeatureGenerator::~FeatureGenerator()
{
  for (int i = 0; i < (int)m_modules.size(); i++)
    delete m_modules[i];
}

void
FeatureGenerator::open(const std::string &filename, bool raw_audio)
{
  if (m_file != NULL)
    close();

  if (raw_audio > 0)
    m_audio_format = AF_RAW;
  else
    m_audio_format = AF_AUTO;

  m_file = fopen(filename.c_str(), "rb");
  if (m_file == NULL)
    throw std::string("could not open file ") + filename + ": " +
      strerror(errno);

  for (int i = 0; i < (int)m_modules.size(); i++)
    m_modules[i]->reset();
  
  assert( m_base_module != NULL );
  m_base_module->set_file(m_file);
}


void
FeatureGenerator::close(void)
{
  if (m_file != NULL) {
    m_base_module->discard_file();
    fclose(m_file);
    m_file = NULL;
  }
}


void
FeatureGenerator::load_configuration(FILE *file)
{
  assert(m_modules.empty());
  std::string line;
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
    if (name.find_first_of(" \t\n") != std::string::npos)
      throw std::string("module name may not contain whitespaces");
  
    FeatureModule *module = NULL;
    if (type == FFTModule::type_str())
      module = new FFTModule(this);
    else if (type == PreModule::type_str())
      module = new PreModule();
    else if (type == MelModule::type_str())
      module = new MelModule(this);
    else if (type == PowerModule::type_str())
      module = new PowerModule();
    else if (type == DCTModule::type_str())
      module = new DCTModule();
    else if (type == DeltaModule::type_str())
      module = new DeltaModule();
    else if (type == NormalizationModule::type_str())
      module = new NormalizationModule();
    else if (type == LinTransformModule::type_str())
      module = new LinTransformModule();
    else if (type == MergerModule::type_str())
      module = new MergerModule();
    else if (type == MeanSubtractorModule::type_str())
      module = new MeanSubtractorModule();
    else if (type == ConcatModule::type_str())
      module = new ConcatModule();
    else if (type == VtlnModule::type_str())
      module = new VtlnModule();
    else if (type == SRNormModule::type_str())
      module = new SRNormModule();
    else
      throw std::string("Unknown module type '") + type + std::string("'");
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

    if (has_sources) {
      std::vector<std::string> sources;
      config.get("sources", sources);
      assert(!sources.empty());
      for (int i = 0; i < (int)sources.size(); i++) {
	ModuleMap::iterator it = m_module_map.find(sources[i]);
	if (it == m_module_map.end())
	  throw std::string("unknown source module: ") + sources[i];
	module->add_source(it->second);
      }
    }
    
    module->set_config(config);
  }

  check_model_structure();
}


void
FeatureGenerator::write_configuration(FILE *file)
{
  assert(!m_modules.empty());
  for (int i = 0; i < (int)m_modules.size(); i++) {
    FeatureModule *module = m_modules[i];

    ModuleConfig config;
    module->get_config(config);

    if (!module->sources().empty()) {
      std::vector<std::string> sources;
      for (int i = 0; i < (int)module->sources().size(); i++)
	sources.push_back(module->sources().at(i)->name());
      config.set("sources", sources);
    }

    fputs("module\n", file);
    config.write(file, 0);
    fputs("\n", file);
  }
}

FeatureModule*
FeatureGenerator::module(const std::string &name)
{
  ModuleMap::iterator it = m_module_map.find(name);
  if (it == m_module_map.end())
    throw std::string("unknown module requested: ") + name;
  return it->second;
}


void
FeatureGenerator::check_model_structure()
{
  if (m_modules.empty())
    throw std::string("no feature modules defined");

  std::set<FeatureModule*> reached;
  std::vector<FeatureModule*> stack;
  stack.push_back(m_last_module);

  while (!stack.empty()) {
    FeatureModule *module = stack.back();
    stack.pop_back();
    for (int i = 0; i < (int)module->sources().size(); i++) {
      FeatureModule *source = module->sources().at(i);
      std::pair<std::set<FeatureModule*>::iterator, bool> ret =
	reached.insert(source);
      if (ret.second)
	stack.push_back(source);
    }
  }

  assert(!m_modules.empty());
  for (int i = 0; i < (int)m_modules.size() - 1; i++)
    if (reached.find(m_modules[i]) == reached.end())
      fprintf(stderr, "WARNING: module %s (type %s) not used as input\n", 
	      m_modules[i]->name().c_str(), m_modules[i]->type_str().c_str());
}
