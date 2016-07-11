#include <fcntl.h>
#include <stdio.h>

#include <set>
#include <errno.h>
#include <string.h>
#include "ModuleConfig.hh"
#include "str.hh"
#include "FeatureGenerator.hh"
#include "FeatureModules.hh"


namespace aku {

FeatureGenerator::FeatureGenerator(void) :
  m_base_module(NULL),
  m_last_module(NULL),
  m_file(NULL),
  m_dont_fclose(false),
  m_eof_on_last_frame(false)
{
}

FeatureGenerator::~FeatureGenerator()
{
  for (int i = 0; i < (int)m_modules.size(); i++)
    delete m_modules[i];
}

void
FeatureGenerator::open(const std::string &filename)
{
/* Old implementation 7.4.2010 varjokal
  if (m_file != NULL)
    close();

  FILE *file = fopen(filename.c_str(), "rb");
  if (file == NULL)
    throw std::string("could not open file ") + filename + ": " +
      strerror(errno);

  open(file, false);
*/

  if (m_file != NULL)
    close();
  for (int i = 0; i < (int)m_modules.size(); i++)
    m_modules[i]->reset();

  assert( m_base_module != NULL );
  m_base_module->set_fname(filename.c_str());
}

void
FeatureGenerator::open_fd(const int fd)
{
  if (m_file != NULL)
    close();

  FILE *file = fdopen(fd, "rb");
  if (file == NULL) {
    throw std::string("could not open fd ") + ": " +
      strerror(errno);
  }
  open(file, false, false);
}


void
FeatureGenerator::open(FILE *file, bool dont_fclose, bool stream)
{
  if (m_file != NULL)
    close();
  m_file = file;
  m_dont_fclose = dont_fclose;

  for (int i = 0; i < (int)m_modules.size(); i++)
    m_modules[i]->reset();

  assert( m_base_module != NULL );
  m_base_module->set_file(m_file, stream);
}

void
FeatureGenerator::close(void)
{
  if (m_file != NULL) {
    m_base_module->discard_file();
    if (!m_dont_fclose)
      fclose(m_file);
    m_file = NULL;
  }
}


void
FeatureGenerator::load_configuration(FILE *file)
{
  //assert(m_modules.empty());
  if (!m_modules.empty()) {
    fprintf(stdout, "FeatureGenerator: loading a new feature configuration\n");
    this->close_configuration();
  }
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
    if (type == AudioFileModule::type_str())
      module = new AudioFileModule(this);
    else if (type == FFTModule::type_str())
      module = new FFTModule();
    else if (type == PreModule::type_str())
      module = new PreModule();
    else if (type == MelModule::type_str())
      module = new MelModule(this);
    else if (type == PowerModule::type_str())
      module = new PowerModule();
    else if (type == MelPowerModule::type_str())
      module = new MelPowerModule();
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
    else if (type == QuantEqModule::type_str())
      module = new QuantEqModule();
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

  compute_init_buffers();
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

void
FeatureGenerator::close_configuration()
{
  for (int i = 0; i < (int)m_modules.size(); i++)
    delete m_modules[i];
  m_modules.clear();
  m_module_map.clear();
  m_base_module = NULL;
  m_last_module = NULL;
}


FeatureModule*
FeatureGenerator::module(const std::string &name)
{
  ModuleMap::iterator it = m_module_map.find(name);
  if (it == m_module_map.end())
    throw std::string("unknown module requested: ") + name;
  return it->second;
}

const FeatureVec
FeatureGenerator::generate(int frame)
{
  assert( m_last_module != NULL );
  const FeatureVec temp = m_last_module->at(frame);
  m_eof_on_last_frame = m_base_module->eof(frame);
  return temp;
}

int
FeatureGenerator::last_frame()
{
  return m_base_module->last_frame();
}

int
FeatureGenerator::sample_rate()
{
  assert( m_base_module != nullptr );
  return m_base_module->sample_rate();
}

float
FeatureGenerator::frame_rate()
{
  assert( m_base_module != nullptr );
  return m_base_module->frame_rate();
}

int
FeatureGenerator::dim()
{
  assert( m_last_module != nullptr );
  return m_last_module->dim();
}

void // private
FeatureGenerator::compute_init_buffers()
{
  // Compute number of targets for each module.
  //

  std::vector<int> target_counts(m_modules.size(), 0);
  std::map<FeatureModule*, int> index_map;
  for (int i = 0; i < (int)m_modules.size(); i++)
    index_map[m_modules[i]] = i;

  for (int i = 0; i < (int)m_modules.size(); i++) {
    FeatureModule *module = m_modules[i];
    for (int j = 0; j < (int)module->sources().size(); j++) {
      int src_index = index_map[module->sources()[j]];
      target_counts[src_index]++;
    }
  }

  // Find bottle-neck modules, i.e. modules that are not in a branch.
  // Below it is assumed that m_modules is sorted topologically so
  // that sources are always before targets.
  //
  std::vector<bool> bottle_neck(m_modules.size(), false);
  int cur_branch_level = 0;
  for (int i = m_modules.size() - 1; i >= 0; i--) {
    FeatureModule *module = m_modules[i];
    if (target_counts.at(i) >= 2)
      cur_branch_level -= target_counts.at(i) - 1;
    assert(cur_branch_level >= 0);
    if (cur_branch_level == 0)
      bottle_neck.at(i) = true;
    if (module->sources().size() >= 2)
      cur_branch_level += module->sources().size() - 1;
  }

  // Every target-branching module M must have buffer offsets that
  // include the largest offsets between M and the next bottle-neck
  // module.  Otherwise, duplicated computation is done when buffers
  // are filled for the first time (thus the name init_offset_left and
  // right).  Some target-branching modules could actually have
  // smaller buffers, but it would be more complicated to compute the
  // minimal size.
  //

  for (int i = m_modules.size() - 1; i >= 0; i--) {
    FeatureModule *module = m_modules[i];

    if (!bottle_neck[i]) {
      for (int j = 0; j < (int)module->sources().size(); j++) {
        FeatureModule *src_module = module->sources()[j];
        src_module->update_init_offsets(*module);
      }
    }
  }
}


void // private
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

void
FeatureGenerator::print_dot_graph(FILE *file)
{
  fprintf(file, "digraph features {\n");
  fprintf(file, "rankdir=RL;\n");
  for (int i = 0; i < (int)m_modules.size(); i++) {
    FeatureModule *module = m_modules[i];
    module->print_dot_node(file);
  }

  for (int i = 0; i < (int)m_modules.size(); i++) {
    FeatureModule *module = m_modules[i];
    for (int j = 0; j < (int)module->sources().size(); j++) {
      fprintf(file, "\t%s -> %s;\n", module->name().c_str(),
              module->sources()[j]->name().c_str());
    }
  }

  fprintf(file, "}\n");
}

}
