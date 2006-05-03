#include <assert.h>
#include <vector>
#include "str.hh"
#include "ModuleConfig.hh"

bool
ModuleConfig::exists(const std::string &name) const 
{
  return m_value_map.find(name) != m_value_map.end();
}

void
ModuleConfig::set(const std::string &name, int value)
{
  assert(name.find_first_of(" \t\n") == std::string::npos);
  m_value_map[name] = str::fmt(64, "%d", value);
}

void
ModuleConfig::set(const std::string &name, float value)
{
  assert(name.find_first_of(" \t\n") == std::string::npos);
  m_value_map[name] = str::fmt(64, "%g", value);
}

void
ModuleConfig::set(const std::string &name, const std::string &str)
{
  assert(name.find_first_of(" \t\n") == std::string::npos);
  assert(str.find_first_of("\n") == std::string::npos);
  m_value_map[name] = str;
}

void
ModuleConfig::set(const std::string &name, const std::vector<int> &vec)
{
  assert(name.find_first_of(" \t\n") == std::string::npos);
  std::string value_str;
  for (int i = 0; i < (int)vec.size(); i++) {
    if (i != 0)
      value_str.append(" ");
    value_str.append(str::fmt(64, "%d", vec[i]));
  }
  m_value_map[name] = value_str;
}

void
ModuleConfig::set(const std::string &name, const std::vector<float> &vec)
{
  assert(name.find_first_of(" \t\n") == std::string::npos);
  std::string value_str;
  for (int i = 0; i < (int)vec.size(); i++) {
    if (i != 0)
      value_str.append(" ");
    value_str.append(str::fmt(64, "%g", vec[i]));
  }
  m_value_map[name] = value_str;
}

void
ModuleConfig::set(const std::string &name, const std::vector<std::string> &vec)
{
  assert(name.find_first_of(" \t\n") == std::string::npos);
  std::string value_str;
  for (int i = 0; i < (int)vec.size(); i++) {
    assert(vec[i].find_first_of(" \t\n") == std::string::npos);
    if (i != 0)
      value_str.append(" ");
    value_str.append(vec[i]);
  }
  m_value_map[name] = value_str;
}

bool
ModuleConfig::get(const std::string &name, int &value) const
{
  ValueMap::const_iterator it = m_value_map.find(name);
  if (it == m_value_map.end())
    return false;

  bool ok = true;
  value = str::str2long(&it->second, &ok);
  if (!ok)
    throw std::string("invalid integer value: ") + it->second;
  return true;
}

bool
ModuleConfig::get(const std::string &name, float &value) const
{
  ValueMap::const_iterator it = m_value_map.find(name);
  if (it == m_value_map.end())
    return false;

  bool ok = true;
  value = str::str2float(&it->second, &ok);
  if (!ok)
    throw std::string("invalid float value: ") + it->second;
  return true;
}

bool
ModuleConfig::get(const std::string &name, std::string &str) const
{
  ValueMap::const_iterator it = m_value_map.find(name);
  if (it == m_value_map.end())
    return false;
  str = it->second;
  return true;
}

bool
ModuleConfig::get(const std::string &name, std::vector<int> &vec) const
{
  ValueMap::const_iterator it = m_value_map.find(name);
  if (it == m_value_map.end())
    return false;

  std::vector<std::string> fields;
  str::split(&it->second, " \t", true, &fields);
  vec.resize(fields.size());
  bool ok = true;
  for (int i = 0; i < (int)fields.size(); i++)
    vec[i] = str::str2long(&fields[i], &ok);
  if (!ok)
    throw std::string("invalid value in integer vector: ") + it->second;
  return true;
}

bool
ModuleConfig::get(const std::string &name, std::vector<float> &vec) const
{
  ValueMap::const_iterator it = m_value_map.find(name);
  if (it == m_value_map.end())
    return false;

  std::vector<std::string> fields;
  str::split(&it->second, " \t", true, &fields);
  vec.resize(fields.size());
  bool ok = true;
  for (int i = 0; i < (int)fields.size(); i++)
    vec[i] = str::str2float(&fields[i], &ok);
  if (!ok)
    throw std::string("invalid value in float vector: ") + it->second;
  return true;
}

bool
ModuleConfig::get(const std::string &name, std::vector<std::string> &vec) const
{
  ValueMap::const_iterator it = m_value_map.find(name);
  if (it == m_value_map.end())
    return false;
  str::split(&it->second, " \t", true, &vec);
  return true;
}

void
ModuleConfig::read(FILE *file)
{
  m_num_lines_read = 0;
  bool first_line = true;
  std::string line;
  std::vector<std::string> fields;
  while (1) {
    if (!str::read_line(&line, file, true))
      throw std::string("unexpected end of module config file");
    m_num_lines_read++;

    str::clean(&line, " \t");
    if (line.empty())
      continue;
    
    if (first_line) {
      if (line != "{")
	throw std::string("'{' expected in module config file: ") + line;
      first_line = false;
      continue;
    }

    if (line == "}")
      break;

    str::split(&line, " \t", true, &fields, 2);
    assert(fields.size() >= 1 && fields.size() <= 2);
    if (fields.size() == 1)
      throw std::string("value missing in module config file: ") + line;
    if (m_value_map.find(fields[0]) != m_value_map.end())
      throw std::string("value redefined in module config file: ") + line;
    
    m_value_map[fields[0]] = fields[1];
  }
}

void
ModuleConfig::write(FILE *file, int indent) const
{
  assert(indent >= 0);
  std::string indent_str(indent, ' ');
  for (ValueMap::const_iterator it = m_value_map.begin(); 
       it != m_value_map.end(); it++)
  {
    if (indent > 0) 
      fputs(indent_str.c_str(), file);
    fputs(it->first.c_str(), file);
    fputc(' ', file);
    fputs(it->second.c_str(), file);
    fputc('\n', file);
  }
}
