#include <vector>
#include "str.hh"
#include "ConfigParser.hh"

int
ConfigParser::get_int(const std::string &value)
{
  bool ok = true;
  int ret = str::str2long(&value, &ok);
  if (!ok)
    throw std::string("invalid integer value: ") + value;
  return ret;
}

float
ConfigParser::get_float(const std::string &value)
{
  bool ok = true;
  float ret = str::str2float(&value, &ok);
  if (!ok)
    throw std::string("invalid float value: ") + value;
  return ret;
}

void
ConfigParser::get_int_vec(const std::string &value, std::vector<int> &vec)
{
  str::split(&value, " \t", true, &m_fields);
  vec.resize(m_fields.size());
  bool ok = true;
  for (int i = 0; i < (int)m_fields.size(); i++)
    vec[i] = str::str2long(&m_fields[i], &ok);
  if (!ok)
    throw std::string("invalid value in integer vector: ") + value;
}

void
ConfigParser::get_float_vec(const std::string &value, std::vector<float> &vec)
{
  str::split(&value, " \t", true, &m_fields);
  vec.resize(m_fields.size());
  bool ok = true;
  for (int i = 0; i < (int)m_fields.size(); i++)
    vec[i] = str::str2float(&m_fields[i], &ok);
  if (!ok)
    throw std::string("invalid value in float vector: ") + value;
}

