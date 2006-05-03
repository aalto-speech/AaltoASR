#ifndef MODULECONFIG_HH
#define MODULECONFIG_HH

#include <map>
#include <vector>
#include <string>

class ModuleConfig {
public:
  bool exists(const std::string &name) const;

  void set(const std::string &name, int value);
  void set(const std::string &name, float value);
  void set(const std::string &name, const std::string &str);
  void set(const std::string &name, const std::vector<int> &vec);
  void set(const std::string &name, const std::vector<float> &vec);
  void set(const std::string &name, const std::vector<std::string> &vec);

  // Leaves 'value' unchanged if 'name' not defined
  bool get(const std::string &name, int &value) const;
  bool get(const std::string &name, float &value) const;
  bool get(const std::string &name, std::string &str) const;
  bool get(const std::string &name, std::vector<int> &vec) const;
  bool get(const std::string &name, std::vector<float> &vec) const;
  bool get(const std::string &name, std::vector<std::string> &vec) const;
  
  /** Writes the configuration in file. */
  void write(FILE *file, int indent = 0) const;

  /** Reads configuration from file. */
  void read(FILE *file);

  /** Return the number of lines read on last call of \ref read() */
  int num_lines_read() const { return m_num_lines_read; }

private:
  typedef std::map<std::string, std::string> ValueMap;
  ValueMap m_value_map;
  int m_num_lines_read;
};


#endif /* MODULECONFIG_HH */
