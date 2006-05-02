#ifndef CONFIGPARSER_HH
#define CONFIGPARSER_HH

class ConfigParser {
public:
  int get_int(const std::string &value);
  float get_float(const std::string &value);
  void get_int_vec(const std::string &value, std::vector<int> &vec);
  void get_float_vec(const std::string &value, std::vector<float> &vec);
private: 
  std::vector<std::string> m_fields;
};

#endif /* CONFIGPARSER_HH */
