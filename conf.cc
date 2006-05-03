#include <deque>
#include <assert.h>
#include "conf.hh"
#include "str.hh"
#include "io.hh"

namespace conf {

  int
  Option::get_int() const
  {
    char *endptr;
    int return_value = strtol(value.c_str(), &endptr, 10);
    if (endptr == value.c_str() || *endptr != 0) {
      fprintf(stderr, "invalid value for option %s: %s\n", name.c_str(), 
	      value.c_str());
      exit(1);
    }
    return return_value;
  }

  float
  Option::get_float() const
  {
    char *endptr;
    double return_value = strtof(value.c_str(), &endptr);
    if (endptr == value.c_str() || *endptr != 0) {
      fprintf(stderr, "invalid value for option %s: %s\n", name.c_str(), 
	      value.c_str());
      exit(1);
    }
    return return_value;
  }

  double
  Option::get_double() const
  {
    char *endptr;
    double return_value = strtod(value.c_str(), &endptr);
    if (endptr == value.c_str() || *endptr != 0) {
      fprintf(stderr, "invalid value for option %s: %s\n", name.c_str(), 
	      value.c_str());
      exit(1);
    }
    return return_value;
  }

  const std::string &
  Option::get_str() const 
  {
    return value;
  }

  const char *
  Option::get_c_str() const 
  {
    return value.c_str();
  }

  Config::Config()
    : longest_name_length(0)
  {
  }

  Config&
  Config::operator()(std::string usage)
  {
    usage_line = usage;
    return *this;
  }

  Config&
  Config::operator()(unsigned char short_name, std::string long_name,
		     std::string type, std::string default_value, 
		     std::string help)
  {
    // Initialize new option
    Option o;
    o.short_name = short_name;
    o.long_name = long_name;
    o.value = default_value;
    o.required = false;
    o.needs_argument = false;
    o.help = help;

    // Remove the possible equal sign part from the long name used as
    // a map key.
    std::string long_name_map_key(long_name);
    int equal_pos = long_name_map_key.find('=');
    if (equal_pos >= 0)
      long_name_map_key.erase(equal_pos);

    // Parse the type string
    std::vector<std::string> types;
    str::split(&type, " \t", true, &types);
    for (int i = 0; i < (int)types.size(); i++) {
      if (types[i] == "arg")
	o.needs_argument = true;
      else if (types[i] == "must")
	o.required = true;
      else {
	fprintf(stderr, "invalid option type %s for option -%c --%s\n",
		types[i].c_str(), short_name, long_name.c_str());
	abort();
      }
    }
  
    // Add structure to configuration
    if (short_name != 0) {
      if (short_map.find(short_name) != short_map.end()) {
	fprintf(stderr, "trying to add option -%c twice\n", short_name);
	exit(1);
      }
      short_map[short_name] = options.size();
      o.name.append("-");
      o.name.append(1, short_name);
    }
    if (long_name_map_key != "") {
      if (long_map.find(long_name_map_key) != long_map.end()) {
	fprintf(stderr, "trying to add option --%s twice\n", 
		long_name_map_key.c_str());
	exit(1);
      }
      long_map[long_name_map_key] = options.size();
      if ((int)long_name.length() > longest_name_length)
	longest_name_length = long_name.length();
      if (o.name.length() > 0)
	o.name.append(" --");
      else
	o.name.append("--");
      o.name.append(long_name_map_key);
    }
    options.push_back(o);

    return *this;
  }

  // private aux function used by parse() and read()
  void
  Config::parse_aux(std::deque<std::string> &argument_queue, bool override)
  {
    bool options_allowed = true; // Have not seen "--" yet.
    std::deque<int> options_pending; // Options waiting for parameter

    while (!argument_queue.empty()) {
      std::string arg(argument_queue.front());
      argument_queue.pop_front();

      // Check if the argument is an option at all
      if (options_allowed && options_pending.empty() && 
	  arg.length() > 1 && arg[0] == '-')
      {

	// End of options "--"
	if (arg == "--") {
	  options_allowed = false;
	  continue;
	}

	// Short option?
	if (arg[1] != '-') {

	  // Parse grouped short options
	  for (int c = 1; c < (int)arg.length(); c++) {

	    // Check if the option is valid
	    unsigned char short_name = arg[c];
	    ShortMap::iterator it = short_map.find(short_name);
	    if (it == short_map.end()) {
	      fprintf(stderr, "invalid option -%c\n", short_name);
	      exit(1);
	    }

	    Option &option = options[it->second];
	    if (option.needs_argument)
	      options_pending.push_back(it->second);
	    else
	      option.specified = true;
	  }
	}

	// Long option?
	else {
	
	  std::string long_name(arg.substr(2));
	  std::string parameter;
	  int equal_pos = long_name.find('=');
	  if (equal_pos >= 0) {
	    parameter = long_name.substr(equal_pos + 1);
	    argument_queue.push_front(parameter);
	    long_name.erase(equal_pos);
	  }
	  LongMap::iterator it = long_map.find(long_name);
	  if (it == long_map.end()) {
	    fprintf(stderr, "invalid option --%s\n", long_name.c_str());
	    exit(1);
	  }
	  
	  Option &option = options[it->second];
	  if (option.needs_argument)
	    options_pending.push_back(it->second);
	  else
	    option.specified = true;
	}

      }

      // Otherwise just a parameter (possibly for a previous option)
      else {

	// Just an ordinary command line parameter
	if (options_pending.empty())
	  arguments.push_back(arg);

	// Parameter for an option
	else {
	  Option &option = options[options_pending.front()];
	  options_pending.pop_front();
	  if (!option.specified || override)
	    option.value = arg;
	  option.specified = true;
	}
      }
    }

    // Check if the last option is missing arguments
    if (!options_pending.empty()) {
      std::string option_name;
      Option &option = options[options_pending.front()];
      if (option.short_name != 0) {
	option_name.append(" -");
	option_name.append(1, option.short_name);
      }
      if (option.long_name != "") {
	option_name.append(" --");
	option_name.append(option.long_name);
      }
      fprintf(stderr, "option%s lacks an argument\n", option_name.c_str());
      exit(1);
    }
  }

  void
  Config::parse(int argc, char *argv[], bool override)
  {
    std::deque<std::string> argument_queue;
    for (int arg_index = 1; arg_index < argc; arg_index++)
      argument_queue.push_back(argv[arg_index]);
    parse_aux(argument_queue, override);
  }

  void
  Config::default_parse(int argc, char *argv[])
  {
    parse(argc, argv);

    if ((*this)["help"].specified) {
      fputs(help_string().c_str(), stdout);
      exit(0);
    }
    check_required();
  }

  void
  Config::read(FILE *file, bool override)
  {
    // Read and split the file in fields
    std::string text;
    if (!str::read_file(&text, file)) {
      perror("could not read config file");
      exit(1);
    }
    std::vector<std::string> fields;
    str::split_with_quotes(&text, " \t\n", true, &fields);
    
    // Prepare for parsing
    std::deque<std::string> argument_queue(fields.begin(), fields.end());
    parse_aux(argument_queue, override);
  }

  void
  Config::check_required() const
  {
    for (int i = 0; i < (int)options.size(); i++) {
      const Option &option = options[i];
      std::string option_name;
      if (option.short_name != 0) {
	option_name.append(" -");
	option_name.append(1, option.short_name);
      }
      if (option.long_name != "") {
	option_name.append(" --");
	option_name.append(option.long_name);
      }
      if (option.required && !option.specified) {
	fprintf(stderr, "option%s required\n", option_name.c_str());
	print_help(stderr, 1);
      }
    }
  }

  void
  Config::print_help(FILE *file, int exit_value) const
  {
    fputs(help_string().c_str(), file);
    exit(exit_value);
  }

  std::string 
  Config::help_string() const
  {
    std::string help;
    help.reserve(1024);
    help.append(usage_line);
    
    // Iterate options
    for (int i = 0; i < (int)options.size(); i++) {
      const Option& option = options[i];
      
      help.append("  ");

      // Append short name
      if (option.short_name != 0) {
	help.append("-");
	help.append(1, option.short_name);
      }
      else
	help.append("  ");
     
      // Append long name
      if (option.long_name != "") {
	if (option.short_name != 0)
	  help.append(", ");
	else
	  help.append("  ");

	help.append("--");
	help.append(option.long_name);
	help.append(longest_name_length - option.long_name.length(), ' ');
      }
      else if (longest_name_length > 0)
	help.append(longest_name_length + 4, ' ');
      
      // Append the help text
      help.append("  ");
      help.append(option.help);
      help.append("\n");
    }

    return help;
  }

  const Option&
  Config::operator[](unsigned char short_name) const
  {
    ShortMap::const_iterator it = short_map.find(short_name);
    if (it == short_map.end()) {
      fprintf(stderr, "Config::get(): unknown option %c\n", short_name);
      abort();
    }
    return options[it->second];
  }

  const Option&
  Config::operator[](std::string long_name) const
  {
    LongMap::const_iterator it = long_map.find(long_name);
    if (it == long_map.end()) {
      fprintf(stderr, "Config::get(): unknown option %s\n", long_name.c_str());
      abort();
    }
    return options[it->second];
  }
};
