#ifndef CONF_HH
#define CONF_HH

#include <vector>
#include <deque>
#include <string>
#include <map>

#include <stdio.h>

/** Configuration options, command line parameters and config files. 
 *
 * The command line options are parsed with the following features:
 * \li Short option names start with a hyphen followed by a character
 * ("-h").
 * \li If a short option needs a parameter, it must be given in
 * the next command line argument ("-i 10").
 * \li An argument consisting of just a hyphen and nothing else is not
 * interpreted as an option.
 * \li Short options can be grouped after a single hyphen ("-hi 10" or
 * "-ih 10").
 * \li Long option names start with two hyphens ("--foobar").
 * \li The parameter can be specified in two ways: "--int 10" or
 * "--int=10"
 * \li An argument consisting of just two hyphens "--" is
 * skipped, and all arguments after that are interpreted as
 * non-options.
 * \li The configuration options can also be read from a
 * file: see \ref Config::read()
 * 
 *
 * Examples with options "-h", "-i INT", and "-s STR":
 * \li Valid: -h -i 10 -s str --string str --string=str -hic 10 str
 * \li Invalid: -i10 -sstr
 * \li Invalid: --integer --string 10 str
 *
 * Example:
 * \include ex_conf.cc
 */
namespace conf {

  typedef std::map<std::string, int> ChoiceMapType;

  class Choice {
  public:
    /** Create an empty choice. */
    Choice() { }

    /** Adds a choice option with an interpreted value
     * \param choice_name = Choice name string
     * \param value = Interpreted value
     */
    Choice& operator()(std::string choice_name, int value) {
      choice_map.insert(ChoiceMapType::value_type(choice_name, value));
      return *this;
    }

    /** Parse the given choice.
     * \param choice = Given choice option
     * \param result = Where the result value is stored to
     * \returns true if choice was found and result set, false otherwise.
     */
    bool parse(std::string choice, int &result) {
      ChoiceMapType::iterator it = choice_map.find(choice);
      if (it == choice_map.end())
        return false;
      result = (*it).second;
      return true;
    }

  private:
     ChoiceMapType choice_map;
  };
  
  /** An option */
  struct Option {
    Option() : short_name(0), required(false), needs_argument(false), 
	       specified(false) { }
    unsigned char short_name; //!< The short name used for command line
    std::string long_name; //!< The long name used also in configuration files
    std::string value; //!< The value parsed from command line or file
    bool required; //!< Is the option required
    bool needs_argument; //!< Does the option need an argument
    bool specified; //!< Has the user specified the option
    std::string help; //!< The help string of the option

    /** The name string of the option: "-h --help", for example. */
    std::string name; 

    int get_int() const; //!< Return the integer value of the option
    float get_float() const; //!< Return the float value of the option
    double get_double() const; //!< Return the double value of the option
    const std::string &get_str() const; //!< Return the string of the option
    const char *get_c_str() const; //!< Return the string of the option
  };

  /** A class for defining, storing and querying options. 
   *
   * \bug An option added to config can not be deleted anymore.
   **/
  class Config {
  public:

    /** Create an empty config. */
    Config();

    /** Add a usage line for the help screen. */
    Config& operator()(std::string usage);

    /** Add a new option. 
     * \param short_name = the short name of the option used after a hyphen
     * \param long_name = the long name of the option used after two hyphens
     * \param type = a space seperated list of type specifiers: 
     *   \li "arg" = argument must follow the option
     *   \li "must" = the argument must be specified by the user
     * \param default_value = the default value of the option
     * \param help = the help string of the option
     */
    Config& operator()(unsigned char short_name,
		       std::string long_name,
		       std::string type = "",
		       std::string default_value = "",
		       std::string help = "");

    /** Parse command line arguments. 
     * \param argc = the number of arguments
     * \param argv = the arguments
     * \param override = should we override options specified already
     */
    void parse(int argc, char *argv[], bool override = true);

    /** Default parsing with "help" option.
     * \param argc = the number of arguments
     * \param argv = the arguments
     */
    void default_parse(int argc, char *argv[]);

    /** Read a config file.
     *
     * The config file can contain command line arguments almost as if
     * they were specified on the command line.  The only difference
     * is that quotation marks \, ', " are interpreted by the
     * function. All quotation marks can be used to quote other
     * quotation marks, and backslash can be used to quote backslash
     * too.  Unless quoted, whitespace (tab, space, newline) separate
     * arguments.
     *
     * \param file = file to read from
     * \param override = should we override options specified already
     */
    void read(FILE *file, bool override = false);
    
    /** Check if all required options are specified. */
    void check_required() const;

    /** Prints usage information. */
    void print_help(FILE *file = stdout, int exit_value = 0) const;

    /** Returns the usage information */
    std::string help_string() const;

    /** Get a value of an option. */
    const Option& operator[](unsigned char short_name) const;

    /** Get a value of an option. */
    const Option& operator[](std::string long_name) const;

    /** A type for mapping short option names to option indices. */
    typedef std::map<unsigned char, int> ShortMap;

    /** A type for mapping long option names to option indices. */
    typedef std::map<std::string, int> LongMap;

    std::string usage_line; //!< Usage line printed in help if specified
    std::vector<Option> options; //!< Options added to the configuration
    ShortMap short_map; //!< Mapping short names to options
    LongMap long_map; //!< Mapping long names to options
    std::vector<std::string> arguments; //!< Rest of the non-option arguments
    int longest_name_length; //!< The length of the longest long name

  private:
    /** The actual parsing of command line (or config file) options */
    void parse_aux(std::deque<std::string> &argument_queue, bool override);
  };

};

#endif /* CONF_HH */
