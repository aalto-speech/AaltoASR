#ifndef STR_HH
#define STR_HH

#include <cstddef> // NULL
#include <cstring>
#include <climits>
#include <cfloat>
#include <assert.h>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <math.h>


/** Functions for handling strings of the Standard Template Library.
 * \bug We assume that the newline is '\n', which can cause problems
 * in Windows.
 */
namespace str {

  /** Convert anything to std::string using std::ostringstream. */
  template <typename T>
  inline std::string str(const T &t)
  {
    std::ostringstream o;
    o << t;
    return o.str();
  }

  /** Format a string. 
   * \param size = maximum size of the resulting string (cut rest)
   * \param fmt = the format string as in standard printf()
   * \return the formatted string
   */
  inline std::string fmt(size_t size, const char *fmt, ...)
  {
    va_list ap;
    va_start(ap, fmt);
    char *buf = new char[size];
    vsnprintf(buf, size, fmt, ap);
    va_end(ap);

    std::string str(buf);
    delete[] buf;
    return str;
  }

  /** Remove leading and trailing characters from a string. 
   * \param str = the string to be cleaned
   * \param chars = the characters to clean from the string
   */
  inline void clean(std::string &str, const char *chars = " \n\t")
  {
    int i;

    // Clean trailing chars
    //
    for (i = str.length(); i > 0; i--)
      if (!strchr(chars, str[i - 1]))
	break;
    str.erase(i);

    // Clean leading chars
    //
    for (i = 0; i < (int)str.length(); i++)
      if (!strchr(chars, str[i]))
	break;
    str.erase(0, i);
  }

  /** Return and remove a substring from the beginning of the string
   * until one of the given characters are found.  If not found, the
   * whole string is popped.
   *
   * \param str = the string to be popped
   * \param chars = the characters to be searched
   * \param include = if false, the found character is not included in
   * the return value (but is still removed)
   */

  inline std::string pop(std::string &str, const std::string &chars, 
                         bool include = false)
  {
    std::string ret;
    std::string::size_type pos = str.find_first_of(chars);
    if (pos == str.npos)
      ret.swap(str);
    else {
      if (include)
        ret = str.substr(0, pos + 1);
      else
        ret = str.substr(0, pos);
      str.erase(0, pos + 1);
    }
    return ret;
  }

  /** Remove leading and trailing characters from a string. 
   * \param str = the string to be cleaned
   * \param chars = the characters to clean from the string
   * \return the cleaned string
   */
  inline std::string cleaned(const std::string &str, 
                             const char *chars = " \n\t")
  {
    std::string ret = str;
    clean(ret, chars);
    return ret;
  }

  /** Remove selected characters from the string. 
   * \param str = the string to be modified
   * \param chars = the characters to be removed
   */
  inline void remove(std::string &str, const char *chars)
  {
    size_t src = 0;
    size_t tgt = 0;
    while (1) {
      if (src == str.length())
        break;
      str[tgt] = str[src];
      if (!strchr(chars, str[src]))
        tgt++;
      src++;
    }
    str.resize(tgt);
  }

  /** Split a string to fields.  If \c num_fields is positive, it
   * specifies the maximum number of fields.  If the line contains
   * more fields, they are included in the last field as such (also
   * possible delimiters without processing).
   *
   * \param str = the string to be splitted
   * \param delims = the delimiter characters
   * \param group = group subsequent delimiters as one delimeter
   * \param num_fields = the maximum number of fields to return
   * \return fields the vector containing the resulting fields
   */
  inline std::vector<std::string>
  split(const std::string &str, const char *delims, bool group, 
        unsigned int num_fields = 0)
  {
    std::vector<std::string> fields;
    size_t begin = 0;
    size_t end = 0;

    fields.clear();
    bool delim_pending = false;
    while (begin < str.length()) {
      // If 'fields' fields was requested and, this is the last field,
      // include the rest.
      if (num_fields > 0 && fields.size() == num_fields - 1) {
        delim_pending = false;
	fields.push_back(str.substr(begin));
	break;
      }

      // Find the string before the next delim
      delim_pending = false;
      while (end < str.length()) {
        if (strchr(delims, str[end])) {
          delim_pending = true;
          break;
        }
	end++;
      }
      fields.push_back(str.substr(begin, end - begin));

      // Eat the delim or group of delims
      end++;
      if (group)
	while (end < str.length() && strchr(delims, str[end]))
	  end++;

      begin = end;
    }

    if (delim_pending)
      fields.push_back("");

    return fields;
  }

  /** Split string into lines.
   *
   * \param str = string to split
   * \param skip_empty = ignore empty lines
   * \param delims = line delimiter charaters
   * \return vector of lines
   */
  inline std::vector<std::string>
  split_lines(const std::string &str, bool skip_empty = true, 
              const char *delims = "\n\r")
  {
    std::vector<std::string> lines = str::split(str, "\n\r", skip_empty);
    if (!lines.empty()) {
      if (lines.back().empty())
        lines.pop_back();
    }
    return lines;
  }

  /** Convert a string to signed long.  Arbitrary amount of white
   * space is accepted before the number as explained on strtol(3)
   * manual page.  Anything after the number is considered as an
   * conversion error.
   *
   * \param str = the string to convert
   * \param base = base of the coversion
   * \throw std::out_of_range if argument was out of range
   * \throw std::invalid_argument if conversion error occured
   */
  inline long str2long(const std::string &str, int base = 10)
  {
    char *endptr;
    const char *c_str = str.c_str();

    long value = strtol(c_str, &endptr, base);
    if (value == LONG_MIN || value == LONG_MAX)
      throw std::out_of_range("str::str2long: argument out of range");
    if (*c_str == '\0' || *endptr != '\0')
      throw std::invalid_argument("str::str2long: invalid string");

    return value;
  }

  /** Convert a string to unsigned long.  Arbitrary amount of white
   * space is accepted before the number as explained on strtol(3)
   * manual page.  Anything after the number is considered as an
   * conversion error.
   *
   * \param str = the string to convert
   * \param base = base of the coversion
   * \throw std::out_of_range if argument was out of range
   * \throw std::invalid_argument if conversion error occured
   */
  inline unsigned long str2ulong(const std::string &str, int base = 10)
  {
    char *endptr;
    const char *c_str = str.c_str();

    unsigned long value = strtoul(c_str, &endptr, base);
    if (value == ULONG_MAX)
      throw std::out_of_range("str::str2ulong: argument out of range");
    if (*c_str == '\0' || *endptr != '\0')
      throw std::invalid_argument("str::str2ulong: invalid string");

    return value;
  }

  /** Convert a string to float.  Arbitrary amount of white space is
   * accepted before the float as explained on strtod(3) manual page.
   * Anything after the number is considered as an conversion error.
   *
   * \param str = the string to convert
   * \throw std::out_of_range if argument was out of range
   * \throw std::invalid_argument if conversion error occured
   */
  inline float str2float(const std::string &str)
  {
    char *endptr;
    const char *c_str = str.c_str();

    double dbl_value = strtod(c_str, &endptr);

    if (*c_str == '\0' || *endptr != '\0')
      throw std::invalid_argument("str::str2float: invalid string");
    if (dbl_value == -HUGE_VAL || dbl_value == HUGE_VAL ||
        dbl_value < -FLT_MAX || dbl_value > FLT_MAX)
      throw std::out_of_range("str::str2float: argument out of range");
    float value = (float)dbl_value;

    return value;
  }


  /** Convert a string to a vector of integers using str2long(). 
   * \param str = the string to convert
   * \return a vector of integers
   * \throw exceptions as described in str2long()
   */
  template <class T>
  std::vector<T> long_vec(const std::string &str)
  {
    std::vector<std::string> str_vec(
      str::split(str::cleaned(str), " \t\n", true));
    std::vector<T> vec(str_vec.size());
    for (size_t i = 0; i < str_vec.size(); i++)
      vec[i] = str::str2long(str_vec[i]);
    return vec;
  }

  /** Convert a string to a vector of floats using str2float(). 
   * \param str = the string to convert
   * \return a vector of floats
   * \throw exceptions as described in str2float()
   */
  inline std::vector<float> float_vec(const std::string &str)
  {
    std::vector<std::string> str_vec(
      str::split(str::cleaned(str), " \t\n", true));
    std::vector<float> vec(str_vec.size());
    for (size_t i = 0; i < str_vec.size(); i++)
      vec[i] = str::str2float(str_vec[i]);
    return vec;
  }

  /** Return a strin with the possible trailing newline removed
   * \param str = string to chomp
   * \param chars = characters to remove
   * \return the string without trailing newline
   */
  inline std::string chomped(const std::string &str, const char *chars = "\n")
  {
    if (!str.empty() && strchr(chars, str[str.length() - 1]))
      return str.substr(0, str.length() - 1);
    return str;
  }

  /** Remove the possible trailing newline from a string
   * \param str = string to chomp
   * \param chars = characters to remove
   * \return reference to modified \c str 
   */
  inline std::string &chomp(std::string &str, const char *chars = "\n")
  {
    if (!str.empty() && strchr(chars, str[str.length() - 1]))
      str.erase(str.end() - 1);
    return str;
  }

  /** Read a string of given length from a file.
   * \param str = the string to read to
   * \param length = the bytes to read
   * \param file = the file to read from
   * \return false if could not read length bytes
   */
  inline bool read_string(std::string &str, size_t length, FILE *file)
  {
    assert(length >= 0);

    str.erase();
    if (length == 0)
      return true;
    str.reserve(length);

    // Read the string
    char buf[4096];
    size_t buf_size = 4096;
    while (length > 0) {
      if (length < buf_size)
        buf_size = length;
      size_t ret = fread(buf, buf_size, 1, file);
      if (ret != 1)
        return false;
    
      str.append(buf, buf_size);
      length -= buf_size;
    }

    return true;
  }

  /** Read a line from a file.
   * \param str = the string to read to
   * \param file = the file to read from
   * \param do_chomp = should we chomp the possible newline at the end
   * \return false if end of file reached (\c str will be empty)
   * \throw str::io_error if read failed
   */
  inline bool 
  read_line(std::string &str, FILE *file = stdin, bool do_chomp = false)
  {
    char *ptr;
    char buf[4096];

    str.erase();
    while (1) {
      ptr = fgets(buf, 4096, file);
      if (!ptr)
	break;

      str.append(buf);
      if (str[str.length() - 1] == '\n') {
	break;
      }
    }

    if (ferror(file) || str.length() == 0)
      return false;

    if (do_chomp)
      chomp(str, "\n");

    return true;
  }
  
  /** Read file in a string. 
   *
   * \param file = file to read from
   * \param rewind = rewind to beginning of the file before reading
   * \return string containing the file contents
   * \throw std::runtime_error if io error occurs
   */
  inline std::string
  read_file(FILE *file = stdin, bool rewind = false)
  {
    if (rewind)
      ::rewind(file);

    const size_t buf_size = 4096;
    char buf[buf_size];
    bool end = false;
    std::string str;
    while (!end) {
      size_t ret = fread(buf, 1, buf_size, file);
      if (ret < buf_size) {
        if (ferror(file))
          throw std::runtime_error("str::read_file() read error");
        end = true;
      }
      str.append(buf, ret);
    }
    return str;
  }

  /** Read file in a string. 
   * \param file_name = name of the file to read
   * \return string containing the file contents
   */
  inline std::string
  read_file(std::string file_name)
  {
    std::string str;
    FILE *file = fopen(file_name.c_str(), "r");
    if (file == NULL)
      throw std::runtime_error("str::read_file() open error");
    try {
      str = read_file(file);
    }
    catch (...) {
      fclose(file);
      throw;
    }
    fclose(file);
    return str;
  }

  /** Create temporary file containing a string.  The seek position is
   * rewinded to the beginning of the file.  The file is deleted
   * automatically upon close.  See system's tmpfile() for details. */
  inline FILE*
  temp_file(const std::string &str)
  {
    FILE *file = tmpfile();
    if (file == NULL)
      throw std::runtime_error("temp_file(): tmpfile() failed");
    fputs(str.c_str(), file);
    rewind(file);
    return file;
  }

  /** Pop a delimited substring from the end of a string.  Possible
   * delimiters in the end of string are removed before popping the
   * the substring.  Possible delimiter before the returned substring
   * is removed.
   *
   * \param str = the string to modify
   * \param delims = the delimiting characters
   * \return the last substring delimited by the delimiting characters
   */
  inline std::string
  pop_back(std::string &str, const char *delims = " \t")
  {
    while (!str.empty()) {
      std::string::size_type pos = str.find_last_of(delims);
      if (pos == str.npos) {
        std::string ret = str;
        str.clear();
        return ret;
      }
      if (pos == str.length() - 1) {
        str.resize(str.length() - 1);
        continue;
      }
      std::string ret = str.substr(pos + 1);
      str.resize(pos);
      return ret;
    }
    return str;
  }


  /** Pop a delimited substring from the beginning of a string.
   * Possible delimiters at the beginning of string are removed before
   * popping the the substring.  Possible delimiter after the returned
   * substring is removed.
   *
   * \param str = the string to modify
   * \param delims = the delimiting characters
   * \return the first substring delimited by the delimiting characters
   */
  inline std::string
  pop_front(std::string &str, const char *delims = " \t")
  {
    while (!str.empty()) {
      std::string::size_type pos = str.find_first_of(delims);
      if (pos == str.npos) {
        std::string ret = str;
        str.clear();
        return ret;
      }
      if (pos == 0) {
        str.erase(str.begin());
        continue;
      }
      std::string ret = str.substr(0, pos);
      str.erase(0, pos + 1);
      return ret;
    }
    return str;
  }

  /** Create vector from one parameter. */
  inline std::vector<std::string>
  vec(const std::string &s1)
  {
    return std::vector<std::string>(1, s1);
  }


  /** Create vector from two parameters. */
  inline std::vector<std::string>
  vec(const std::string &s1, const std::string &s2)
  {
    std::vector<std::string> vec(2);
    vec[0] = s1;
    vec[1] = s2;
    return vec;
  }

  /** Create vector from three parameters. */
  inline std::vector<std::string>
  vec(const std::string &s1, const std::string &s2, const std::string &s3)
  {
    std::vector<std::string> vec(3);
    vec[0] = s1;
    vec[1] = s2;
    vec[2] = s3;
    return vec;
  }

  /** Create vector from four parameters. */
  inline std::vector<std::string>
  vec(const std::string &s1, const std::string &s2,
      const std::string &s3, const std::string &s4)
  {
    std::vector<std::string> vec(4);
    vec[0] = s1;
    vec[1] = s2;
    vec[2] = s3;
    vec[3] = s4;
    return vec;
  }


  inline std::string str(const std::vector<int> &vec)
  {
    std::string str;
    for (std::vector<int>::size_type i = 0; i < vec.size(); i++)
      str.append(fmt(256, "%s%d", i > 0 ? " " : "", vec[i]));
    return str;
  }
  

};

#endif /* STR_HH */
