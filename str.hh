#ifndef STR_HH
#define STR_HH

#include <string>
#include <vector>
#include <stdio.h>

/** Functions for handling strings of the Standard Template Library. */
namespace str {

  /** Format a string. */
  std::string fmt(size_t size, const char *fmt, ...);

  /** Read a line from a file.
   * \param str = the string to read to
   * \param file = the file to read from
   * \param do_chomp = should we chomp the possible newline at the end
   * \return false if no more lines in the file
   */
  bool read_line(std::string *str, FILE *file = stdin, bool do_chomp = false);

  /** Read a string of given length from a file.
   * \param str = the string to read to
   * \param length = the bytes to read
   * \param file = the file to read from
   * \return false if could not read length bytes
   */
  bool read_string(std::string *str, size_t length, FILE *file = stdin);

  /** Read a file to string.
   * \param str = the string to read to
   * \param file = the file to read from
   * \param length = the maximum size of the file (0 = read all)
   * \return false if read failed for some reason (see errno for details)
   */
  bool read_file(std::string *str, FILE *file, size_t length = 0);

  /** Remove the possible trailing newline from a string. */
  void chomp(std::string *str);

  /** Remove leading and trailing characters from a string. 
   * \param str = the string to be cleaned
   * \param chars = the characters to clean from the string
   */
  void clean(std::string *str, const char *chars);

  /** Split a string to fields.  If \c num_fields is positive, it
   * specifies the maximum number of fields.  If the line contains
   * more fields, they are included in the last field as such (also
   * possible delimiters without processing).
   *
   * \param str = the string to be splitted
   * \param delims = the delimiter characters
   * \param group = group subsequent delimiters as one delimeter
   * \param fields = the vector containing the resulting fields
   * \param num_fields = the maximum number of fields to return
   */
  void
  split(const std::string *str, const char *delims, bool group,
	std::vector<std::string> *fields, int num_fields = 0);

  /** Split a string to fields while processing quotes \, ', ".  
   *
   * The function modifies \c str by removing quotes and preserving
   * only the first delimiter from groups of delimiters. All quotation
   * marks can be used to quote other quotation marks, and backslash
   * can be used to quote backslash too.
   *
   * \warning Delimeters must not include quotation chars.
   *
   * \param str = the string to be splitted
   * \param delims = the delimiter characters
   * \param group = group subsequent delimiters as one delimeter
   * \param fields = the vector containing the resulting fields
   */
  void
  split_with_quotes(std::string *str, const char *delims, bool group,
		    std::vector<std::string> *fields);

  /** Convert a string to numeric value. 
   * \param str = the string to convert
   * \param ok = set to false if the whole string could not be converted, 
   * otherwise the value is not changed
   * \return the numeric value.
   */
  /*@{*/
  long str2long(const char *str, bool *ok);
  double str2float(const char *str, bool *ok);
  long str2long(const std::string *str, bool *ok);
  double str2float(const std::string *str, bool *ok);
  /*@}*/
};

#endif /* STR_HH */
