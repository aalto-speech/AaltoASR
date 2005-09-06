#include <math.h>
#include <string.h>
#include <cassert>
#include "str.hh"

namespace str {

  bool
  read_line(std::string *str, FILE *file, bool do_chomp)
  {
    char *ptr;
    char buf[4096];

    str->erase();
    while (1) {
      ptr = fgets(buf, 4096, file);
      if (!ptr)
	break;

      str->append(buf);
      if ((*str)[str->length() - 1] == '\n') {
	break;
      }
    }

    if (ferror(file) || str->length() == 0)
      return false;

    if (do_chomp)
      chomp(str);

    return true;
  }

  bool
  read_string(std::string *str, size_t length, FILE *file, bool append)
  {
    if (!append)
      str->erase();
    if (length == 0)
      return true;
    str->reserve(str->length() + length);

    // Read the string one bufferful at time
    char buf[4096];
    size_t buf_size = 4096;
    size_t bytes_read = 0;
    while (length > 0) {

      // Read the buffer full
      if (length < buf_size)
	buf_size = length;
      size_t ret = fread(buf, 1, buf_size, file);
      assert(ret <= length);

      // Error or end of file?
      if (ret < buf_size && ferror(file))
	return false;
      if (ret == 0)
	break;
    
      str->append(buf, ret);
      bytes_read += ret;
      length -= ret;
    }

    // Got anything from the file?
    if (bytes_read == 0)
      return false;

    return true;
  }

  bool read_file(std::string *str, FILE *file, size_t length)
  {
    const int chunk_size = 4096;
    char buf[chunk_size];
    str->clear();

    // Read file buffer by buffer and process
    while (1) {

      // Grow the string for the next chunk
      size_t pos = str->length();
      size_t new_size = pos + chunk_size;
      if (length > 0 && new_size > length)
	new_size = length;
      str->resize(new_size);

      // Read the next chunk
      int bytes_read = fread(buf, 1, str->length() - pos, file);
      for (int i = 0; i < bytes_read; i++)
	(*str)[pos + i] = buf[i];
      if (bytes_read < chunk_size) {
	str->resize(pos + bytes_read);
	if (ferror(file))
	  return false;
	return true;
      }

      if (str->length() == length)
	return true;
    }
  }

  void
  chomp(std::string *str)
  {
    if (str->empty())
      return;

    if ((*str)[str->length() - 1] == '\n')
      str->resize(str->length() - 1);
  }

  void
  clean(std::string *str, const char *chars)
  {
    int i;

    // Clean trailing chars
    for (i = str->length(); i > 0; i--)
      if (!strchr(chars, (*str)[i - 1]))
	break;
    str->erase(i);

    // Clean leading chars
    for (i = 0; i < (int)str->length(); i++)
      if (!strchr(chars, (*str)[i]))
	break;
    str->erase(0, i);
  }

  void
  split(const std::string *str, const char *delims, bool group,
	std::vector<std::string> *fields, int num_fields)
  {
    int begin = 0;
    int end = 0;

    fields->clear();
    while (begin < (int)str->length()) {

      // If 'fields' fields was requested and, this is the last field,
      // include the rest.
      if (num_fields > 0 && (int)fields->size() == num_fields - 1) {
	fields->push_back(str->substr(begin));
	break;
      }

      // Find the string before the next delim
      while (end < (int)str->length() && !strchr(delims, (*str)[end]))
	end++;
      fields->push_back(str->substr(begin, end - begin));

      // Eat the delim or group of delims
      end++;
      if (group)
	while (end < (int)str->length() && strchr(delims, (*str)[end]))
	  end++;

      begin = end;
    }
  }

  void
  split_with_quotes(std::string *str, const char *delims, bool group,
		    std::vector<std::string> *fields)
  {
    enum { NONE, BACKSLASH, SINGLE, DOUBLE } mode = NONE;
    size_t begin = 0;
    size_t src = 0;
    size_t tgt = 0;

    fields->clear();
    while (src < str->length()) {

      // Find the string before the next delim while processing
      // quotes.
      while (src < str->length()) {

	switch ((*str)[src]) {

	case '\\':
	  if (mode == NONE)
	    mode = BACKSLASH;
	  else
	    (*str)[tgt++] = '\\';
	  break;

	case '\'':
	  if (mode == NONE)
	    mode = SINGLE;
	  else if (mode == SINGLE)
	    mode = NONE;
	  else
	    (*str)[tgt++] = '\'';
	  break;

	case '"':
	  if (mode == NONE)
	    mode = DOUBLE;
	  else if (mode == DOUBLE)
	    mode = NONE;
	  else
	    (*str)[tgt++] = '"';
	  break;

	default:
	  (*str)[tgt++] = (*str)[src];
	  if (mode == NONE && strchr(delims, (*str)[src])) {
	    src++;
	    tgt--;
	    goto end_of_field;
	  }
	  break;
	}
	if (mode == BACKSLASH && (*str)[src] != '\\')
	  mode = NONE;
	src++;
      }
	
    end_of_field:
      fields->push_back(str->substr(begin, tgt - begin));
      tgt++;

      // Eat the delim or group of delims
      if (group)
	while (src < str->length() && strchr(delims, (*str)[src]))
	  src++;
      begin = tgt;
    }
  }

  long
  str2long(const char *str, bool *ok)
  {
    char *endptr;

    long value = strtol(str, &endptr, 10);
    if (value == LONG_MIN || value == LONG_MAX) {
      fprintf(stderr, "str2long(): value out of range\n");
      exit(1);
    }

    if (*str == '\0' || *endptr != '\0')
      *ok = false;

    return value;
  }

  double
  str2float(const char *str, bool *ok)
  {
    char *endptr;

    float value = strtod(str, &endptr);
#ifdef HUGE_VALF
    if (value == HUGE_VALF || value == -HUGE_VALF) {
      fprintf(stderr, "str2float(): value out of range\n");
      exit(1);
    }
#else
    if (value == HUGE_VAL || value == -HUGE_VAL) {
      fprintf(stderr, "str2float(): value out of range\n");
      exit(1);
    }
#endif

    if (*str == '\0' || *endptr != '\0')
      *ok = false;

    return value;
  }

  long 
  str2long(const std::string *str, bool *ok) 
  { 
    return str2long(str->c_str(), ok); 
  }

  double 
  str2float(const std::string *str, bool *ok) 
  { 
    return str2float(str->c_str(), ok); 
  }

}
