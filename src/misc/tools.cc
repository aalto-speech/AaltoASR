#include <string.h>
#include <assert.h>
#include "tools.hh"

bool
read_line(std::string *str, FILE *file)
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

  return true;
}

bool
read_string(std::string *str, size_t length, FILE *file)
{
  assert(length >= 0);

  str->erase();
  if (length == 0)
    return true;
  str->reserve(length);

  // Read the string
  char buf[4096];
  size_t buf_size = 4096;
  while (length > 0) {
    if (length < buf_size)
      buf_size = length;
    size_t ret = fread(buf, buf_size, 1, file);
    if (ret != 1)
      return false;
    
    str->append(buf, buf_size);
    length -= buf_size;
  }

  return true;
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
  for (i = 0; i < str->length(); i++)
    if (!strchr(chars, (*str)[i]))
      break;
  str->erase(0, i);
}

void
split(const std::string &str, const char *delims, bool group,
      std::vector<std::string> *strings)
{
  int begin = 0;
  int end = 0;

  strings->clear();
  while (begin < str.length()) {

    // Find the string before the next delim
    while (end < str.length() && !strchr(delims, str[end]))
      end++;
    strings->push_back(str.substr(begin, end - begin));

    // Eat the delim or group of delims
    end++;
    if (group)
      while (end < str.length() && strchr(delims, str[end]))
	end++;

    begin = end;
  }
}
