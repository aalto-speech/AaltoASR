#ifndef TOOLS_HH
#define TOOLS_HH

#include <string>
#include <vector>
#include <stdio.h>

bool read_line(std::string *str, FILE *file = stdin);
bool read_string(std::string *str, size_t length, FILE *file = stdin);
void chomp(std::string *str);
void clean(std::string *str, const char *chars);
void split(const std::string &str, const char *chars, bool group, 
	   std::vector<std::string> *strings);

#endif /* TOOLS_HH */
