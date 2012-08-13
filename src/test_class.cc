#include <cstdlib>
#include <iostream>
#include <sstream>
#include <cmath>
#include <cassert>

#include "WordClasses.hh"

using namespace std;

int main(int argc, char * argv[])
{
	WordClasses classes;
	istringstream def("CLASS-47210 0.01 own\nCLASS-47210 cabinet's\nCLASS-47215 0.0508012 apologize");
	def >> classes;
	const WordClasses::Membership & m = classes.get_membership("own");
	assert(m.log_prob == -2);
	assert(classes.get_class_name(m.class_id) == "CLASS-47210");
	return EXIT_SUCCESS;
}
