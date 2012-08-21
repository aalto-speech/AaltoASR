#include <cstdlib>
#include <iostream>
#include <sstream>
#include <cmath>
#include <cassert>

#include "WordClasses.hh"
#include "Toolbox.hh"

using namespace std;

int main(int argc, char * argv[])
{
	Vocabulary voc;
	voc.add_word("own");
	voc.add_word("cabinet's");
	voc.add_word("apologize");

	WordClasses classes;
	istringstream def("CLASS-47210 0.01 own\nCLASS-47210 cabinet's\nCLASS-47215 0.0508012 apologize");
	classes.read(def, voc);

	const WordClasses::Membership & m = classes.get_membership(1);
	assert(m.log_prob == -2);
	assert(classes.get_class_name(m.class_id) == "CLASS-47210");

	Toolbox t;
	t.select_decoder(0);
	t.hmm_read("/share/puhe/jpylkkon/hmms/wsj_284_ml_gain4000_occ400_17.11.2010_25.ph");
	t.lex_read("/share/puhe/jpylkkon/wsj/lm_20k.lex");
	t.read_word_classes("/share/puhe/funesomo12/classes/myicsifshswb.classes");

	cout << "All good." << endl;
	return EXIT_SUCCESS;
}
