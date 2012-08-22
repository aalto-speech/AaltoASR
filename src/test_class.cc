#include <cstdlib>
#include <iostream>
#include <sstream>
#include <cmath>
#include <cassert>

#include "WordClasses.hh"
#include "Toolbox.hh"

using namespace std;

bool log_probs_equal(float a, float b)
{
    return fabs(a - b) < 0.00001;
}

void test_search()
{
	Toolbox t;
	t.set_verbose(1);
	t.select_decoder(0);
	t.hmm_read("/share/puhe/funesomo12/models/icsifsh-pPMWF-stg120k_13.7.2012_25.ph");
	t.lex_read("/share/puhe/funesomo12/dictionaries/cmu-cmups.lex");
	t.read_word_classes("/share/puhe/funesomo12/models/myicsifshswb.classes");
	t.ngram_read("/share/puhe/funesomo12/models/myicsifshswb-c.4bo.bin", 1);

	const std::vector<LMHistory::Word> & words = t.tp_search().get_word_repository();
	const Vocabulary & voc = t.tp_search().get_vocabulary();
	assert(t.tp_search().get_word_classes() != NULL);
	const WordClasses & classes = *t.tp_search().get_word_classes();
	assert(t.tp_search().get_ngram() != NULL);
	const TreeGram & lm = *t.tp_search().get_ngram();

	int costa_id = voc.word_index("costa");
	LMHistory::Word costa_word = words[costa_id];
	assert(costa_word.word_id() == costa_id);
	assert(log_probs_equal(costa_word.cm_log_prob(), -1.0693265));
	int costa_class_id = classes.get_membership(costa_id).class_id;
	assert(classes.get_class_name(costa_class_id) == "CLASS-33515");
	int costa_lm_id = lm.word_index("CLASS-33515");
	assert(costa_word.lm_id() == costa_lm_id);
	assert(costa_word.num_components() == 1);
	assert(costa_word.component_lm_id(0) == costa_lm_id);

	int rica_id = voc.word_index("rica");
	LMHistory::Word rica_word = words[rica_id];
	assert(rica_word.word_id() == rica_id);
	assert(log_probs_equal(rica_word.cm_log_prob(), -1.375761233));
	int rica_class_id = classes.get_membership(rica_id).class_id;
	assert(classes.get_class_name(rica_class_id) == "CLASS-00710");
	int rica_lm_id = lm.word_index("CLASS-00710");
	assert(rica_word.lm_id() == rica_lm_id);
	assert(rica_word.num_components() == 1);
	assert(rica_word.component_lm_id(0) == rica_lm_id);

	int costa_rica_id = voc.word_index("costa_rica");
	LMHistory::Word costa_rica_word = words[costa_rica_id];
	assert(costa_rica_word.word_id() == costa_rica_id);
	assert(log_probs_equal(costa_rica_word.cm_log_prob(), -2.445087733));
	assert(classes.get_membership(costa_rica_id).class_id == -1);
	assert(costa_rica_word.lm_id() == -1);
	assert(costa_rica_word.num_components() == 2);
	assert(costa_rica_word.component_lm_id(0) == costa_lm_id);
	assert(costa_rica_word.component_lm_id(1) == rica_lm_id);
}

int main(int argc, char * argv[])
{
	try {
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

		test_search();

		cout << "All good." << endl;
		return EXIT_SUCCESS;
	}
	catch (TPNowayLexReader::UnknownHmm & e) {
		cerr << "Unknown phone in lexicon: " << e.phone() << endl;
		return EXIT_FAILURE;
	}
}
