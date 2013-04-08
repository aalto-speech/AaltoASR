#include <iostream>
#include <sstream>
#include <algorithm>
#include <cmath>

#include "WordClasses.hh"

using namespace std;

WordClasses::WordClasses()
{
}

void WordClasses::read(std::istream & is, Vocabulary & vocabulary)
{
	m_memberships.resize(vocabulary.num_words());

	while (is.good()) {
		string line;
		getline(is, line);
		istringstream iss(line);

		string class_name;
		iss >> class_name;
		if (!iss.good()) {
			continue;  // Ignore empty lines.
		}

		float probability;
		iss >> probability;
		if (!iss.good()) {
			// Not an error if probability is missing.
			probability = 1;
			iss.clear();
		}

		ws(iss);
		if (!iss.good()) {
			// Expansion is missing.
			throw WordClasses::ParseError("Error reading class definitions.");
		}
		string expansion;
		getline(iss, expansion);

		int word_id = vocabulary.add_word(expansion);
		add_class_expansion(class_name, probability, word_id);
	}
}

void WordClasses::add_class_expansion(
		const std::string & class_name,
		float probability,
		word_id_type word_id)
{
	Membership membership;

	class_names_type::const_iterator iter =
			find(m_class_names.begin(), m_class_names.end(), class_name);
	if (iter == m_class_names.end()) {
		membership.class_id = m_class_names.size();
		m_class_names.push_back(class_name);
	}
	else {
		membership.class_id = iter - m_class_names.begin();
	}

	membership.log_prob = log10(probability);

	if (word_id >= m_memberships.size()) {
		m_memberships.resize(word_id + 1);
	}
	m_memberships[word_id] = membership;
}

const WordClasses::Membership & WordClasses::get_membership(
		word_id_type word_id) const
{
	return m_memberships.at(word_id);
}

const std::string & WordClasses::get_class_name(
		class_id_type class_id) const
{
	return m_class_names.at(class_id);
}
