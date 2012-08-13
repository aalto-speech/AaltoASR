#include <iostream>
#include <sstream>
#include <algorithm>
#include <cmath>

#include "WordClasses.hh"

using namespace std;

WordClasses::WordClasses()
{
}

void WordClasses::add_class_expansion(
		const std::string & class_name,
		float probability,
		const std::string & expansion)
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
	m_memberships[expansion] = membership;
}

const WordClasses::Membership & WordClasses::get_membership(
		const std::string & word) const
{
	memberships_type::const_iterator iter = m_memberships.find(word);
	if (iter == m_memberships.end()) {
		throw invalid_argument("WordClasses::get_membership");
	}
	return iter->second;
}

const std::string & WordClasses::get_class_name(
		class_id_type class_id) const
{
	return m_class_names.at(class_id);
}

std::istream & operator>>(std::istream & is, WordClasses & wc)
{
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
		wc.add_class_expansion(class_name, probability, expansion);
	}
	return is;
}
