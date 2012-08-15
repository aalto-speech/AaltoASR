#ifndef WORDCLASSES_HH
#define WORDCLASSES_HH

#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <stdexcept>

#include "Vocabulary.hh"

class WordClasses
{
private:
	typedef std::vector<std::string> class_names_type;

public:
	typedef class_names_type::size_type class_id_type;
	typedef int word_id_type;

	struct ParseError : public std::runtime_error {
		ParseError(const std::string & what_arg) :
			std::runtime_error(what_arg) {}
	};

	struct Membership {
		class_id_type class_id;
		float log_prob;
	};

	WordClasses();

	bool empty() const
	{ return m_class_names.empty(); }

	int num_oovs() const
	{ return m_num_oovs; }

	/// \brief Reads word class definitions from a text stream.
	///
	/// Reads the possible expansions of word classes and their respective
	/// probabilities. Each expansion appears on a separate line as
	///
	///   class [p] word1 word2 ...
	///
	/// where class names a word class, p gives the probability for the class
	/// expansion, and word1 word2 ... defines the word string that the class
	/// expands to. If p is omitted it is assumed to be 1.
	///
	/// \param is Input stream for reading the textual definitions.
	/// \param vocabulary Vocabulary for translating words in word IDs.
	///
	/// \exception ParseError If unable to parse a definition.
	///
	void read(std::istream & is, const Vocabulary & vocabulary);

	void add_class_expansion(
			const std::string & class_name,
			float probability,
			word_id_type word_id);

	const Membership & get_membership(word_id_type word_id) const;

	const std::string & get_class_name(class_id_type class_id) const;

private:
	typedef std::map<word_id_type, Membership> memberships_type;

	class_names_type m_class_names;

	/// A mapping from word ID to class membership.
	memberships_type m_memberships;

	/// Number of OOVs encountered in class definitions.
	int m_num_oovs;
};

#endif
