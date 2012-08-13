#ifndef WORDCLASSES_HH
#define WORDCLASSES_HH

#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <stdexcept>

class WordClasses
{
private:
	typedef std::vector<std::string> class_names_type;

public:
	typedef class_names_type::size_type class_id_type;

	struct ParseError : public std::runtime_error {
		ParseError(const std::string & what_arg) :
			std::runtime_error(what_arg) {}
	};

	struct Membership {
		class_id_type class_id;
		float log_prob;
	};

	WordClasses();

	void add_class_expansion(
			const std::string & class_name,
			float probability,
			const std::string & expansion);

	const Membership & get_membership(const std::string & word) const;

	const std::string & get_class_name(class_id_type class_id) const;

private:
	typedef std::map<std::string, Membership> memberships_type;

	class_names_type m_class_names;
	memberships_type m_memberships;
};

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
/// \exception ParseError If unable to parse a definition.
///
std::istream & operator>>(std::istream & is, WordClasses & wc);

#endif
