#ifndef LMHISTORY_HH
#define LMHISTORY_HH

#include <vector>

#include "config.hh"
#include "history.hh"

struct LMHistory
{
	class Word
	{
	public:
		/// \brief The default constructor creates a NULL word (word ID and LM ID
		/// -1).
		///
		Word();

		/// \brief Sets the dictionary and language model ID of the word.
		///
		/// \param word_id Used to uniquely identify every word in the vocabulary,
		/// and to retrieve their written form from the dictionary.
		/// \param lm_id Used when computing LM scores. If the language model is
		/// based on classes, this is the ID of the class of the word in question.
		/// If the word is not found in the language model, this is -1.
		///
		void set_ids(int word_id, int lm_id);

		/// \brief Sets the class membership log probability.
		///
		/// \param cm_log_prob If the language model is based on classes, this is
		/// the class membership log probability. When using multiwords, this is the
		/// sum of the individual class membership log probabilities.
		///
		void set_cm_log_prob(float cm_log_prob);

#ifdef ENABLE_MULTIWORD_SUPPORT
		/// \brief Adds a component word. For regular words, the entire word has to
		/// be added as a single component.
		///
		/// \param lm_id Used when computing LM scores. If the language model is
		/// based on classes, this is the ID of the class of the component word.
		/// If the word is not found in the language model, this is -1.
		/// \param cm_log_prob If the language model is based on classes, this is
		/// the class membership log probability of the component. Otherwise 0.
		///
		void add_component(int lm_id);
#endif

		/// \brief Returns the ID of the word in the dictionary.
		///
		/// Word ID is used to uniquely identify every word in the vocabulary, and
		/// to retrieve their written form from the dictionary.
		///
		int word_id() const
		{
			return m_word_id;
		}

		/// \brief Returns the ID of the word (or class) in the language model.
		///
		/// Language model IDs are used when computing LM scores. If the language
		/// model is based on classes, this is the ID of the class of the word in
		/// question. If the word is not found in the language model, this is -1.
		///
		int lm_id() const
		{
			return m_lm_id;
		}

		/// \brief Returns the log probability for the class membership, when the
		/// language model is based on classes. Otherwise returns 0. If this is a
		/// multiword, returns the sum of the class membership log probabilities of
		/// the component words.
		///
		/// We assume each word is a member of at most one class.
		///
		float cm_log_prob() const
		{
			return m_cm_log_prob;
		}

#ifdef ENABLE_MULTIWORD_SUPPORT
		/// \brief Returns the number of components in a multiword (1 for regular
		/// words).
		///
		int num_components() const
		{
			return m_component_lm_ids.size();
		}

		/// \brief Returns the word ID for component \a index in the language model
		/// (or the whole word if this is not a multiword).
		///
		int component_lm_id(int index) const
		{
			return m_component_lm_ids[index];
		}
#endif

	private:
		/// Word ID in the dictionary.
		int m_word_id;

		/// Word ID in the language model.
		int m_lm_id;

		/// The log probability for the class membership, or 0 if not using a class-
		/// based language model.
		float m_cm_log_prob;

#ifdef ENABLE_MULTIWORD_SUPPORT
		/// Word IDs in the language model of the component words.
		std::vector<int> m_component_lm_ids;
#endif
	};

	LMHistory(const Word & last_word, LMHistory * previous);

	const Word & last() const
	{
		return m_last_word;
	}

public:
	LMHistory * previous;
	int reference_count;
	bool printed;
	int word_start_frame;
	int word_first_silence_frame;  // "end frame", initialized to -1.

private:
	// A reference to TokenPassSearch::m_word_lookup table.
	const Word & m_last_word;
};

inline LMHistory::LMHistory(const Word & last_word, LMHistory * previous) :
		previous(previous), reference_count(0), printed(false), word_start_frame(
				0), word_first_silence_frame(-1), m_last_word(last_word)
{
	if (previous)
		hist::link(previous);
}

#endif
