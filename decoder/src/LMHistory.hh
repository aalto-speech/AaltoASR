#ifndef LMHISTORY_HH
#define LMHISTORY_HH

#include <vector>

#include "config.hh"
#include "history.hh"

class LMHistory
{
public:
  class Word
  {
  public:
    struct ID
    {
      /// Word ID in the dictionary.
      int word_id;

      /// Word (or class) ID in the language model, or -1 if not available.
      int lm_id;

      /// Word (or class) ID in the lookahead language model, or 0 if not
      /// available.
      int lookahead_lm_id;
    };

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
    /// \param lookahead_lm_id Used when computing the lookahead score. If
    /// the language model is based on classes, this is the ID of the class
    /// of the word in question. If the word is not found in the lookahead
    /// LM, this is 0.
    ///
    void set_ids(int word_id, int lm_id, int lookahead_lm_id);

    /// \brief Sets the class membership log probability.
    ///
    /// \param cm_log_prob If the language model is based on classes, this is
    /// the class membership log probability. When using multiwords, this is the
    /// sum of the individual class membership log probabilities. Otherwise 0.
    ///
    void set_cm_log_prob(float cm_log_prob);

#ifdef ENABLE_MULTIWORD_SUPPORT
    /// \brief Adds a component word. For regular words, the entire word has to
    /// be added as a single component.
    ///
    /// \param word_id Used to uniquely identify every word in the
    /// vocabulary. We could perform the language model computations without
    /// this information, but it's used for mostly historic reasons in the
    /// lookahead code.
    /// \param lm_id Used when computing LM scores. If the language model is
    /// based on classes, this is the ID of the class of the component word.
    /// If the word is not found in the language model, this is -1.
    /// \param lookahead_lm_id Used when computing the lookahead score. If
    /// the language model is based on classes, this is the ID of the class
    /// of the component word. If the word is not found in the lookahead
    /// LM, this is 0.
    ///
    void add_component(int word_id, int lm_id, int lookahead_lm_id);
#endif

    /// \brief Returns the ID of the word in the dictionary.
    ///
    /// Word ID is used to uniquely identify every word in the vocabulary, and
    /// to retrieve their written form from the dictionary.
    ///
    int word_id() const
    {
      return m_id.word_id;
    }

    /// \brief Returns the ID of the word (or its class) in the language
    /// model.
    ///
    /// Language model IDs are used when computing LM scores. If the language
    /// model is based on classes, this is the ID of the class of the word in
    /// question. If the word is not found in the language model, this is -1.
    ///
    int lm_id() const
    {
      return m_id.lm_id;
    }

    /// \brief Returns the ID of the word (or its class) in the lookahead
    /// language model.
    ///
    /// Lookahead LM IDs are used when computing the lookahead score as the
    /// maximum score of possible word ends in the lookahead model. If the
    /// language model is based on classes, this is the ID of the class of
    /// the word in question. If the word is not found in the lookahead LM,
    /// this is 0.
    ///
    int lookahead_lm_id() const
    {
      return m_id.lookahead_lm_id;
    }

    const ID & id() const
    {
      return m_id;
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
      return m_components.size();
    }

    /// \brief Returns the ID for component \a index.
    ///
    const ID & component(int index) const
    {
      return m_components[index];
    }
#endif

  private:
    ID m_id;

    /// The log probability for the class membership, or 0 if not using a
    /// class-based language model.
    float m_cm_log_prob;

#ifdef ENABLE_MULTIWORD_SUPPORT
    /// Word IDs in the language model of the component words.
    std::vector<ID> m_components;
#endif
  };

  class ConstReverseIterator
  {
  public:
    ConstReverseIterator(const LMHistory * history);

    ConstReverseIterator & operator++();

    const Word::ID & operator*() const;

    const Word::ID * operator->() const;

    bool operator==(const ConstReverseIterator & x) const;

  private:
    const LMHistory * m_history;
#ifdef ENABLE_MULTIWORD_SUPPORT
    int m_component_index;
#endif
  };

  LMHistory(const Word * last_word, LMHistory * previous);
  LMHistory();

  const Word & last() const
  {
    return *last_word;
  }

  ConstReverseIterator rbegin() const;

  ConstReverseIterator rend() const;

  LMHistory * previous;
  int reference_count;
  bool printed;
  int word_start_frame;
  int word_first_silence_frame;  // "end frame", initialized to -1.

  // A reference to TokenPassSearch::m_word_lookup table.
  const Word * last_word;
};

inline LMHistory::LMHistory(const Word * last_word, LMHistory * previous) :
  previous(previous), reference_count(0), printed(false), word_start_frame(
									   0), word_first_silence_frame(-1), last_word(last_word)
{
  if (previous)
    hist::link(previous);
}

inline LMHistory::LMHistory() :
  previous(NULL), reference_count(0), printed(false), word_start_frame(0), 
  word_first_silence_frame(-1), last_word(NULL)
{}

inline LMHistory::ConstReverseIterator &
LMHistory::ConstReverseIterator::operator++()
{
#ifdef ENABLE_MULTIWORD_SUPPORT
  if (m_component_index > 0) {
    --m_component_index;
  }
  else {
    m_history = m_history->previous;
    m_component_index = m_history->last().num_components() - 1;
  }
#else
  m_history = m_history->previous;
#endif
  return *this;
}

inline const LMHistory::Word::ID &
LMHistory::ConstReverseIterator::operator*() const
{
#ifdef ENABLE_MULTIWORD_SUPPORT
  if (m_component_index < 0) {
    // No component words.
    return m_history->last().id();
  }
  else {
    return m_history->last().component(m_component_index);
  }
#else
  return m_history->last().id();
#endif
}

inline const LMHistory::Word::ID *
LMHistory::ConstReverseIterator::operator->() const
{
  return &(operator*());
}

inline bool LMHistory::ConstReverseIterator::operator==(
							const LMHistory::ConstReverseIterator & x) const
{
  if (m_history == NULL)
    return x.m_history == NULL;
  if (m_history != x.m_history)
    return false;
#ifdef ENABLE_MULTIWORD_SUPPORT
  return m_component_index == x.m_component_index;
#else
  return true;
#endif
}

#endif
