#include "LMHistory.hh"

using namespace std;

LMHistory::Word::Word() :
		m_cm_log_prob(0)
{
	m_id.word_id = -1;
	m_id.lm_id = -1;
	m_id.lookahead_lm_id = 0;
}

void LMHistory::Word::set_ids(int word_id, int lm_id, int lookahead_lm_id)
{
	m_id.word_id = word_id;
	m_id.lm_id = lm_id;
	m_id.lookahead_lm_id = lookahead_lm_id;
}

void LMHistory::Word::set_cm_log_prob(float cm_log_prob)
{
	m_cm_log_prob = cm_log_prob;
}

#ifdef ENABLE_MULTIWORD_SUPPORT
void LMHistory::Word::add_component(int word_id, int lm_id, int lookahead_lm_id)
{
	ID component_id;
	component_id.word_id = word_id;
	component_id.lm_id = lm_id;
	component_id.lookahead_lm_id = lookahead_lm_id;
	m_components.push_back(component_id);
}
#endif

LMHistory::ConstReverseIterator LMHistory::rbegin() const
{
	return ConstReverseIterator(this);
}

LMHistory::ConstReverseIterator LMHistory::rend() const
{
	return ConstReverseIterator(NULL);
}

LMHistory::ConstReverseIterator::ConstReverseIterator(const LMHistory * history) :
		m_history(history)
{
#ifdef ENABLE_MULTIWORD_SUPPORT
	if (history != NULL) {
		m_component_index = history->last().num_components() - 1;
	}
	else {
		m_component_index = -1;
	}
#endif
}
