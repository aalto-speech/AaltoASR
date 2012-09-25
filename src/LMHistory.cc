#include "LMHistory.hh"

using namespace std;

LMHistory::Word::Word() :
		m_word_id(-1), m_lm_id(-1), m_lookahead_lm_id(0), m_cm_log_prob(0)
{
}

void LMHistory::Word::set_ids(int word_id, int lm_id, int lookahead_lm_id)
{
	m_word_id = word_id;
	m_lm_id = lm_id;
	m_lookahead_lm_id = lookahead_lm_id;
}

void LMHistory::Word::set_cm_log_prob(float cm_log_prob)
{
	m_cm_log_prob = cm_log_prob;
}

#ifdef ENABLE_MULTIWORD_SUPPORT
void LMHistory::Word::add_component(int word_id, int lm_id, int lookahead_lm_id)
{
	Component x;
	x.word_id = word_id;
	x.lm_id = lm_id;
	x.lookahead_lm_id = lookahead_lm_id;
	m_components.push_back(x);
}
#endif
