#include <cstddef>  // NULL
#include <stdio.h>
#include <algorithm>
#include <stdexcept>
#include <iostream>

#include "TPLexPrefixTree.hh"

using namespace std;

int safe_tolower(int c)
{
	if (c == 'Å')
		return 'å';
	else if (c == 'Ä')
		return 'ä';
	else if (c == 'Ö')
		return 'ö';
	return tolower(c);
}

int safe_toupper(int c)
{
	if (c == 'å')
		return 'Å';
	else if (c == 'ä')
		return 'Ä';
	else if (c == 'ö')
		return 'Ö';
	return toupper(c);
}

TPLexPrefixTree::TPLexPrefixTree(std::map<std::string, int> &hmm_map,
		std::vector<Hmm> &hmms) :
	m_words(0), m_verbose(0), m_lm_lookahead(0), m_silence_is_word(true),
			m_hmm_map(hmm_map), m_hmms(hmms)
{
	initialize_nodes();
	m_lm_buf_count = 0;
	m_cross_word_triphones = true;
	m_optional_short_silence = true;
	m_short_silence_state = NULL;
	m_word_boundary_id = -1;
	m_ignore_case = false;
}

struct delete_node
{
	void operator()(TPLexPrefixTree::Node * ptr)
	{
		if (ptr != NULL)
			delete ptr;
	}
};

TPLexPrefixTree::~TPLexPrefixTree()
{
	for_each(m_nodes.begin(), m_nodes.end(), delete_node());
	m_nodes.clear();
}

void TPLexPrefixTree::set_lm_lookahead(int lm_lookahead)
{
	if (m_silence_node != NULL) {
		cerr << "WARNING: TPLexPrefixTree::set_lm_lookahead() called after reading lexicon." << endl;
		cerr << "WARNING: Lookahead setting will not be apply to the existing lexicon." << endl;
	}
	m_lm_lookahead = lm_lookahead;
}

void TPLexPrefixTree::initialize_lex_tree(void)
{
	m_words = 0;
	initialize_nodes();
	//	m_lm_lookahead = 0;

	m_lm_buf_count = 0;

	m_short_silence_state = NULL;
	m_word_boundary_id = -1;

	free_cross_word_network_connection_points();
	m_silence_arcs.clear();

	if (m_cross_word_triphones)
		create_cross_word_network();
}

void TPLexPrefixTree::add_word(std::vector<Hmm*> &hmm_list, int word_id)
{
	int i, j, k;
	node_vector hmm_state_nodes;
	node_vector source_nodes, sink_nodes;
	std::vector<float> source_trans_log_probs;
	std::vector<float> sink_trans_log_probs;
	int word_end;
	Arc end_arc;
	bool silence = false;

	// NOTE! HMM states are indexed with respect to their mixture model.
	// This is a quick solution to allow tied HMM states. The right way
	// to do this would be to load HMMs so that different HMMs would
	// really share the same states. Now it is merely assumed that the
	// (local) transitions in the states with the same mixtures are the same.

	// Invariants: source_nodes.size() == source_trans_log_probs.size(),
	//             sink_nodes.size() == sink_trans_log_probs.size()

	if (hmm_list.size() == 1 && hmm_list[0]->states.size() == 3
			&& hmm_list[0]->label == "_") {
		// Short silence
		m_short_silence_state = &hmm_list[0]->state(2);
		// Check the self transition
		assert( m_short_silence_state->transitions.size() == 2 &&
				m_short_silence_state->transitions[0].target == 2 );
		return;
	}

	if (hmm_list.size() == 1 && hmm_list[0]->label == "__")
		silence = true;

	source_nodes.push_back(m_root_node);
	source_trans_log_probs.push_back(0);
	word_end = -1;
	for (i = 0; i < hmm_list.size(); i++) {
		assert( source_nodes.size() > 0 );

		if (i == hmm_list.size() - 1) // Last node
		{
			// Added a check that the last phone is a triphone (label is five characters).
			//   2012-05-07 / SE
			if (m_cross_word_triphones && !silence && (hmm_list[i]->label.size() == 5)) {
				// Link to cross word network.
				// First, add a null node with the word ID.
				Node *wid_node = new Node(word_id);
				Arc temp_arc;
				std::vector<Arc> new_arcs;
				wid_node->node_id = m_nodes.size();
				wid_node->flags = NODE_USE_WORD_END_BEAM;
				if (i == 0)
					wid_node->flags |= NODE_FIRST_STATE_OF_WORD;
				m_nodes.push_back(wid_node);
				temp_arc.next = wid_node;
				if (i == 0)
					wid_node->flags |= NODE_FIRST_STATE_OF_WORD;
				for (j = 0; j < source_nodes.size(); j++) {
					temp_arc.log_prob = source_trans_log_probs[j];
					source_nodes[j]->arcs.push_back(temp_arc);
				}

				// Link null node to cross word network
				std::string temp1(hmm_list[i]->label, 0, 1);
				std::string temp2(hmm_list[i]->label, 2, 1);
				std::string key = temp1 + temp2;
				link_node_to_fan_network(key, new_arcs, true, false, 0);
				for (j = 0; j < new_arcs.size(); j++)
					wid_node->arcs.push_back(new_arcs[j]);
				if (i == 0) {
					// If word has only one HMM, make another instance of the word
					// inside the cross word network.
					add_single_hmm_word_for_cross_word_modeling(hmm_list[0],
							word_id);
				}
				// Added a check that the first phone is a triphone (label is five characters).
				//   2012-05-07 / SE
				else if ((i == 1) && (hmm_list[0]->label.size() == 5)) {
					// Word has two HMMs, mark the null node as a connection
					// point for linking from cross word network
					add_fan_in_connection_node(wid_node, hmm_list[0]->label);
				}
				source_nodes.clear();
				source_trans_log_probs.clear();
				break; // Skip last state (handled in the cross word network).
			}
			if (!silence || m_silence_is_word)
				word_end = word_id;
		}

		// Set up a table for storing the nodes corresponding to the HMM states
		hmm_state_nodes.clear();
		hmm_state_nodes.insert(hmm_state_nodes.end(),
				hmm_list[i]->states.size() - 2, NULL);
		sink_nodes.clear();
		sink_trans_log_probs.clear();

		// Expand previous sink nodes with new source transitions
		for (j = 0; j < source_nodes.size(); j++) // Iterate previous sink nodes
		{
			for (k = 0; k < hmm_list[i]->state(0).transitions.size(); k++) {
				expand_lexical_tree(source_nodes[j], hmm_list[i],
						hmm_list[i]->state(0).transitions[k],
						source_trans_log_probs[j], word_end, hmm_state_nodes,
						sink_nodes, sink_trans_log_probs, 0);
			}
		}

		// Added a check that the first phone is a triphone (label is five characters).
		//   2012-05-07 / SE
		if (m_cross_word_triphones && (i == 1) && (hmm_list.size() > 2) && (hmm_list[0]->label.size() == 5)) {
			// Mark the state for linking cross word network back to the
			// lexical tree.
			add_fan_in_connection_node(hmm_state_nodes[0], hmm_list[0]->label);
		}
		if (silence && i == 0) {
			m_silence_node = hmm_state_nodes[0];
			m_silence_node->flags |= NODE_SILENCE_FIRST;
			if (m_silence_is_word)
				m_silence_node->flags |= NODE_FIRST_STATE_OF_WORD;
		}
		else if (i == 0)
			hmm_state_nodes[0]->flags |= NODE_FIRST_STATE_OF_WORD;

		// Expand other states, from left to right
		for (j = 2; j < hmm_list[i]->states.size(); j++) {
			assert( hmm_state_nodes[j-2] != NULL );
			for (k = 0; k < hmm_list[i]->state(j).transitions.size(); k++) {
				expand_lexical_tree(hmm_state_nodes[j - 2], hmm_list[i],
						hmm_list[i]->state(j).transitions[k], 0, word_end,
						hmm_state_nodes, sink_nodes, sink_trans_log_probs, 0);
			}
		}
		source_nodes = sink_nodes;
		source_trans_log_probs = sink_trans_log_probs;
	}

	// Mark end of the word node (sink state), add to the word end list
	// and link to the end node.
	if (!m_cross_word_triphones || silence) {
		assert( source_nodes.size() == 1 ); // The sink state
		end_arc.next = (silence && m_cross_word_triphones ? m_root_node
				: m_end_node);
		end_arc.log_prob = get_out_transition_log_prob(source_nodes.front());
		source_nodes.front()->arcs.push_back(end_arc);
		if (silence)
			m_last_silence_node = hmm_state_nodes.back();
	}

	m_words = word_id + 1;
}

void TPLexPrefixTree::expand_lexical_tree(Node *source, Hmm *hmm,
		HmmTransition &t, float cur_trans_log_prob, int word_end,
		node_vector &hmm_state_nodes, node_vector &sink_nodes, std::vector<
				float> &sink_trans_log_probs, unsigned short flags)
{
	int i;

	// Check if we are going to sink state
	if (hmm->is_sink(t.target)) {
		if (word_end == -1) {
			// Mark the source node as a next sink and return
			sink_nodes.push_back(source);
			sink_trans_log_probs.push_back(cur_trans_log_prob + t.log_prob);
		}
		else {
			// Make explicit sink state for word end
			if (sink_nodes.size() == 0) {
				Node * sink = new Node(word_end);
				sink->node_id = m_nodes.size();
				sink->flags = NODE_USE_WORD_END_BEAM | flags;
				m_nodes.push_back(sink);
				sink_nodes.push_back(sink);
			}
			// Add new arc and return
			Arc temp;
			temp.next = sink_nodes.front();
			temp.log_prob = cur_trans_log_prob + t.log_prob;
			source->arcs.push_back(temp);
		}
		return;
	}

	// Find out if the node is already linked
	for (i = 0; i < source->arcs.size(); i++) {
		if (source->arcs[i].next->state != NULL
				&& source->arcs[i].next->state->model
						== hmm->state(t.target).model) {
			// Already linked to something, check the link is correct
			if (hmm_state_nodes[t.target - 2] == NULL
					|| hmm_state_nodes[t.target - 2] == source->arcs[i].next) {
				// Link ok, assume the transition is OK so nothing to be done
				hmm_state_nodes[t.target - 2] = source->arcs[i].next;
				return;
			}
		}
	}

	// The node has not been linked
	if (hmm_state_nodes[t.target - 2] == NULL) {
		// The state does not even exist, so create it.
		// FIXME! Can the order of processing the transitions and source states
		// affect the existence of a node?

		hmm_state_nodes[t.target - 2] = new Node(-1, &hmm->state(t.target));
		hmm_state_nodes[t.target - 2]->node_id = m_nodes.size();
		hmm_state_nodes[t.target - 2]->flags = flags;
		m_nodes.push_back(hmm_state_nodes[t.target - 2]);
	}

	// Add new arc
	Arc temp;
	temp.next = hmm_state_nodes[t.target - 2];
	temp.log_prob = cur_trans_log_prob + t.log_prob;
	source->arcs.push_back(temp);
}

void TPLexPrefixTree::finish_tree(void)
{
	assert( m_silence_node != NULL );

	if (m_cross_word_triphones) {
		// Link the fan points to create a cross word network
		string_to_nodes_map::const_iterator fan_out_it =
				m_fan_out_last_nodes.begin();
		while (fan_out_it != m_fan_out_last_nodes.end()) {
			const node_vector & nlist = fan_out_it->second;
			for (int i = 0; i < nlist.size(); i++) {
				link_fan_out_node_to_fan_in(nlist[i], fan_out_it->first);
			}
			++fan_out_it;
		}

		link_fan_in_nodes();
		// FIXME: LM lookahead for one phoneme morphemes inside the
		// cross word network is missing!
	}
	else {
		// Replace arcs to the end node in the word ends to the root to make the
		// tree re-entrant.
		for (int i = 0; i < m_nodes.size(); i++) {
			for (int j = 0; j < m_nodes[i]->arcs.size(); j++) {
				if (m_nodes[i]->arcs[j].next == m_end_node)
					m_nodes[i]->arcs[j].next = m_root_node;
			}
		}
	}

	// Link silence arcs
	for (int i = 0; i < m_silence_arcs.size(); i++) {
		m_silence_arcs[i].node->arcs[m_silence_arcs[i].arc_index].next
				= m_silence_node;
	}
	m_silence_arcs.clear();

	// Link start node to silence node
	Arc arc;
	arc.next = m_silence_node;
	arc.log_prob = 0;
	m_start_node->arcs.push_back(arc);

	// Propagate word ID:s towards the root node and add LM lookahead
	// list to every branch (if this option is used)
	for (int i = 0; i < m_root_node->arcs.size(); i++)
		post_process_lex_branch(m_root_node->arcs[i].next, NULL);

	if (m_cross_word_triphones) {
		string_to_nodes_map::const_iterator it;
		int i;
		it = m_fan_in_entry_nodes.begin();
		while (it != m_fan_in_entry_nodes.end()) {
			const node_vector & nlist = it->second;
			for (i = 0; i < nlist.size(); i++) {
				if (!post_process_fan_triphone(nlist[i], NULL, true)) {
					if (m_verbose > 1)
						fprintf(stderr, "Removed a fan-in node from key %s\n",
								(*it).first.c_str());
				}
			}
			++it;
		}
		/*it = m_fan_out_last_nodes.begin(); // NOTE: Only last nodes!
		 while (it != m_fan_out_last_nodes.end())
		 {
		 nlist = (*it).second;
		 for (i = 0; i < nlist->size(); i++)
		 {
		 if (!post_process_fan_triphone((*nlist)[i], NULL, false))
		 fprintf(stderr, "Removed a fan-out node from key %s\n",
		 (*it).first.c_str());
		 }
		 ++it;
		 }*/
	}

	// FIXME! Should the word id lists for LM lookahead be sorted to increase
	// processor cache utility?

	analyze_cross_word_network();

	int nodes = 0, arcs = 0;
	count_prefix_tree_size(m_root_node, &nodes, &arcs);
	if (m_verbose > 1) {
		fprintf(stderr, "Prefix tree: %d nodes, %d arcs\n", nodes, arcs);
	}

	free_cross_word_network_connection_points();
	debug_prune_dead_ends(m_root_node);

	// fprintf(stderr, "WARNING: silence loop not added\n");
	// debug_add_silence_loop();
}

void TPLexPrefixTree::post_process_lex_branch(Node *node,
		std::vector<int> *lm_la_list)
{
	int out_trans_count;
	Node *original_node = node;
	Node *real_next = NULL;
	node_vector prev_nodes;
	int i;

	if (!m_silence_is_word && node == m_silence_node)// Word LM, no word ID
	{
		return;
	}

	assert(((node->flags)&(NODE_FAN_IN|NODE_FAN_OUT)) == 0);

	// Skip nodes without branches
	for (;;) {
		if (node->word_id != -1) {
			// Final node
			int word_id = node->word_id;
			if (prev_nodes.size() > 0) {
				original_node->word_id = word_id;
				//original_node->flags |= NODE_USE_WORD_END_BEAM;
				node->word_id = -1;
				node->flags |= NODE_AFTER_WORD_ID | NODE_USE_WORD_END_BEAM;
				if (node->state == NULL) {
					if (node->arcs.size() == 1
							&& prev_nodes.back()->arcs.size() == 2) {
						// Final node was a NULL node, we don't need it anymore
						if (prev_nodes.back()->arcs[0].next
								!= prev_nodes.back()) {
							prev_nodes.back()->arcs[0].next
									= node->arcs[0].next;
							prev_nodes.back()->arcs[0].log_prob
									+= node->arcs[0].log_prob;
						}
						else {
							prev_nodes.back()->arcs[1].next
									= node->arcs[0].next;
							prev_nodes.back()->arcs[1].log_prob
									+= node->arcs[0].log_prob;
						}
					}
				}
				for (i = 1; i < prev_nodes.size(); i++)
					prev_nodes[i]->flags |= NODE_AFTER_WORD_ID
							| NODE_USE_WORD_END_BEAM;
			}
			if (m_lm_lookahead && lm_la_list != NULL) {
				// Add word to LM lookahead list
				lm_la_list->push_back(word_id);
			}
			return;
		}

		out_trans_count = 0;
		for (i = 0; i < node->arcs.size(); i++) {
			if (node->arcs[i].next != node) // Skip self transitions
			{
				real_next = node->arcs[i].next;
				if (++out_trans_count > 1)
					break;
			}
		}
		prev_nodes.push_back(node);
		if (out_trans_count < 2 && (!m_cross_word_triphones
				|| !(real_next->flags & NODE_FAN_IN_CONNECTION))) {
			node = real_next; // Only one transition to another node, skip
		}
		else
			break;
	}
	for (i = 0; i < node->arcs.size(); i++) {
		if (node->arcs[i].next != node) {
			post_process_lex_branch(node->arcs[i].next,
					&original_node->possible_word_id_list);
		}
	}
	if (m_lm_lookahead) {
		if (original_node->possible_word_id_list.size() > 0)
			m_lm_buf_count++;
		if (lm_la_list != NULL) {
			for (i = 0; i < original_node->possible_word_id_list.size(); i++)
				lm_la_list->push_back(original_node->possible_word_id_list[i]);
		}
	}
}

// Returns false if the node should be removed (no links from there).
bool TPLexPrefixTree::post_process_fan_triphone(Node *node,
		std::vector<int> *lm_la_list, bool fan_in)
{
	int out_trans_count, arc_count;
	std::vector<Arc>::iterator arc_it;
	std::vector<int> *new_lm_la_list;
	int i;

	if (node->word_id != -1) {
		// Possible word end (and no longer in a fan-in node).
		if (m_lm_lookahead && lm_la_list != NULL) {
			if (find(lm_la_list->begin(), lm_la_list->end(), node->word_id)
					== lm_la_list->end())
				lm_la_list->push_back(node->word_id);
		}
		return true;
	}
	if (!m_lm_lookahead && ((fan_in && !(node->flags & NODE_FAN_IN))
			|| (!fan_in && !(node->flags & NODE_FAN_OUT)))) {
		// No longer in a fan-in/out node. We don't use this if we have
		// LM lookahead, as we want to reach possible word ends.
		return true;
	}
	if (m_lm_lookahead && node->possible_word_id_list.size() > 0) {
		// Already filled this node's list of possible word IDs, copy it.
		if (lm_la_list != NULL) {
			// Add words to LM lookahead list
			for (i = 0; i < node->possible_word_id_list.size(); i++) {
				if (find(lm_la_list->begin(), lm_la_list->end(),
						node->possible_word_id_list[i]) == lm_la_list->end())
					lm_la_list->push_back(node->possible_word_id_list[i]);
			}
		}
		return true;
	}

	out_trans_count = 0;
	for (i = 0; i < node->arcs.size(); i++) {
		if (node->arcs[i].next != node) // Skip self transitions
		{
			if (++out_trans_count > 1)
				break;
		}
	}
	if (out_trans_count == 0) {
		node->arcs.clear();
		return false; // No transitions, remove the node.
	}
	else if (out_trans_count < 2 && lm_la_list != NULL) {
		// Only one transition to another node, don't add LM lookahead.
		new_lm_la_list = lm_la_list;
	}
	else
		new_lm_la_list = &node->possible_word_id_list;

	arc_count = 0;
	arc_it = node->arcs.begin();
	while (arc_it != node->arcs.end()) {
		if ((*arc_it).next != node) {
			if (post_process_fan_triphone((*arc_it).next, new_lm_la_list,
					fan_in)) {
				arc_count++;
				++arc_it;
			}
			else {
				// Node was removed
				node->arcs.erase(arc_it);
			}
		}
		else
			++arc_it;
	}
	if (arc_count == 0) {
		// There's no transitions out from this node, remove it.
		node->arcs.clear();
		return false;
	}
	if (m_lm_lookahead) {
		if (lm_la_list != NULL && new_lm_la_list != lm_la_list) {
			for (i = 0; i < new_lm_la_list->size(); i++) {
				if (find(lm_la_list->begin(), lm_la_list->end(),
						(*new_lm_la_list)[i]) == lm_la_list->end())
					lm_la_list->push_back((*new_lm_la_list)[i]);
			}
		}
	}
	return true;
}

void TPLexPrefixTree::set_sentence_boundary(int sentence_start_id,
		int sentence_end_id)
{
	// Add nodes containing the sentence start and end word ids
	TPLexPrefixTree::Node * sentence_end_node = new Node(sentence_end_id);
	sentence_end_node->node_id = m_nodes.size();
	sentence_end_node->flags |= NODE_FIRST_STATE_OF_WORD;
	sentence_end_node->state = m_last_silence_node->state;
	m_nodes.push_back(sentence_end_node);

	Arc arc;
	arc.next = sentence_end_node;
	arc.log_prob = get_out_transition_log_prob(m_last_silence_node);
	m_last_silence_node->arcs.push_back(arc);

	arc.next = m_root_node;
	arc.log_prob = 0;
	sentence_end_node->arcs.push_back(arc);

	// In the end of the recognition, the tokens in the final nodes are
	// appended with a sentence end symbol.
	m_silence_node->flags |= NODE_FINAL;
	m_last_silence_node->flags |= NODE_FINAL;
}

void TPLexPrefixTree::initialize_nodes()
{
	for_each(m_nodes.begin(), m_nodes.end(), delete_node());
	m_nodes.clear();
	m_root_node = new Node(-1);
	m_root_node->node_id = 0;
	m_root_node->flags = NODE_USE_WORD_END_BEAM;
	m_nodes.push_back(m_root_node);
	m_end_node = new Node(-1);
	m_end_node->node_id = 1;
	m_nodes.push_back(m_end_node);
	m_start_node = new Node(-1);
	m_start_node->node_id = 2;
	m_nodes.push_back(m_start_node);
	m_silence_node = NULL;
	m_last_silence_node = NULL;
}

void TPLexPrefixTree::create_cross_word_network()
{
	std::map<std::string, int>::const_iterator it;

	it = m_hmm_map.begin();
	while (it != m_hmm_map.end()) {
		if ((*it).first.size() == 5) // a-b+c
		{
			std::string b((*it).first, 0, 1);
			std::string e((*it).first, 4, 1);
			if (b != "_" && b != "=" && e != "=") {
				add_hmm_to_fan_network((*it).second, false);
			}
		}
		++it;
	}
}

// FIXME: Only works with strict left to right HMMs with one source and
// sink state.
void TPLexPrefixTree::add_hmm_to_fan_network(int hmm_id, bool fan_out)
{
	Hmm *hmm = &m_hmms[hmm_id];
	Node *last_node;
	int j, k;
	node_vector hmm_state_nodes;
	node_vector sink_nodes;
	std::vector<float> sink_trans_log_probs;
	unsigned short flags;

	hmm_state_nodes.insert(hmm_state_nodes.end(), hmm->states.size() - 2, NULL);
	if (fan_out) {
		flags = NODE_FAN_OUT | NODE_AFTER_WORD_ID | NODE_USE_WORD_END_BEAM;
		hmm_state_nodes[0] = get_fan_out_entry_node(&hmm->state(2), hmm->label);
		hmm_state_nodes[0]->flags |= NODE_FAN_OUT_FIRST;
		hmm_state_nodes[hmm->states.size() - 3] = get_fan_out_last_node(
				&hmm->state(hmm->states.size() - 1), hmm->label);
	}
	else {
		flags = NODE_FAN_IN;//|NODE_USE_WORD_END_BEAM; // | NODE_AFTER_WORD_ID;
		hmm_state_nodes[0] = get_fan_in_entry_node(&hmm->state(2), hmm->label);
		hmm_state_nodes[0]->flags |= NODE_FAN_IN_FIRST
				| NODE_FIRST_STATE_OF_WORD;
		hmm_state_nodes[hmm->states.size() - 3] = get_fan_in_last_node(
				&hmm->state(hmm->states.size() - 1), hmm->label);
	}

	// Expand the nodes
	for (j = 2; j < hmm->states.size(); j++) {
		assert( hmm_state_nodes[j-2] != NULL );
		for (k = 0; k < hmm->state(j).transitions.size(); k++) {
			expand_lexical_tree(hmm_state_nodes[j - 2], hmm,
					hmm->state(j).transitions[k], 0, -1, hmm_state_nodes,
					sink_nodes, sink_trans_log_probs, flags);
		}
	}
	assert(sink_nodes.size() == 1);
	last_node = sink_nodes.front();
	assert( last_node == hmm_state_nodes[hmm->states.size()-3] );
	if (!m_optional_short_silence && ((m_lm_lookahead && fan_out)
			|| (!m_lm_lookahead && !fan_out)))
		last_node->flags |= NODE_INSERT_WORD_BOUNDARY;
}

void TPLexPrefixTree::link_fan_out_node_to_fan_in(Node *node,
		const std::string &key)
{
	std::string out_right(key, 1, 1);

	if (out_right == "_") {
		// Link to the silence node.
		node->flags &= ~NODE_INSERT_WORD_BOUNDARY; // Clear this flag
		int j;
		for (j = 0; j < node->arcs.size(); j++)
			if (node->arcs[j].next == NULL) // Silence target is NULL at the moment
				break;
		if (j == node->arcs.size()) {
			Arc temp_arc;
			NodeArcId temp_id;
			temp_arc.next = NULL;
			temp_arc.log_prob = get_out_transition_log_prob(node);
			node->arcs.push_back(temp_arc);
			temp_id.node = node;
			temp_id.arc_index = node->arcs.size() - 1;
			m_silence_arcs.push_back(temp_id);
		}
	}
	else {
		link_node_to_fan_network(key, node->arcs, false, true,
				get_out_transition_log_prob(node));
		if (m_optional_short_silence) {
			Arc temp_arc;
			Node *silence = get_short_silence_node();
			temp_arc.next = silence;
			temp_arc.log_prob = get_out_transition_log_prob(node);
			node->arcs.push_back(temp_arc);
			link_node_to_fan_network(key, silence->arcs, false, true,
					get_out_transition_log_prob(silence));
		}
	}
}

void TPLexPrefixTree::link_node_to_fan_network(const std::string &key,
		std::vector<Arc> &source_arcs, bool fan_out, bool ignore_length,
		float out_transition_log_prob)
{
	node_vector *target_nodes;
	std::string new_key;
	Arc temp_arc;
	int i, j;
	if (fan_out) {
		target_nodes = &get_fan_node_list(key, m_fan_out_entry_nodes);
		if (target_nodes->size() == 0) {
			// Fan out nodes are created on demand
			std::string left(key, 0, 1);
			std::string mid(key, 1, 1);
			std::map<std::string, int>::const_iterator it = m_hmm_map.begin();
			while (it != m_hmm_map.end()) {
				if ((*it).first.size() == 5) {
					std::string b((*it).first, 0, 1);
					std::string m((*it).first, 2, 1);
					std::string e((*it).first, 4, 1);
					if (b == left && m == mid && e != "=") {
						add_hmm_to_fan_network((*it).second, true);
					}
				}
				++it;
			}
		}
	}
	else {
		new_key = key;
		if (ignore_length && m_ignore_case) {
			std::transform(new_key.begin(), new_key.end(), new_key.begin(),
					safe_tolower);
		}
		target_nodes = &get_fan_node_list(new_key, m_fan_in_entry_nodes);
	}

	for (i = 0; i < target_nodes->size(); i++) {
		for (j = 0; j < source_arcs.size(); j++) {
			if (source_arcs[j].next == (*target_nodes)[i])
				break;
		}
		if (j == source_arcs.size()) {
			// Link
			temp_arc.next = (*target_nodes)[i];
			temp_arc.log_prob = out_transition_log_prob;
			source_arcs.push_back(temp_arc);
		}
	}
	if (!fan_out && ignore_length && m_ignore_case) // Obsolete?!?
	{
		// Try with long length
		// NOTE: if (m_ignore_case) used to be here for the transform only,
		// moved to apply to the whole block as it should be obsolete
		std::transform(++new_key.begin(), new_key.end(), ++new_key.begin(),
				safe_toupper);
		node_vector & target_nodes = get_fan_node_list(new_key,
				m_fan_in_entry_nodes);
		for (i = 0; i < target_nodes.size(); i++) {
			for (j = 0; j < source_arcs.size(); j++) {
				if (source_arcs[j].next == target_nodes[i])
					break;
			}
			if (j == source_arcs.size()) {
				// Link
				temp_arc.next = target_nodes[i];
				temp_arc.log_prob = out_transition_log_prob;
				source_arcs.push_back(temp_arc);
			}
		}
	}
}

void TPLexPrefixTree::add_single_hmm_word_for_cross_word_modeling(Hmm *hmm,
		int word_id)
{
	// Create another instance of a null node after the fan_in network and
	// link it back to fan in.
	string_to_nodes_map::const_iterator it;
	std::string middle(hmm->label, 2, 1);
	Node *wid_node;
	Arc temp_arc;
	NodeArcId node_arc_id;
	int i;

	it = m_fan_in_last_nodes.begin();
	while (it != m_fan_in_last_nodes.end()) {
		std::string left((*it).first, 0, 1);
		if (left == middle) {
			std::string right((*it).first, 1, 1);
			std::string in_key = left + right;
			if (m_ignore_case)
				std::transform(in_key.begin(), in_key.end(), in_key.begin(),
						safe_tolower);
			// Create a null node
			wid_node = new Node(word_id);
			wid_node->node_id = m_nodes.size();
			wid_node->flags = NODE_USE_WORD_END_BEAM;
			m_nodes.push_back(wid_node);
			temp_arc.next = wid_node;
			const node_vector & nlist = it->second;
			for (i = 0; i < nlist.size(); i++) {
				temp_arc.log_prob = get_out_transition_log_prob(nlist[i]);
				nlist[i]->arcs.push_back(temp_arc);
			}
			if (right == "_") {
				temp_arc.next = NULL;
				wid_node->arcs.push_back(temp_arc);
				node_arc_id.node = wid_node;
				node_arc_id.arc_index = wid_node->arcs.size() - 1;
				m_silence_arcs.push_back(node_arc_id);
			}
			else {
				// If LM lookahead is in use, word boundary is inserted at the
				// last state of the fan out branch.
				if (m_lm_lookahead && !m_optional_short_silence)
					wid_node->flags |= NODE_INSERT_WORD_BOUNDARY;
				link_node_to_fan_network(in_key, wid_node->arcs, false, true, 0);
				if (m_optional_short_silence) {
					Arc temp_arc;
					Node *silence = get_short_silence_node();
					temp_arc.next = silence;
					temp_arc.log_prob = 0;
					wid_node->arcs.push_back(temp_arc);
					link_node_to_fan_network(in_key, silence->arcs, false,
							true, get_out_transition_log_prob(silence));
				}
			}
		}
		++it;
	}
}

void TPLexPrefixTree::link_fan_in_nodes(void)
{
	// At this point, cross word network is ready and words have been linked
	// to the fan_out layer. Also single HMM words have been linked back
	// to the fan_in layer. What is left is to link the last states of
	// the fan_in layer back to the beginning of the lexical prefix tree.
	string_to_nodes_map::const_iterator it;
	int i;
	it = m_fan_in_last_nodes.begin();
	while (it != m_fan_in_last_nodes.end()) {
		const node_vector & nlist = it->second;
		for (i = 0; i < nlist.size(); i++)
			create_lex_tree_links_from_fan_in(nlist[i], it->first);
		++it;
	}
}

void TPLexPrefixTree::create_lex_tree_links_from_fan_in(Node *fan_in_node,
		const std::string &key)
{
	Arc temp_arc;
	int j, k;

	std::string out_right(key, 1, 1);
	// Skip silences, as this node has either already been linked to
	// silence while adding a one-HMM word, or it is unused.
	if (out_right != "_") {
		string_to_nodes_map::const_iterator it =
				m_fan_in_connection_nodes.find(key);
		if (it != m_fan_in_connection_nodes.end()) {
			for (j = 0; j < it->second.size(); j++) {
				// Check the link does not exist already
				for (k = 0; k < fan_in_node->arcs.size(); k++)
					if (fan_in_node->arcs[k].next == it->second[j])
						break;
				if (k == fan_in_node->arcs.size()) {
					// Link
					temp_arc.next = it->second[j];
					temp_arc.log_prob
							= get_out_transition_log_prob(fan_in_node);
					fan_in_node->arcs.push_back(temp_arc);
				}
			}
		}
	}
}

void TPLexPrefixTree::analyze_cross_word_network(void)
{
	int num_out_nodes, num_in_nodes;
	int num_out_arcs, num_in_arcs;
	int temp_nodes, temp_arcs;
	int i;
	string_to_nodes_map::const_iterator it;

	num_out_nodes = num_in_nodes = 0;
	num_out_arcs = num_in_arcs = 0;

	it = m_fan_out_entry_nodes.begin();
	while (it != m_fan_out_entry_nodes.end()) {
		for (i = 0; i < it->second.size(); i++)
			count_fan_size(it->second[i], NODE_FAN_OUT, &temp_nodes, &temp_arcs);
		num_out_nodes += temp_nodes;
		num_out_arcs += temp_arcs;
		++it;
	}
	it = m_fan_in_entry_nodes.begin();
	while (it != m_fan_in_entry_nodes.end()) {
		for (i = 0; i < it->second.size(); i++)
			count_fan_size(it->second[i], NODE_FAN_IN, &temp_nodes, &temp_arcs);
		num_in_nodes += temp_nodes;
		num_in_arcs += temp_arcs;
		++it;
	}
	if (m_verbose > 1) {
		fprintf(stderr, "FAN OUT: %d nodes, %d arcs\n", num_out_nodes,
				num_out_arcs);
		fprintf(stderr, "FAN IN:  %d nodes, %d arcs\n", num_in_nodes,
				num_in_arcs);
	}
}

void TPLexPrefixTree::count_fan_size(Node *node, unsigned short flag,
		int *num_nodes, int *num_arcs)
{
	int i;
	int temp_nodes, temp_arcs;

	*num_nodes = 0;
	*num_arcs = 0;
	if (!(node->flags & flag))
		return;
	*num_nodes = 1;
	*num_arcs = node->arcs.size();
	for (i = 0; i < node->arcs.size(); i++) {
		if (node != node->arcs[i].next) {
			count_fan_size(node->arcs[i].next, flag, &temp_nodes, &temp_arcs);
			*num_nodes += temp_nodes;
			*num_arcs += temp_arcs;
		}
	}
}

void TPLexPrefixTree::count_prefix_tree_size(Node *node, int *num_nodes,
		int *num_arcs)
{
	int i;
	if (node->flags & (NODE_FAN_OUT | NODE_FAN_IN))
		return;
	(*num_nodes)++;
	(*num_arcs) += node->arcs.size();
	for (i = 0; i < node->arcs.size(); i++) {
		if (node->arcs[i].next != node && node->arcs[i].next != m_root_node) {
			count_prefix_tree_size(node->arcs[i].next, num_nodes, num_arcs);
		}
	}
}

void TPLexPrefixTree::free_cross_word_network_connection_points(void)
{
	m_fan_out_entry_nodes.clear();
	m_fan_out_last_nodes.clear();
	m_fan_in_entry_nodes.clear();
	m_fan_in_last_nodes.clear();
	m_fan_in_connection_nodes.clear();
}

TPLexPrefixTree::Node*
TPLexPrefixTree::get_short_silence_node(void)
{
	Arc temp_arc;
	assert( m_short_silence_state != NULL );
	Node *silence = new Node(m_word_boundary_id, m_short_silence_state);
	silence->node_id = m_nodes.size();
	silence->flags = NODE_FAN_OUT | NODE_USE_WORD_END_BEAM | NODE_FINAL;
	if (m_silence_is_word)
		silence->flags |= NODE_FIRST_STATE_OF_WORD;
	m_nodes.push_back(silence);
	// Make self transition
	temp_arc.next = silence;
	temp_arc.log_prob = m_short_silence_state->transitions[0].log_prob;
	silence->arcs.push_back(temp_arc);

	return silence;
}

TPLexPrefixTree::Node*
TPLexPrefixTree::get_fan_out_entry_node(HmmState *state,
		const std::string &label)
{
	std::string temp1(label, 0, 1);
	std::string temp2(label, 2, 1);
	std::string key = temp1 + temp2;
	node_vector & nlist = get_fan_node_list(key, m_fan_out_entry_nodes);
	Node *node;

	node = get_fan_state_node(state, nlist);
	node->flags = NODE_FAN_OUT | NODE_AFTER_WORD_ID | NODE_USE_WORD_END_BEAM;
	return node;
}

TPLexPrefixTree::Node*
TPLexPrefixTree::get_fan_out_last_node(HmmState *state,
		const std::string &label)
{
	std::string temp1(label, 2, 1);
	std::string temp2(label, 4, 1);
	if (m_ignore_case)
		std::transform(temp1.begin(), temp1.end(), temp1.begin(), safe_tolower);
	std::string key = temp1 + temp2;
	node_vector & nlist = get_fan_node_list(key, m_fan_out_last_nodes);
	Node *node;

	node = get_fan_state_node(state, nlist);
	node->flags = NODE_FAN_OUT | NODE_AFTER_WORD_ID | NODE_USE_WORD_END_BEAM;
	return node;
}

TPLexPrefixTree::Node*
TPLexPrefixTree::get_fan_in_entry_node(HmmState *state,
		const std::string &label)
{
	std::string temp1(label, 0, 1);
	std::string temp2(label, 2, 1);
	std::string key = temp1 + temp2;
	node_vector & nlist = get_fan_node_list(key, m_fan_in_entry_nodes);
	Node *node;

	node = get_fan_state_node(state, nlist);
	node->flags = NODE_FAN_IN;
	return node;
}

TPLexPrefixTree::Node*
TPLexPrefixTree::get_fan_in_last_node(HmmState *state, const std::string &label)
{
	std::string temp1(label, 2, 1);
	std::string temp2(label, 4, 1);
	std::string key = temp1 + temp2;
	node_vector & nlist = get_fan_node_list(key, m_fan_in_last_nodes);
	Node *node;

	node = get_fan_state_node(state, nlist);
	node->flags = NODE_FAN_IN;
	return node;
}

TPLexPrefixTree::Node*
TPLexPrefixTree::get_fan_state_node(HmmState *state, node_vector & nodes)
{
	Node *new_node;
	for (int i = 0; i < nodes.size(); i++) {
		if (nodes[i] == NULL)
			throw logic_error("TPLexPrefixTree::get_fan_state_node");
		if (nodes[i]->state->model == state->model) {
			return nodes[i];
		}
	}
	// Node did not exist, create it
	new_node = new Node(-1, state);
	new_node->node_id = m_nodes.size();
	m_nodes.push_back(new_node);
	nodes.push_back(new_node);
	return new_node;
}

TPLexPrefixTree::node_vector &
TPLexPrefixTree::get_fan_node_list(const std::string &key,
		string_to_nodes_map &nmap)
{
	string_to_nodes_map::const_iterator it;

	return nmap[key];
}

void TPLexPrefixTree::add_fan_in_connection_node(Node *node,
		const std::string &prev_label)
{
	std::string temp1(prev_label, 2, 1);
	std::string temp2(prev_label, 4, 1);
	std::string map_index = temp1 + temp2;
	string_to_nodes_map::const_iterator it;

	node->flags |= NODE_FAN_IN_CONNECTION;
	m_fan_in_connection_nodes[map_index].push_back(node);
}

float TPLexPrefixTree::get_out_transition_log_prob(Node *node)
{
	for (int i = 0; i < node->arcs.size(); i++)
		if (node->arcs[i].next == node) {
			// Self transition, compute the out transition
			return log10(1 - pow(10, node->arcs[i].log_prob));
		}
	return 0;
}

void TPLexPrefixTree::prune_lookahead_buffers(int min_delta, int max_depth)
{
	if (m_verbose > 1)
		printf("LM lookahead buffers before pruning: %d\n", m_lm_buf_count);
	m_lm_buf_count = 0;
	for (int i = 0; i < m_root_node->arcs.size(); i++)
		prune_lm_la_buffer(min_delta, max_depth, m_root_node->arcs[i].next, -1,
				0);
	if (m_verbose > 1)
		printf("LM lookahead buffers after pruning: %d\n", m_lm_buf_count);
}

void TPLexPrefixTree::prune_lm_la_buffer(int delta_thr, int depth_thr,
		Node *node, int last_size, int cur_depth)
{
	int i;
	int cur_size = last_size;

	if (!m_silence_is_word && node == m_silence_node) // Word LM, no word ID
		return;
	if (node->word_id != -1)
		return; // No more LM lookahead

	if (node->possible_word_id_list.size() > 0) {
		// Determine if we want to remove this buffer
		if (last_size > 0 && last_size - node->possible_word_id_list.size()
				<= delta_thr) {
			// Not enough change from last lookahead node, remove
			node->possible_word_id_list.clear();
		}
		else if (cur_depth >= depth_thr) {
			// Gone past the maximum depth
			node->possible_word_id_list.clear();
		}
		else {
			cur_depth++;
			cur_size = node->possible_word_id_list.size();
			m_lm_buf_count++;
		}
	}

	for (i = 0; i < node->arcs.size(); i++) {
		if (node->arcs[i].next != node) {
			prune_lm_la_buffer(delta_thr, depth_thr, node->arcs[i].next,
					cur_size, cur_depth);
		}
	}
}

void TPLexPrefixTree::set_lm_lookahead_cache_sizes(int cache_size)
{
	for (int i = 0; i < m_nodes.size(); i++)
		if (m_nodes[i]->possible_word_id_list.size() > 0)
			m_nodes[i]->lm_lookahead_buffer.set_max_items(cache_size);
}

void TPLexPrefixTree::clear_node_token_lists(void)
{
	for (int i = 0; i < m_nodes.size(); i++) {
		m_nodes[i]->token_list = NULL;
	}
}

void TPLexPrefixTree::print_node_info(int node)
{
	printf("word_id = %d\n", m_nodes[node]->word_id);
	printf("model = %d\n", (m_nodes[node]->state == NULL ? -1
			: m_nodes[node]->state->model));
	printf("flags: %04x\n", m_nodes[node]->flags);
	printf("LM lookahead: %zd possible word(s)\n",
			m_nodes[node]->possible_word_id_list.size());
	printf("%zd arc(s):\n", m_nodes[node]->arcs.size());
	for (int i = 0; i < m_nodes[node]->arcs.size(); i++) {
		printf(" -> %d (%d), transition: %.2f\n",
				m_nodes[node]->arcs[i].next->node_id,
				(m_nodes[node]->arcs[i].next->state == NULL ? -1
						: m_nodes[node]->arcs[i].next->state->model),
				m_nodes[node]->arcs[i].log_prob);
	}
}

void TPLexPrefixTree::print_lookahead_info(int node, const Vocabulary &voc)
{
	printf("Possible word ends: ");
	if (m_nodes[node]->possible_word_id_list.size() == 0)
		printf("N/A\n");
	else {
		printf("%zd\n", m_nodes[node]->possible_word_id_list.size());
		for (int i = 0; i < m_nodes[node]->possible_word_id_list.size(); i++)
			printf(" %d (%s)\n", m_nodes[node]->possible_word_id_list[i],
					voc.word(m_nodes[node]->possible_word_id_list[i]).c_str());
	}
}

void TPLexPrefixTree::debug_prune_dead_ends(Node *node)
{
	for (int i = 0; i < node->arcs.size(); i++) {
		node->flags |= NODE_DEBUG_PRUNED;
		Node *target = node->arcs[i].next;
		if (!(target->flags & NODE_DEBUG_PRUNED))
			debug_prune_dead_ends(target);
		if (target->arcs.empty()) {
			node->arcs[i] = node->arcs.back();
			node->arcs.pop_back();
			i--;
		}
	}
	if (node->arcs.size() == 1 && node->arcs[0].next == node)
		node->arcs.clear();

	/*if (node->arcs.empty() && !(node->flags & NODE_FINAL))
	 fprintf(stderr, "debug_prune_dead_ends: pruned node %d\n", node->node_id);*/
}

void TPLexPrefixTree::debug_add_silence_loop()
{
	fprintf(stderr, "DEBUG WARNING: adding hardcoded loop in silence\n");
	Node *node = m_silence_node;
	Node *prev_node = NULL;
	for (int i = 0; i < 2; i++) {
		prev_node = node;
		for (int a = 0; a < node->arcs.size(); a++) {
			if (node->arcs[a].next != node) {
				node = node->arcs[a].next;
				break;
			}
		}
		assert(node != prev_node);
	}
	Arc arc;
	arc.log_prob = 0;
	arc.next = prev_node;
	node->arcs.push_back(arc);
}
