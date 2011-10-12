#include <assert.h>
#include "Rescore.hh"

Rescore::Rescore()
  : m_tree_gram(NULL),
    m_src_lattice(NULL),
    m_sentence_start_label("<s>"),
    m_sentence_end_label("</s>"),
    m_null_label("!NULL")
{
}

Lattice::Node& // private
Rescore::find_or_create_node(int node_id, Context &context)
{
  // Check if the context is defined already
  for (int c = 0; c < (int)m_node_contexts[node_id].size(); c++) {
    Context &old_context = m_node_contexts[node_id][c];
    if (old_context == context)
      return m_rescored_lattice.node(old_context.node_id);
  }

  // Context not found, create a new node and context
  Lattice::Node &node = m_rescored_lattice.new_node();
  Context new_context = context;
  new_context.node_id = node.id;
  m_node_contexts[node_id].push_back(new_context);
  return node;
}

void
Rescore::sort_nodes()
{
  // Initialize
  std::vector<bool> flagged(m_src_lattice->num_nodes(), false);
  flagged[m_src_lattice->final_node_id] = true;
  std::vector<int> stack;
  for (int i = 0; i < m_src_lattice->num_nodes(); i++)
    if (!flagged[i])
      stack.push_back(i);
  m_sorted_nodes.resize(m_src_lattice->num_nodes());
  m_sorted_nodes.back() = m_src_lattice->final_node_id;
  int next_sorted_index = m_sorted_nodes.size() - 2;

  // Traverse through lattice
  while (!stack.empty()) {
    int node_id = stack.back();
    stack.pop_back();
    assert(!flagged[node_id]);

    // Check if all children are flagged
    Lattice::Node &node = m_src_lattice->node(node_id);
    bool children_flagged = true;
    for (int i = 0; i < (int)node.arcs.size(); i++) {
      int tgt_node_id = node.arcs[i].target_node_id;
      if (!flagged[tgt_node_id]) {
	children_flagged = false;
	break;
      }
    }

    // Postpone node if some child is not flagged
    if (!children_flagged) {
      stack.insert(stack.begin(), node_id);
      continue;
    }

    flagged[node_id] = true;
    m_sorted_nodes[next_sorted_index--] = node_id;
  }
}

void
Rescore::rescore(Lattice *src_lattice, TreeGram *tree_gram, bool quiet)
{
  m_src_lattice = src_lattice;
  m_tree_gram = tree_gram;
  m_rescored_lattice.clear();
  m_sentence_end_id = tree_gram->word_index(m_sentence_end_label);

  // Create a new final node for source lattice and add sentence end
  // arc.
  {
    Lattice::Node &new_final_node = m_src_lattice->new_node();
    m_src_lattice->new_arc(m_src_lattice->final_node_id, new_final_node.id,
			   m_sentence_end_label, 0, 0);
    m_src_lattice->final_node_id = new_final_node.id;
  }

  // Initialize rescored lattice and node contexts
  {
    Lattice::Node &node = m_rescored_lattice.new_node();
    m_rescored_lattice.initial_node_id = node.id;
    m_node_contexts.clear();
    m_node_contexts.resize(src_lattice->num_nodes());
    Context context;
    context.gram.push_back(tree_gram->word_index(m_sentence_start_label));
    context.node_id = node.id;
    m_node_contexts[src_lattice->initial_node_id].push_back(context);
  }

  // Traverse source lattice in topological order
  if (!quiet)
    fprintf(stderr, "sorting...");
  sort_nodes();
  if (!quiet)
    fprintf(stderr, "rescoring...");
  for (int s = 0; s < (int)m_sorted_nodes.size(); s++) {
    int src_id = m_sorted_nodes[s];
    Lattice::Node &src_node = m_src_lattice->node(src_id);

    // Process all arcs from the source node
    for (int a = 0; a < (int)src_node.arcs.size(); a++) {
      Lattice::Arc &arc = src_node.arcs[a];
      int tgt_id = arc.target_node_id;

      // Process all contexts of the source node
      for (int c = 0; c < (int)m_node_contexts[src_id].size(); c++) {

	// Compute the language model probability and cut the context
	// to the maximum length needed by the model.
	Context &src_context = m_node_contexts[src_id][c];
	Context tgt_context = src_context;
	float lm_log_prob = 0;
	if (arc.label != m_null_label) {
	  int word_id = tree_gram->word_index(arc.label);
	  tgt_context.gram.push_back(word_id);
	  lm_log_prob = tree_gram->log_prob(tgt_context.gram);

	  while ((int)tgt_context.gram.size() > 
		 tree_gram->last_history_length())
	    tgt_context.gram.pop_front();
	}

	// Create the resulting lattice (final state has </s> context)
	bool final_node = false;
	if (tgt_context.gram.back() == m_sentence_end_id) {
	  tgt_context.gram.erase(tgt_context.gram.begin(), 
				 tgt_context.gram.end() - 1);
	  final_node = true;
	}
	Lattice::Node &new_tgt_node = find_or_create_node(tgt_id, tgt_context);
	m_rescored_lattice.final_node_id = new_tgt_node.id;
	Lattice::Node &new_src_node = 
	  m_rescored_lattice.node(src_context.node_id);
	m_rescored_lattice.new_arc(new_src_node.id, new_tgt_node.id,
				   arc.label, arc.ac_log_prob, lm_log_prob);
      }
    }
  }
}
