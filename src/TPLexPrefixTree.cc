#include <stdio.h>

#include "TPLexPrefixTree.hh"


TPLexPrefixTree::TPLexPrefixTree()
  : m_words(0),
    m_verbose(0),
    m_lm_lookahead(0)
{
  m_root_node = new Node(-1);
  m_root_node->node_id = 0;
  node_list.push_back(m_root_node);
  m_end_node = new Node(-1);
  m_end_node->node_id = 1;
  node_list.push_back(m_end_node);
}


void
TPLexPrefixTree::add_word(std::vector<Hmm*> &hmm_list, int word_id)
{
  int i, j, k;
  std::vector<Node*> hmm_state_nodes;
  std::vector<Node*> source_nodes, sink_nodes;
  std::vector<float> source_trans_log_probs;
  std::vector<float> sink_trans_log_probs;
  int word_end;
  Arc end_arc;

  // NOTE! HMM states are indexed with respect to their mixture model.
  // This is a quick solution to allow tied hmm states. The right way
  // to do this would be to load HMMs so that different HMMs would
  // really share the same states. Now it is merely assumed that the
  // (local) transitions in the states with the same mixtures are the same.

  // Invariants: source_nodes.size() == source_trans_log_probs.size(),
  //             sink_nodes.size() == sink_trans_log_probs.size()

  source_nodes.push_back(m_root_node);
  source_trans_log_probs.push_back(0);
  word_end = -1;
  for (i = 0; i < hmm_list.size(); i++)
  {
    assert( source_nodes.size() > 0 );

    if (i == hmm_list.size()-1)
      word_end = word_id;

    // Set up a table for storing the nodes corresponding to the HMM states
    hmm_state_nodes.clear();
    hmm_state_nodes.insert(hmm_state_nodes.end(), hmm_list[i]->states.size()-2,
                           NULL);
    sink_nodes.clear();
    sink_trans_log_probs.clear();

    // Expand previous sink nodes with new source transitions
    for (j = 0; j < source_nodes.size(); j++) // Iterate previous sink nodes
    {
      for (k = 0; k < hmm_list[i]->state(0).transitions.size(); k++)
      {
        expand_lexical_tree(source_nodes[j], hmm_list[i],
                            hmm_list[i]->state(0).transitions[k],
                            source_trans_log_probs[j],
                            word_end,
                            hmm_state_nodes,
                            sink_nodes, sink_trans_log_probs);
      }
    }

    if (m_lm_lookahead == 1 && i == 0)
    {
      // Language model lookahead in first subtree nodes only
      hmm_state_nodes[0]->possible_word_id_list.push_back(word_id);
    }

    // Expand other states, from left to right
    for (j = 2; j < hmm_list[i]->states.size(); j++)
    {
      assert( hmm_state_nodes[j-2] != NULL );
      for (k = 0; k < hmm_list[i]->state(j).transitions.size(); k++)
      {
        expand_lexical_tree(hmm_state_nodes[j-2], hmm_list[i],
                            hmm_list[i]->state(j).transitions[k],
                            0, word_end,
                            hmm_state_nodes,
                            sink_nodes, sink_trans_log_probs);
      }
    }
    source_nodes = sink_nodes;
    source_trans_log_probs = sink_trans_log_probs;
  }

  // Mark end of the word node (sink state), add to the word end list and link
  // to the end node.
  assert( source_nodes.size() == 1 ); // The sink state
  word_end_list.push_back(source_nodes.front());
  end_arc.next = m_end_node;
  end_arc.log_prob = 0;
  source_nodes.front()->arcs.push_back(end_arc);
  m_words = word_id+1;
}


void
TPLexPrefixTree::expand_lexical_tree(Node *source, Hmm *hmm,
                                     HmmTransition &t,
                                     float cur_trans_log_prob,
                                     int word_end,
                                     std::vector<Node*> &hmm_state_nodes,
                                     std::vector<Node*> &sink_nodes,
                                     std::vector<float> &sink_trans_log_probs)
{
  int i;

  // Check if we are going to sink state
  if (hmm->is_sink(t.target))
  {
    if (word_end == -1)
    {
      // Mark the source node as a next sink and return
      sink_nodes.push_back(source);
      sink_trans_log_probs.push_back(cur_trans_log_prob + t.log_prob);
    }
    else
    {
      // Make explicit sink state for word end
      if (sink_nodes.size() == 0)
      {
        Node *sink;
        sink = new Node(word_end);
        sink->node_id = node_list.size();
        node_list.push_back(sink);
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
  for (i = 0; i < source->arcs.size(); i++)
  {
    if (source->arcs[i].next->state != NULL &&
        source->arcs[i].next->state->model == hmm->state(t.target).model)
    {
      // Already linked, assume the transition is OK so nothing to be done
      hmm_state_nodes[t.target-2] = source->arcs[i].next;
      return;
    }
  }

  // The node has not been linked
  if (hmm_state_nodes[t.target-2] == NULL)
  {
    // The state does not even exist, so create it.
    // FIXME! Can the order of processing the transitions and source states
    // affect the existence of a node?

    hmm_state_nodes[t.target-2] = new Node(-1, &hmm->state(t.target));
    hmm_state_nodes[t.target-2]->node_id = node_list.size();
    node_list.push_back(hmm_state_nodes[t.target-2]);
  }

  // Add new arc
  Arc temp;
  temp.next = hmm_state_nodes[t.target-2];
  temp.log_prob = cur_trans_log_prob + t.log_prob;
  source->arcs.push_back(temp);
}


void
TPLexPrefixTree::finish_tree(void)
{
  // Replace arcs to the end node in the word ends to the root to make the
  // tree re-entrant
  if (m_verbose > 1)
    printf("%d words in the lexicon\n", word_end_list.size());
  
  for (int i = 0; i < word_end_list.size(); i++)
  {
    for (int j = 0; j < word_end_list[i]->arcs.size(); j++)
    {
      if (word_end_list[i]->arcs[j].next == m_end_node)
        word_end_list[i]->arcs[j].next = m_root_node;
    }
  }

  // FIXME! Should the word id lists for lm lookahead be sorted to increase
  // cache utility?
}


void
TPLexPrefixTree::clear_node_token_lists(void)
{
  for (int i = 0; i < node_list.size(); i++)
  {
    node_list[i]->token_list = NULL;
  }
}


void
TPLexPrefixTree::print_node_info(int node)
{
  printf("word_id = %d\n", node_list[node]->word_id);
  printf("model = %d\n", (node_list[node]->state==NULL?-1:
                          node_list[node]->state->model));
  printf("%d arc(s):\n", node_list[node]->arcs.size());
  for (int i = 0; i < node_list[node]->arcs.size(); i++)
  {
    printf(" -> %d (%d), transition: %.2f\n",
           node_list[node]->arcs[i].next->node_id,
           (node_list[node]->arcs[i].next->state==NULL?-1:
            node_list[node]->arcs[i].next->state->model),
           node_list[node]->arcs[i].log_prob);
  }
}

void
TPLexPrefixTree::print_lookahead_info(int node, const Vocabulary &voc)
{
  printf("Possible word ends: ");
  if (node_list[node]->possible_word_id_list.size() == 0)
    printf("N/A\n");
  else
  {
    printf("%d\n", node_list[node]->possible_word_id_list.size());
    for (int i=0; i < node_list[node]->possible_word_id_list.size(); i++)
      printf(" %d (%s)\n", node_list[node]->possible_word_id_list[i],
             voc.word(node_list[node]->possible_word_id_list[i]).c_str());
  }
}
