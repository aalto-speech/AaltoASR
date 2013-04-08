#ifndef WORDGRAPH_HH
#define WORDGRAPH_HH

#include <cstddef>  // NULL
#include <cfloat>
#include <assert.h>
#include <vector>

/** A structure for maintaining WordGraphs during recognition.  Each
 * node of the graph stores the incoming arcs.  Weights are stored in
 * the arcs.  Each node contains the weight of the best path reaching
 * the node.
 *
 * \bugs Explain the general idea of the word graph structure and
 * garbage collection.
 *
 * Negative arc and node indices mean null.  Prepare to meet negative
 * indices in the \c arcs and \c nodes vectors, too.
 *
 * \note Currently, the word indices stored in the graph are words
 * from the lexicon.  The language model may have a different
 * vocabulary.
 */
struct WordGraph {

  /** An arc in the graph. 
   * 
   * The arcs coming to a node are gathered as a linked list through
   * the \c sibling_arc variable.
   */
  struct Arc {
    /** The default constructor. */
    Arc() : source_node_id(-1), sibling_arc(-1), am_weight(0), lm_weight(0) { }

    /** Create an arc. 
     * \param sibling_arc = the next arc in the linked list
     * \param source_node_id = the source node of the arc
     * \param am_weight = the acoustic weight of the arc
     * \param lm_weight = the language model weight of the arc
     */
    Arc(int sibling_arc, int source_node_id, float am_weight = 0,
        float lm_weight = 0)
      : source_node_id(source_node_id), sibling_arc(sibling_arc), 
        am_weight(am_weight), lm_weight(lm_weight) { }

    int source_node_id; //!< The source node of the arc.
    int sibling_arc; //!< The next arc ending to the same node. 
    float am_weight; //!< The acoustic weight of the arc.
    float lm_weight; //!< The language model weight of the arc.
  };


  /** A node in the graph. */
  struct Node {
    Node(int frame, int symbol, int lex_node_id, 
         float path_weight = -FLT_MAX) 
      : first_arc(-1), reachable(true), frame(frame), symbol(symbol), 
        lex_node_id(lex_node_id), path_weight(path_weight), 
        reference_count(0) { }
    int first_arc; //!< The first arc coming to this node.
    bool reachable; //!< A flag for garbage collection.
    int frame; //!< The frame of the node, i.e., the ending time of the word
    int symbol; //!< The symbol (for example word_id) which ends in this node.
    int lex_node_id; //!< The lexicon node where the word ended.
    float path_weight; //!< Weight of the best path reaching this node.
    int reference_count; //!< Reference count for garbage collection
  };


  /** The default constructor. */
  WordGraph() { }

  /** Add a new node to the graph
   * \param word_id = the word id used to create the node index
   * \param frame = the frame used to create the node index
   * \param lex_node_id = the lexicon node in which the word ends
   * \param path_weight = the weight of the best path to the node
   * \return the resulting node index
   */
  int add_node(int frame, int symbol, int lex_node_id, 
               float path_weight = -FLT_MAX)
  {
    /* No reusable nodes, create a new. */
    if (m_free_node_indices.empty()) {
      nodes.push_back(Node(frame, symbol, lex_node_id, path_weight));
      nodes.back().reference_count = 0;
      return nodes.size() - 1;
    }

    /* Reuse an old node. */
    int node_index = m_free_node_indices.back();
    m_free_node_indices.pop_back();
    nodes[node_index] = Node(frame, symbol, lex_node_id, path_weight);
    nodes[node_index].reference_count = 0;
    return node_index;
  }

  /** Insert a new arc to the graph.  Stores only the best path among
   * the paths that start from and end up in the same node.
   *
   * Word pair approximation reduces lattice size by 5 - 20 % but also increases
   * errors when decoding the lattice, because this may remove an arc from the
   * best path.
   *
   * \bugs Checks quite slowly if the arc exists already, but probably
   * that does not matter much.
   *
   * \param source_node_id = the index of the source node
   * \param target_node_id = the index of the target node
   * \param am_weight = the acoustic weight of the arc
   * \param lm_weight = the language model weight of the arc
   * \param word_pair_approx If true, stores only the best path among the paths
   * that end up in the same node and have the same symbol in the source node.
   */
  void add_arc(int source_node_id, int target_node_id, float am_weight = 0, 
               float lm_weight = 0, bool word_pair_approx = true)
  {
    float weight = am_weight + lm_weight;
    Node &src_node = nodes[source_node_id];
    Node &tgt_node = nodes[target_node_id];
    float path_weight = src_node.path_weight + weight;

    // FIXME: this might be slow. Check that the arc does not exist already.
    for (int a = tgt_node.first_arc; a >= 0; a = arcs[a].sibling_arc) {
      Arc &arc = arcs[a];
      Node &old_src_node = nodes[arc.source_node_id];

      bool match = arc.source_node_id == source_node_id;
      if (word_pair_approx) {
        match = match || ((old_src_node.symbol == src_node.symbol) &&
                          (old_src_node.lex_node_id == src_node.lex_node_id));
      }

      if (match) {
        float old_path_weight = old_src_node.path_weight + arc.am_weight + arc.lm_weight;
        if (path_weight > old_path_weight) {
          unlink(arc.source_node_id);
          arc.am_weight = am_weight;
          arc.lm_weight = lm_weight;
          arc.source_node_id = source_node_id;
          if (path_weight > tgt_node.path_weight)
            tgt_node.path_weight = path_weight;
          link(arc.source_node_id);
        }
        return;
      }
    }

    // Insert the new arc
    int arc_index;
    if (m_free_arc_indices.empty()) {
      arc_index = arcs.size();
      arcs.push_back(Arc(tgt_node.first_arc, source_node_id, 
                         am_weight, lm_weight));
    }
    else {
      arc_index = m_free_arc_indices.back();
      m_free_arc_indices.pop_back();
      arcs[arc_index] = Arc(tgt_node.first_arc, source_node_id, 
                            am_weight, lm_weight);
    }
    tgt_node.first_arc = arc_index;
    if (path_weight > tgt_node.path_weight)
      tgt_node.path_weight = path_weight;
    link(source_node_id);
  }

  /** Marks all nodes unreachable. */
  void reset_reachability(bool value = false)
  {
    for (size_t i = 0; i < nodes.size(); i++)
      nodes[i].reachable = value;
  }

  /** Mark nodes reachable backwards from the given node. 
   * 
   * Performs a depth-first search from the given node, and marks all
   * reachable nodes.  Backtracks when a hits a node which is already
   * marked reachable.
   *
   * \param node = the first reachable node
   */
  void mark_reachable_nodes(int node)
  {
    std::vector<int> stack;
    stack.push_back(node);

    // Perform a depth-first search for reachable nodes
    while (!stack.empty()) {

      // Pop node from stack
      int node = stack.back();
      stack.pop_back();

      // Mark node reachable and put children to stack
      nodes[node].reachable = true;
      int a = nodes[node].first_arc;
      while (a >= 0) {
        Arc &arc = arcs[a];
        if (!nodes[arc.source_node_id].reachable)
          stack.push_back(arc.source_node_id);
        a = arc.sibling_arc;
      }
    }
  }

  /** Increase the reference count of the node. */
  void link(int node_index) {
    nodes[node_index].reference_count++;
  }
  
  /** Unlink a node and unlink recursively backwards if reference
   * count becomes zero. */
  void unlink(int node_index) 
  {
    assert(m_unlink_stack.empty());
    m_unlink_stack.push_back(node_index);
    while (!m_unlink_stack.empty()) {

      /* Pop node and decrease its reference count. */
      node_index = m_unlink_stack.back();
      m_unlink_stack.pop_back();
      Node &node = nodes[node_index];
      node.reference_count--;

      /* Reference count zero?  Unlink all arcs and target nodes. */
      if (node.reference_count == 0) {
        while (node.first_arc >= 0) {
          m_free_arc_indices.push_back(node.first_arc);
          Arc &arc = arcs[node.first_arc];
          m_unlink_stack.push_back(arc.source_node_id);
          node.first_arc = arc.sibling_arc;
        }
        m_free_node_indices.push_back(node_index);
      }
    }
  }

  /** Reset the structure to initial state. */
  void reset()
  {
    arcs.clear();
    nodes.clear();
    m_free_node_indices.clear();
    m_free_arc_indices.clear();
  }

  std::vector<Arc> arcs; //!< All arcs of the graph.
  std::vector<Node> nodes; //!< All nodes of the graph.

private:
  std::vector<int> m_free_arc_indices; //!< Indices of the reusable arcs
  std::vector<int> m_free_node_indices; //!< Indices of the reusable nodes

  /** Stack of nodes to be unlinked.  This is used only inside
   * unlink() function, but is defined as a class member so that the
   * vector need not be allocated every time unlink() is called.  The
   * vector should always be empty outside unlink() function. */
  std::vector<int> m_unlink_stack; 
};

#endif /* WORDGRAPH_HH */
