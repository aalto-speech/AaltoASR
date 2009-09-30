#ifndef LATTICE_HH
#define LATTICE_HH

#include <vector>
#include <string>
#include <stdio.h>

/** Lattice containing nodes and arcs */
class Lattice {
public:

  /** Directed arc between nodes. */
  struct Arc {
    /** Constructor */
    Arc(int target_node_id, std::string label, float ac_log_prob, 
	float lm_log_prob);
    int target_node_id; //!< Index of the target node
    std::string label; //!< Label of the arc (0 = null)
    float ac_log_prob; //!< Acoustic probability
    float lm_log_prob; //!< Language model probability
  };

  /** Node of the lattice. */
  struct Node {
    Node(int id) : id(id) { }
    int id; //!< The index of the node in the \c nodes vector.
    std::vector<Arc> arcs; //!< Arcs leaving this node
  };

  /** Default constructor. */
  Lattice();

  /** Clear lattice. */
  void clear();

  /** Access a node */
  Node &node(int index) { return m_nodes[index]; }
  
  /** The number of nodes. */
  int num_nodes() { return m_nodes.size(); }

  /** The number of arcs. */
  int num_arcs() { return m_num_arcs; }

  /** Create a new node in the lattice. */
  Node &new_node();

  /** Create an arc. */
  void new_arc(int S, int E, std::string W, float a, float l);

  /** Read lattice from file in HTK format */
  void read(FILE *file);
  
  /** Write lattice in HTK format */
  void write(FILE *file);

  int initial_node_id; //!< Initial node id;
  int final_node_id; //!< Final node id;

private:
  std::vector<Node> m_nodes; //!< Node of the lattice
  int m_num_arcs; //!< Number of arcs in the lattice
};

#endif /* LATTICE_HH */
