#ifndef RESCORE_HH
#define RESCORE_HH

#include "TreeGram.hh"
#include "Lattice.hh"

/** A class for expanding and rescoring lattices. */
class Rescore {
public:
  /** Context structure for expanding lattices. */
  struct Context {
    TreeGram::Gram gram; //!< Gram specifying the context
    int node_id; //!< Node id in the rescored lattice corresponding to context
    bool operator==(const Context &c) { return gram == c.gram; } //!< Compare
  };

  /** Default constructor. */
  Rescore();

  /** Expand and rescore the lattice with a language model. */
  void rescore(Lattice *src_lattice, TreeGram *tree_gram);

  /** Get the rescored lattice. */
  Lattice &rescored_lattice() { return m_rescored_lattice; }

private:

  /** Sort nodes of the source lattice topologically. */
  void sort_nodes();

  /** Create a new node corresponding to the context for the rescored
      lattice if necessary, and return the corresponding node. */
  Lattice::Node &find_or_create_node(int node_id, Context &context);

  TreeGram *m_tree_gram; //!< Language model used in rescoring
  Lattice *m_src_lattice; //!< The lattice to be rescored
  Lattice m_rescored_lattice; //!< The result lattice of the rescoring
  std::string m_sentence_start_label; //!< Sentence start label in LM
  std::string m_sentence_end_label; //!< Sentence end label in LM
  int m_sentence_end_id; //!< Sentence end label id in LM
  std::string m_null_label; //!< Null arc label

  //!< Node ids of the source lattice in sorted order.
  std::vector<int> m_sorted_nodes;

  //!< A vector containing a context vector for each source lattice node.
  std::vector<std::vector<Context> > m_node_contexts;
};


#endif /* RESCORE_HH */
