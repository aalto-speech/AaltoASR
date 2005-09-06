#ifndef MORPHSET_HH
#define MORPHSET_HH

#include <string>

/** A structure containing a set of morphs in a letter-tree format.
 * Input letters and output morphs are stored in arcs.  Nodes are just
 * placeholders for arcs. */
struct MorphSet {

  struct Node;

  /** Arc of a morph tree. */
  struct Arc {
    Arc(char letter, std::string morph, Node *target_node, Arc *sibling_arc)
      : letter(letter), morph(morph), target_node(target_node), 
	sibling_arc(sibling_arc) { }

    char letter; //!< Letter of the morph
    std::string morph; //!< Non-zero if complete morph
    Node *target_node; //!< Target node
    Arc *sibling_arc; //!< Pointer to another arc from the source node.
  };

  /** Node of a morph tree. */
  struct Node {
    Node() : first_arc(NULL) { }
    Node(Arc *first_arc) : first_arc(first_arc) { }
    Arc *first_arc; //!< The first arc of a list of arcs
  };

  
  Node root_node; //!< The root of the morph tree

  /** Insert a letter to a node (or follow an existing arc). 
   * \param letter = a letter to insert to the node
   * \param morph = a possible morph corresponding to this node (can be empty)
   * \param node = a node to which the letter is inserted
   * \return pointer to the created or existing node
   */
  Node *insert(char letter, const std::string &morph, Node *node);

  /** Read a morph set (one morph per line) */
  void read(FILE *file);

  /** Print the contents of the tree. */
  void show(FILE *file);

};

#endif /* MORPHSET_HH */
