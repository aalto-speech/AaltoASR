#ifndef WORDGRAPH_HH
#define WORDGRAPH_HH

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <vector>
#include "Vocabulary.hh"

/** 
 * WordGraph reads Sphinx-3 lattice files
 * @file wordgraph.h
 * @date 25.6.2003
 */
class WordGraph {

 public:

  typedef enum { NODES, EDGES } ReadMode;

  struct RangeError : public std::exception {
    virtual const char *what() const throw()
    { return "WordGraph: range error"; }
  };

  class Node {
  public:
    int frame;
    int word_id;
    std::vector<int> edges;
  };
  
  class Edge {
  public:
    int target_node;
    float ac_log_prob;
  };
  
  /**
   * @param file stream to read lattice from
   * @param vocab vocabulary to convert words->int
   */
  WordGraph(FILE *file, Vocabulary *vocab);

  /**
   * Destroys lattice
   */
  ~WordGraph();
  
  /**
   * reads word graph from a file
   * @return 1 if read succesful, 0 otherwise
   */
  int read();
  
  /**
   * @param index node number to get
   * @return Node& with corresponding index
   * if index out of range, throws RangeError
   */
  WordGraph::Node & node(int index);
  
  /**
   * @param index edge number to get
   * @return Edge& with corresponding index
   * if index out of range, throws RangeError
   */
  WordGraph::Edge & edge(int index);

  /**
   * @param frame frame starting time
   * @return vector of nodes that start from the given frame
   * if frame out of range, throws RangeError
   */
  std::vector<WordGraph::Node> & frame(int frame);

  /**
   * @return number of nodes in lattice
   */
  int node_count();

  /**
   * @return number of edges in lattice
   */
  int edge_count();

  /**
   * @return number of frames
   */
  int frame_count();

 private:

  bool ok;
  FILE *m_file;
  Vocabulary *m_vocab;
  ReadMode m_read_mode;
  std::vector<WordGraph::Node> m_nodes;
  std::vector<WordGraph::Edge> m_edges;
  std::vector<std::vector<WordGraph::Node> > m_frames;
  int m_frame_count;
  int m_node_count;
  int m_edge_count;
  int m_best_seg_ascr;
};

#endif /* WORDGRAPH_HH */
