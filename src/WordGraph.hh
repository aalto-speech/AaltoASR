#ifndef WORDGRAPH_HH
#define WORDGRAPH_HH

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "Vocabulary.hh"

typedef enum { NODES, EDGES } ReadMode;

/** 
 * WordGraph imports lattice files
 * @file wordgraph.h
 * @date 10.6.2003
 */
class WordGraph {

 public:
  
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
   * if index out of range, result undefined
   */
  WordGraph::Node & node(int index);
  
  /**
   * @param index edge number to get
   * @return Edge& with corresponding index
   * if index out of range, result undefined
   */
  WordGraph::Edge & edge(int index);

  /**
   * @param frame frame starting time
   * @return vector of node*s that start from the given frame
   */
  std::vector<WordGraph::Node *> & frame(int frame);

  /**
   * @return number of nodes in lattice
   */
  int nodes();

  /**
   * @return number of edges in lattice
   */
  int edges();

  /**
   * @return number of frames
   */
  int frames();

 private:

  FILE *file;
  Vocabulary *vocab;
  bool ok_flag;
  ReadMode read_mode;
  WordGraph::Node **node_table;
  std::vector<WordGraph::Node *> **frame_table;
  std::vector<WordGraph::Edge *> edge_vector;
  int frame_count;
  int node_count;
  int edge_count;
  int best_seg_ascr; // ?
};

#endif /* WORDGRAPH_HH */



























