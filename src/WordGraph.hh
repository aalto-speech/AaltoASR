#ifndef WORDGRAPH_H
#define WORDGRAPH_H

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "Vocabulary.hh"

typedef enum { NODES, EDGES } ReadMode;

/** 
 * WordGraph imports lattice files
 * @file wordgraph.h
 * @date 6.6.2003
 */
class WordGraph {

 public:
  
  class Node {
  public:
    int frame;
    int word_id;
    int pronunciation;
    std::vector<int> edges;
  };
  
  class Edge {
  public:
    int target_node;
    float ac_log_prob;
  };
  
  /**
   * @param file stream to read lattice from
   */
  WordGraph(FILE *file, Vocabulary *vocab);

  /**
   * Destroys lattice
   */
  ~WordGraph();
  
  /**
   * constructs word graph from a file
   * @return 1 if read succesful, 0 otherwise
   */
  int read();
  
  /**
   * @param index node number to get
   * @return Node& with corresponding index
   * if index out of range, result undefined
   */
  Node &node(int index);
  
  /**
   * @param index edge number to get
   * @return Edge& with corresponding index
   * if index out of range, result undefined
   */
  Edge &edge(int index);

  /**
   * @return number of nodes in lattice
   */
  int nodes();

  /**
   * @return number of edges in lattice
   */
  int edges();

  /**
   * @param frame frame starting time
   * @return vector of node*s that start from the given frame
   */
  std::vector<Node*> frame_nodes(int frame);

 private:

  FILE *file;
  Vocabulary *vocab;
  bool ok_flag;
  ReadMode read_mode;
  WordGraph::Node **node_table;
  std::vector<WordGraph::Edge *> edge_vector;
  int frame_count;
  int node_count;
  int edge_count;
  int best_seg_ascr; // ?
};

#endif /* WORDGRAPH_H */



























