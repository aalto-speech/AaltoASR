#ifndef TREEGRAMARPAREADER_HH
#define TREEGRAMARPAREADER_HH

#include <stdio.h>
#include "TreeGram.hh"
#include <map>

#ifdef USE_CL
#include "ClusterMap.hh"
#endif

class TreeGramArpaReader {
public:
  TreeGramArpaReader();
  void read(FILE *file, TreeGram *tree_gram);
  void write(FILE *file, TreeGram *tree_gram);

private:
  void read_error();

  std::vector<int> m_counts;
  int m_lineno;
};

#endif /* TREEGRAMARPAREADER_HH */
