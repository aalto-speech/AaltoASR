#ifndef TREEGRAMARPAREADER_HH
#define TREEGRAMARPAREADER_HH

#include <map>
#include <stdio.h>

#include "TreeGram.hh"

class TreeGramArpaReader {
public:
  TreeGramArpaReader();
  void read(FILE *file, TreeGram *tree_gram);

private:
  void read_error();

  std::vector<int> m_counts;
  int m_lineno;
};

#endif /* TREEGRAMARPAREADER_HH */
