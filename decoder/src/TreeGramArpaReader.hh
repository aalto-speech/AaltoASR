// Routines for reading and writing arpa format files from and to the 
// internal prefix tree format.
#ifndef TREEGRAMARPAREADER_HH
#define TREEGRAMARPAREADER_HH

#include <stdio.h>
#include "TreeGram.hh"

class TreeGramArpaReader {
public:
  TreeGramArpaReader();
  void read(FILE *file, TreeGram *tree_gram, bool add_missing_unigrams=false);
  void write(FILE *file, TreeGram *tree_gram);
  void write_interpolated(FILE *file, TreeGram *treegram);
};

#endif /* TREEGRAMARPAREADER_HH */
