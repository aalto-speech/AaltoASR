#ifndef LATTICER_HH
#define LATTICER_HH

#include <string>
#include <stdio.h>
#include "MorphSet.hh"

/** A class for segmenting a text corpus into a morph lattice that
 * contains all possible morph paths through the text. */ 
class Latticer {
public:
  /** Default constructor. */
  Latticer();

  /** Create a latticer with a morph set and text corpus file. */
  Latticer(const MorphSet *morph_set, FILE *file);

  /** Create the lattice by segmenting the text corpus into morphs. */
  void create_lattice(FILE *output = stdout);

  MorphSet *morph_set; //!< The morph set to use for segmenting the text
  FILE *input; //!< The file from which the text corpus is read

  std::string text; //!< A buffer containing part of the text
};

#endif /* LATTICER_HH */
