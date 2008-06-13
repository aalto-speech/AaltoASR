#ifndef BINNGRAMREADER_HH
#define BINNGRAMREADER_HH

#include <stdio.h>
#include "Ngram.hh"
#include "Vocabulary.hh"
#include "misc/Endian.hh"

class BinNgramReader {
public:
  void read(FILE *file, Ngram *ng);
  void write(FILE *file, Ngram *ng, bool reflip=true);
private:
  void flip_endian(Ngram *ng);
};
#endif
