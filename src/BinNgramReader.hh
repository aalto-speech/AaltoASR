#ifndef BINNGRAMREADER_HH
#define BINNGRAMREADER_HH

#include <stdio.h>
#include "Ngram.hh"
#include "Vocabulary.hh"
#include "Endian.hh"


class BinNgramReader{
public:
  void read(FILE *in, Ngram *ng);
  void write(FILE *out, Ngram *ng, bool reflip=true);
private:
  void flip_endian(Ngram *ng);
};
#endif
