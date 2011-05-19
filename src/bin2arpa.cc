#include <stdio.h>

#include "TreeGram.hh"
#include "TreeGramArpaReader.hh"

int main(int argc, char *argv[]) 
{
  TreeGramArpaReader arpa;
  TreeGram gram;

  fputs("reading binary from stdin, writing arpa to stdout\n", stderr);

  gram.read(stdin, true);
  arpa.write(stdout, &gram);
}
