#include <stdio.h>

#include "TreeGram.hh"
#include "TreeGramArpaReader.hh"

int main(int argc, char *argv[]) 
{
  TreeGramArpaReader reader;
  TreeGram gram;

  fputs("reading arpa from stdin, writing binary to stdout\n", stderr);

  reader.read(stdin, &gram);
  gram.write(stdout, true);
}
