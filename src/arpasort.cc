#include "TreeGramArpaReader.hh"

int
main(int argc, char *argv[])
{
  TreeGramArpaReader reader;
  TreeGram gram;

  reader.read(stdin, &gram);
  reader.write(stdout, &gram);
}
