#include <iostream>

#include "ArpaNgramReader.hh"

int
main(int argc, char *argv[])
{
  ArpaNgramReader r;

  r.read(std::cin);

  Ngram &n = r.ngram();
}
