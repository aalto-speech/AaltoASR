#include <fstream>
#include <iostream>

#include "ArpaNgramReader.hh"

int
main(int argc, char *argv[])
{
  Vocabulary v;
  {
    std::ifstream in("vocab");
    if (!in) {
      std::cerr << "could not read vocabulary" << std::endl;
      exit(1);
    }
    v.read(in);
  }

  ArpaNgramReader r(v);

  try {
    r.read(std::cin);
  }
  catch (std::exception &e) {
    std::cerr << e.what() << " on line " << r.lineno() << std::endl;
    exit(1);
  }

  Ngram &n = r.ngram();
}
