#include <fstream>
#include <iostream>

#include "ArpaNgramReader.hh"

int
main(int argc, char *argv[])
{
  // Vocabulary
  Vocabulary v;
  {
    std::cout << "reading vocab" << std::endl;
    std::ifstream in(argv[1]);
    if (!in) {
      std::cerr << "could not read vocabulary" << std::endl;
      exit(1);
    }
    v.read(in);
  }

  // Language model
  ArpaNgramReader r(v);
  try {
    std::cout << "reading arpa" << std::endl;
    std::ifstream in(argv[2]);
    if (!in) {
      std::cerr << "could not read vocabulary" << std::endl;
      exit(1);
    }
    r.read(in);
  }
  catch (std::exception &e) {
    std::cerr << e.what() << " on line " << r.lineno() << std::endl;
    exit(1);
  }

  // Test
  Ngram &n = r.ngram();
  std::string str;
  str.reserve(128);

  int order = 3;

  std::deque<int> history(order, -1);
  while (std::cin >> str) {
    int word = v.index(str);

    history.pop_front();
    history.push_back(word);

    
  }
}
