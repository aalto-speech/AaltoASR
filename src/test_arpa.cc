#include <math.h>
#include <fstream>
#include <iostream>
#include <deque>

#include "ArpaNgramReader.hh"

int
main(int argc, char *argv[])
{
  // Vocabulary
  Vocabulary v;
  v.read(argv[1]);

  // Language model
  ArpaNgramReader r(v);
  try {
    r.read(argv[2]);
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

  std::cout << "evaluating" << std::endl;
  std::deque<int> history;
  while (std::cin >> str) {
    int word = v.index(str);

    while (history.size() >= order)
      history.pop_front();
    history.push_back(word);

    std::cout << n.log_prob(history.begin(), history.end()) << std::endl;
  }
}
