#include <fstream>
#include <iostream>
#include <deque>

#include <cmath>

#include "ArpaNgramReader.hh"

int
main(int argc, char *argv[])
{
  // Language model
  ArpaNgramReader r;
  try {
    r.set_oov(argv[1]);
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

  int order = n.order();

  std::cout << "evaluating" << std::endl;
  std::deque<int> history;
  while (std::cin >> str) {
    int word = n.index(str);

    while (history.size() >= order)
      history.pop_front();
    history.push_back(word);

    std::cout << n.log_prob(history.begin(), history.end()) << std::endl;
  }
}
