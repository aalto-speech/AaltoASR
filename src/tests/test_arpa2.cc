#include <math.h>
#include <fstream>
#include <iostream>
#include <deque>

#include "ArpaNgramReader.hh"

int
main(int argc, char *argv[])
{
  // Language model
  ArpaNgramReader r;
  try {
    std::cerr << "ngram" << std::endl;
    r.read(argv[1]);
  }
  catch (std::exception &e) {
    std::cerr << e.what() << " on line " << r.lineno() << std::endl;
    exit(1);
  }

  // Test
  Ngram &n = r.ngram();
  std::string str;
  str.reserve(128);

  std::cerr << "evaluating" << std::endl;
  std::deque<int> history;
  while (std::cin >> str) {
    int word = n.index(str);

    while (history.size() >= n.order())
      history.pop_front();
    history.push_back(word);

    std::cout.precision(10);
    std::cout << n.log_prob(history.begin(), history.end()) 
	      << "\t" << word << "\t" << str << std::endl;
  }
}
