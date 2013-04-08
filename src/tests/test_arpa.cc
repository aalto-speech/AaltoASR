#include "ArpaNgramReader.hh"

class Main {
public:
  Main();
  void print_raw();
  void print(std::vector<int> &history, int order, int max_order);
  void run(int argc, char *argv[]);
  ArpaNgramReader r;
  Ngram &n;
};

Main::Main()
  : r(), n(r.ngram())
{
}

void
Main::print_raw()
{
  for (int i = 0; i < n.nodes(); i++) {
    Ngram::Node *node = n.node(i);
    std::cout << i 
	      << " " << n.word(node->word)
	      << " " << node->first
	      << std::endl;
  }
}

void
Main::print(std::vector<int> &history, int order, int max_order)
{
  Ngram::Node *node = n.node(history.back());
  if (order == max_order) {
    std::cout << node->log_prob;
    for (int i = 0; i < history.size(); i++) {
      Ngram::Node *node = n.node(history[i]);
      std::cout << " " << n.word(node->word);
    }
    if (node->back_off != 0)
      std::cout << " " << node->back_off;
    std::cout << " (" << node->first << ")" << std::endl;
  }

  else if (node->first >= 0) {
    int end = (node+1)->first;
    if (end == -1)
      end = n.nodes();
    for (int i = node->first; i < end; i++) {
      history.push_back(i);
      print(history, order + 1, max_order);
      history.pop_back();
    }
  }
}

void
Main::run(int argc, char *argv[])
{
  r.read(argv[1]);
  std::vector<int> history;

  std::cerr << "nodes: " << n.nodes() << std::endl;

//  print_raw();

  std::cout << "\\data\\" << std::endl;

  for (int j = 0; j < n.size() + 1; j++) {
    history.clear();
    history.push_back(j);
    print(history, 1, n.order());
  }

  std::cout << "\\end\\" << std::endl;
}

int
main(int argc, char *argv[])
{
  Main m;
  try {
    m.run(argc, argv);
  }
  catch (std::exception &e) {
    std::cerr << e.what() << std::endl;
    std::cerr << m.r.lineno() << std::endl;
    exit(1);
  }
}
