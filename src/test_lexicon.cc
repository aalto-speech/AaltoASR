#include <fstream>

#include "NowayHmmReader.hh"
#include "NowayLexiconReader.hh"

using namespace std;

class Main {
 public:
  Main() : hr(), lr(hr.hmm_map(), hr.hmms()) { }
  void print_node(const Lexicon::Node *node, int level, bool print_spaces);
  void run();
  NowayHmmReader hr;
  NowayLexiconReader lr;
};

void
Main::print_node(const Lexicon::Node *node, int level, bool print_spaces)
{
  const int width = 2;

  if (node->hmm_id >= 0) {
    if (print_spaces)
      for (int i = 0; i < level; i++)
        cout << " ";
    cout << hr.hmms()[node->hmm_id].label;
    cout << (node->word_id == -1 ? " " : "*");
    level += width;
  }
  int i;
  for (i = 0; i < node->next.size(); i++)
    print_node(node->next[i], level, i != 0);
  if (i == 0)
    cout << endl;
}

void
Main::run()
{
  {
    std::ifstream in("test.hmm");
    hr.read(in);
  }

  {
    std::ifstream in("test.lex");
    lr.read(in);
  }

  print_node(lr.lexicon().root(), 0, false);
}

int
main(int argc, char *argv[])
{
  try {
    Main m;
    m.run();
  } 
  catch (std::exception &e) {
    std::cerr << e.what() << std::endl;
  }
}
