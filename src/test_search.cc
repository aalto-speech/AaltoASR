#include <iomanip>
#include <fstream>

#include <errno.h>

#include "Timer.hh"
#include "LnaReaderCircular.hh"
#include "NowayHmmReader.hh"
#include "NowayLexiconReader.hh"
#include "Expander.hh"
#include "Search.hh"

using namespace std;

class Main {
public:
  Main();
  void run();
  void print_token(Lexicon::Token *token);
  NowayHmmReader hr;
  NowayLexiconReader lr;
  LnaReaderCircular lna;
};

Main::Main()
  : hr(),
    lr(hr.hmm_map(), hr.hmms()), 
    lna()
{
}

void
Main::print_token(Lexicon::Token *token)
{
  std::cout.setf(std::cout.right, std::cout.adjustfield);
  std::cout.setf(std::cout.fixed, std::cout.floatfield);
  std::cout.precision(2);

  std::cout << token->log_prob << std::endl;

  std::vector<Lexicon::Path*> paths;
  for (Lexicon::Path *path = token->path; path != NULL; path = path->prev)
    paths.push_back(path);
  for (int i = paths.size() - 1; i >= 0; i--) {
    Lexicon::Path *path = paths[i];
    int start = path->frame;
    int end = token->frame;
    if (i > 0)
      end = paths[i-1]->frame;
    double log_prob;
    if (i > 0)
      log_prob = paths[i-1]->log_prob - paths[i]->log_prob;
    else
      log_prob = token->log_prob - paths[i]->log_prob;
    std::cout.setf(std::cout.left, std::cout.adjustfield);
    std::cout 
      << setw(8) << start
      << setw(8) << end
      << setw(8) << end - start
      << setw(4) << hr.hmms()[paths[i]->hmm_id].label;
    std::cout << setw(10) << -log_prob << std::endl;
  }
}

void
Main::run()
{
  {
    std::cout << "load hmms" << std::endl;
    std::ifstream in("/home/neuro/thirsima/share/synt/pk_synt5.pho_mod");
    if (!in) {
      std::cerr << "could not read hmm file" << std::endl;
      exit(1);
    }
    hr.read(in);
  }

  {
    std::cout << "load lexicon" << std::endl;
//    std::ifstream in("/home/neuro/thirsima/share/synt/iso64000.lex");
//    std::ifstream in("/home/neuro/thirsima/share/synt/words20000.lex");
    std::ifstream in("/home/neuro/thirsima/share/synt/1000.lex");
//    std::ifstream in("tavu.lex");
//    std::ifstream in("synt.lex");
//    std::ifstream in("/home/neuro/thirsima/share/synt/pk_synt5.lex");
    if (!in) {
      std::cerr << "could not open lex file" << std::endl;
      exit(1);
    }
    try {
      lr.read(in);
    } 
    catch (std::exception &e) {
      std::cerr << e.what() << std::endl
		<< lr.word() << std::endl;
      exit(1);
    }
  }

  // 16k frames buffer
  lna.open("/home/neuro/thirsima/share/synt/pk_synt5.lna", 76, 1024*16);

  std::cout << "expand" << std::endl;

  Expander ex(hr.hmms(), lr.lexicon(), lna);

//    Timer timer;
//    timer.start();
//    timer.stop();
//    std::cout << std::endl << timer.sec() << " seconds" << std::endl;

  ex.set_token_limit(2000);

  Search search(ex, lr.vocabulary(), 200);

  search.run();

}

int
main(int argc, char *argv[])
{
  try {
    Main m;
    m.run();
  }
  catch (exception &e) {
    cerr << e.what() << std::endl;
    cerr << strerror(errno) << std::endl;
    exit(1);
  }

  exit(0);
}

