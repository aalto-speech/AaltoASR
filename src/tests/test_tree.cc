#include "TreeGram.hh"
#include "TreeGramArpaReader.hh"

int
main(int argc, char *argv[])
{
  TreeGram g;
  TreeGramArpaReader r;

  // Read arpa or binary
  if (argc > 1 && atoi(argv[1]) == 1)
    r.read(stdin, &g);
  else
    g.read(stdin);

  // Write arpa or binary
  if (argc > 2) {
    if (atoi(argv[2]) == 1)
      r.write(stdout, &g);
    else if (atoi(argv[2]) == 2) {
      TreeGram::Iterator it(&g);
      while (it.next()) {
	printf("%g", it.node().log_prob);
	for (int i = 1; i <= it.order(); i++)
	  printf(" %s", g.word(it.node(i).word).c_str());
	printf(" %g\n", it.node().back_off);
      }
    }
  }
  else 
    g.write(stdout, false);
}
