#include "TreeGram.hh"
#include "TreeGramArpaReader.hh"

int
main(int argc, char *argv[])
{
  TreeGram g;

  // Read the model
  {
    TreeGramArpaReader r;
    r.read(stdin, &g);
  }

  // Print the model
  {
    TreeGram::Iterator i(&g);
    while (i.next()) {
      printf("( ");
      for (int o = 1; o <= i.order(); o++)
	printf("%d ", i.node(o).word);
      printf(") %g %g\n", i.node().log_prob, i.node().back_off);
    }
  }
}
