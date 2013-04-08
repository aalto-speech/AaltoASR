#include "TreeGram.hh"
#include "TreeGramArpaReader.hh"
#include "tools.hh"

int
main(int argc, char *argv[])
{
  
  TreeGram g;
  TreeGramArpaReader r;

  FILE *f = fopen(argv[1], "r");
  if (!f) {
    fprintf(stderr, "could not open %s\n", argv[1]);
    exit(1);
  }
  r.read(f, &g);
  fclose(f);

  std::string line;
  std::vector<std::string> vec;
  TreeGram::Iterator it;

  while (1) {
    // Read line
    printf(">");
    fflush(stdout);
    if (!read_line(&line, stdin))
      break;
    chomp(&line);
    split(line, " \t", true, &vec);

    // Go to gram
    if (vec[0] == "go") {
      TreeGram::Gram gram;
      for (int i = 1; i < vec.size(); i++) {
	int index = g.word_index(vec[i]);
	gram.push_back(index);
      }
      it = g.iterator(gram);
    }

    // Next
    if (vec[0] == "mic") {
      int delta;
      if (vec.size() < 2)
	delta = 1;
      else
	delta = atoi(vec[1].c_str());
      if (!it.move_in_context(delta))
	printf("can not go further\n");
    }

    // Up
    if (vec[0] == "up") {
      if (!it.up())
	printf("can not go further\n");
    }

    // Down
    if (vec[0] == "down") {
      if (!it.down())
	printf("can not go further\n");
    }

    // Print iterator
    for (int i = 1; i <= it.order(); i++) {
      printf("%s(%d) ", g.word(it.node(i).word).c_str(), it.node(i).word);
    }
    printf("\n");
  }
}
