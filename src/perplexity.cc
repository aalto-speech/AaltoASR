#include <math.h>
#include "tools.hh"
#include "TreeGram.hh"
#include "TreeGramArpaReader.hh"

int
main(int argc, char *argv[])
{
  TreeGram tree_gram;
  TreeGramArpaReader reader;
  
  std::string name(argv[1]);

  FILE *file = fopen(name.c_str(), "r");
  if (!file) {
    fprintf(stderr, "could not open %s\n", name.c_str());
    exit(1);
  }

//  tree_gram.read(file);
  reader.read(file, &tree_gram);

  TreeGram::Gram gram;
  std::string line;
  std::vector<std::string> words;
  while (read_line(&line, stdin)) {
    chomp(&line);
    split(line, " \t", true, &words);

    for (int i = 0; i < words.size(); i++) {
      if (gram.size() >= tree_gram.order())
	gram.pop_front();
      gram.push_back(tree_gram.word_index(words[i]));
      printf("%s %g\n", words[i].c_str(), pow(10, tree_gram.log_prob(gram)));
    }
  }
}
