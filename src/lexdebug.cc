#include <vector>
#include <math.h>
#include "Toolbox.hh"

typedef TPLexPrefixTree::Node Node;
Toolbox t;

void
print_tree(Node *root)
{
  std::vector<Node*> stack(1, root);

  printf("digraph lexicon {\n");
  while (!stack.empty()) {
    Node *node = stack.back();
    stack.pop_back();
    if (node->flags & NODE_DEBUG_PRINTED)
      continue;

    const char *color = node->state == NULL ? "red" : "blue";
    const char *fill_color = node->word_id == -1 ? "white" : "gray";
    
    printf("\t%d [label=\"%d\\n%s\\n%04X\",style=filled,color=%s,fillcolor=%s];\n", 
	   node->node_id, node->node_id, 
	   node->word_id == -1 ? "-" : t.word(node->word_id).c_str(), 
	   node->flags & 0x3fff, color, fill_color);
    node->flags |= NODE_DEBUG_PRINTED;

    for (int i = 0; i < node->arcs.size(); i++) {
      printf("\t\t%d -> %d;\n", node->node_id, node->arcs[i].next->node_id);
      stack.push_back(node->arcs[i].next);
    }
  }
  printf("}\n");
}

int
main(int argc, char *argv[])
{
  if (argc != 3) {
    fprintf(stderr, "usage: lexdebug HMM LEX\n");
    exit(1);
  }

  try {

    // Create decoder
    t.select_decoder(0); // Token-pass
    t.set_cross_word_triphones(true);
    t.set_lm_lookahead(1);

    // Load files
    t.set_word_boundary("<w>");
    t.hmm_read(argv[1]);
    t.lex_read(argv[2]);
    t.set_sentence_boundary("<s>", "</s>");

    TPLexPrefixTree &lex = t.debug_get_tp_lex();
    Node *root = lex.start_node();
    lex.debug_prune_dead_ends(root);
    print_tree(root);
  }
  catch (std::exception &e) {
    fprintf(stderr, "exception: %s\n", e.what());
  }
}
