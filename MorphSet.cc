#include <vector>
#include "MorphSet.hh"
#include "str.hh"

MorphSet::Node*
MorphSet::insert(char letter, const std::string &morph, Node *node)
{
  // Find an existing arc
  Arc *arc = node->first_arc;
  while (arc != NULL) {
    if (arc->letter == letter)
      break;
    arc = arc->sibling_arc;
  }

  if (arc == NULL) {
    node->first_arc = new Arc(letter, morph, new Node(NULL), node->first_arc);
    arc = node->first_arc;
  }
  else if (morph.length() > 0) {
    if (arc->morph.length() > 0) {
      fprintf(stderr, 
	      "ERROR: MorphSet::insert(): trying to redefine morph %s\n", 
	      morph.c_str());
      exit(1);
    }
    arc->morph = morph;
  }

  return arc->target_node;
}

void
MorphSet::read(FILE *file)
{
  std::string line;
  while (str::read_line(&line, file, true)) {

    // Skip empty lines
    str::clean(&line, " \t\r\n");
    if (line.length() == 0)
      continue;

    // Create arcs
    Node *node = &root_node;
    for (int i = 0; i < (int)line.length(); i++)
      node = insert(line[i], i < (int)line.length() - 1 ? "" : line, node);
  }
}

void
MorphSet::show(FILE *file)
{
  std::vector<Node*> stack(1, &root_node);
  while (!stack.empty()) {
    Node *node = stack.back();
    fprintf(file, "node %p\n", node);
    stack.pop_back();
    Arc *arc = node->first_arc;
    while (arc != NULL) {
      fprintf(file, "  arc %c \"%s\" %p\n", arc->letter, arc->morph.c_str(),
	      arc->target_node);
      stack.push_back(arc->target_node);
      arc = arc->sibling_arc;
    }
  }
}
