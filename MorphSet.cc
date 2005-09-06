#include <vector>
#include "MorphSet.hh"
#include "str.hh"

MorphSet::MorphSet() : max_morph_length(0) { }

MorphSet::Node*
MorphSet::insert(char letter, const std::string &morph, Node *node)
{
  // Find a possible existing arc with the letter
  Arc *arc = node->first_arc;
  while (arc != NULL) {
    if (arc->letter == letter)
      break;
    arc = arc->sibling_arc;
  }

  // No existing arc: create a new arc
  if (arc == NULL) {
    node->first_arc = new Arc(letter, morph, new Node(NULL), node->first_arc);
    arc = node->first_arc;
  }

  // Update the existing arc if morph was set 
  else if (morph.length() > 0) {
    if (arc->morph.length() > 0) {
      fprintf(stderr, 
	      "ERROR: MorphSet::insert(): trying to redefine morph %s\n", 
	      morph.c_str());
      exit(1);
    }
    arc->morph = morph;
  }

  // Maintain the length of the longest morph
  if (morph.length() > max_morph_length)
    max_morph_length = morph.length;

  return arc->target_node;
}

MorphSet::Arc*
MorphSet::find_arc(char letter, const Node *node)
{
  Arc *arc = node->first_arc;
  while (arc != NULL) {
    if (arc->letter == letter)
      break;
    arc = arc->sibling_arc;
  }
  return arc;
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
MorphSet::write(FILE *file)
{
  fprintf(file, 
	  "digraph morph {\n"
	  "charset=Latin1;\n"
	  "node [label=foo];\n");

  // Depth-first search through the tree
  std::vector<Node*> stack(1, &root_node);
  std::string label;
  std::string color;
  while (!stack.empty()) {

    // Get the node
    Node *node = stack.back();
    stack.pop_back();
    
    // Iterate arcs of the node
    Arc *arc = node->first_arc;
    while (arc != NULL) {
      
      // Label of the arc
      label = arc->letter;
      color = "black";
      if (arc->morph.length() > 0) {
	label.append("\\n");
	label.append(arc->morph);
	color = "blue";
      }

      fprintf(file, "n%p -> n%p [label=\"%s\",color=%s];\n", 
	      node, arc->target_node, label.c_str(), color.c_str());
      stack.push_back(arc->target_node);
      arc = arc->sibling_arc;
    }
  }

  fprintf(file, "}\n");
}
