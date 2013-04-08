#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include "str.hh"
#include "Latticer.hh"

Latticer::Latticer() : 
  morph_set(NULL), 
  word_boundary_label("<w>") 
{ 
}

void
Latticer::create_lattice(FILE *input, FILE *output)
{
  text.reserve(morph_set->max_morph_length * 2);

  int src_node_pos = 1;
  int last_pos = 0;
  bool eof_reached = false;
  bool was_word_boundary = false;

  fprintf(output, "0 1 %s\n", word_boundary_label.c_str());

  while (1) {

    // Ensure that we have enough text in the text buffer
    if (!eof_reached && (int)text.length() <= morph_set->max_morph_length) {
      bool ok = str::read_string(&text, morph_set->max_morph_length, 
				 input, true);

      // Error or end of file?
      if (!ok) {
	if (ferror(input)) {
	  perror("ERROR: Latticer::create_lattice(): read failed");
	  exit(1);
	}
	eof_reached = true;
	text.append(" ");
      }
    }
    
    // Everything processed
    if (text.length() == 0) {
      assert(eof_reached);
      break;
    }

    // Insert a single word boundary on whitespace
    if (strchr(" \n\r\t", text[0]) != NULL) {
      text.erase(text.begin());
      if (!was_word_boundary) {
	fprintf(output, "%d %d %s\n", src_node_pos, src_node_pos + 1,
		word_boundary_label.c_str());
	src_node_pos++;
	if (src_node_pos > last_pos)
	  last_pos = src_node_pos;
      }
      was_word_boundary = true;
      continue;
    }
    else
      was_word_boundary = false;

    // Output all morphs in the morph set that match the text
    MorphSet::Node *src_node = &morph_set->root_node;
    size_t pos = 0;
    while (src_node != NULL && pos < text.length()) {
      MorphSet::Arc *arc = morph_set->find_arc(text[pos], src_node);
      if (arc == NULL)
	break;

      // Output a possible morph 
      if (arc->morph.length() > 0) {
	int tgt_node_pos = src_node_pos + pos + 1;
	fprintf(output, "%d %d %s\n", src_node_pos, tgt_node_pos,
		arc->morph.c_str());
	if (tgt_node_pos > last_pos)
	  last_pos = tgt_node_pos;
      }

      src_node = arc->target_node;
      pos++;
    }

    // Move one character forward
    src_node_pos++;
    text.erase(text.begin());
  }
  fprintf(output, "%d\n", last_pos);
}
