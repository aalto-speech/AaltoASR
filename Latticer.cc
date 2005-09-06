#include "Latticer.hh"

Latticer::Latticer() : morph_set(NULL), input(NULL) { }

Latticer::Latticer(const MorphSet *morph_set, FILE *input)
  : morph_set(morph_set), input(input) { }

void
Latticer::create_lattice(FILE *output)
{
  text.reserve(morph_set->max_morph_length * 2);

  bool eof_reached = false;
  while (1) {

    // Ensure that we have enough text in the text buffer
    if (!eof_reached && text.length() <= morph_set->max_morph_length) {
      bool ok = str::read_string(&text, morph_set->max_morph_length, 
				 input, true);

      // Error or end of file?
      if (!ok) {
	if (ferror(input)) {
	  perror("ERROR: Latticer::create_lattice(): read failed");
	  exit(1);
	}
	eof_reached = true;
      }
    }
    
    // Everything processed
    if (text.length() == 0)
      break;

    // Find all morphs in the morph set that match the text
    MorphSet::Node *node = morph_set->root_node;
    size_t pos = 0;
    while (node != NULL && pos < text.length()) {
      node = morph_set->find_arc(text

      pos++;
    }
  }
}
