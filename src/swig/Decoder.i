# -*- tab-width: 2 -*-

%module Decoder
%{
#include "Toolbox.hh"
%}

%exception {
	try {
		$action
	}
	catch (std::exception &e) {
		std::cerr << e.what() << std::endl;
		exit(1);
	}
}

%typemap(python,in) std::string& {
  if (!PyString_Check($input)) {
    PyErr_SetString(PyExc_TypeError, "not a string");
    return NULL;
  }
  $1 = new std::string(PyString_AsString($input),
		       PyString_Size($input));
}

%typemap(python,freearg) std::string& {
  delete $1;
}

%typemap(python,out) std::string& {
  $result = Py_BuildValue("s#",$1->c_str(),$1->size());
}

class Hypo {
};

%addmethods Hypo {
  double log_prob() { return self->log_prob; }
  int frame() { return self->frame; }
}

class HypoStack {
public:
  Hypo &at(int index);
	int size();
	bool empty();
	void sort();
	void prune(int top);
	double best_log_prob();
	int best_index();	
};

class Toolbox {
public:
  Toolbox();
  
  void hmm_read(const char *file);
  void lex_read(const char *file);
  const std::string &lex_word();
  const std::string &lex_phone();
  void ngram_read(const char *file);
	int ngram_lineno();

  // Lna
  void lna_open(const char *file, int models, int size);
  void lna_close();
	void lna_seek(int frame);

	// Expander
	void expand(int frame, int frames);
	const std::string &best_word();
	void print_words(int words);
	int find_word(const std::string &word);

	// Search
  void init(int expand_window, int stacks, int reserved_hypos);
	void sort(int frame, int top);
  bool expand_stack(int frame);
	void move_buffer(int frame);
	void go(int frame);
	bool run();
	void runto(int frame);

  int first_frame();
  int last_frame();
  HypoStack &stack(int frame);
	void prune(int frame, int top);
	int paths();

	void set_forced_end(bool forced_end);
  void set_hypo_limit(int hypo_limit);
  void set_word_limit(int word_limit);
  void set_word_beam(double word_beam);
  void set_lm_scale(double lm_scale);
  void set_lm_offset(double lm_offset);
  void set_token_limit(int limit);
	void set_state_beam(double beam);
	void set_hypo_beam(double beam);
	void set_global_beam(double beam);
  void set_max_state_duration(int duration);
	void set_verbose(bool verbose);

  void print_hypo(Hypo &hypo);
};
