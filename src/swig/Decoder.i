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

class HypoStack {
public:
  Hypo &at(int index);
	int size();
};

class Toolbox {
public:
  Toolbox();
  
  void hmm_read(const char *file);
  void lex_read(const char *file);
  void ngram_read(const char *file);

  // Lna
  void lna_open(const char *file, int models, int size);
  void lna_close();

	// Search
  void init(int frames, int hypos);
  bool expand(int frame);
  int earliest_frame();
  int last_frame();
  HypoStack &stack(int frame);
  void set_hypo_limit(int hypo_limit);
  void set_word_limit(int word_limit);
  void set_lm_scale(double lm_scale);
  void set_token_limit(int limit);
  void set_max_state_duration(int duration);

  void print_hypo(Hypo &hypo);
};
