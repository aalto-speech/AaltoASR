# -*- tab-width: 2 -*-

%module Decoder
%{
#include <iostream>
#include <exception>
#include "Hmm.hh"
#include "Lexicon.hh"
#include "Vocabulary.hh"
#include "LnaReader.hh"
%}

%exception LnaReader::go_to(int) {
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

%include Hmm.hh
%include Lexicon.hh
%include Vocabulary.hh
%include LnaReader.hh
