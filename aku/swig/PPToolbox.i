# -*- tab-width: 2 -*-

%include exception.i

%module PPToolbox
%{
// SWIG defines c_abs which conflicts with c_abs
// imported by f2c.h of Lapack++.
#undef c_abs
#include "PhoneProbsToolbox.hh"
%}

%exception {
  try {
    $action
  }
  catch (std::exception &e) {
    std::cerr << e.what() << std::endl;
    SWIG_exception(SWIG_RuntimeError, "Exception");
  }
  catch (std::string &e) {
    std::cerr << "Exception: ";
    std::cerr << e << std::endl;
    SWIG_exception(SWIG_RuntimeError, "Exception");
  }
  catch (...) {
    SWIG_exception(SWIG_RuntimeError, "Unknown exception");
  }
}

#if defined(SWIGPYTHON)
%typemap(in) std::string& {
  if (!PyString_Check($input)) {
    PyErr_SetString(PyExc_TypeError, "not a string");
    return NULL;
  }
  $1 = new std::string(PyString_AsString($input),
           PyString_Size($input));
}

%typemap(freearg) std::string& {
  delete $1;
}

%typemap(out) std::string& {
  $result = Py_BuildValue("s#",$1->c_str(),$1->size());
}

%typemap(in) FILE* {
	if (!(PyFile_Check($input))) {
		PyErr_SetString(PyExc_TypeError, "not a file pointer");
		return NULL;
	}
	$1=PyFile_AsFile($input);
}
#endif

class PPToolbox {
public:
  void read_configuration(const std::string &cfgname);
  void read_models(const std::string &base);
  void generate_to_fd(const int in, const int out, const bool raw_flag);
  void generate(const std::string &input_name, const std::string &output_name, const bool raw_flag);
  void set_clustering(const std::string &clfile_name, double eval_minc, double eval_ming);
  //set_clustering() //FIXME: implement to speed up

  //set_raw_flag(bool x);
  //set_lnabytes(int x);

private:
  conf::Config config;
  FeatureGenerator gen;
  HmmSet model;
  std::vector<float> obs_log_probs;

  void write_int(FILE *fp, unsigned int i);
};
