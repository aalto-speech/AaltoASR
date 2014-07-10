# -*- tab-width: 2 -*-

%include exception.i

%module FstDecoder
%{
#include <cstddef>
%}

%exception {
  try {
    $action
  }
  catch (std::exception &e) {
    std::cerr << e.what() << std::endl;
    SWIG_exception(SWIG_RuntimeError, "Exception");
  }
  catch (...){
    SWIG_exception(SWIG_RuntimeError, "Unknown exception received by swig");
  }
}

#if defined(SWIGPYTHON)
%include "std_string.i"
%include "std_vector.i"

typedef std::string bytestype;
%typemap(out) bytestype& {
  $result = PyBytes_FromStringAndSize(static_cast<const char*>($1->c_str()),$1->size());
}

// Instantiate templates used 
%template(StringVector) std::vector<std::string>;
%template(FloatVector) std::vector<float>;
#endif

// A decoder for FSTs

// The recognizer operates on iso-8859-15 charset, we cannot return the string directly
// Let us return it as bytes, which can then be decoded in python to get a real 
// python string s.decode('iso-8859-15') 
typedef std::string bytestype;
%typemap(out) bytestype {
  $result = PyBytes_FromStringAndSize(static_cast<const char*>($1.c_str()),$1.size());
}

// Ignored argument.
%typemap(in, numinputs=0) float *confidence_retval (float temp) {
    $1 = &temp;
}

%typemap(argout) float *confidence_retval {
  %append_output(PyFloat_FromDouble((double) (*$1)));
}

%module FstDecoder
%include FstSearch.hh

// Needed to make the FstConfidence inheritance work
%template(FstSearchC) FstSearch_base<FstConfidenceToken>;

%include FstConfidence.hh
%{
  #include "FstConfidence.hh"
%}



