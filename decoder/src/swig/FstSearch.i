%include "exception.i"
%include "std_string.i"

typedef std::string bytestype;
%typemap(out) bytestype {
	// This is for modern pythons (>=2.7)
  $result = PyBytes_FromStringAndSize(static_cast<const char*>($1.c_str()),$1.size());
	// This is for python <= 2.4
  //$result = Py_BuildValue("s#",$1->c_str(),$1->size());
}

%module FstSearch
%include FstSearch.hh
%{
  #include "FstSearch.hh"
%}

