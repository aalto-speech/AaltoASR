%include "exception.i"
%include "std_string.i"

// The recognizer operates on iso-8859-15 charset, we cannot return the string directly
// Let us return it as bytes, which can then be decoded in python to get a real 
// python string s.decode('iso-8859-15') 
typedef std::string bytestype;
%typemap(out) bytestype {
  $result = PyBytes_FromStringAndSize(static_cast<const char*>($1.c_str()),$1.size());
}

%module FstSearch
%include FstSearch.hh
%{
  #include "FstSearch.hh"
%}

