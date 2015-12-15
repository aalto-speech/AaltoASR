# -*- tab-width: 2 -*-

%include exception.i

%module Decoder
%{
#include <cstddef>
#include "fsalm/LM.hh"
#include "Toolbox.hh"
using namespace fsalm;
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

#ifdef SWIGPYTHON


%include "std_string.i"
%include "std_vector.i"

typedef std::string bytestype;
%typemap(out) bytestype& {
	// This is for modern pythons (>=2.7)
  $result = PyBytes_FromStringAndSize(static_cast<const char*>($1->c_str()),$1->size());
	// This is for python <= 2.4
  //$result = Py_BuildValue("s#",$1->c_str(),$1->size());
}
// This migth work for modern pythons only (>=3.0)
//#if defined(PY3)
typedef std::vector<std::pair<std::string,std::pair<double,double> > > timed_token_stream_type;
%typemap(out) timed_token_stream_type& {
  $result = PyList_New(0);
  for(size_t i = 0; i < $1->size(); i++){
    PyObject *str,*start_time,*end_time,*new_token;

    str = PyBytes_FromStringAndSize(static_cast<const char*>($1->at(i).first.c_str()),$1->at(i).first.size());
    start_time = PyFloat_FromDouble(static_cast<const double>($1->at(i).second.first));
    end_time = PyFloat_FromDouble(static_cast<const double>($1->at(i).second.second));
    new_token = PyDict_New();
    PyDict_SetItem(new_token,PyString_FromString("token"),str);
    PyDict_SetItem(new_token,PyString_FromString("starttime"),start_time);
    PyDict_SetItem(new_token,PyString_FromString("endtime"),end_time);
    PyList_Append($result,new_token);
    Py_DECREF(str);
    Py_DECREF(start_time);
    Py_DECREF(end_time);
    Py_DECREF(new_token);
  }
}
//#endif


// Instantiate templates used 
%template(StringVector) std::vector<std::string>;
%template(FloatVector) std::vector<float>;

#if !defined(PY3)
// This is disabled, since doesn't exists for py3 any more
%typemap(in) FILE* {
	if (!(PyFile_Check($input))) {
		PyErr_SetString(PyExc_TypeError, "not a file pointer");
		return NULL;
	}
	$1=PyFile_AsFile($input);
}
#endif
#endif

%apply float *OUTPUT { float *score };

class LM {
public:
	LM();
  void read_arpa(FILE *file);
  void read(FILE *file);
	int initial_node_id() const;
  int walk(int node_id, int symbol, float *score) const;
};

%extend LM {
	int sym(const std::string &str) { return self->symbol_map().index(str); }
}

class Toolbox {
public:
  Toolbox(const char *hmm_path, const char *dur_path);
  ~Toolbox();

  const std::vector<Hmm> &hmms();
  void lex_read(const char *file);
  const std::string &lex_word();
  const std::string &lex_phone();

  void interpolated_ngram_read(const std::vector<std::string>, const std::vector<float>);
  void interpolated_lookahead_ngram_read(const std::vector<std::string>, const std::vector<float>);

  int ngram_read(const char *file, const bool binary, const bool quiet);
  int ngram_read(const char *file, const bool binary);
  int ngram_read(const char *file);
  void htk_lattice_grammar_read(const char *file, bool quiet);
  void fsa_lm_read(const char *file, bool binary, bool quiet);
  void fsa_lm_read(const char *file, bool binary);
  void read_word_classes(const char *file);
  void read_lookahead_ngram(const char *file, const bool binary, bool quiet);
  void read_lookahead_ngram(const char *file, const bool binary);
  void read_lookahead_ngram(const char *file);

  // lna
  void lna_open(const char *file, int size);
  void lna_open_fd(const int fd, int size);
  void lna_close();
  void lna_seek(int frame);
  Acoustics &acoustics();
  void use_one_frame_acoustics();
  void set_one_frame(int frame, const std::vector<float> log_probs);

  void reset(int frame);
  void set_end(int frame);
  bool run();
  int frame();

  void write_word_graph(const std::string &file_name);
  void print_best_lm_history();
  void print_best_lm_history_to_file(FILE *out);
  const bytestype &best_hypo_string(bool print_all, bool output_time);
  const timed_token_stream_type &best_timed_hypo_string(bool print_all);
  void write_state_segmentation(const std::string &file);

  void set_prune_similar(int prune_similar);
  void set_lm_scale(float lm_scale);
  void set_token_limit(int limit);
  void set_duration_scale(float scale);
  void set_transition_scale(float scale);
  void set_global_beam(float beam);
  void set_word_end_beam(float beam);
  void set_eq_depth_beam(float beam);
  void set_eq_word_count_beam(float beam);
  void set_fan_in_beam(float beam);
  void set_fan_out_beam(float beam);
  void set_tp_state_beam(float beam);
  void set_split_multiwords(bool b);
  void set_cross_word_triphones(bool cw_triphones);
  void set_silence_is_word(bool b);
  void set_ignore_case(bool b);		
  void set_lm_lookahead(int lmlh);
  void set_insertion_penalty(float ip);
  void set_print_text_result(int print);
  void set_print_state_segmentation(int print);
  void set_keep_state_segmentation(int keep);
  void set_verbose(int verbose);
  void set_print_probs(bool print_probs);
  void set_word_boundary(const std::string &word);
  void set_sentence_boundary(const std::string &start, const std::string &end);
  void set_generate_word_graph(bool value);
  void set_use_word_pair_approximation(bool value);
  void set_use_lm_cache(bool value);
  void set_require_sentence_end(bool s);
  void set_remove_pronunciation_id(bool remove);

  void set_optional_short_silence(bool state);

  void prune_lm_lookahead_buffers(int min_delta, int max_depth);

  void write_word_history(const std::string &file_name);
  void print_lm_history();

  void print_tp_lex_node(int node);
  void print_tp_lex_lookahead(int node);

  void debug_print_best_lm_history();
};
