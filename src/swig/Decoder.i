# -*- tab-width: 2 -*-

%include exception.i

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
    SWIG_exception(SWIG_RuntimeError, "Exception");
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

%typemap(python,in) FILE* {
	if (!(PyFile_Check($input))) {
		PyErr_SetString(PyExc_TypeError, "not a file pointer");
		return NULL;
	}
	$1=PyFile_AsFile($input);
}

class Hypo {
};

%extend Hypo {
  float log_prob() { return self->log_prob; }
  int frame() { return self->frame; }
}

class HypoStack {
public:
  Hypo &at(int index);
  Hypo &front();
  Hypo &back();
  int size();
  bool empty();
  int find_similar(const Hypo &hypo, int words);
  void sorted_insert(const Hypo &hypo);
};

class Expander {
  Expander(const std::vector<Hmm> &hmms, Lexicon &lexicon, Acoustics &m_acoustics);
  void expand(int start_frame, int frames);
  void set_forced_end(bool forced_end);
  void set_token_limit(int limit);
  void set_beam(float beam);
  void set_max_state_duration(int duration);
  void sort_words();
  const std::vector<Word*> &words();
  Word* word(int index);
};

class Toolbox {
public:
  Toolbox();
  
  void hmm_read(const char *hmm_file);
  void duration_read(const char *dur_file);
  const std::vector<Hmm> &hmms();
  void lex_read(const char *file);
  const std::string &lex_word();
  const std::string &lex_phone();
  void ngram_read(const char *file, float weight, const bool binary);
  void ngram_read(const char *file, float weight);
  void read_lookahead_ngram(const char *file, const bool binary);
  void read_lookahead_ngram(const char *file);

  // Lna
  void lna_open(const char *file, int size);
  void lna_close();
  void lna_seek(int frame);
  Acoustics &acoustics();
  void use_one_frame_acoustics();
  void set_one_frame(int frame, const std::vector<float> log_probs);

  // Expander
  void expand(int frame, int frames);
  const std::string &best_word();
  void print_words(int words);
  int find_word(const std::string &word);

  // Search
  void init(int expand_window);
	void reset(int frame);
	void set_end(int frame);
  bool expand_stack(int frame);
	void expand_words(int frame, const std::string &words);
  void go(int frame);
  bool run();
  bool runto(int frame);
	bool recognize_segment(int start_frame, int end_frame);

  int frame();
  int first_frame();
  int last_frame();
  HypoStack &stack(int frame);
  int paths();

	void write_word_graph(const std::string &file_name);
	void print_best_lm_history();
	void print_best_lm_history_to_file(FILE *out);
	void select_decoder(int stack_dec);

  void set_forced_end(bool forced_end);
  void set_hypo_limit(int hypo_limit);
  void set_prune_similar(int prune_similar);
  void set_word_limit(int word_limit);
  void set_word_beam(float word_beam);
  void set_lm_scale(float lm_scale);
  void set_lm_offset(float lm_offset);
  void set_unk_offset(float unk_offset);
  void set_token_limit(int limit);
  void set_state_beam(float beam);
  void set_duration_scale(float scale);
  void set_transition_scale(float scale);
  void set_rabiner_post_mode(int mode);
  void set_hypo_beam(float beam);
  void set_global_beam(float beam);
	void set_word_end_beam(float beam);
	void set_eq_depth_beam(float beam);
  void set_eq_word_count_beam(float beam);
	void set_fan_in_beam(float beam);
	void set_fan_out_beam(float beam);
	void set_tp_state_beam(float beam);
  void set_max_state_duration(int duration);
  void set_cross_word_triphones(bool cw_triphones);
  void set_silence_is_word(bool b);
	void set_ignore_case(bool b);		
  void set_lm_lookahead(int lmlh);
	void set_insertion_penalty(float ip);
  void set_print_text_result(int print);
  void set_print_state_segmentation(int print);
  void set_verbose(int verbose);
  void set_print_probs(bool print_probs);
  void set_print_indices(bool print_indices);
  void set_print_frames(bool print_frames);
  void set_multiple_endings(int multiple_endings);
  void set_word_boundary(const std::string &word);
  void set_sentence_boundary(const std::string &start, const std::string &end);
  void set_dummy_word_boundaries(bool value);
	void set_generate_word_graph(bool value);
  void set_require_sentence_end(bool s);

  void set_optional_short_silence(bool state);

  void prune_lm_lookahead_buffers(int min_delta, int max_depth);

	void print_prunings();
  void print_hypo(Hypo &hypo);
  void print_sure();
	void write_word_history(const std::string &file_name);
	void print_lm_history();
	
	void print_tp_lex_node(int node);
  void print_tp_lex_lookahead(int node);

	void debug_print_best_lm_history();
};
