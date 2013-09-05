%include "exception.i"
%include "std_string.i"
%module FstSearch

%{
class FstSearch {
public:
  FstSearch(const char * search_fst_fname, const char * hmm_path, const char * dur_path = NULL);
  ~FstSearch();

  std::string run();
  // FIXME: These functions are direct copies from Toolbox, code duplication !
  void hmm_read(const char *file);
  void duration_read(const char *dur_file);
  void lna_open(const char *file, int size);
  void lna_open_fd(const int fd, int size);
  void lna_close();

  float duration_scale;
  float beam;
  int token_limit;
  float transition_scale;
};
%}
