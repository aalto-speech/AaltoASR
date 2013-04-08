#include "io.hh"

namespace io {

  Stream::Stream()
    : file(NULL),
      is_pipe(false),
      close_allowed(true)
  {
  }

  Stream::Stream(std::string file_name, std::string mode, bool allow_close)
    : file(NULL),
      is_pipe(false),
      close_allowed(true)
  {
    open(file_name, mode, allow_close);
  }


  Stream::~Stream()
  {
    close();
  }

  void
  Stream::open(std::string file_name, std::string mode, bool allow_close)
  {
    close();

    if (file_name.length() == 0) {
      fprintf(stderr, "can not open empty filename\n");
      exit(1);
    }

    close_allowed = allow_close;

    // Standard input or output
    if (file_name == "-") {
      if (mode == "r")
	file = stdin;
      else if (mode == "w")
	file = stdout;
      else {
	fprintf(stderr, "Stream::open(): "
		"invalid mode %s for standard input or output\n", 
		mode.c_str());
	exit(1);
      }
      close_allowed = false;
    }

    // A pipe for reading
    else if (file_name[file_name.length()-1] == '|') {
      if (mode != "r") {
	fprintf(stderr, "invalid mode %s for input pipe %s\n",
		mode.c_str(), file_name.c_str());
	exit(1);
      }
      file_name.erase(file_name.length() - 1, 1);
      file = popen(file_name.c_str(), mode.c_str());
      if (file == NULL) {
	fprintf(stderr, "could not open input pipe %s\n", file_name.c_str());
	exit(1);
      }
      is_pipe = true;
    }

    // A pipe for writing
    else if (file_name[0] == '|') {
      if (mode != "w") {
	fprintf(stderr, "invalid mode %s for output pipe %s\n",
		mode.c_str(), file_name.c_str());
	exit(1);
      }
      file_name.erase(0, 1);
      file = popen(file_name.c_str(), mode.c_str());
      if (file == NULL) {
	fprintf(stderr, "could not open output pipe %s\n", file_name.c_str());
	exit(1);
      }
      is_pipe = true;
    }

    // A gzipped file
    else if (file_name.length() >= 3 &&
	file_name.substr(file_name.length() - 3) == ".gz") 
    {
      std::string pipe;
      if (mode == "r")
	pipe = "gzip -dc '" + file_name + "'";
      else if (mode == "w")
	pipe = "gzip > '" + file_name + "'";
      else {
	fprintf(stderr, "invalid mode %s for gzipped file\n", mode.c_str());
      }
      file = popen(pipe.c_str(), mode.c_str());
      if (file == NULL) {
	fprintf(stderr, "could not open mode %s for pipe: %s\n", mode.c_str(),
		pipe.c_str());
	exit(1);
      }
      is_pipe = true;
    }

    // An ordinary file
    else {
      file = fopen(file_name.c_str(), mode.c_str());
      if (file == NULL) {
	fprintf(stderr, "could not open mode %s for file %s\n", 
		mode.c_str(), file_name.c_str());
	exit(1);
      }
      is_pipe = false;
    }

  }

  void
  Stream::close()
  {
    if (file == NULL || !close_allowed)
      return;
    if (is_pipe)
      pclose(file);
    else
      fclose(file);
    is_pipe = false;
    file = NULL;
  }

}
