#include <errno.h>
#include <sys/stat.h>
#include "TreeGram.hh"
#include "Lattice.hh"
#include "Rescore.hh"
#include "conf.hh"
#include "io.hh"
#include "str.hh"

conf::Config config;

/** Read a list of file names from a file. */
std::vector<std::string>
read_file_list(FILE *file)
{
  std::vector<std::string> files;
  std::string line;
  while (str::read_line(&line, file)) {
    str::clean(&line, " \t\n");
    if (line.empty())
      continue;
    files.push_back(line);
  }
  return files;
}

std::string
strip_dir(std::string path)
{
  std::string::size_type pos = path.find_last_of('/');
  if (pos != path.npos)
    path.erase(0, pos + 1);
  return path;
}

bool
file_exists(std::string file)
{
  struct stat sb;
  int ret = stat(file.c_str(), &sb);
  if (ret < 0) {
    if (errno == ENOENT)
      return false;
    fprintf(stderr, "stat() failed on target file %s\n", file.c_str());
    perror(NULL);
    exit(1);
  }
  return true;
}

/** The good old main. */
int
main(int argc, char *argv[])
{
  config("usage: lattice_rescore [OPTION...]\n")
    ('C', "config=FILE", "arg", "", "configuration file")
    ('f', "force", "", "", "force overwriting existing files")
    ('h', "help", "", "", "display help")
    ('l', "lm=FILE", "arg must", "", "language model used in rescoring")
    ('i', "in=FILE", "arg", "", "input lattice")
    ('I', "in-list=FILE", "arg", "", "input list of lattices")
    ('o', "out=FILE", "arg", "", "output lattice file")
    ('O', "out-dir=DIR", "arg", "", "output directory")
    ('p', "post-process=FILE", "arg", "", 
     "run a post-processor for each output file")
    ;
  config.parse(argc, argv);
  if (config["help"].specified) {
    fputs(config.help_string().c_str(), stdout);
    exit(0);
  }
  if (config["config"].specified)
    config.read(io::Stream(config["config"].get_str(), "r").file);

  if (config.arguments.size() != 0) {
    fputs(config.help_string().c_str(), stderr);
    exit(1);
  }

  // Read the language model
  fprintf(stderr, "reading the language model...");
  TreeGram tree_gram;
  if (config["lm"].specified)
    tree_gram.read(io::Stream(config["lm"].get_str(), "r").file);
  fprintf(stderr, "\n");

  // Parse input lattices
  std::vector<std::string> input_files;
  if (config["in"].specified && config["in-list"].specified) {
    fprintf(stderr, "ERROR: do not specify input lattice and input list\n");
    exit(1);
  }
  if (!config["in"].specified && !config["in-list"].specified) {
    fprintf(stderr, "ERROR: must specify input lattice or input list\n");
    exit(1);
  }
  if (config["in"].specified)
    input_files.push_back(config["in"].get_str());
  else if (config["in-list"].specified)
    input_files = read_file_list(
      io::Stream(config["in-list"].get_str(), "r").file);

  // Create output directory
  if (config["out-dir"].specified)
    mkdir(config["out-dir"].get_c_str(), 0777);

  // Rescore lattices
  Rescore rescore;
  Lattice src_lattice;
  for (int i = 0; i < (int)input_files.size(); i++) {
    std::string output_file;
    if (config["out"].specified)
      output_file = config["out"].get_str();
    else if (config["out-dir"].specified)
      output_file = 
        config["out-dir"].get_str() + "/" + strip_dir(input_files[i]);
    if (file_exists(output_file) && !config["force"].specified) {
      fprintf(stderr, "skipped existing file %s\n", output_file.c_str());
      continue;
    }

    fprintf(stderr, "processing %s...", input_files[i].c_str());
    src_lattice.read(io::Stream(input_files[i], "r").file);
    rescore.rescore(&src_lattice, &tree_gram);

    fprintf(stderr, "writing %s...", output_file.c_str());
    rescore.rescored_lattice().write(io::Stream(output_file, "w").file);
    fprintf(stderr, "\n");

    if (config["post-process"].specified) {
      std::string cmd = config["post-process"].get_str() + 
        " \"" + output_file + "\"";
      fprintf(stderr, "running post-processor: %s\n", cmd.c_str());
      int ret = system(cmd.c_str());
      if (ret < 0) {
        fprintf(stderr, "WARNING: command failed\n");
      }
    }
  }
}
