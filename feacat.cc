#include <iostream>
#include <iomanip>
#include <vector>

#include <stdlib.h>

#include "parse_input.h"
#include "FeatureGenerator.hh"

const char *in_file;
const char *cfg_file = NULL;

int raw_sample_rate;
int raw;
int raw_head;
int knn;
int info;
int delta;
int deltaw;
int accel;
int accelw;
int rustu;
int power;
int concat;
int concatw;
bool triangular; 
int win_size;
float start_time;
float end_time;


int num;
int buf;
int random_test;
int seed;
int noprint;

int start_frame;
int end_frame;

bool cms;
int cms_win_len;


FeatureGenerator gen;

const char *usage = 
"usage: feacat OPTIONS [OPTIONS...]\n"
"mandatory:\n"
"  -in FILE      audio file in supported format\n"
"options:\n"
"  -win INT      window size (default: 1000)\n"
"  -info INT     info level\n"
"  -start FLOAT  start time in seconds\n"
"  -end FLOAT    end time in seconds\n"
"  -cfg FILE     load configuration and transformations from a file\n"
"  -raw_output   output in raw headerless 32-bit float format\n"
"  -raw_head     output in raw 32-bit float format with headers\n"
"  -raw INT       Raw sound files with given sample rate\n"
"  -cms INT       Cepstral mean subtraction (default window length: 150)\n"
"  -ntriangular  Use original Mel filters (otherwise use triangulars)\n" 
"debugging:\n"
"  -num          print frame numbers\n"
"  -random INT   number of tests\n"
"  -seed INT     seed for random number generator\n"
"  -noprint      do not print features\n"
"features:\n"
"  -delta\n"
"  -deltaw INT   number of neighboring frames in delta computation\n"
"  -accel        compute acceleration coefficients ie. deltadeltas\n"
"  -accelw INT   number of neighboring frames in deltadelta computation\n"
"  -rustu        not important\n"
"  -power        include power feature\n"
"  -concat       not important\n"
"  -concatw INT  not important\n"
;

void
parse_options(int argc, char *argv[])
{
  if (pi_extract_parameter(argc, argv, "-h", OPTION2) || 
      pi_extract_parameter(argc, argv, "-help", OPTION2)) {
    std::cout << usage;
    exit(0);
  }

  in_file = pi_extract_parameter(argc, argv, "-in", ALWAYS);

  info = pi_oatoi(pi_extract_parameter(argc, argv, "-info", OPTION),0);
  win_size = pi_oatoi(pi_extract_parameter(argc, argv, "-win", OPTION), 1000);
  start_time = pi_oatof(pi_extract_parameter(argc, argv, "-start", OPTION),0);
  end_time = pi_oatof(pi_extract_parameter(argc, argv, "-end", OPTION),0);
  cfg_file = pi_extract_parameter(argc, argv, "-cfg", OPTION);
  raw = !!pi_extract_parameter(argc, argv, "-raw_output", OPTION2);
  raw_head = !!pi_extract_parameter(argc, argv, "-raw_head", OPTION2);
  cms = !!pi_extract_parameter(argc, argv, "-cms", OPTION2);
  cms_win_len = pi_oatoi(pi_extract_parameter(argc, argv, "-cms",OPTION), 150);
  num = !!pi_extract_parameter(argc, argv, "-num", OPTION2);
  random_test = pi_oatoi(pi_extract_parameter(argc, argv, "-random", OPTION), 0);
  seed = pi_oatoi(pi_extract_parameter(argc, argv, "-seed", OPTION), 0);
  noprint = !!pi_extract_parameter(argc, argv, "-noprint", OPTION2);

  delta = !!pi_extract_parameter(argc, argv, "-delta", OPTION2);
  accel = !!pi_extract_parameter(argc, argv, "-accel", OPTION2);
  deltaw = pi_oatoi(pi_extract_parameter(argc, argv, "-deltaw", OPTION), 2);
  accelw = pi_oatoi(pi_extract_parameter(argc, argv, "-accelw", OPTION), 2);
  rustu = !!pi_extract_parameter(argc, argv, "-rustu", OPTION2);
  power = !!pi_extract_parameter(argc, argv, "-power", OPTION2);
  concat = !!pi_extract_parameter(argc, argv, "-concat", OPTION2);
  concatw = pi_oatoi(pi_extract_parameter(argc, argv, "-concatw", OPTION), 3);
  
  triangular = !pi_extract_parameter(argc, argv, "-ntriangular", OPTION2);
  
  raw_sample_rate = pi_oatoi(pi_extract_parameter(argc, argv, "-raw",OPTION),0);

  start_frame = (int)(start_time * 125.0); // FIXME: magic number
  end_frame = (int)(end_time * 125.0); // FIXME: magic number

  if (info > 0) {
    std::cerr << "start frame: " << start_frame << std::endl
	      << "end frame: " << end_frame << std::endl;
  }

  srand48(seed);
}

void
print_feature(ConstFeatureVec &fea)
{
  if (noprint)
    return;

  // Raw output
  if (raw) {
    // Ensure 32-bit float format
    if (sizeof(float) != 4) {
      fprintf(stderr, "ABORT: sizeof(float) != 4\n");
      exit(1);
    }
    for (int i = 0; i < fea.dim(); i++) {
      float tmp = fea[i];
      fwrite(&tmp, sizeof(float), 1, stdout);
    }
  }

  // ASCII output
  else {

    for (int i=0; i < fea.dim(); i++) {
      std::cout << std::setw(8) << fea[i];
    }

    std::cout << std::endl;
  }
}

int
rnd(int a, int b)
{
  assert(b > a);
  return lrand48() % (b - a) + a;
}

int
main(int argc, char *argv[])
{
  try {
    parse_options(argc, argv);

    std::cout.setf(std::cout.fixed, std::cout.floatfield);
    std::cout.setf(std::cout.right, std::cout.adjustfield);
    std::cout.precision(2);

    /*gen.set_delta(delta, deltaw);
    gen.set_accel(accel, accelw);
    gen.set_rustu(rustu);
    gen.set_power(power);
    gen.set_concat(concat, concatw);
    gen.allow_growing(true);
    gen.set_triangular(triangular); 
    if (cfg_file != NULL) gen.load_configuration(cfg_file);
    else {
      std::vector<float> mean;
      std::vector<float> var;
      gen.set_feature_normalization(mean, var);
    }
    if (cms_win_len==0) cms_win_len=150; // TODO: someone fix the parameter extraction 
    gen.set_cms(cms, cms_win_len);*/

    gen.load_configuration("");
    
    gen.open(in_file, raw_sample_rate);
    if (raw_head) {
      raw=1;
      int dim=gen.dim();
      fwrite(&dim, sizeof(char), 1, stdout);
    }

/*    int buffer_start = start_frame;
    int buffer_end;
    while (1) {
      if (random_test > 0) {
	buffer_start = rnd(start_frame, end_frame);
	buffer_end = buffer_start + rnd(win_size / 2, win_size);

	gen.generate(buffer_start, buffer_end);
	
	if (buffer_start > gen.end_frame()) {
	  end_frame = gen.end_frame();
	  std::cerr << "warning: fixing end_frame to " << end_frame 
		    << std::endl;
	  continue;
	}
	if (gen.start_frame() == gen.end_frame())
	  continue;

	if (buffer_end > gen.end_frame())
	  buffer_end = gen.end_frame();
	if (buffer_end > end_frame)
	  buffer_end = end_frame;

	assert(buffer_end > buffer_start);
	int frame = rnd(gen.start_frame(), buffer_end);
	print_feature(frame);

	random_test--;
	if (random_test == 0)
	  break;
      }
      else {
	buffer_end = buffer_start + win_size;
	if (end_frame > 0 && buffer_end > end_frame)
	  buffer_end = end_frame;
	
	gen.generate(buffer_start, buffer_end);
	if (gen.start_frame() == gen.end_frame())
	  break;
	
	for (int i = gen.start_frame(); i < gen.end_frame(); i++){
	  print_feature(i);
	  }
	buffer_start = buffer_end;
      }
      }*/
    for (int f = start_frame; f < end_frame; f++)
    {
      ConstFeatureVec fea = gen.generate(f);
      if (gen.eof())
        break;
      print_feature(fea);
    }
  }
  catch (std::exception &e) {
    std::cerr << "exception: " << e.what() << std::endl;
  }
  catch (...) {
    std::cerr << "exception" << std::endl;
  }
}
