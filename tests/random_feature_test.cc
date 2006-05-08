#include <sys/times.h>
#include <stdlib.h>
#include "io.hh"
#include "FeatureGenerator.hh"

void
print_feature(const FeatureVec &vec)
{
  for (int i = 0; i < vec.dim(); i++)
    printf("%.2f\t", vec[i]);
  printf("\n");
}

int
main(int argc, char *argv[])
{
  try {
    if (argc != 6 && argc != 7) {
      fprintf(stderr, "usage: random_feature_test "
	      "WAV CONFIG START END NUM_TESTS [SEED]\n");
      exit(1);
    }

    struct tms dummy;
    int seed = times(&dummy);
    if (argc == 7)
      seed = atoi(argv[6]);
    srand48(seed);

    FeatureGenerator gen;
    gen.load_configuration(io::Stream(argv[2]));
    gen.open(argv[1], false);
    int start = atoi(argv[3]);
    int end = atoi(argv[4]);
    int num_tests = atoi(argv[5]);
    if (start >= end) {
      fprintf(stderr, "invalid range: start %d, end %d\n", start, end);
      exit(1);
    }

    FeatureBuffer buf;
    buf.resize(end - start, gen.dim());
    for (int f = start; f < end; f++) {
      const FeatureVec &vec = gen.generate(f);
      buf[f - start].copy(vec);
    }

    for (int t = 0; t < num_tests; t++) {
      int frame = start + lrand48() % (end - start);
      const FeatureVec &vec = gen.generate(frame);
      for (int i = 0; i < vec.dim(); i++) {
	const FeatureVec &buf_vec = buf[frame - start];
	if (vec[i] != buf_vec[i]) {
	  printf("features differ in frame %d with seed %d\n", frame, seed);
	  print_feature(vec);
	  print_feature(buf_vec);
	  exit(1);
	}
      }
    }

    printf("test successful\n");
  }
  catch (std::string &str) {
    fprintf(stderr, "caught exception: %s\n", str.c_str());
    abort();
  }
}
