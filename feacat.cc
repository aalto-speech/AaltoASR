#include "io.hh"
#include "conf.hh"
#include "FeatureGenerator.hh"
#include "SpeakerConfig.hh"

conf::Config config;
FeatureGenerator gen;
bool raw_output = false;
SpeakerConfig speaker_conf(gen);

void
print_feature(const FeatureVec &fea)
{
  // Raw output
  if (raw_output) {
    for (int i = 0; i < fea.dim(); i++) {
      float tmp = fea[i];
      fwrite(&tmp, sizeof(float), 1, stdout);
    }
  }

  // ASCII output
  else {
    for (int i=0; i < fea.dim(); i++)
      printf("%8.2f ", fea[i]);
    printf("\n");
  }
}

int
main(int argc, char *argv[])
{
  assert(sizeof(float) == 4);
  assert(sizeof(int) == 4);

  try {
    config("usage: feacat [OPTION...] FILE\n")
      ('h', "help", "", "", "display help")
      ('c', "config=FILE", "arg must", "", "read feature configuration")
      ('w', "write-config=FILE", "arg", "", "write feature configuration")
      ('R', "raw-input", "", "", "raw audio input")
      ('\0', "raw-output", "", "", "raw float output")
      ('H', "header", "", "", "write a header (feature dim, 32 bits) in raw output")
      ('s', "start-frame=INT", "arg", "", "audio start frame")
      ('e', "end-frame=INT", "arg", "", "audio end frame")
      ('S', "speakers=FILE", "arg", "", "speaker configuration file")
      ('d', "speaker-id=NAME", "arg", "", "speaker ID")
      ('u', "utterance-id=NAME", "arg", "", "utterance ID")
      ;
    config.default_parse(argc, argv);
    if (config.arguments.size() != 1)
      config.print_help(stderr, 1);
    raw_output = config["raw-output"].specified;

    if (config["header"].specified && !raw_output)
    {
      fprintf(stderr, "Warning: header is only written in raw output mode\n");
    }

    gen.load_configuration(io::Stream(config["config"].get_str()));
    io::Stream audio_stream(config.arguments[0]);
    gen.open(audio_stream, config["raw-input"].specified, true);

    if (config["speakers"].specified)
    {
      speaker_conf.read_speaker_file(io::Stream(config["speakers"].get_str()));
      speaker_conf.set_speaker(config["speaker-id"].get_str());
      if (config["utterance-id"].specified)
        speaker_conf.set_utterance(config["utterance-id"].get_str());
    }

    if (config["write-config"].specified)
      gen.write_configuration(io::Stream(config["write-config"].get_str(),
                                         "w"));

    if (raw_output && config["header"].specified)
    {
      int dim = gen.dim();
      fwrite(&dim, sizeof(int), 1, stdout);
    }

    int start_frame = 0;
    int end_frame = INT_MAX;
    if (config["start-frame"].specified)
      start_frame = config["start-frame"].get_int();
    if (config["end-frame"].specified)
      end_frame = config["end-frame"].get_int();
    if (start_frame < end_frame)
    {
      for (int f = start_frame; f <= end_frame; f++) {
        const FeatureVec fea = gen.generate(f);
        if (end_frame == INT_MAX && gen.eof())
          break;
        print_feature(fea);
      }
    }
    else
    {
      for (int f = start_frame; f >= end_frame; f--) {
        const FeatureVec fea = gen.generate(f);
        print_feature(fea);
      }
    }

    gen.close();
  }
  catch (std::exception &e) {
    fprintf(stderr, "exception: %s\n", e.what());
    abort();
  }
  catch (std::string &str) {
    fprintf(stderr, "exception: %s\n", str.c_str());
    abort();
  }
}
