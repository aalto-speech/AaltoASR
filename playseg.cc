#include <exception>
#include <sndfile.h>
#include <math.h>
#include "AudioPlayer.hh"
#include "io.hh"
#include "conf.hh"
#include "str.hh"

conf::Config config;
AudioPlayer audio_player;
SF_INFO sf_info;
SNDFILE *audio_file = NULL;
float time_unit;
float offset;

void
play_segment(int start_sample, int end_sample)
{
  if (end_sample <= start_sample) {
    printf("end_time less or equal to start_time, skipping\n");
    return;
  }

  // Seek to correct place
  sf_count_t ret = sf_seek(audio_file, start_sample, SEEK_SET);
  if (ret < 0) {
    printf("tried to seek outside the file, skipping\n");
    return;
  }

  // Read audio
  int samples_to_read = end_sample - start_sample;
  short *buf = new short[samples_to_read];
  int samples_read = sf_read_short(audio_file, buf, samples_to_read);
  audio_player.enqueue(buf, samples_read);
}

void
open_audio_file(std::string file_name)
{
  // Close old audio file if still open
  if (audio_file != NULL) {
    int ret = sf_close(audio_file);
    if (ret != SF_ERR_NO_ERROR) {
      fprintf(stderr, "ERROR: sf_close() failed\n");
      exit(1);
    }
  }

  audio_file = sf_open(file_name.c_str(), SFM_READ, &sf_info);
  if (audio_file == NULL) {
    fprintf(stderr, "WARNING: could not open audio file %s\n", 
	    file_name.c_str());
    return;
  }
}

void
handle_input()
{
  std::string line;
  std::vector<std::string> fields;

  while (str::read_line(&line, stdin, true)) {

    // Clean whitespace and parse into whitespace-seperated fields
    str::clean(&line, " \t");
    str::split(&line, " \t", true, &fields);

    // Treat line as a file name if only one field
    if (fields.size() == 1) {
      open_audio_file(fields[0]);
      continue;
    }

    if (fields.size() < 2) {
      printf("skipping invalid line:\n%s\n", line.c_str());
      continue;
    }

    // Interpret the first two fields as starting and ending times
    bool ok = true;
    float start_time = str::str2float(&fields[0], &ok);
    float end_time = str::str2float(&fields[1], &ok);
    if (!ok) {
      printf("skipping invalid line:\n%s\n", line.c_str());
      continue;
    }

    // Compute the position in the audio file and play
    int start_sample = lrint((start_time + offset) * time_unit  * 
			     sf_info.samplerate);
    int end_sample = lrint((end_time + offset) * time_unit * 
			   sf_info.samplerate);
    play_segment(start_sample, end_sample);
  }
}

int
main(int argc, char *argv[])
{
  config("usage: playseg [OPTION...] [AUDIOFILE]\n")
    ('C', "config=FILE", "arg", "", "configuration file")
    ('h', "help", "", "", "display help")
    ('o', "offset=FLOAT", "arg", "0", "offset in seconds (default: 0)")
    ('O', "output", "", "", "write raw waveform to output file")
    ('t', "time-unit=FLOAT", "arg", "0.0000625", 
     "time unit (s) in input (default: 0.0000625)");
  config.parse(argc, argv);
  if (config["help"].specified) {
    fputs(config.help_string().c_str(), stdout);
    exit(0);
  }
  if (config["config"].specified)
    config.read(io::Stream(config["config"].get_str(), "r").file);
  config.check_required();

  time_unit = config["time-unit"].get_float();
  offset = config["offset"].get_float();
    
  // Open initial audio file
  if (config.arguments.empty())
    printf("no active audio file yet\n");
  else
    open_audio_file(config.arguments[0]);

  try {
    handle_input();
  }
  catch (std::exception &e) {
    fprintf(stderr, "caught exception: %s\n", e.what());
  }
}
