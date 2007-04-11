#include <errno.h>

#include "Recipe.hh"
#include "str.hh"

Recipe::Info::Info()
  : start_time(0), end_time(0), start_line(0), end_line(0),
    speaker_id(""), utterance_id("")
{
}

void
Recipe::clear()
{
  infos.clear();
}

void
Recipe::read(FILE *f)
{
  std::string line;
  std::vector<std::string> field;

  while (1) {
    bool ok = str::read_line(&line, f);

    // Do we have an error or eof?
    if (!ok) {
      if (ferror(f)) {
	fprintf(stderr, "Recipe: str::read_file(): read error: %s\n",
		strerror(errno));
	exit(1);
      }

      if (feof(f))
	return; // RETURN
    }

    // Parse string in fields and skip empty or commented lines
    str::clean(&line, "\n\t ");
    if (line.length() == 0 || line[0] == '#')
      continue;
    str::split(&line, " \t", true, &field);
    
    // Add info in the vector
    infos.push_back(Info());
    Info &info = infos.back();
    if (field.size() > 0)
      info.audio_path = field[0];
    if (field.size() > 1)
      info.phn_path = field[1];
    if (field.size() > 2)
      info.phn_out_path = field[2];
    if (field.size() > 3)
      info.start_time = atof(field[3].c_str());
    if (field.size() > 4)
      info.end_time = atof(field[4].c_str());
    if (field.size() > 5)
      info.start_line = atoi(field[5].c_str());
    if (field.size() > 6)
      info.end_line = atoi(field[6].c_str());
    if (field.size() > 7)
      info.speaker_id = field[7];
    if (field.size() > 8)
      info.speaker_id = field[8];
  }
}


PhnReader*
Recipe::Info::init_phn_files(HmmSet *model, bool relative_sample_nums,
                              bool out_phn, FeatureGenerator *fea_gen,
                              bool raw_audio, PhnReader *phn_reader)
{
  float frame_rate;

  // Open the audio file
  fea_gen->open(audio_path, raw_audio);

  // Initialize the PhnReader
  if (phn_reader == NULL)
    phn_reader = new PhnReader;
  if (model == NULL)
    phn_reader->set_state_num_labels(true);
  phn_reader->set_relative_sample_numbers(relative_sample_nums);
  frame_rate = fea_gen->frame_rate();
  phn_reader->set_frame_rate(frame_rate);

  // Open the segmentation
  phn_reader->open(out_phn?phn_out_path:phn_path);

  if (start_time > 0 || end_time > 0)
  {
    phn_reader->set_frame_limits((int)(start_time * frame_rate), 
                                (int)(end_time * frame_rate));
  }
  if (start_line > 0 || end_line > 0)
  {
    phn_reader->set_line_limits(start_line, end_line);
  }
  
  return phn_reader;
}
