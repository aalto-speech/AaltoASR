#include <errno.h>

#include <string.h>
#include "Recipe.hh"
#include "str.hh"


namespace aku {

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
Recipe::read(FILE *f, int num_batches, int batch_index, bool cluster_speakers)
{
  std::string line;
  std::vector<std::string> field;
  std::vector<std::string> line_buffer;
  std::map<std::string, std::string> key_value_map;
  std::vector<std::string> key_value;
  int cur_index = 1;
  int cur_line = 0;
  int target_lines;
  int i, j;
  std::map<std::string,std::string>::const_iterator it;
  std::string cur_speaker = "", new_speaker;

  if (num_batches > 1 && (batch_index < 1 || batch_index > num_batches))
    throw std::string("Invalid batch index");

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
	break;
    }

    // Parse string in fields and skip empty or commented lines
    str::clean(&line, "\n\t ");
    if (line.length() == 0 || line[0] == '#')
      continue;
    line_buffer.push_back(line);
  }

  int batch_remainder = 0;
  if (num_batches <= 1)
    target_lines = line_buffer.size();
  else
  {
    target_lines = (int)line_buffer.size()/num_batches;
    batch_remainder = line_buffer.size()%num_batches;
  }
  int extra_line = 1;
  if (target_lines < 1)
  {
    target_lines = 1;
    extra_line = 0;
  }
  if (batch_remainder == 0)
    extra_line = 0;
  
  for (i = 0; i < (int)line_buffer.size(); i++)
  {
    // Parse the recipe line
    str::split(&line_buffer[i], " \t", true, &field);
    for (j = 0; j < (int)field.size(); j++)
    {
      str::split(&field[j], "=", false, &key_value);
      if ((int)key_value.size() != 2)
        throw std::string("Invalid recipe line: ") + line_buffer[i];
      if ((int)key_value.size() == 2)
        key_value_map[key_value[0]] = key_value[1];
    }

    // Check if batch index needs updating
    if (num_batches > 1 && cur_index < num_batches)
    {
      if ((it = key_value_map.find("speaker")) != key_value_map.end())
        new_speaker = (*it).second;
      else
        new_speaker = "";
      if (cur_line >= target_lines + extra_line &&
          (!cluster_speakers || cur_speaker.size() == 0 ||
           cur_speaker != new_speaker))
      {
        cur_index++;
        if (cur_index > batch_index)
          break;
        cur_line -= target_lines + extra_line;
        if (cur_index > batch_remainder)
          extra_line = 0;
      }
      cur_speaker = new_speaker;
    }
    
    if (num_batches <= 1 || cur_index == batch_index)
    {
      // This line belongs to the current batch, save it
      infos.push_back(Info());
      Info &info = infos.back();
      if ((it = key_value_map.find("audio")) != key_value_map.end())
        info.audio_path = (*it).second;
      if ((it = key_value_map.find("alt-audio")) != key_value_map.end())
        info.alt_audio_path = (*it).second;
      if ((it = key_value_map.find("transcript")) != key_value_map.end())
        info.transcript_path = (*it).second;
      if ((it = key_value_map.find("alignment")) != key_value_map.end())
        info.alignment_path = (*it).second;
      if ((it = key_value_map.find("hmmnet")) != key_value_map.end())
        info.hmmnet_path = (*it).second;
      if ((it = key_value_map.find("den-hmmnet")) != key_value_map.end())
        info.den_hmmnet_path = (*it).second;
      if ((it = key_value_map.find("lna")) != key_value_map.end())
        info.lna_path = (*it).second;
      if ((it = key_value_map.find("start-time")) != key_value_map.end())
        info.start_time = atof((*it).second.c_str());
      if ((it = key_value_map.find("end-time")) != key_value_map.end())
        info.end_time = atof((*it).second.c_str());
      if ((it = key_value_map.find("start-line")) != key_value_map.end())
        info.start_line = atoi((*it).second.c_str());
      if ((it = key_value_map.find("end-line")) != key_value_map.end())
        info.end_line = atoi((*it).second.c_str());
      if ((it = key_value_map.find("speaker")) != key_value_map.end())
        info.speaker_id = (*it).second;
      if ((it = key_value_map.find("utterance")) != key_value_map.end())
        info.utterance_id = (*it).second;
    }
    
    cur_line++;
  }
}


PhnReader*
Recipe::Info::init_phn_files(HmmSet *model, bool relative_sample_nums,
                             bool state_num_labels, bool out_phn,
                             FeatureGenerator *fea_gen,
                             PhnReader *phn_reader)
{
  float frame_rate = 125; // Default value

  if (fea_gen != NULL)
  {
    // Open the audio file
    fea_gen->open(audio_path);
  }

  // Initialize the PhnReader
  if (phn_reader == NULL)
  {
    if (model == NULL)
      throw std::string("recipe::Info::init_phn_files: HMM model is required if phn_reader==NULL");
    phn_reader = new PhnReader(model);
  }
  phn_reader->set_state_num_labels(state_num_labels);
  phn_reader->set_relative_sample_numbers(relative_sample_nums);

  if (fea_gen != NULL)
    frame_rate = fea_gen->frame_rate();
  phn_reader->set_frame_rate(frame_rate);

  // Open the segmentation
  phn_reader->open(out_phn?alignment_path:transcript_path);

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


HmmNetBaumWelch*
Recipe::Info::init_hmmnet_files(HmmSet *model, bool den_hmmnet,
                                FeatureGenerator *fea_gen,
                                HmmNetBaumWelch *hnbw)
{
  // Open the audio file
  fea_gen->open(audio_path);

  // Initialize the HmmNetBaumWelch
  if (hnbw == NULL)
  {
    if (model == NULL)
      throw std::string("Recipe::Info::init_hmmnet_files: HMM model is required if hnbw==NULL");
     hnbw= new HmmNetBaumWelch(*fea_gen, *model);
  }

  // Open the HMM network
  if (den_hmmnet) {
    hnbw->open(den_hmmnet_path);
  }
  else {
	if (hmmnet_path.empty())
	  throw std::string("Recipe::Info::init_hmmnet_files: hmmnet not specified in recipe.");
    hnbw->open(hmmnet_path);
  }

  if (start_time > 0 || end_time > 0)
  {
    float frame_rate = fea_gen->frame_rate();
    hnbw->set_frame_limits((int)(start_time * frame_rate), 
                           (int)(end_time * frame_rate));
  }

  return hnbw;
}

}
