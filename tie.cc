#include <fstream>
#include <string>
#include <iostream>

#include "io.hh"
#include "str.hh"
#include "conf.hh"
#include "FeatureGenerator.hh"
#include "Recipe.hh"
#include "TriphoneSet.hh"
#include "PhnReader.hh"
#include "SpeakerConfig.hh"


int win_size;
int info;
bool raw_flag;
bool durstat;
float start_time, end_time;
int start_frame, end_frame;

conf::Config config;
Recipe recipe;
FeatureGenerator fea_gen;
PhnReader phn_reader;
SpeakerConfig speaker_config(fea_gen);

bool fill_missing_contexts;
bool ignore_context_length;
bool skip_short_silence_context;
bool set_speakers = false;

bool triphone_phn;
TriphoneSet triphone_set;


int my_tolower(int c)
{
  if (c == 'Å')
    return 'å';
  else if (c == 'Ä')
    return 'ä';
  else if (c == 'Ö')
    return 'ö';
  return tolower(c);
}

std::string transform_context(const std::string &context)
{
  if (context[0] == '_')
    return "_";
  if (ignore_context_length)
    return context;
  std::string temp = context;
  std::transform(temp.begin(), temp.end(), temp.begin(), my_tolower);
  return temp;
}

std::string extract_left_context(const std::string &tri)
{
  return tri.substr(0, tri.rfind('-'));
}

std::string extract_right_context(const std::string &tri)
{
  return tri.substr(tri.find('+')+1);
}

std::string extract_center_pho(const std::string &tri)
{
  std::string temp = tri.substr(tri.rfind('-')+1);
  return temp.substr(0, temp.find('+'));
}


void
open_files(const std::string audio_file, const std::string phn_file, 
	   int first_phn_line, int last_phn_line)
{
  int first_sample;

  // Open sph file
  fea_gen.open(audio_file, raw_flag);

  // Open transcription
  phn_reader.open(phn_file);
  // Note: PHN files always use time multiplied by 16000
  phn_reader.set_sample_limit((int)(start_time * 16000), 
			      (int)(end_time * 16000));
  phn_reader.set_line_limit(first_phn_line, last_phn_line, 
			    &first_sample);

  if (first_sample > 16000 * start_time) {
    start_time = (float)(first_sample / 16000);
    start_frame = (int)(fea_gen.frame_rate() * start_time);
  }
}

void
collect_triphone_stats(PhnReader &phn_reader,
                       int start_frame, int end_frame,
                       std::string speaker, std::string utterance)
{
  PhnReader::Phn cur_phn;
  int phn_start_frame, phn_end_frame;
  std::string prev_label = "__";
  std::string cur_label;
  std::string next_label;
  std::deque<PhnReader::Phn> phn_sequence;
  bool eof_encountered = false;
  
  if (set_speakers && speaker.size() > 0)
  {
    speaker_config.set_speaker(speaker);
    if (utterance.size() > 0)
      speaker_config.set_utterance(utterance);
  }

  eof_encountered = !phn_reader.next(cur_phn);
  cur_label = cur_phn.label[0];
  
  while (!eof_encountered || phn_sequence.size() > 0)
  {
    if (cur_phn.state < 0)
      throw std::string("Triphone tying requires state segmented phn files!");

    if (phn_sequence.size() == 0)
    {
      // Read new phn lines until a new label is found
      PhnReader::Phn temp;
      next_label = "__"; // Used if EOF is encountered
      eof_encountered = !phn_reader.next(temp);
      while (!eof_encountered)
      {
        if (!skip_short_silence_context || temp.label[0] != "_")
        {
          phn_sequence.push_back(temp);
          if (temp.state == 0)
          {
            // Found the next new label
            next_label = temp.label[0];
            break;
          }
        }
        eof_encountered = !phn_reader.next(temp);
      }
    }

    phn_start_frame=(int)((double)cur_phn.start/16000.0*fea_gen.frame_rate()
                          +0.5);
    phn_end_frame=(int)((double)cur_phn.end/16000.0*fea_gen.frame_rate()+0.5);
    if (phn_start_frame < start_frame)
    {
      assert( phn_end_frame > start_frame );
      phn_start_frame = start_frame;
    }
    if (end_frame != 0 && phn_end_frame > end_frame)
    {
      assert( phn_start_frame < end_frame );
      phn_end_frame = end_frame;
    }

    if (set_speakers && cur_phn.speaker.size() > 0 &&
        cur_phn.speaker != speaker_config.get_cur_speaker())
    {
      speaker_config.set_speaker(speaker);
      if (utterance.size() > 0)
        speaker_config.set_utterance(utterance);
    }

    if (triphone_phn)
    {
      std::string left = extract_left_context(cur_label);
      std::string center = extract_center_pho(cur_label);
      std::string right = extract_right_context(cur_label);

      if (center[0] != '_')
      {
        for (int f = phn_start_frame; f < phn_end_frame; f++)
          triphone_set.add_feature(fea_gen.generate(f), left,
                                   center, right, cur_phn.state);
      }
    }
    else
    {
      if (cur_label[0] != '_')
      {
        std::string left=transform_context(prev_label);
        std::string right=transform_context(next_label);

        /*std::string temp = left+"-"+cur_label+"+"+right;
        fprintf(stderr, "%i %i %s . %d\n", phn_start_frame, phn_end_frame,
        temp.c_str(), cur_phn.state);*/
        
        for (int f = phn_start_frame; f < phn_end_frame; f++)
          triphone_set.add_feature(fea_gen.generate(f), left,
                                   cur_label, right, cur_phn.state);
      }
    }

    if (phn_sequence.size() > 0)
    {
      cur_phn = phn_sequence.front();
      phn_sequence.pop_front();
      if (cur_phn.state == 0) // New label
      {
        prev_label = cur_label;
        cur_label = next_label;
        assert( phn_sequence.size() == 0 );
      }
    }
  }
}


void
save_triphone_tying(const std::string &filename)
{
  FILE *fp;
  int state_num;

  // Save silence models
  if ((fp = fopen(filename.c_str(), "w")) == NULL)
  {
    fprintf(stderr, "Could not open file %s for writing.\n", filename.c_str());
    exit(1);
  }
  fprintf(fp, "_ 1 0\n__ 3 1 2 3\n");
  fclose(fp);
  
  state_num = triphone_set.save_to_basebind(filename, 4);
}


int
main(int argc, char *argv[])
{
  try {
    config("usage: tie [OPTION...]\n")
      ('h', "help", "", "", "display help")
      ('c', "config=FILE", "arg must", "", "feature configuration")
      ('r', "recipe=FILE", "arg must", "", "recipe file")
      ('O', "ophn", "", "", "use output phns for training")
      ('u', "rule=FILE", "arg must", "", "rule set for triphone state tying")
      ('o', "out=FILE", "arg", "", "output filename for basebind")
      ('R', "raw-input", "", "", "raw audio input")
      ('\0', "missing", "", "", "also tie missing triphone contexts ")
      ('\0', "count=INT", "arg", "0", "minimum feature count for state clusters")
      ('\0', "lh=FLOAT", "arg", "0", "minimum likelihood gain for cluster splitting")
      ('\0', "laward=FLOAT", "arg", "0", "likelihood award from applying the length rule")
      ('\0', "sc", "", "", "triphone context is used over short silences \'_\'")
      ('\0', "il", "", "", "ignore center phoneme length (case sensitive phonemes")
      ('\0', "icl", "", "", "ignore context length (case sensitive context)")
      ('\0', "tri", "", "", "triphone phn")
      ('S', "speakers=FILE", "arg", "", "speaker configuration file")
      ('\0', "sphn", "", "", "phns with speaker ID's in use")
      ('i', "info=INT", "arg", "0", "info level")
      ;
    config.default_parse(argc, argv);
    
    info = config["info"].get_int();
    raw_flag = config["raw-input"].specified;
    fea_gen.load_configuration(io::Stream(config["config"].get_str()));

    // Read recipe file
    recipe.read(io::Stream(config["recipe"].get_str()));

    skip_short_silence_context = config["sc"].specified;
    triphone_phn = config["tri"].specified;

    if (config["count"].specified)
      triphone_set.set_min_count(config["count"].get_int());
    if (config["lh"].specified)
      triphone_set.set_min_likelihood_gain(config["lh"].get_float());
    triphone_set.set_length_award(config["laward"].get_float());
    fill_missing_contexts = config["missing"].specified;
    triphone_set.set_ignore_length(config["il"].specified);
    ignore_context_length = config["icl"].specified;
    triphone_set.set_ignore_context_length(ignore_context_length);
    triphone_set.load_rule_set(config["rule"].get_str());

    phn_reader.set_speaker_phns(config["sphn"].specified);
    
    if (config["speakers"].specified)
    {
      speaker_config.read_speaker_file(
        io::Stream(config["speakers"].get_str()));
      set_speakers = true;
    }

    // Initialize triphone tying
    triphone_set.set_dimension(fea_gen.dim());
    triphone_set.set_info(info);
    
    for (int f = 0; f < (int)recipe.infos.size(); f++)
    {
      if (info > 0)
      {
        fprintf(stderr, "Processing file: %s", 
                recipe.infos[f].audio_path.c_str());
        if (recipe.infos[f].start_time || recipe.infos[f].end_time) 
          fprintf(stderr," (%.2f-%.2f)",recipe.infos[f].start_time,
                  recipe.infos[f].end_time);
        fprintf(stderr,"\n");
      }
    
      start_time = recipe.infos[f].start_time;
      end_time = recipe.infos[f].end_time;
      start_frame = (int)(start_time * fea_gen.frame_rate());
      end_frame = (int)(end_time * fea_gen.frame_rate());
    
      // Open the audio and phn files from the given list.
      open_files(recipe.infos[f].audio_path.c_str(),
                 (config["ophn"].specified?recipe.infos[f].phn_out_path:
                  recipe.infos[f].phn_path),
                 recipe.infos[f].start_line,
                 recipe.infos[f].end_line);

      collect_triphone_stats(phn_reader, start_frame, end_frame,
                             recipe.infos[f].speaker_id,
                             recipe.infos[f].utterance_id);

      fea_gen.close();
      phn_reader.close();
    }
    
    triphone_set.finish_triphone_statistics();
    if (fill_missing_contexts)
      triphone_set.fill_missing_contexts(false);
    triphone_set.tie_triphones();
    
    save_triphone_tying(config["out"].get_str());
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
