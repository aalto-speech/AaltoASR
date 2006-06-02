#include <math.h>
#include <string>

#include "io.hh"
#include "str.hh"
#include "conf.hh"
#include "HmmSet.hh"
#include "FeatureGenerator.hh"
#include "PhnReader.hh"
#include "Recipe.hh"
#include "SpeakerConfig.hh"
#include "MllrTrainer.hh"

int info;
bool raw_flag;
float start_time, end_time;
int start_frame, end_frame;
bool state_num_labels;

conf::Config config;
Recipe recipe;
HmmSet model;
FeatureGenerator fea_gen;
PhnReader phn_reader;
SpeakerConfig speaker_conf(fea_gen);

typedef std::map<std::string, MllrTrainer*> MllrTrainerMap;
MllrTrainerMap mllr_trainers;
bool ordered_spk; // files for a speaker are arranged successively

LinTransformModule *mllr_module;
std::string cur_speaker;


void
open_files(const std::string &audio_file, const std::string &phn_file, 
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
calculate_mllr_transform(const std::string &speaker)
{
  std::string cur_speaker = speaker_conf.get_cur_speaker();

  if (cur_speaker != speaker)
    speaker_conf.set_speaker(speaker);
  (mllr_trainers[speaker])->calculate_transform(mllr_module);
  if (cur_speaker != speaker)
    speaker_conf.set_speaker(cur_speaker);
}


void
set_speaker(std::string new_speaker)
{
  if (new_speaker == cur_speaker)
    return;
  
  // Check if this is the first time this speaker is encountered
  if (mllr_trainers.count(new_speaker) == 0)
  {
    // If all files for a speaker are processed successively
    // we can calculate transformation here and free memory!
    if (ordered_spk && speaker_conf.get_cur_speaker().size() > 0)
    {
      calculate_mllr_transform(speaker_conf.get_cur_speaker());
      delete mllr_trainers[speaker_conf.get_cur_speaker()];
      mllr_trainers.clear();
    }
    
    // Initialize for new speaker
    mllr_trainers[new_speaker] = new MllrTrainer(model, fea_gen);
  }

  cur_speaker = new_speaker;

  // Change speaker to FeatureGenerator
  speaker_conf.set_speaker(cur_speaker);
}


void
train_mllr(int start_frame, int end_frame, std::string speaker)
{
  Hmm hmm;
  HmmState state;
  PhnReader::Phn phn;
  int state_index;
  int phn_start_frame, phn_end_frame;
  int f;

  if (speaker.size() > 0)
    set_speaker(speaker);
  else
    cur_speaker = "";

  while (phn_reader.next(phn))
  {
    if (phn.state < 0)
      throw std::string("State segmented phn file is needed");

    phn_start_frame = (int)((float)phn.start/16000.0*fea_gen.frame_rate());
    phn_end_frame = (int)((float)phn.end/16000.0*fea_gen.frame_rate());
    if (phn_start_frame < start_frame)
    {
      assert( phn_end_frame > start_frame );
      phn_start_frame = start_frame;
    }
    if (phn_end_frame > end_frame)
    {
      assert( phn_start_frame < end_frame );
      phn_end_frame = end_frame;
    }

    if (phn.speaker.size() > 0 && phn.speaker != cur_speaker)
      set_speaker(phn.speaker);

    if (cur_speaker.size() == 0)
      throw std::string("Speaker ID is missing");

    if (state_num_labels)
      state = model.state(phn.state);
    else
    {
      hmm = model.hmm(model.hmm_index(phn.label[0]));
      state_index = hmm.state(phn.state);
      state = model.state(state_index);
    }

    for (f = phn_start_frame; f < phn_end_frame; f++)
    {
      FeatureVec fea_vec = fea_gen.generate(f);
      if (fea_gen.eof())
        break;

      (mllr_trainers[cur_speaker])->find_probs(&state, fea_vec);
    }
    if (f < phn_end_frame)
      break; // EOF in FeatureGenerator
  }
}


int
main(int argc, char *argv[])
{
  try {
    config("usage: mllr [OPTION...]\n")
      ('h', "help", "", "", "display help")
      ('b', "base=BASENAME", "arg", "", "base filename for model files")
      ('g', "gk=FILE", "arg", "", "Gaussian kernels")
      ('m', "mc=FILE", "arg", "", "kernel indices for states")
      ('p', "ph=FILE", "arg", "", "HMM definitions")
      ('c', "config=FILE", "arg must", "", "feature configuration")
      ('r', "recipe=FILE", "arg must", "", "recipe file")
      ('O', "ophn", "", "", "use output phns for adaptation")
      ('R', "raw-input", "", "", "raw audio input")
      ('M', "mllr=MODULE", "arg must", "", "MLLR module name")
      ('S', "speakers=FILE", "arg must", "", "speaker configuration input file")
      ('o', "out=FILE", "arg", "", "output speaker configuration file")
      ('\0', "sphn", "", "", "phn-files with speaker ID's in use")
      ('\0', "snl", "", "", "phn-files with state number labels")
      ('\0', "ords","", "", "files for each speaker are arranged successively")
      ('i', "info=INT", "arg", "0", "info level")
      ;
    config.default_parse(argc, argv);
    
    info = config["info"].get_int();
    raw_flag = config["raw-input"].specified;
    fea_gen.load_configuration(io::Stream(config["config"].get_str()));

    if (config["base"].specified)
    {
      model.read_all(config["base"].get_str());
    }
    else if (config["gk"].specified && config["mc"].specified &&
             config["ph"].specified)
    {
      model.read_gk(config["gk"].get_str());
      model.read_mc(config["mc"].get_str());
      model.read_ph(config["ph"].get_str());
    }
    else
    {
      throw std::string("Must give either --base or all --gk, --mc and --ph");
    }

    ordered_spk = config["ords"].specified;
    
    // Read recipe file
    recipe.read(io::Stream(config["recipe"].get_str()));

    mllr_module = dynamic_cast< LinTransformModule* >
      (fea_gen.module(config["mllr"].get_str()));
    if (mllr_module == NULL)
      throw std::string("Module ") + config["mllr"].get_str() +
        std::string(" is not a linear transformation module");

    phn_reader.set_speaker_phns(config["sphn"].specified);

    state_num_labels = config["snl"].specified;
    phn_reader.set_state_num_labels(state_num_labels);

    // Check the dimension
    if (model.dim() != fea_gen.dim()) {
      throw str::fmt(128,
                     "gaussian dimension is %d but feature dimension is %d",
                     model.dim(), fea_gen.dim());
    }

    speaker_conf.read_speaker_file(io::Stream(config["speakers"].get_str()));

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
      open_files(recipe.infos[f].audio_path, 
                 (config["ophn"].specified?recipe.infos[f].phn_out_path:
                  recipe.infos[f].phn_path),
                 (config["ophn"].specified?0:recipe.infos[f].start_line),
                 (config["ophn"].specified?0:recipe.infos[f].end_line));

      if (!config["sphn"].specified &&
          recipe.infos[f].speaker_id.size() == 0)
        throw std::string("Speaker ID is missing");

      train_mllr(start_frame, end_frame, recipe.infos[f].speaker_id);

      fea_gen.close();
      phn_reader.close();
    }

    // Compute the MLLR transformations
    for (MllrTrainerMap::const_iterator it = mllr_trainers.begin();
         it != mllr_trainers.end(); it++)
    {
      calculate_mllr_transform((*it).first);
    }
    
    // Write new speaker configuration
    if (config["out"].specified)
      speaker_conf.write_speaker_file(
        io::Stream(config["out"].get_str(), "w"));
  }
  catch (HmmSet::UnknownHmm &e) {
    fprintf(stderr, 
	    "Unknown HMM in transcription\n");
    abort();
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
