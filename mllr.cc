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
int phn_deviation = 0;
bool seg_files;

conf::Config config;
Recipe recipe;
HmmSet model;
FeatureGenerator fea_gen;
PhnReader *phn_reader;
SpeakerConfig speaker_conf(fea_gen);

typedef std::map<std::string, MllrTrainer*> MllrTrainerMap;
MllrTrainerMap mllr_trainers;
bool ordered_spk; // files for a speaker are arranged successively

LinTransformModule *mllr_module;
std::string cur_speaker;


void
calculate_mllr_transform(const std::string &speaker)
{
  std::string cur_speaker = speaker_conf.get_cur_speaker();

  if (cur_speaker != speaker)
    speaker_conf.set_speaker(speaker);
  (mllr_trainers[speaker])->calculate_transform(mllr_module);
  (mllr_trainers[speaker])->clear_stats();
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
train_mllr(int start_frame, int end_frame, std::string &speaker,
           std::string &utterance)
{
  Hmm hmm;
  HmmState state;
  PhnReader::Phn phn;
  int state_index;
  int phn_start_frame, phn_end_frame;
  int f;

  if (speaker.size() > 0)
  {
    set_speaker(speaker);
    if (utterance.size() > 0)
      speaker_conf.set_utterance(utterance);
  }
  else
    cur_speaker = "";

  while (phn_reader->next_phn_line(phn))
  {
    if (phn.state < 0)
      throw std::string("State segmented phn file is needed");

    phn_start_frame = (int)((double)phn.start/16000.0*fea_gen.frame_rate()
                            +0.5);
    phn_end_frame = (int)((double)phn.end/16000.0*fea_gen.frame_rate()+0.5);

    phn_start_frame = phn_start_frame + phn_deviation;
    phn_end_frame = phn_end_frame + phn_deviation;

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

    if (phn.speaker.size() > 0 && phn.speaker != cur_speaker)
    {
      set_speaker(phn.speaker);
      if (utterance.size() > 0)
        speaker_conf.set_utterance(utterance);
    }

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
      ('g', "gk=FILE", "arg", "", "Mixture base distributions")
      ('m', "mc=FILE", "arg", "", "Mixture coefficients for the states")
      ('p', "ph=FILE", "arg", "", "HMM definitions")
      ('c', "config=FILE", "arg must", "", "feature configuration")
      ('r', "recipe=FILE", "arg must", "", "recipe file")
      ('O', "ophn", "", "", "use output phns for adaptation")
      ('H', "hmmnet", "", "", "use HMM networks for training")      
      ('R', "raw-input", "", "", "raw audio input")
      ('M', "mllr=MODULE", "arg must", "", "MLLR module name")
      ('S', "speakers=FILE", "arg must", "", "speaker configuration input file")
      ('o', "out=FILE", "arg", "", "output speaker configuration file")
      ('\0', "seg", "", "", "decoder given state sequence hypothesis")
      ('\0', "sphn", "", "", "phn-files with speaker ID's in use")
      ('\0', "snl", "", "", "phn-files with state number labels")
      ('\0', "ords","", "", "files for each speaker are arranged successively")
      ('B', "batch=INT", "arg", "0", "number of batch processes with the same recipe")
      ('I', "bindex=INT", "arg", "0", "batch process index")
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

    // Some required switches
    state_num_labels = config["snl"].specified;
    ordered_spk = config["ords"].specified;
    seg_files = config["seg"].specified;
    
    // Read recipe file
    recipe.read(io::Stream(config["recipe"].get_str()),
                config["batch"].get_int(), config["bindex"].get_int(),
                true);    

    // Read linear transformation
    mllr_module = dynamic_cast< LinTransformModule* >
      (fea_gen.module(config["mllr"].get_str()));
    if (mllr_module == NULL)
      throw std::string("Module ") + config["mllr"].get_str() +
        std::string(" is not a linear transformation module");

    // Check the dimension
    if (model.dim() != fea_gen.dim()) {
      throw str::fmt(128,
                     "gaussian dimension is %d but feature dimension is %d",
                     model.dim(), fea_gen.dim());
    }

    speaker_conf.read_speaker_file(io::Stream(config["speakers"].get_str()));

    // Process each recipe line
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

      // Create phn_reader
      bool skip;
      phn_reader = 
        recipe.infos[f].init_phn_files(&model, false, false,
                                       config["ophn"].specified, &fea_gen,
                                       config["raw-input"].specified, NULL);
      phn_reader->set_speaker_phns(config["sphn"].specified);
      phn_reader->set_state_num_labels(state_num_labels);

      if (!phn_reader->init_utterance_segmentation())
      {
        fprintf(stderr, "Could not initialize the utterance for PhnReader.");
        fprintf(stderr,"Current file was: %s\n",
                recipe.infos[f].audio_path.c_str());
        skip = true;
      }
      
      if (!config["sphn"].specified &&
          recipe.infos[f].speaker_id.size() == 0)
        throw std::string("Speaker ID is missing");

      train_mllr(start_frame, end_frame, recipe.infos[f].speaker_id,
                 recipe.infos[f].utterance_id);

      fea_gen.close();
      phn_reader->close();
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
