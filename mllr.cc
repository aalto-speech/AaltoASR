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

conf::Config config;
Recipe recipe;
HmmSet model;
FeatureGenerator fea_gen;
SpeakerConfig speaker_conf(fea_gen);

typedef std::map<std::string, MllrTrainer*> MllrTrainerMap;
MllrTrainerMap mllr_trainers;
bool ordered_spk; // files for a speaker are arranged successively

LinTransformModule *mllr_module;
std::string cur_speaker;

std::set<std::string> updated_speakers;

double total_log_likelihood = 0;


void
calculate_mllr_transform(const std::string &speaker)
{
  std::string cur_speaker = speaker_conf.get_cur_speaker();

  if (cur_speaker != speaker)
    speaker_conf.set_speaker(speaker);
  (mllr_trainers[speaker])->calculate_transform(mllr_module);
  (mllr_trainers[speaker])->clear_stats();
  updated_speakers.insert(speaker);
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
train_mllr(Segmentator *seg)
{
  HmmState state;
  int i;
  
  while (seg->next_frame())
  {
    const std::vector<Segmentator::IndexProbPair> &pdfs =
      seg->pdf_probs();
    FeatureVec fea_vec = fea_gen.generate(seg->current_frame());
    if (fea_gen.eof())
      break; // EOF in FeatureGenerator

    for (i = 0; i < (int)pdfs.size(); i++)
    {
      state = model.state(pdfs[i].index);
      (mllr_trainers[cur_speaker])->find_probs(pdfs[i].prob, &state, fea_vec);
    }
  }
  if (seg->computes_total_log_likelihood())
    total_log_likelihood += seg->get_total_log_likelihood();
}


int
main(int argc, char *argv[])
{
  Segmentator *segmentator;
  
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
      ('E', "extvit", "", "", "Use extended Viterbi over HMM networks")
      ('V', "vit", "", "", "Use Viterbi over HMM networks")
      ('M', "mllr=MODULE", "arg must", "", "MLLR module name")
      ('S', "speakers=FILE", "arg must", "", "speaker configuration input file")
      ('o', "out=FILE", "arg", "", "output speaker configuration file")
      ('F', "fw-beam=FLOAT", "arg", "0", "Forward beam (for HMM networks)")
      ('W', "bw-beam=FLOAT", "arg", "0", "Backward beam (for HMM networks)")
      ('\0', "snl", "", "", "phn-files with state number labels")
      ('\0', "rsamp", "", "", "phn sample numbers are relative to start time")
      ('\0', "ords","", "", "files for each speaker are arranged successively")
      ('B', "batch=INT", "arg", "0", "number of batch processes with the same recipe")
      ('I', "bindex=INT", "arg", "0", "batch process index")
      ('i', "info=INT", "arg", "0", "info level")
      ;
    config.default_parse(argc, argv);
    
    info = config["info"].get_int();
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

    if ((config["vit"].specified || config["extvit"].specified) &&
        !config["hmmnet"].specified)
      throw std::string("--vit and --extvit require --hmmnet");

    ordered_spk = config["ords"].specified;
    
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

      bool skip = false;

      if (recipe.infos[f].speaker_id.size() == 0)
        throw std::string("Speaker ID is missing");

      if (config["hmmnet"].specified)
      {
        // Open files and configure
        HmmNetBaumWelch* lattice = recipe.infos[f].init_hmmnet_files(
          &model, false, &fea_gen, NULL);
        lattice->set_pruning_thresholds(config["bw-beam"].get_float(),
                                        config["fw-beam"].get_float());

        if (config["vit"].specified)
          lattice->set_mode(HmmNetBaumWelch::MODE_VITERBI);
        if (config["extvit"].specified)
          lattice->set_mode(HmmNetBaumWelch::MODE_EXTENDED_VITERBI);

        set_speaker(recipe.infos[f].speaker_id);
        if (recipe.infos[f].utterance_id.size() > 0)
          speaker_conf.set_utterance(recipe.infos[f].utterance_id);
        
        double orig_beam = lattice->get_backward_beam();
        int counter = 1;
        while (!lattice->init_utterance_segmentation())
        {
          if (counter >= 5)
          {
            fprintf(stderr, "Could not run Baum-Welch for file %s\n",
                    recipe.infos[f].audio_path.c_str());
            fprintf(stderr, "The HMM network may be incorrect or initial beam too low.\n");
            skip = true;
            break;
          }
          fprintf(stderr,
                  "Warning: Backward phase failed, increasing beam to %.1f\n",
                  ++counter*orig_beam);
          lattice->set_pruning_thresholds(counter*orig_beam, 0);
        }
        segmentator = lattice;
      }
      else
      {
        // Create phn_reader
        PhnReader *phn_reader = 
          recipe.infos[f].init_phn_files(&model,
                                         config["rsamp"].specified,
                                         config["snl"].specified,
                                         config["ophn"].specified, &fea_gen,
                                         NULL);
        
        set_speaker(recipe.infos[f].speaker_id);
        if (recipe.infos[f].utterance_id.size() > 0)
          speaker_conf.set_utterance(recipe.infos[f].utterance_id);
        
        if (!phn_reader->init_utterance_segmentation())
        {
          fprintf(stderr, "Could not initialize the utterance for PhnReader.");
          fprintf(stderr,"Current file was: %s\n",
                  recipe.infos[f].audio_path.c_str());
          skip = true;
        }
        segmentator = phn_reader;
      }

      train_mllr(segmentator);

      fea_gen.close();
      segmentator->close();
      delete segmentator;
    }

    // Compute the MLLR transformations
    for (MllrTrainerMap::const_iterator it = mllr_trainers.begin();
         it != mllr_trainers.end(); it++)
    {
      calculate_mllr_transform((*it).first);
    }
    
    // Write new speaker configuration
    if (config["out"].specified)
    {
      std::set<std::string> *speaker_set = NULL, *utterance_set = NULL;
      std::set<std::string> empty_ut;

      if (config["batch"].get_int() > 1)
      {
        if (config["bindex"].get_int() == 1)
          updated_speakers.insert(std::string("default"));
        speaker_set = &updated_speakers;
        utterance_set = &empty_ut;
      }

      speaker_conf.write_speaker_file(
        io::Stream(config["out"].get_str(), "w"), speaker_set, utterance_set);
    }

    if (info > 0 && total_log_likelihood != 0)
    {
      fprintf(stderr, "Total log likelihood: %f\n", total_log_likelihood);
    }
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
