#include <fstream>
#include <string>
#include <string.h>
#include <iostream>

#include "io.hh"
#include "str.hh"
#include "conf.hh"
#include "HmmSet.hh"
#include "FeatureGenerator.hh"
#include "Recipe.hh"
#include "SpeakerConfig.hh"
#include "util.hh"

std::string out_file;
std::string save_summary_file;

int info;
int accum_pos;
bool raw_flag;
bool transtat;
bool durstat;
float start_time, end_time;
int start_frame, end_frame;
double total_log_likelihood=0;

conf::Config config;
Recipe recipe;
HmmSet model;
FeatureGenerator fea_gen;
SpeakerConfig speaker_config(fea_gen);


void
train(HmmSet *model, Segmentator *segmentator)
{
  int frame;

  while (segmentator->next_frame()) {

    // Fetch the current feature vector
    frame = segmentator->current_frame();
    FeatureVec feature = fea_gen.generate(frame);    
    model->reset_cache();
    
    // Accumulate all possible states distributions for this frame
    const std::vector<Segmentator::IndexProbPair> &pdfs
      = segmentator->pdf_probs();
    for (int i = 0; i < (int)pdfs.size(); i++) {
      model->accumulate_distribution(feature, pdfs[i].index,
                                     pdfs[i].prob, accum_pos);
      if (!segmentator->computes_total_log_likelihood())
        total_log_likelihood += util::safe_log(
          pdfs[i].prob*model->state_likelihood(pdfs[i].index, feature));
    }
    
    // Accumulate also transition probabilities if desired
    if (transtat) {

      const std::vector<Segmentator::IndexProbPair> &transitions
        = segmentator->transition_probs();
      
      for (int i = 0; i < (int)transitions.size(); i++) {
        model->accumulate_transition(transitions[i].index,
                                     transitions[i].prob);
        if (!segmentator->computes_total_log_likelihood())
        {
          HmmTransition &t = model->transition(transitions[i].index);
          total_log_likelihood +=
            util::safe_log(transitions[i].prob*t.prob);
        }
      }
    }
  }
  if (segmentator->computes_total_log_likelihood())
    total_log_likelihood += segmentator->get_total_log_likelihood();
}


int
main(int argc, char *argv[])
{
  Segmentator *segmentator;
  try {
    config("usage: stats [OPTION...]\n")
      ('h', "help", "", "", "display help")
      ('b', "base=BASENAME", "arg", "", "base filename for model files")
      ('g', "gk=FILE", "arg", "", "Mixture base distributions")
      ('m', "mc=FILE", "arg", "", "Mixture coefficients for the states")
      ('p', "ph=FILE", "arg", "", "HMM definitions")
      ('c', "config=FILE", "arg must", "", "feature configuration")
      ('r', "recipe=FILE", "arg must", "", "recipe file")
      ('O', "ophn", "", "", "use output phns for training")
      ('H', "hmmnet", "", "", "use HMM networks for training")
      ('D', "den-hmmnet", "", "", "use denominator HMM networks for training")
      ('o', "out=BASENAME", "arg must", "", "base filename for output statistics")
      ('R', "raw-input", "", "", "raw audio input")
      ('t', "transitions", "", "", "collect also state transition statistics")
      ('d', "durstat", "", "", "collect also duration statistics")
      ('F', "fw-beam=FLOAT", "arg", "0", "Forward beam (for lattice-based training)")
      ('W', "bw-beam=FLOAT", "arg", "0", "Backward beam (for lattice-based training)")
      ('A', "ac-scale=FLOAT", "arg", "1", "Acoustic scaling (for lattice-based training)")
      ('s', "savesum=FILE", "arg", "", "save summary information (loglikelihood etc.)")
      ('S', "speakers=FILE", "arg", "", "speaker configuration file")
      ('B', "batch=INT", "arg", "0", "number of batch processes with the same recipe")
      ('I', "bindex=INT", "arg", "0", "batch process index")
      ('i', "info=INT", "arg", "0", "info level")
      ('\0', "mllt", "", "", "maximum likelihood linear transformation")
      ;
    config.default_parse(argc, argv);
    
    info = config["info"].get_int();
    raw_flag = config["raw-input"].specified;
    fea_gen.load_configuration(io::Stream(config["config"].get_str()));

    // Initialize the model for accumulating statistics
    if (config["base"].specified)
    {
      model.read_all(config["base"].get_str(), config["mllt"].specified);
    }
    else if (config["gk"].specified && config["mc"].specified &&
             config["ph"].specified)
    {
      model.read_gk(config["gk"].get_str(), config["mllt"].specified);
      model.read_mc(config["mc"].get_str());
      model.read_ph(config["ph"].get_str());
    }
    else
    {
      throw std::string("Must give either --base or all --gk, --mc and --ph");
    }
    out_file = config["out"].get_str();

    if (config["savesum"].specified)
      save_summary_file = config["savesum"].get_str();

    if (config["batch"].specified^config["bindex"].specified)
      throw std::string("Must give both --batch and --bindex");

    // Check for state transition statistics
    transtat = config["transitions"].specified;
    if (transtat)
      fprintf(stderr, "You have defined --transitions option: state transition statistics will be collected as well\n");

    // Check for duration statistics
    durstat = config["durstat"].specified;
    if (durstat)
      fprintf(stderr, "You have defined --durstat option: duration statistics will be collected as well\n");
    
    // Check the dimension
    if (model.dim() != fea_gen.dim()) {
      throw str::fmt(128, 
		     "gaussian dimension is %d but feature dimension is %d",
                     model.dim(), fea_gen.dim());
    }

    // Load speaker configurations
    if (config["speakers"].specified)
    {
      speaker_config.read_speaker_file(
        io::Stream(config["speakers"].get_str()));
    }

    // Read recipe file
    recipe.read(io::Stream(config["recipe"].get_str()),
                config["batch"].get_int(), config["bindex"].get_int(),
                true);

    // Configure the model for accumulating
    if (config["den-hmmnet"].specified) {
      model.set_estimation_mode(PDF::MMI);
      accum_pos=1;
    }
    else {
      model.set_estimation_mode(PDF::ML);
      accum_pos=0;
    }
    model.start_accumulating();        

    // Process each recipe line
    for (int f = 0; f < (int)recipe.infos.size(); f++)
    {

      // Print file name, start and end times to stderr
      if (info > 0)
      {
        fprintf(stderr, "Processing file: %s",
                recipe.infos[f].audio_path.c_str());
        if (recipe.infos[f].start_time || recipe.infos[f].end_time)
          fprintf(stderr," (%.2f-%.2f)",recipe.infos[f].start_time,
                  recipe.infos[f].end_time);
        fprintf(stderr,"\n");
      }

      if (config["speakers"].specified)
      {
        speaker_config.set_speaker(recipe.infos[f].speaker_id);
        if (recipe.infos[f].utterance_id.size() > 0)
          speaker_config.set_utterance(recipe.infos[f].utterance_id);
      }

      bool skip = false;

      if (config["hmmnet"].specified || config["den-hmmnet"].specified)
      {
        // Open files and configure
        HmmNetBaumWelch* lattice =
          recipe.infos[f].init_hmmnet_files(&model,
                                            config["den-hmmnet"].specified,
                                            &fea_gen, raw_flag, NULL);
        lattice->set_collect_transition_probs(transtat);
        lattice->set_pruning_thresholds(config["bw-beam"].get_float(), config["fw-beam"].get_float());
        if (config["ac-scale"].specified)
          lattice->set_acoustic_scaling(config["ac-scale"].get_float());
        
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
        PhnReader* phnreader = 
          recipe.infos[f].init_phn_files(&model, false, false,
                                         config["ophn"].specified, &fea_gen,
                                         config["raw-input"].specified, NULL);
        phnreader->set_collect_transition_probs(transtat);
        segmentator = phnreader;
        if (!segmentator->init_utterance_segmentation())
        {
          fprintf(stderr, "Could not initialize the utterance for PhnReader.");
          fprintf(stderr,"Current file was: %s\n",
                  recipe.infos[f].audio_path.c_str());
          skip = true;
        }
      }

      if (!skip)
      {
        // Train
        train(&model, segmentator);
      }
	
      // Clean up
      delete segmentator;
      fea_gen.close();
    }
    
    if (info > 0)
    {
      fprintf(stderr, "Finished collecting statistics (%i/%i), writing models\n",
	      config["batch"].get_int(), config["bindex"].get_int());
      fprintf(stderr, "Total log likelihood: %f\n", total_log_likelihood);
    }

    // Write statistics to file dump and clean up
    model.dump_statistics(out_file);
    model.stop_accumulating();
  }


  // Handle errors
  catch (HmmSet::UnknownHmm &e) {
    fprintf(stderr, 
	    "Unknown HMM in transcription, "
	    "writing incompletely taught models\n");
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
