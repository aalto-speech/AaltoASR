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

  
std::string out_file;
std::string save_summary_file;

int info;
bool raw_flag;
bool transtat;
bool durstat;
float start_time, end_time;
int start_frame, end_frame;

conf::Config config;
Recipe recipe;
HmmSet ml_model;
HmmSet mmi_model;
FeatureGenerator fea_gen;


// FIXME: does this go correctly?
void
train(HmmSet *model, Segmentator *segmentator)
{
  int frame, relative_target;
  
  segmentator->init_utterance_segmentation();

  while (segmentator->next_frame()) {

    // Fetch the current feature vector
    frame = segmentator->current_frame();
    FeatureVec feature = fea_gen.generate(frame);    
    
    // Accumulate all possible states distributions for this frame
    const std::vector<Segmentator::StateProbPair> &states
      = segmentator->state_probs();
    for (int i = 0; i < (int)states.size(); i++)
      model->accumulate_distribution(feature, states[i].state_index, states[i].prob);
    
    // Accumulate also transition probabilities if desired
    if (transtat) {
      
      const Segmentator::TransitionMap &transitions
	= segmentator->transition_probs();
      
      for (Segmentator::TransitionMap::const_iterator it = transitions.begin();
	   it != transitions.end(); it++)
	{
	  relative_target = 0;
	  if ((*it).first.from != (*it).first.to) relative_target = 1;
	  model->accumulate_transition((*it).first.from, relative_target, (*it).second);
	}
    }
  }
}


int
main(int argc, char *argv[])
{

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
      ;
    config.default_parse(argc, argv);
    
    info = config["info"].get_int();
    raw_flag = config["raw-input"].specified;
    fea_gen.load_configuration(io::Stream(config["config"].get_str()));


    // Initialize the model for accumulating ML statistics
    if (config["base"].specified)
    {
      ml_model.read_all(config["base"].get_str());
    }
    else if (config["gk"].specified && config["mc"].specified &&
             config["ph"].specified)
    {
      ml_model.read_gk(config["gk"].get_str());
      ml_model.read_mc(config["mc"].get_str());
      ml_model.read_ph(config["ph"].get_str());
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
    if (ml_model.dim() != fea_gen.dim()) {
      throw str::fmt(128, 
		     "gaussian dimension is %d but feature dimension is %d",
                     ml_model.dim(), fea_gen.dim());
    }

    // Read recipe file
    recipe.read(io::Stream(config["recipe"].get_str()),
                config["batch"].get_int(), config["bindex"].get_int(),
                true);


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
          
      // If lattice path set, let's do MMI
      if (recipe.infos[f].hmmnet_path != "") {

	// If this is the first recipe line, initialize the model for training
	if (mmi_model.dim() == 0) {
	  if (config["base"].specified)
	    {
	      mmi_model.read_all(config["base"].get_str());
	    }
	  else if (config["gk"].specified && config["mc"].specified &&
		   config["ph"].specified)
	    {
	      mmi_model.read_gk(config["gk"].get_str());
	      mmi_model.read_mc(config["mc"].get_str());
	      mmi_model.read_ph(config["ph"].get_str());
	    }
	}

	// Open files and configure
	HmmNetBaumWelch* lattice = recipe.infos[f].init_hmmnet_files(&mmi_model, &fea_gen,
								     raw_flag, NULL);
	lattice->set_collect_transition_probs(transtat);
	lattice->set_pruning_thresholds(config["bw-beam"].get_float(), config["fw-beam"].get_float());
	if (config["ac-scale"].specified)
	  lattice->set_acoustic_scaling(config["ac-scale"].get_float());
	
	// Train MMI
	mmi_model.start_accumulating();
	train(&mmi_model, lattice);
	
	// Clean up
	delete lattice;
	fea_gen.close();
      }

      // ML statistics should be accumulated anyway
      PhnReader* phnreader = 
	recipe.infos[f].init_phn_files(&ml_model, true, config["ophn"].specified, &fea_gen,
				       config["raw-input"].specified, NULL);
      phnreader->set_collect_transition_probs(transtat);
      
      // Train ML
      ml_model.start_accumulating();
      train(&ml_model, phnreader);
      
      // Clean up
      delete phnreader;
      fea_gen.close();
    }
    
    if (info > 0)
      fprintf(stderr, "Finished collecting statistics (%i/%i), writing models\n",
	      config["batch"].get_int(), config["bindex"].get_int());

    // Write statistics to file dump and clean up
    if (mmi_model.dim() != 0) {
      mmi_model.dump_statistics(out_file+"_mmi");
      mmi_model.stop_accumulating();
    }
    ml_model.dump_statistics(out_file);
    ml_model.stop_accumulating();
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
