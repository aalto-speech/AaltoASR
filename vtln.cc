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

#define TINY 1e-10

  
std::string save_summary_file;

int info;

float grid_start;
float grid_step;
int grid_size;
bool relative_grid;

conf::Config config;
Recipe recipe;
HmmSet model;
FeatureGenerator fea_gen;
SpeakerConfig speaker_conf(fea_gen);

VtlnModule *vtln_module;
std::string cur_speaker;
int cur_warp_index;

typedef struct {
  float center; // Center warp value
  std::vector<float> warp_factors;
  std::vector<double> log_likelihoods;
} SpeakerStats;

typedef std::map<std::string, SpeakerStats> SpeakerStatsMap;
SpeakerStatsMap speaker_stats;



void
set_speaker(std::string speaker, std::string utterance, int grid_iter)
{
  float new_warp;
  int i;
  
  cur_speaker = speaker;

  assert( cur_speaker.size() > 0 );

  speaker_conf.set_speaker(speaker);
  if (utterance.size() > 0)
    speaker_conf.set_utterance(utterance);
    
  SpeakerStatsMap::iterator it = speaker_stats.find(speaker);
  if (it == speaker_stats.end())
  {
    // New speaker encountered
    SpeakerStats new_speaker;
      
    if (relative_grid)
      new_speaker.center = vtln_module->get_warp_factor();
    else
      new_speaker.center = 1;
    speaker_stats[cur_speaker] = new_speaker;
  }
    
  new_warp = speaker_stats[cur_speaker].center + grid_start +
    grid_iter*grid_step;
  vtln_module->set_warp_factor(new_warp);
    
  for (i = 0; i < (int)speaker_stats[cur_speaker].warp_factors.size(); i++)
  {
    if (fabs(new_warp - speaker_stats[cur_speaker].warp_factors[i])
        < TINY)
      break;
  }
  if (i == (int)speaker_stats[cur_speaker].warp_factors.size())
  {
    // New warp factor
    speaker_stats[cur_speaker].warp_factors.push_back(new_warp);
    speaker_stats[cur_speaker].log_likelihoods.push_back(0);
  }
  cur_warp_index = i;
}


void
compute_vtln_log_likelihoods(Segmentator *seg, std::string &speaker,
                             std::string &utterance)
{
  int grid_iter;
  int i;

  for (grid_iter = 0; grid_iter < grid_size; grid_iter++)
  {
    set_speaker(speaker, utterance, grid_iter);    
    seg->reset();
    seg->init_utterance_segmentation();

    while (seg->next_frame())
    {
      const std::vector<Segmentator::StateProbPair> &states =
        seg->state_probs();
      FeatureVec fea_vec = fea_gen.generate(seg->current_frame());
      if (fea_gen.eof())
        break; // EOF in FeatureGenerator
      
      for (i = 0; i < (int)states.size(); i++)
      {
        // Get probabilities
        speaker_stats[cur_speaker].log_likelihoods[cur_warp_index] += 
          log(states[i].prob*model.compute_state_likelihood(states[i].state_index,fea_vec));
      }
    }
  }
}


void
save_vtln_stats(FILE *fp)
{
  for (SpeakerStatsMap::iterator it = speaker_stats.begin();
      it != speaker_stats.end(); it++)
  {
    fprintf(fp, "[%s]\n", (*it).first.c_str());
    for (int i = 0; i < (int)(*it).second.warp_factors.size(); i++)
    {
      fprintf(fp, "%.3f: %.3f\n", (*it).second.warp_factors[i],
              (*it).second.log_likelihoods[i]);
    }
    fprintf(fp, "\n");
  }
}


void
find_best_warp_factors(void)
{
  for (SpeakerStatsMap::iterator it = speaker_stats.begin();
      it != speaker_stats.end(); it++)
  {
    assert( (*it).second.warp_factors.size() > 0 &&
            (*it).second.warp_factors.size() ==
            (*it).second.log_likelihoods.size() );
    float best_wf = (*it).second.warp_factors[0];
    double best_ll = (*it).second.log_likelihoods[0];
    for (int i = 1; i < (int)(*it).second.warp_factors.size(); i++)
    {
      if ((*it).second.log_likelihoods[i] > best_ll)
      {
        best_ll = (*it).second.log_likelihoods[i];
        best_wf = (*it).second.warp_factors[i];
      }
    }

    speaker_conf.set_speaker((*it).first);
    vtln_module->set_warp_factor(best_wf);
  }
}


int
main(int argc, char *argv[])
{
  PhnReader phn_reader;

  try {
    config("usage: vtln [OPTION...]\n")
      ('h', "help", "", "", "display help")
      ('b', "base=BASENAME", "arg", "", "base filename for model files")
      ('g', "gk=FILE", "arg", "", "Gaussian kernels")
      ('m', "mc=FILE", "arg", "", "kernel indices for states")
      ('p', "ph=FILE", "arg", "", "HMM definitions")
      ('c', "config=FILE", "arg must", "", "feature configuration")
      ('r', "recipe=FILE", "arg must", "", "recipe file")
      ('R', "raw-input", "", "", "raw audio input")
      ('v', "vtln=MODULE", "arg must", "", "VTLN module name")
      ('S', "speakers=FILE", "arg must", "", "speaker configuration input file")
      ('o', "out=FILE", "arg", "", "output speaker configuration file")
      ('s', "savesum=FILE", "arg", "", "save summary information (loglikelihoods)")
      ('\0', "snl", "", "", "phn-files with state number labels")
      ('\0', "rsamp", "", "", "phn sample numbers are relative to start time")
      ('\0', "grid-size=INT", "arg", "21", "warping grid size (default: 21/5)")
      ('\0', "grid-rad=FLOAT", "arg", "0.1", "radius of warping grid (default: 0.1/0.03)")
      ('\0', "relative", "", "", "relative warping grid (and smaller grid defaults)")
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

    if (config["savesum"].specified)
      save_summary_file = config["savesum"].get_str();

    if (config["batch"].specified^config["bindex"].specified)
      throw std::string("Must give both --batch and --bindex");
    
    // Read recipe file
    recipe.read(io::Stream(config["recipe"].get_str()),
                config["batch"].get_int(), config["bindex"].get_int(),
                true);

    vtln_module = dynamic_cast< VtlnModule* >
      (fea_gen.module(config["vtln"].get_str()));
    if (vtln_module == NULL)
      throw std::string("Module ") + config["vtln"].get_str() +
        std::string(" is not a VTLN module");
    
    grid_start = config["grid-rad"].get_float();
    grid_size = std::max(config["grid-size"].get_int(), 1);
    grid_step = 2*grid_start/std::max(grid_size-1, 1);
    relative_grid = config["relative"].specified;
    if (relative_grid)
    {
      if (!config["grid-rad"].specified)
        grid_start = 0.03;
      if (!config["grid-size"].specified)
        grid_size = 5;
      grid_step = 2*grid_start/std::max(grid_size-1, 1);
    }
    grid_start = -grid_start;

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
    
      // Open the audio and phn files from the given list.
      recipe.infos[f].init_phn_files((config["snl"].specified?NULL:&model),
                                     config["rsamp"].specified,
                                     true, &fea_gen,
                                     config["raw-input"].specified,
                                     &phn_reader);
      if (recipe.infos[f].speaker_id.size() == 0)
        throw std::string("Speaker ID is missing");

      compute_vtln_log_likelihoods(&phn_reader, recipe.infos[f].speaker_id,
                                   recipe.infos[f].utterance_id);

      fea_gen.close();
      phn_reader.close();
    }

    // Find the best warp factors from statistics
    find_best_warp_factors();

    if (config["savesum"].specified)
    {
      // Save the statistics
      save_vtln_stats(io::Stream(save_summary_file, "w"));
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
