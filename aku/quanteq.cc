#include <limits.h>
#include <float.h>
#include <math.h>
#include <string>
#include <algorithm>

#include "io.hh"
#include "str.hh"
#include "conf.hh"
#include "FeatureGenerator.hh"
#include "FeatureModules.hh"
#include "Recipe.hh"
#include "SpeakerConfig.hh"

using namespace aku;

#define TINY 1e-10

int info;

// Grid search parameters
float grid_alpha_step;
float grid_gamma_end;
float grid_gamma_step;
int num_quant;

conf::Config config;
Recipe recipe;
FeatureGenerator fea_gen;
SpeakerConfig utterance_conf(fea_gen);
std::vector<std::vector<float> > fea_mat; // Matrix of features
std::vector<std::vector<float> > quant;   // Matrix of quantiles
std::vector<float> quant_train;          // Training quantiles (channel independent)

QuantEqModule *quanteq_module;

void
compute_quantiles(void)
{
  int qind;

  quant.clear();
  quant.resize(fea_mat.size());

  for (int c = 0; c < (int) fea_mat.size(); c++) {
    quant[c].clear();
    quant[c].resize(num_quant);

    for (int q = 0; q < num_quant; q++) {
      qind = (int) ceil(fea_mat[c].size() * (float) (q+1) / num_quant) - 1;
      nth_element(fea_mat[c].begin(), fea_mat[c].begin() + qind, fea_mat[c].end());
      quant[c][q] = fea_mat[c][qind];

      // Check lower bound
      if (quant[c][q] < quant_train[q])
      {
        quant[c][q] = quant_train[q];
      }
    }
  }
}

void
find_best_params(void)
{
  float score;
  float best_score;
  std::vector<float> alphavec;
  std::vector<float> gammavec;
  std::vector<float> quant_max;

  alphavec.clear();
  gammavec.clear();
  quant_max.clear();
  alphavec.resize((int) quant.size());
  gammavec.resize((int) quant.size());
  quant_max.resize((int) quant.size());

  // Perform a grid search to find best values for alpha and gamma for each channel
  for (int c = 0; c < (int) quant.size(); c++) {
    best_score = FLT_MAX;
    for (float alpha = 0; alpha <= 1; alpha = alpha + grid_alpha_step) {
      for (float gamma = 0; gamma <= grid_gamma_end; gamma = gamma + grid_gamma_step) {

        score = 0;
        for (int q = 0; q < (int) quant[c].size()-1; q++) {
          score = score + pow( (quant[c].back() * ( alpha * pow(quant[c][q] / quant[c].back(), gamma) + (1-alpha) * (quant[c][q] / quant[c].back()) ) - quant_train[q]) , 2);
        }
        if (score < best_score) {
          best_score = score;
          alphavec[c] = alpha;
          gammavec[c] = gamma;
        }
      }
    }
    quant_max[c] = quant[c].back();
  }


quanteq_module->set_alpha(alphavec);
quanteq_module->set_gamma(gammavec);
quanteq_module->set_quant_max(quant_max);
}

int
main(int argc, char *argv[])
{
  std::set<std::string> utterances;
  try {
    config("usage: quanteq [OPTION...]\n")
      ('h', "help", "", "", "display help")
      ('c', "config=FILE", "arg must", "", "feature configuration")
      ('r', "recipe=FILE", "arg must", "", "recipe file")
      ('q', "quanteq=MODULE", "arg must", "", "QuantEq module name")
      ('S', "utterances=FILE", "arg must", "", "utterances configuration input file")
      ('o', "out=FILE", "arg", "", "output utterance configuration file")
      ('\0', "num-quant=INT", "arg", "4", "Number of quantiles (default: 4")
      ('\0', "grid-alpha-step=FLOAT", "arg", "0.01", "step for grid search of alpha (default: 0.01")
      ('\0', "grid-gamma-step=FLOAT", "arg", "0.01", "step for grid search of gamma (default: 0.01")
      ('\0', "grid-gamma-end=FLOAT", "arg", "3", "maximum value for grid search of gamma (default: 3")
      ('B', "batch=INT", "arg", "0", "number of batch processes with the same recipe")
      ('I', "bindex=INT", "arg", "0", "batch process index")
      ('i', "info=INT", "arg", "0", "info level")
      ;
    config.default_parse(argc, argv);

    if (config["batch"].get_int() > 1 && config["bindex"].get_int() == 1)
      utterances.insert(std::string("default"));
    
    info = config["info"].get_int();
    fea_gen.load_configuration(io::Stream(config["config"].get_str()));

    if (config["batch"].specified^config["bindex"].specified)
      throw std::string("Must give both --batch and --bindex");

    fea_mat.clear();
    fea_mat.resize(fea_gen.dim());
    
    // Read recipe file
    recipe.read(io::Stream(config["recipe"].get_str()),
                config["batch"].get_int(), config["bindex"].get_int(),
                true);

    quanteq_module = dynamic_cast< QuantEqModule* >
      (fea_gen.module(config["quanteq"].get_str()));
    if (quanteq_module == NULL)
      throw std::string("Module ") + config["quanteq"].get_str() +
        std::string(" is not a QUANTEQ module");

    // Set grid search parameters
    grid_alpha_step = config["grid-alpha-step"].get_float();
    grid_gamma_end = config["grid-gamma-end"].get_float();
    grid_gamma_step = config["grid-gamma-step"].get_float();
    num_quant = config["num-quant"].get_int();
    quant_train = quanteq_module->get_quant_train();

    utterance_conf.read_speaker_file(io::Stream(config["utterances"].get_str()));

    // Go through recipe file one line at a time
    for (int f = 0; f < (int)recipe.infos.size(); f++)
    {
      if (info > 0)
      {
        fprintf(stderr, "Processing file: %s (%d/%d)", 
                recipe.infos[f].audio_path.c_str(), f+1, (int)recipe.infos.size());
        if (recipe.infos[f].start_time || recipe.infos[f].end_time) 
          fprintf(stderr," (%.2f-%.2f)",recipe.infos[f].start_time,
                  recipe.infos[f].end_time);
        fprintf(stderr,"\n");
      }

      // Open the audio file (this way?)
      fea_gen.open(recipe.infos[f].audio_path.c_str());
      utterance_conf.set_utterance(recipe.infos[f].utterance_id);
      utterances.insert(recipe.infos[f].utterance_id);

      // Extract features
      int start_frame = 0;
      if (recipe.infos[f].start_time > 0)
        start_frame = (int) (recipe.infos[f].start_time * fea_gen.frame_rate());
      int end_frame = INT_MAX;
      if (recipe.infos[f].end_time > 0)
        end_frame = (int) (recipe.infos[f].end_time * fea_gen.frame_rate());
      int cur_frame = start_frame;

      while (cur_frame <= end_frame) {
        const FeatureVec fea = fea_gen.generate(cur_frame);
        if (fea_gen.eof())
          break;
        // Save features in a matrix for quantiles calculation
        for (int c = 0; c < fea.dim(); c++)
        {
          fea_mat[c].push_back(fea[c]);
        }

        cur_frame++;
      }
      
      fea_gen.close();

      // Compute quantiles of the utterance
      compute_quantiles();

      // Find best parameters with grid search
      find_best_params();      
    }
    // Write new utterance configuration
    if (config["out"].specified)
    {
      std::set<std::string> *utterance_set = NULL;
      if (config["batch"].get_int() > 1)
        utterance_set = &utterances;
      
      utterance_conf.write_speaker_file(
        io::Stream(config["out"].get_str(), "w"), NULL, utterance_set);
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
