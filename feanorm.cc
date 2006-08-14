#include <math.h>
#include <vector>
#include <string>

#include "io.hh"
#include "conf.hh"
#include "Recipe.hh"
#include "FeatureGenerator.hh"
#include "SpeakerConfig.hh"

const char *recipe_file;
const char *out_file;

conf::Config config;
FeatureGenerator gen;
SpeakerConfig m_speaker_config(gen); // Speaker configuration handler

int
main(int argc, char *argv[])
{
  NormalizationModule *norm_mod = NULL;
  FeatureModule *norm_mod_src = NULL;
  std::vector<double> block_mean_acc, global_mean_acc;
  std::vector<double> block_var_acc, global_var_acc;
  std::vector<double> block_cov_acc, global_cov_acc;
  double global_acc_count;
  std::vector<float> mean, scale;
  bool raw_flag;
  bool cov_flag;
  int info;
  int block_size;
  int cur_block_size;
  int start_frame, end_frame;
  int dim;
  
  try {
    config("usage: feanorm [OPTION...]\n")
      ('h', "help", "", "", "display help")
      ('r', "recipe=FILE", "arg must", "", "recipe file")
      ('c', "config=FILE", "arg must", "", "read feature configuration")
      ('w', "write-config=FILE", "arg", "", "write feature configuration")
      ('R', "raw-input", "", "", "raw audio input")
      ('M', "module=NAME", "arg", "", "normalization module name")
      ('b', "block=INT", "arg", "1000", "block size (for reducing round-off errors)")
      ('P', "print", "", "", "print mean and scale to stdout")
      ('\0', "cov", "", "", "estimate and print covariance matrix")
      ('S', "speakers=FILE", "arg", "", "speaker configuration file")
      ('i', "info=INT", "arg", "0", "info level")
      ;
    config.default_parse(argc, argv);

    info = config["info"].get_int();
    raw_flag = config["raw-input"].specified;
    gen.load_configuration(io::Stream(config["config"].get_str()));

    dim = gen.dim();

    if (config["module"].specified)
    {
      norm_mod = dynamic_cast< NormalizationModule* >
        (gen.module(config["module"].get_str()));
      if (norm_mod == NULL)
        throw std::string("Module ") + config["module"].get_str() +
          std::string(" is not a normalization module");
      dim = norm_mod->dim();
      std::vector<FeatureModule*> sources = norm_mod->sources();
      assert( sources.front()->dim() == dim );
      norm_mod_src = sources.front();
    }
    else if (config["write-config"].specified)
    {
      fprintf(stderr, "Warning: No --module given, configuration will be written unaltered\n");
    }

    block_size = config["block"].get_int();

    cov_flag = config["cov"].specified;

    if (config["speakers"].specified)
      m_speaker_config.read_speaker_file(
        io::Stream(config["speakers"].get_str()));

    // Initialize accumulators
    block_mean_acc.resize(dim);
    block_var_acc.resize(dim);
    global_mean_acc.resize(dim, 0);
    global_var_acc.resize(dim, 0);
    global_acc_count = 0;

    if (cov_flag)
    {
      block_cov_acc.resize(dim*dim);
      global_cov_acc.resize(dim*dim, 0);
    }

    // Read recipe file
    Recipe recipe;
    recipe.read(io::Stream(config["recipe"].get_str()));

    // Handle each file in the recipe
    for (int recipe_index = 0; recipe_index < (int)recipe.infos.size(); 
	 recipe_index++) 
    {
      if (info > 0)
      {
        fprintf(stderr, "Processing file: %s\n",
	      recipe.infos[recipe_index].audio_path.c_str());
      }
      gen.open(recipe.infos[recipe_index].audio_path, raw_flag);

      if (config["speakers"].specified)
      {
        m_speaker_config.set_speaker(recipe.infos[recipe_index].speaker_id);
        if (recipe.infos[recipe_index].utterance_id.size() > 0)
          m_speaker_config.set_utterance(
            recipe.infos[recipe_index].utterance_id);
      }

      for (int d = 0; d < dim; d++)
      {
        block_mean_acc[d] = 0;
        block_var_acc[d] = 0;
      }
      if (cov_flag)
      {
        for (int d = 0; d < dim*dim; d++)
          block_cov_acc[d] = 0;
      }
      cur_block_size = 0;

      start_frame = (int)(recipe.infos[recipe_index].start_time *
                          gen.frame_rate());
      end_frame = (int)(recipe.infos[recipe_index].end_time *
                        gen.frame_rate());
      if (end_frame == 0)
        end_frame = INT_MAX;
      for (int f = start_frame; f < end_frame; f++)
      {
        const FeatureVec temp_fvec = gen.generate(f);
        if (gen.eof())
        {
          // Accumulate to global variables
          if (cur_block_size > 0)
          {
            for (int d = 0; d < dim; d++)
            {
              global_mean_acc[d] += block_mean_acc[d]/(double)block_size;
              global_var_acc[d] += block_var_acc[d]/(double)block_size;
            }
            if (cov_flag)
            {
              for (int d = 0; d < dim*dim; d++)
                global_cov_acc[d] += block_cov_acc[d]/(double)block_size;
            }
            global_acc_count += (double)cur_block_size/(double)block_size;
          }
          break;
        }

        const FeatureVec vec = (norm_mod_src == NULL? temp_fvec :
                                norm_mod_src->at(f));
        for (int d = 0; d < dim; d++)
        {
          block_mean_acc[d] += vec[d];
          block_var_acc[d] += vec[d]*vec[d];
        }
        if (cov_flag)
        {
          for (int d1 = 0; d1 < dim; d1++)
            for (int d2 = 0; d2 < dim; d2++)
              block_cov_acc[d1*dim+d2] += vec[d1]*vec[d2];
        }
        cur_block_size++;
        if (cur_block_size == block_size)
        {
          // Accumulate to global variables
          for (int d = 0; d < dim; d++)
          {
            global_mean_acc[d] += block_mean_acc[d]/(double)block_size;
            global_var_acc[d] += block_var_acc[d]/(double)block_size;
          }
          if (cov_flag)
          {
            for (int d = 0; d < dim*dim; d++)
              global_cov_acc[d] += block_cov_acc[d]/(double)block_size;
          }
          global_acc_count++;
          // Reset the block accumulators
          for (int d = 0; d < dim; d++)
          {
            block_mean_acc[d] = 0;
            block_var_acc[d] = 0;
          }
          if (cov_flag)
          {
            for (int d = 0; d < dim*dim; d++)
              block_cov_acc[d] = 0;
          }
          cur_block_size = 0;
        }
      }

      gen.close();
    }
    mean.resize(dim);
    for (int d = 0; d < dim; d++)
    {
      global_mean_acc[d] /= global_acc_count;
      mean[d] = global_mean_acc[d];
    }
    scale.resize(dim);
    for (int d = 0; d < dim; d++)
    {
      scale[d] = 1/sqrtf(global_var_acc[d] / global_acc_count -
                         global_mean_acc[d]*global_mean_acc[d]);
    }
    
    if (config["print"].specified)
    {
      printf("mean:\n");
      for (int d = 0; d < dim; d++)
        printf("%8.2f ", mean[d]);
      printf("\n");
      printf("scale:\n");
      for (int d = 0; d < dim; d++)
        printf("%8.2f ", scale[d]);
      printf("\n");

    }

    if (cov_flag)
    {
      for (int d1 = 0; d1 < dim; d1++)
      {
        for (int d2 = 0; d2 < dim; d2++)
        {
          double t = global_cov_acc[d1*dim+d2] / global_acc_count -
            global_mean_acc[d1]*global_mean_acc[d2];
          printf("%f ", t);
        }
        printf("\n");
      }
    }
      
    if (norm_mod != NULL)
      norm_mod->set_normalization(mean, scale);
    if (config["write-config"].specified)
      gen.write_configuration(io::Stream(config["write-config"].get_str(),
                                         "w"));
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
