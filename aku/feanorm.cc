#include <climits>
#include <math.h>
#include <vector>
#include <string>

#include "io.hh"
#include "conf.hh"
#include "Recipe.hh"
#include "FeatureGenerator.hh"
#include "FeatureModules.hh"
#include "SpeakerConfig.hh"

using namespace aku;

const char *recipe_file;
const char *out_file;

conf::Config config;
FeatureGenerator gen;
SpeakerConfig speaker_config(gen); // Speaker configuration handler

std::set<std::string> updated_utterances;

int
main(int argc, char *argv[])
{
  NormalizationModule *norm_mod = NULL;
  FeatureModule *norm_mod_src = NULL;
  LinTransformModule *pca_mod = NULL;
  FeatureModule *pca_mod_src = NULL;
  std::vector<double> block_mean_acc, global_mean_acc;
  std::vector<double> block_var_acc, global_var_acc;
  std::vector<double> block_cov_acc, global_cov_acc;
  std::vector<double> utt_mean_acc;
  std::vector<double> utt_var_acc;
  double global_acc_count;
  std::vector<float> mean, scale;
  bool cov_flag;
  bool pca_flag;
  int info;
  int block_size;
  int cur_block_size;
  int utt_frames;
  int start_frame, end_frame;
  int dim;
  
  try {
    config("usage: feanorm [OPTION...]\n")
      ('h', "help", "", "", "display help")
      ('r', "recipe=FILE", "arg must", "", "recipe file")
      ('c', "config=FILE", "arg must", "", "read feature configuration")
      ('w', "write-config=FILE", "arg", "", "write feature configuration")
      ('M', "module=NAME", "arg", "", "normalization module name")
      ('P', "pca=NAME", "arg", "", "pca module name")
      ('u', "unit-determinant", "", "", "unit determinant for pca transform, by default unit variance for data")
      ('b', "block=INT", "arg", "1000", "block size (for reducing round-off errors)")
      ('\0', "utt=FILE", "arg", "", "estimate utterance normalization and write to a file")
      ('p', "print", "", "", "print mean and variance to stdout")
      ('\0', "cov", "", "", "estimate and print covariance matrix")
      ('S', "speakers=FILE", "arg", "", "speaker configuration file")
      ('i', "info=INT", "arg", "0", "info level")
      ;
    config.default_parse(argc, argv);

    info = config["info"].get_int();
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
    else
    {
      if (config["utt"].specified)
        throw std::string("--utt requires the normalization module (--module)");
      if (config["write-config"].specified)
      {
        fprintf(stderr, "Warning: No --module given, configuration will be written unaltered\n");
      }
    }

    if (config["pca"].specified)
    {
      pca_mod = dynamic_cast< LinTransformModule* >
        (gen.module(config["pca"].get_str()));
      if (pca_mod == NULL)
        throw std::string("Module ") + config["pca"].get_str() +
          std::string(" is not a linear transformation module");
      std::vector<FeatureModule*> sources = pca_mod->sources();
      assert( sources.front()->dim() == dim );
      pca_mod_src = sources.front();
    }
    
    block_size = config["block"].get_int();

    cov_flag = config["cov"].specified;
    pca_flag = config["pca"].specified;

    if (config["speakers"].specified)
      speaker_config.read_speaker_file(
        io::Stream(config["speakers"].get_str()));
    else if (config["utt"].specified)
      throw std::string("--utt requires --speakers");

    // Initialize accumulators
    block_mean_acc.resize(dim);
    block_var_acc.resize(dim);
    utt_mean_acc.resize(dim);
    utt_var_acc.resize(dim);
    global_mean_acc.resize(dim, 0);
    global_var_acc.resize(dim, 0);
    global_acc_count = 0;

    if (cov_flag || pca_flag)
    {
      block_cov_acc.resize(dim*dim);
      global_cov_acc.resize(dim*dim, 0);
    }

    // Read recipe file
    Recipe recipe;
    recipe.read(io::Stream(config["recipe"].get_str()),
                0, 0, false);

    // Handle each file in the recipe
    for (int recipe_index = 0; recipe_index < (int)recipe.infos.size(); 
	 recipe_index++) 
    {
      if (info > 0)
      {
        fprintf(stderr, "Processing file: %s\n",
	      recipe.infos[recipe_index].audio_path.c_str());
      }
      gen.open(recipe.infos[recipe_index].audio_path);

      if (config["speakers"].specified)
      {
        speaker_config.set_speaker(recipe.infos[recipe_index].speaker_id);
        if (recipe.infos[recipe_index].utterance_id.size() > 0)
          speaker_config.set_utterance(
            recipe.infos[recipe_index].utterance_id);
      }

      for (int d = 0; d < dim; d++)
      {
        block_mean_acc[d] = 0;
        block_var_acc[d] = 0;
        utt_mean_acc[d] = 0;
        utt_var_acc[d] = 0;
      }
      if (cov_flag || pca_flag)
      {
        for (int d = 0; d < dim*dim; d++)
          block_cov_acc[d] = 0;
      }
      cur_block_size = 0;
      utt_frames = 0;

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
            if (cov_flag || pca_flag)
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
          utt_mean_acc[d] += vec[d];
          double temp = vec[d]*vec[d];
          block_var_acc[d] += temp;
          utt_var_acc[d] += temp;
        }
        if (cov_flag || pca_flag)
        {
          for (int d1 = 0; d1 < dim; d1++)
            for (int d2 = 0; d2 < dim; d2++)
              block_cov_acc[d1*dim+d2] += vec[d1]*vec[d2];
        }
        cur_block_size++;
        utt_frames++;
        if (cur_block_size == block_size)
        {
          // Accumulate to global variables
          for (int d = 0; d < dim; d++)
          {
            global_mean_acc[d] += block_mean_acc[d]/(double)block_size;
            global_var_acc[d] += block_var_acc[d]/(double)block_size;
          }
          if (cov_flag || pca_flag)
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
          if (cov_flag || pca_flag)
          {
            for (int d = 0; d < dim*dim; d++)
              block_cov_acc[d] = 0;
          }
          cur_block_size = 0;
        }
      }

      gen.close();

      if (config["utt"].specified &&
          recipe.infos[recipe_index].utterance_id.size() > 0)
      {
        // Set utterance normalization to normalization module
        mean.resize(dim);
        scale.resize(dim);
        for (int d = 0; d < dim; d++)
        {
          utt_mean_acc[d] /= (double)utt_frames;
          mean[d] = utt_mean_acc[d];
          double var = sqrtf(utt_var_acc[d] / (double)utt_frames -
                             utt_mean_acc[d]*utt_mean_acc[d]);
          if (var <= 0)
            var = 1;
          scale[d] = 1/var;
        }
        assert( norm_mod != NULL );
        norm_mod->set_normalization(mean, scale);
        updated_utterances.insert(recipe.infos[recipe_index].utterance_id);
      }
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

    Matrix tr_matrix(dim, dim);
    if (pca_flag) {
      Matrix eigvecs(dim, dim);
      Vector eigvals(dim);
      for (int d1 = 0; d1 < dim; d1++)
        for (int d2 = 0; d2 < dim; d2++)
          eigvecs(d1, d2) = global_cov_acc[d1*dim+d2] / global_acc_count -
            global_mean_acc[d1]*global_mean_acc[d2];
      LaEigSolveSymmetricVecIP(eigvecs, eigvals);
      tr_matrix.copy(eigvecs);
      LaVectorLongInt pivots(dim,1);
      LUFactorizeIP(tr_matrix, pivots);
      LaLUInverseIP(tr_matrix, pivots);
      
      // Normalize pca to have unit determinant, mainly for seeding mllt
      if (config["unit-determinant"].specified) {

        // Remove the effect of variance scaling first
        for (int i=0; i<dim; i++)
          for (int j=0; j<dim; j++)
            tr_matrix(i,j) /= scale[j];

        // Scale after that
        Matrix temp_m(tr_matrix);
        LUFactorizeIP(temp_m, pivots);
        double det=1;
        for (int i=0; i<dim; i++)
          det *= temp_m(i,i);
        det = std::fabs(det);
        double sc = pow(det, 1/(double)dim);
        Blas_Scale(1/sc, tr_matrix);
      }

      // Normal case, normalize the data to have unit variance
      else {
        for (int i=0; i<dim; i++)
          for (int j=0; j<dim; j++)
            tr_matrix(i,j) /= sqrt(eigvals(i));
        
        // Remove the effect of variance scaling
        for (int i=0; i<dim; i++)
          for (int j=0; j<dim; j++)
            tr_matrix(i,j) /= scale[j];
      }
    }
    
    if (config["print"].specified)
    {
      printf("mean:\n");
      for (int d = 0; d < dim; d++)
        printf("%f ", mean[d]);
      printf("\n");
      printf("variance:\n");
      for (int d = 0; d < dim; d++)
        printf("%f ", 1/(scale[d]*scale[d]));
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

    if (config["utt"].specified)
    {
      // Write utterance normalizations
      std::set<std::string> *speaker_set = NULL, *utterance_set = NULL;

      // std::set<std::string> empty_spkr;
      // if (config["batch"].get_int() > 1)
      // {
      //   if (config["bindex"].get_int() == 1)
      //     updated_utterances.insert(std::string("default"));
      //   speaker_set = &empty_spkr;
      //   utterance_set = &updated_utterances;
      // }
      
      speaker_config.write_speaker_file(
        io::Stream(config["utt"].get_str(), "w"), speaker_set, utterance_set);
    }

    if (norm_mod != NULL && !config["utt"].specified)
      norm_mod->set_normalization(mean, scale);

    if (pca_mod != NULL) {
      std::vector<float> tr;
      tr.resize(dim*dim);
      for (int i=0; i<dim; i++)
        for (int j=0; j<dim; j++)
          tr[i*dim + j] = tr_matrix(i, j);
      pca_mod->set_transformation_matrix(tr);
    }
    
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
