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
#include "Distributions.hh"

std::string out_file;

int f;
int source_dim;
int target_dim;
int info;
int accum_pos;
int num_feas=0;
int maxmem;
int maxpos;
float start_time, end_time;

conf::Config config;
Recipe recipe;
HmmSet model;
FeatureGenerator fea_gen;
SpeakerConfig speaker_config(fea_gen);
std::string module_name;
std::vector<int> silence_states;

typedef std::pair<int,double> StateLikelihoodPair;
bool less_than_function( const StateLikelihoodPair lhs, const StateLikelihoodPair rhs )
{
  return lhs.second > rhs.second;
};


int
main(int argc, char *argv[])
{
  Segmentator *segmentator;
  try {
    config("usage: lda [OPTION...]\n")
      ('h', "help", "", "", "display help")
      ('p', "ph=FILE", "arg", "", "HMM definitions")
      ('c', "config=FILE", "arg must", "", "feature configuration")
      ('w', "write-config=FILE", "arg", "", "write feature configuration")
      ('r', "recipe=FILE", "arg must", "", "recipe file")
      ('O', "ophn", "", "", "use output phns for training")
      ('H', "hmmnet", "", "", "use HMM networks for training")
      ('d', "dim", "arg", "39", "dimensionality of the projected features (default 39)")
      ('M', "module=NAME", "arg", "", "linear transform module name")
      ('R', "raw-input", "", "", "raw audio input")
      ('F', "fw-beam=FLOAT", "arg", "0", "Forward beam (for HMM networks)")
      ('W', "bw-beam=FLOAT", "arg", "0", "Backward beam (for HMM networks)")
      ('A', "ac-scale=FLOAT", "arg", "1", "Acoustic scaling (for HMM networks)")
      ('V', "vit", "", "", "Use Viterbi over HMM networks")
      ('S', "speakers=FILE", "arg", "", "speaker configuration file")
      ('m', "maxmem=INT", "arg", "3000", "maximum memory usage in MB (default 3000)")
      ('\0', "mingamma=FLOAT", "arg", "50", "minimum gamma value per state (default 50)")
      ('\0', "maxgamma=FLOAT", "arg", "1000000", "gamma values will be ceiled to maxgamma (default 1 000 000)")
      ('\0', "no-silence", "", "", "don't use silence states in estimation")
      ('i', "info=INT", "arg", "0", "info level")
      ;
    config.default_parse(argc, argv);
    
    info = config["info"].get_int();
    double max_gamma = config["maxgamma"].get_double();
    
    // Load the HMM definitions
    if (config["ph"].specified)
      model.read_ph(config["ph"].get_str());
    else
      throw std::string("Must give --ph");

    // Find silence states
    Hmm &short_sil = model.hmm("_");
    Hmm &long_sil = model.hmm("__");
    for (int i=0; i<short_sil.num_states(); i++)
      silence_states.push_back(short_sil.state(i));
    for (int i=0; i<long_sil.num_states(); i++)
      silence_states.push_back(long_sil.state(i));
    
    // Load config
    if (config["config"].specified)
      fea_gen.load_configuration(io::Stream(config["config"].get_str()));
    
    // Check that the transformation exists
    module_name = config["module"].get_str();
    LinTransformModule *lda_module = dynamic_cast< LinTransformModule* >
      (fea_gen.module(module_name));
    if (lda_module == NULL)
      throw std::string("Module ") + module_name + 
        std::string(" is not a transform module");

    // Check the source
    const std::vector<FeatureModule*> source_modules = lda_module->sources();
    FeatureModule *source_module = source_modules[0];
    
    // Dimensionalities
    source_dim = source_module->dim();
    target_dim = config["dim"].get_int();

    // For how many untied states can we collect statistics
    int maxmem = config["maxmem"].get_int();
    int maxpos = int( (double(maxmem) * 1000 * 1000) / (double(source_dim) * source_dim * sizeof(double)));
    maxpos = std::min(maxpos, model.num_states());
    if (info)
      std::cout << "Collecting statistics at maximum for " << maxpos << " states" << std::endl;
    
    // Initialize an accumulator for the whole data
    FullStatisticsAccumulator whole_data_accumulator(source_dim);
    
    // State-based accumulators
    std::vector<FullStatisticsAccumulator*> state_accumulators;
    state_accumulators.resize(model.num_states());
    for (int i=0; i<model.num_states(); i++)
      state_accumulators[i] = NULL;

    // Gamma counts for the states
    std::vector<StateLikelihoodPair> state_gammas;
    state_gammas.resize(model.num_states());
    for (int i=0; i<model.num_states(); i++) {
      state_gammas[i].first=i;
      state_gammas[i].second=0;
    }
    
    // Load speaker configurations
    if (config["speakers"].specified)
    {
      speaker_config.read_speaker_file(
        io::Stream(config["speakers"].get_str()));
    }
    
    // Read recipe file
    recipe.read(io::Stream(config["recipe"].get_str()), 1, 1, true);


    // PASS 1
    
    // Process each recipe line and gamma counts
    for (f = 0; f < (int)recipe.infos.size(); f++)
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
      
      if (config["hmmnet"].specified)
      {
        // Open files and configure
        HmmNetBaumWelch* lattice = recipe.infos[f].init_hmmnet_files(
          &model, config["den-hmmnet"].specified, &fea_gen,
          config["raw-input"].specified, NULL);
        lattice->set_pruning_thresholds(config["bw-beam"].get_float(), config["fw-beam"].get_float());
        if (config["ac-scale"].specified)
          lattice->set_acoustic_scaling(config["ac-scale"].get_float());
        if (config["extvit"].specified)
          lattice->set_mode(HmmNetBaumWelch::MODE_EXTENDED_VITERBI);
        else if (config["vit"].specified)
          lattice->set_mode(HmmNetBaumWelch::MODE_VITERBI);
        
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
        segmentator = phnreader;
        if (!segmentator->init_utterance_segmentation())
        {
          fprintf(stderr, "Could not initialize the utterance for PhnReader.");
          fprintf(stderr,"Current file was: %s\n",
                  recipe.infos[f].audio_path.c_str());
          skip = true;
        }
      }
      
      int frame;
      while (segmentator->next_frame()) {
        
        // Fetch the current feature vector
        frame = segmentator->current_frame();
        FeatureVec feature = source_module->at(frame);
        
        if (fea_gen.eof())
          break; // EOF in FeatureGenerator
        
        // Update all gamma counts for this frame
        const std::vector<Segmentator::IndexProbPair> &pdfs
          = segmentator->pdf_probs();
        for (int i = 0; i < (int)pdfs.size(); i++)
          state_gammas[pdfs[i].index].second += pdfs[i].prob;
      }
      
      // Clean up
      delete segmentator;
      fea_gen.close();
    }

    if (info>0)
      std::cout << "Reserving memory" << std::endl;
    std::sort(state_gammas.begin(), state_gammas.end(), less_than_function);
    int total_states=0;
    // Set up the accumulators
    for (int i=0; i<maxpos; i++)
      if (state_gammas[i].second >= config["mingamma"].get_double()) {
        state_accumulators[state_gammas[i].first] = new FullStatisticsAccumulator(source_dim);
        total_states++;
      }
    // Don't collect statistics for silence states
    if (config["no-silence"].specified) {
      if (info>0)
        std::cout << "Discarding " << silence_states.size() << " silence states" << std::endl;
      for (unsigned int i=0; i<silence_states.size(); i++) {
        delete state_accumulators[silence_states[i]];
        state_accumulators[silence_states[i]] = NULL;
      }
    }
    
    // PASS 2
    
    // Process each recipe line and collect statistics
    if (info)
      std::cout << "Collecting statistics for "
                << total_states << " states" << std::endl;
    for (f = 0; f < (int)recipe.infos.size(); f++)
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
      
      if (config["hmmnet"].specified)
      {
        // Open files and configure
        HmmNetBaumWelch* lattice = recipe.infos[f].init_hmmnet_files(
          &model, config["den-hmmnet"].specified, &fea_gen,
          config["raw-input"].specified, NULL);
        lattice->set_pruning_thresholds(config["bw-beam"].get_float(), config["fw-beam"].get_float());
        if (config["ac-scale"].specified)
          lattice->set_acoustic_scaling(config["ac-scale"].get_float());
        if (config["extvit"].specified)
          lattice->set_mode(HmmNetBaumWelch::MODE_EXTENDED_VITERBI);
        else if (config["vit"].specified)
          lattice->set_mode(HmmNetBaumWelch::MODE_VITERBI);
        
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
        segmentator = phnreader;
        if (!segmentator->init_utterance_segmentation())
        {
          fprintf(stderr, "Could not initialize the utterance for PhnReader.");
          fprintf(stderr,"Current file was: %s\n",
                  recipe.infos[f].audio_path.c_str());
          skip = true;
        }
      }
      
      int frame;
      while (segmentator->next_frame()) {
        
        num_feas++;
        
        // Fetch the current feature vector
        frame = segmentator->current_frame();
        FeatureVec feature = source_module->at(frame);
        
        if (fea_gen.eof())
          break; // EOF in FeatureGenerator
        
        // Accumulate all possible state distributions for this frame
        const std::vector<Segmentator::IndexProbPair> &pdfs
          = segmentator->pdf_probs();
        for (int i = 0; i < (int)pdfs.size(); i++) {
          if (state_accumulators[pdfs[i].index] != NULL) {
            state_accumulators[pdfs[i].index]->accumulate(1, pdfs[i].prob, feature);
            whole_data_accumulator.accumulate(1, pdfs[i].prob, feature);
          }
        }

      }
      
      // Clean up
      delete segmentator;
      fea_gen.close();
    }

    
    // Compute the LDA

    if (info>0)
      std::cout << "Compute the LDA" << std::endl;
    
    // Get statistics for the whole data
    Vector data_mean;
    Matrix data_cov;
    whole_data_accumulator.get_mean_estimate(data_mean);
    whole_data_accumulator.get_covariance_estimate(data_cov);
    
    // B and W
    Matrix B = Matrix::zeros(source_dim, source_dim);
    Matrix W = Matrix::zeros(source_dim, source_dim);
    Vector tmean(source_dim);
    Matrix tcov(source_dim, source_dim);
    for (int i=0; i<model.num_states(); i++) {
      if (state_accumulators[i] != NULL) {
        state_accumulators[i]->get_mean_estimate(tmean);
        state_accumulators[i]->get_covariance_estimate(tcov);
        Blas_Add_Mult(tmean, -1, data_mean);
        Blas_R1_Update(B, tmean, tmean, std::min(state_accumulators[i]->gamma(), max_gamma));
        Blas_Add_Mat_Mult(W, std::min(state_accumulators[i]->gamma(), max_gamma), tcov);
      }
    }

    // Inverse of W
    Matrix W_inverse(W);
    LaVectorLongInt pivots(source_dim,1);
    LUFactorizeIP(W_inverse, pivots);
    LaLUInverseIP(W_inverse, pivots);

    // Eigendecomposition for inverse(W) * B
    Matrix WinvB(source_dim, source_dim);
    Blas_Mat_Mat_Mult(W_inverse, B, WinvB, 1.0, 0.0);
    Matrix eigenvectors(source_dim, source_dim);
    Vector eigenvalues_real(source_dim);
    Vector eigenvalues_imag(source_dim);
    LaEigSolve(WinvB, eigenvalues_real, eigenvalues_imag, eigenvectors);
    Matrix pca = Matrix::zeros(source_dim, target_dim);
    for (int i=0; i<target_dim-1; i++) {
      assert(std::fabs(eigenvalues_real(i+1))<=std::fabs(eigenvalues_real(i)));
      if (eigenvalues_real(i) < 0)
        std::cout << "Warning: a negative eigenvector was selected\n";
    }
    // All row values, only the first columns
    for (int i=0; i<source_dim; i++)
      for (int j=0; j<target_dim; j++)
        pca(i,j) = eigenvectors(i,j);

    Matrix fea_cov = Matrix::zeros(target_dim, target_dim);
    Matrix fea_cov_temp = Matrix::zeros(target_dim, source_dim);
    Blas_Mat_Trans_Mat_Mult(pca, data_cov, fea_cov_temp, 1.0, 0.0);
    Blas_Mat_Mat_Mult(fea_cov_temp, pca, fea_cov, 1.0, 0.0);

    Vector fea_cov_eigvals_real(target_dim);
    Vector fea_cov_eigvals_imag(target_dim);
    Matrix fea_cov_eigvecs(target_dim, target_dim);
    LaEigSolve(fea_cov, fea_cov_eigvals_real, fea_cov_eigvals_imag, fea_cov_eigvecs);
    for (int i=0; i<target_dim; i++)
      fea_cov_eigvals_real(i) = 1/sqrt(fea_cov_eigvals_real(i));
    Matrix fea_cov_eigval_matrix = Matrix::from_diag(fea_cov_eigvals_real);

    // Get the final LDA matrix
    Matrix lda_matrix = Matrix::zeros(target_dim, source_dim);
    Matrix lda_temp_matrix = Matrix::zeros(target_dim, target_dim);
    Blas_Mat_Mat_Trans_Mult(fea_cov_eigval_matrix, fea_cov_eigvecs, lda_temp_matrix, 1.0, 0.0);
    Blas_Mat_Mat_Trans_Mult(lda_temp_matrix, pca, lda_matrix, 1.0, 0.0);

    // Set the transformation to FeatureGenerator
    std::vector<float> tr;
    int pos=0;
    tr.resize(target_dim*source_dim);
    for (int i=0; i<target_dim; i++)
      for (int j=0; j<source_dim; j++)
      {
        tr[pos] = lda_matrix(i,j);
        pos++;
      }
    lda_module->set_transformation_matrix(tr);
    
    // Write out
    if (config["write-config"].specified)
      fea_gen.write_configuration(io::Stream(config["write-config"].get_str(), "w"));
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
