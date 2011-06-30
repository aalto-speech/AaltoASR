#include <fstream>
#include <string>
#include <string.h>
#include <iostream>
#include <math.h>
#include <algorithm>
#include <limits.h>

#include "io.hh"
#include "str.hh"
#include "conf.hh"
#include "HmmSet.hh"
#include "FeatureGenerator.hh"
#include "Recipe.hh"
#include "SpeakerConfig.hh"
#include "util.hh"
#include "SegErrorEvaluator.hh"

using namespace aku;

std::string out_file;

int info;
int accum_pos;
bool transtat = false;
float start_time, end_time;
double total_num_log_likelihood = 0;
double total_den_log_likelihood = 0;
double total_mpe_score = 0;
double total_mpe_num_score = 0;
int num_frames = 0;

bool print_alignments = false;

double mpfe_insertion_penalty = 0;

// Training modes
bool mpe = false;
bool gradient_statistics = false;

bool binary_mpfe = false;

bool compute_mpe_numerator_score = true;


conf::Config config;
Recipe recipe;
HmmSet model;
HmmSet *num_seg_model = NULL;
FeatureGenerator fea_gen;
SpeakerConfig speaker_config(fea_gen, &model);

SegErrorEvaluator error_evaluator;
SegErrorEvaluator::ErrorMode errmode;


void print_alignment_line(FILE *f, float fr, int start, int end,
                          const std::string &label)
{
  // NOTE: phn files assume 16kHz sample rate
  int frame_mult = (int) (16000 / fr);

  if (start < 0)
    return;

  fprintf(f, "%d %d %s\n", start * frame_mult, end * frame_mult,
          label.c_str());
}


void simple_train(HmmSet &model, Segmentator &segmentator,
                  bool accumulate, FILE *alignment_out, bool hmmnets)
{
  int cur_start_frame = -1;
  std::string cur_label = "";

  double orig_beam = 0;
  int counter = 1;
  while (!segmentator.init_utterance_segmentation())
  {
    if (counter == 1)
    {
      fprintf(stderr, "Could not initialize the utterance segmentation.\n");
    }
    if (!hmmnets || counter >= 5)
    {
      fprintf(stderr, "Giving up for this file\n");
      return;
    }
    HmmNetBaumWelch *hmmnet_seg =
      dynamic_cast< HmmNetBaumWelch* > (&segmentator);
    assert( hmmnet_seg != NULL );
    if (counter == 1)
    {
      orig_beam = hmmnet_seg->get_backward_beam();
    }
    counter++;
    fprintf(stderr, "Increasing beam to %.1f\n", counter*orig_beam);
    hmmnet_seg->set_pruning_thresholds(0, counter*orig_beam);
  }
  
  while (segmentator.next_frame())
  {

    // Fetch the current feature vector
    int frame = segmentator.current_frame();
    FeatureVec feature = fea_gen.generate(frame);

    if (fea_gen.eof())
      break; // EOF in FeatureGenerator

    num_frames++;
    
    if (alignment_out != NULL)
    {
      if (cur_start_frame < 0)
      {
        // Initialize
        cur_start_frame = frame;
        cur_label = segmentator.highest_prob_label();
      }
      else if (cur_label != segmentator.highest_prob_label())
      {
        print_alignment_line(alignment_out, fea_gen.frame_rate(),
                             cur_start_frame, frame, cur_label);
        cur_start_frame = frame;
        cur_label = segmentator.highest_prob_label();
      }
    }

    // Accumulate all possible states distributions for this frame
    const Segmentator::IndexProbMap &pdfs = segmentator.pdf_probs();

    for (Segmentator::IndexProbMap::const_iterator it = pdfs.begin();
         it != pdfs.end(); ++it)
    {
      if (accumulate)
      {
        model.accumulate_distribution(feature, (*it).first,
                                      (*it).second, PDF::ML_BUF);
        // model.accumulate_aux_gamma((*it).first, (*it).second,
        //                             PDF::ML_BUF);
      }

      if (!segmentator.computes_total_log_likelihood())
      {
        total_num_log_likelihood += util::safe_log(
          (*it).second*model.state_likelihood((*it).first, feature));
      }
    }
    
    // Accumulate also transition probabilities if desired
    if (transtat && accumulate)
    {
      const Segmentator::IndexProbMap &transitions =
        segmentator.transition_probs();
      
      for (Segmentator::IndexProbMap::const_iterator it = transitions.begin();
         it != transitions.end(); ++it)
      {
        model.accumulate_transition((*it).first, (*it).second);
        if (!segmentator.computes_total_log_likelihood())
        {
          HmmTransition &t = model.transition((*it).first);
          total_num_log_likelihood+=util::safe_log((*it).second*t.prob);
        }
      }
    }
  }

  if (segmentator.computes_total_log_likelihood())
  {
    total_num_log_likelihood += segmentator.get_total_log_likelihood();
    fprintf(stderr, "  %g\n", segmentator.get_total_log_likelihood());
  }

  if (alignment_out != NULL)
  {
    if (cur_start_frame >= 0)
    {
      // Print the pending line
      // FIXME: Is +1 in the last frame correct?
      print_alignment_line(alignment_out, fea_gen.frame_rate(),
                           cur_start_frame, segmentator.current_frame()+1,
                           cur_label);
    }
  }
}


HmmNetBaumWelch::SegmentedLattice*
create_segmented_lattice(HmmNetBaumWelch &seg, float fw_beam, float bw_beam,
                         float ac_scale, int hmmnet_seg_mode)
{
  seg.set_pruning_thresholds(fw_beam, bw_beam);
  seg.set_acoustic_scaling(ac_scale);
  seg.set_mode(hmmnet_seg_mode);
  double orig_beam = seg.get_backward_beam();
  int counter = 1;
  HmmNetBaumWelch::SegmentedLattice *lattice = NULL;
  while ((lattice = seg.create_segmented_lattice()) == NULL)
  {
    if (counter >= 5)
    {
      fprintf(stderr, "Could not run Baum-Welch for file\n");
      fprintf(stderr, "The HMM network may be incorrect or initial beam too low.\n");
      return NULL;
    }
    counter++;
    fprintf(stderr,
            "Warning: Backward phase failed, increasing beam to %.1f\n",
            counter*orig_beam);
    seg.set_pruning_thresholds(0, counter*orig_beam);
  }
  
  // Recompute the scores
  lattice->compute_total_scores();
  return lattice;
}


void
collect_lattice_stats(HmmSet &model, HmmNetBaumWelch &seg,
                      HmmNetBaumWelch::SegmentedLattice *lattice,
                      PDF::StatisticsMode mode)
{
  std::set<int> active_nodes;

  if (!lattice->frame_lattice)
    throw std::string("collect_lattice_stats requires a frame lattice");
  
  active_nodes.insert(lattice->initial_node);
  while (active_nodes.find(lattice->final_node) == active_nodes.end())
  {
    std::set<int> target_nodes;

    num_frames++; // FIXME: Frames not counted with --no-train and DT

    int frame = -1;
    model.reset_cache();
    
    // Propagate the active nodes and collect the statistics
    for (std::set<int>::iterator it = active_nodes.begin();
         it != active_nodes.end(); ++it)
    {
      if (frame == -1)
        frame = lattice->nodes[*it].frame;
      else
        assert( frame == lattice->nodes[*it].frame );
      for (int i = 0; i < (int)lattice->nodes[*it].out_arcs.size(); i++)
      {
        HmmNetBaumWelch::SegmentedArc &arc =
          lattice->arcs[lattice->nodes[*it].out_arcs[i]];
        target_nodes.insert(arc.target_node); // Propagate

        const FeatureVec feature = seg.get_feature(frame);
        HmmTransition &tr = model.transition(arc.transition_index);
        int pdf_index = model.emission_pdf_index(tr.source_index);
        double arc_prob = exp(HmmNetBaumWelch::loglikelihoods.divide(
                                arc.total_score, lattice->total_score));
        
        if (mode & PDF_ML_STATS)
        {
          model.accumulate_distribution(feature, pdf_index, arc_prob,
                                         PDF::ML_BUF);
        }
        if (mode & PDF_MMI_STATS)
        {
          model.accumulate_distribution(feature, pdf_index, arc_prob,
                                        PDF::MMI_BUF);
        }
        double gamma = 0;
        if (mode & (PDF_MPE_NUM_STATS|PDF_MPE_DEN_STATS))
          gamma = (arc.custom_path_score-lattice->total_custom_score)*
            arc_prob;

        if (mode & PDF_MPE_NUM_STATS)
        {
          if (gamma > 0 || gradient_statistics)
            model.accumulate_distribution(feature, pdf_index,
                                           gamma, PDF::MPE_NUM_BUF);
          if (gradient_statistics)
            model.accumulate_aux_gamma(pdf_index, gamma, PDF::MPE_NUM_BUF);
        }
        if (mode & PDF_MPE_DEN_STATS)
        {
          if (gamma <= 0)
            model.accumulate_distribution(feature, pdf_index,
                                          -gamma, PDF::MPE_DEN_BUF);
        }
      }
    }

    active_nodes.swap(target_nodes);
  }
}

int main(int argc, char *argv[])
{
  int hmmnet_seg_mode = 0;
  int hmmnet_num_seg_mode = 0;
  PDF::StatisticsMode stats_mode = 0;
  bool only_ml = true;
  
  std::string gkfile, mcfile, phfile;
  try {
    config("usage: stats [OPTION...]\n")
      ('h', "help", "", "", "display help")
      ('b', "base=BASENAME", "arg", "", "base filename for model files")
      ('g', "gk=FILE", "arg", "", "Mixture base distributions")
      ('m', "mc=FILE", "arg", "", "Mixture coefficients for the states")
      ('p', "ph=FILE", "arg", "", "HMM definitions")
      ('\0', "nsegmc=FILE", "arg", "", "mc-file for segmentating the numerator")
      ('\0', "nseggk=FILE", "arg", "", "gk-file for segmentating the numerator")
      ('c', "config=FILE", "arg must", "", "feature configuration")
      ('r', "recipe=FILE", "arg must", "", "recipe file")
      ('O', "ophn", "", "", "use output phns for training")
      ('H', "hmmnet", "", "", "use HMM networks for training")
      ('o', "out=BASENAME", "arg must", "", "base filename for output statistics")
      ('t', "transitions", "", "", "collect also state transition statistics")
      ('F', "fw-beam=FLOAT", "arg", "0", "Forward beam (for HMM networks)")
      ('W', "bw-beam=FLOAT", "arg", "0", "Backward beam (for HMM networks)")
      ('A', "ac-scale=FLOAT", "arg", "1", "Acoustic scaling (for HMM networks)")
      ('M', "segmode=MODE", "arg", "bw", "Segmentation mode: bw/vit/mpv")
      ('\0', "numseg=MODE", "arg", "", "Numerator segmentation mode")
      ('\0', "ml", "", "", "Collect statistics for ML")
      ('\0', "mmi", "", "", "Collect statistics for MMI")
      ('\0', "mpe", "", "", "Collect statistics for MPE/MWE/MPFE")
      ('\0', "grad", "", "", "Prepare gradient based statistics (with --mpe)")
      ('\0', "mllt", "", "", "maximum likelihood linear transformation (for --ml)")
      ('\0', "errmode=MODE", "arg", "", "For --mpe. Modes: mwe/mpe/mpfe/mpfe-cps/mpfe-pdf")
      ('\0', "nosil", "", "", "Ignore silence arcs in MPE scoring")
      ('S', "speakers=FILE", "arg", "", "speaker configuration file")
      ('U', "uttadap", "", "", "Enable utterance adaptation")
      ('n', "no-train", "", "", "Only collect summary statistics")
      ('a', "alignment", "", "", "save output alignments (only with ML training)")
      ('B', "batch=INT", "arg", "0", "number of batch processes with the same recipe")
      ('I', "bindex=INT", "arg", "0", "batch process index")
      ('i', "info=INT", "arg", "0", "info level");
    config.default_parse(argc, argv);

    info = config["info"].get_int();
    fea_gen.load_configuration(io::Stream(config["config"].get_str()));

    // Initialize the model for accumulating statistics
    if (config["base"].specified) {
      model.read_all(config["base"].get_str());
      gkfile = config["base"].get_str() + ".gk";
      mcfile = config["base"].get_str() + ".mc";
      phfile = config["base"].get_str() + ".ph";
    }
    else if (config["gk"].specified && config["mc"].specified &&
             config["ph"].specified)
    {
      gkfile = config["gk"].get_str();
      model.read_gk(gkfile);
      mcfile = config["mc"].get_str();
      model.read_mc(mcfile);
      phfile = config["ph"].get_str();
      model.read_ph(phfile);
    }
    else {
      throw std::string(
        "Must give either --base or all --gk, --mc and --ph");
    }
    out_file = config["out"].get_str();

    if (config["nseggk"].specified || config["nsegmc"].specified)
    {
      if (!config["hmmnet"].specified)
        throw std::string("Numerator segmentation requires --hmmnet");
      num_seg_model = new HmmSet;
      if (config["nseggk"].specified)
        num_seg_model->read_gk(config["nseggk"].get_str());
      else
        num_seg_model->read_gk(gkfile);
      if (config["nsegmc"].specified)
        num_seg_model->read_mc(config["nsegmc"].get_str());
      else
        num_seg_model->read_mc(mcfile);
      num_seg_model->read_ph(phfile);
    }

    if (config["batch"].specified^config["bindex"].specified)
      throw std::string("Must give both --batch and --bindex");

    // Check for state transition statistics
    transtat = config["transitions"].specified;

    // Check the dimension
    if (model.dim() != fea_gen.dim()) {
      throw str::fmt(128,
                     "gaussian dimension is %d but feature dimension is %d",
                     model.dim(), fea_gen.dim());
    }

    // Load speaker configurations
    if (config["speakers"].specified) {
      speaker_config.read_speaker_file(
        io::Stream(config["speakers"].get_str()));
    }

    // Read recipe file
    recipe.read(io::Stream(config["recipe"].get_str()),
                config["batch"].get_int(), config["bindex"].get_int(),
                true);
    
    if (config["ml"].specified)
    {
      stats_mode |= PDF_ML_STATS;
    }
    if (config["mmi"].specified)
    {
      only_ml = false;
      stats_mode |= PDF_MMI_STATS;
    }
    if (config["mpe"].specified)
    {
      only_ml = false;
      mpe = true;
      if (config["grad"].specified)
      {
        gradient_statistics = true;
        stats_mode |= PDF_MPE_NUM_STATS;
      }
      else
      {
        stats_mode |= PDF_MPE_NUM_STATS|PDF_MPE_DEN_STATS;
      }
      error_evaluator.set_model(&model);
    }

    if (stats_mode == 0)
      throw std::string("At least one mode (--ml, --mmi, --mpe) must be given!");

    if (config["nosil"].specified)
      error_evaluator.set_ignore_silence(true);
    else
      error_evaluator.set_ignore_silence(false);

    if (!only_ml && !config["hmmnet"].specified)
      throw std::string("Discriminative training requires --hmmnet");

    if (config["segmode"].specified && !config["hmmnet"].specified)
      throw std::string("Segmentation modes are supported only with --hmmnet");
    conf::Choice segmode_choice;
    segmode_choice("bw", HmmNetBaumWelch::MODE_BAUM_WELCH)
      ("vit", HmmNetBaumWelch::MODE_VITERBI)
      ("mpv", HmmNetBaumWelch::MODE_MULTIPATH_VITERBI);
    hmmnet_seg_mode = HmmNetBaumWelch::MODE_BAUM_WELCH;
    if (!segmode_choice.parse(config["segmode"].get_str(), hmmnet_seg_mode))
      throw std::string("Invalid segmentation mode ") +
        config["segmode"].get_str();

    hmmnet_num_seg_mode = hmmnet_seg_mode;
    if (config["numseg"].specified &&
        !segmode_choice.parse(config["numseg"].get_str(),
                              hmmnet_num_seg_mode))
            throw std::string("Invalid segmentation mode ") +
              config["numseg"].get_str();

    if (config["errmode"].specified)
    {
      if (!config["mpe"].specified)
        fprintf(stderr, "--errmode ignored without --mpe\n");
      else
      {
        conf::Choice errmode_choice;
        errmode_choice("mwe", SegErrorEvaluator::MWE)
          ("mpe", SegErrorEvaluator::MPE)
          ("mpfe-pdf", SegErrorEvaluator::MPFE_PDF)
          ("mpfe-cps", SegErrorEvaluator::MPFE_CONTEXT_PHONE_STATE)
          ("mpfe", SegErrorEvaluator::MPFE_HYP_CONTEXT_PHONE_STATE)
          ;
        std::string errmode_str = config["errmode"].get_str();
        int result;
        if (!errmode_choice.parse(errmode_str, result))
          throw std::string("Invalid choice for --errmode: ") + errmode_str;
        errmode = (SegErrorEvaluator::ErrorMode)result;
        error_evaluator.set_mode(errmode);
      }
    }
    else if (config["mpe"].specified)
    {
      errmode = SegErrorEvaluator::MPE;
      error_evaluator.set_mode(errmode);
    }

    if (config["alignment"].specified)
    {
      print_alignments = true;
      if (config["ophn"].specified)
        throw std::string("Can not read and write to output PHNs");
      if (config["hmmnet"].specified &&
          hmmnet_num_seg_mode != HmmNetBaumWelch::MODE_VITERBI)
        printf("Warning: Printing alignments will not produce sensible results if using\nhmmnet mode other than viterbi.\n");
      if (!only_ml)
        printf("Warning: Alignment is disabled with discriminative training\n");
    }

    if (config["mllt"].specified)
    {
      if (!only_ml)
        throw std::string("--mllt is only supported with --ml");
      stats_mode |= PDF_ML_FULL_STATS;
    }
    if (!config["no-train"].specified)
      model.start_accumulating(stats_mode);

    // Process each recipe line
    for (int f = 0; f < (int) recipe.infos.size(); f++) {
      // Print file name, start and end times to stderr
      if (info > 0) {
        fprintf(stderr, "Processing file: %s",
                recipe.infos[f].audio_path.c_str());
        if (recipe.infos[f].start_time || recipe.infos[f].end_time)
          fprintf(stderr, " (%.2f-%.2f)", recipe.infos[f].start_time,
                  recipe.infos[f].end_time);
        fprintf(stderr, "\n");
      }

      if (config["speakers"].specified) {
        speaker_config.set_speaker(recipe.infos[f].speaker_id);
        if (config["uttadap"].specified &&
            recipe.infos[f].utterance_id.size() > 0)
          speaker_config.set_utterance(recipe.infos[f].utterance_id);
      }

      FILE *alignment_out = NULL;
      if (print_alignments)
      {
        if ((alignment_out = fopen(recipe.infos[f].alignment_path.c_str(),
                                   "w")) == NULL)
          fprintf(stderr, "Could not open alignment file %s\n",
                  recipe.infos[f].alignment_path.c_str());
      }

      if (!config["hmmnet"].specified)
      {
        assert( only_ml );
        PhnReader* phnreader = 
          recipe.infos[f].init_phn_files(&model, false, false,
                                         config["ophn"].specified, &fea_gen,
                                         NULL);
        phnreader->set_collect_transition_probs(transtat);
        simple_train(model, *phnreader, !config["no-train"].specified,
                     alignment_out, false);
        delete phnreader;
      }
      else
      {
        // Open files and configure
        HmmNetBaumWelch* num_seg = recipe.infos[f].init_hmmnet_files(
          (num_seg_model == NULL ? &model : num_seg_model),
          false, &fea_gen, NULL);

        if (only_ml)
        {
          num_seg->set_collect_transition_probs(transtat);
          num_seg->set_mode(hmmnet_num_seg_mode);
          num_seg->set_pruning_thresholds(config["fw-beam"].get_float(),
                                          config["bw-beam"].get_float());
          num_seg->set_acoustic_scaling(config["ac-scale"].get_float());
          simple_train(model, *num_seg, !config["no-train"].specified,
                       alignment_out, true);
        }
        else
        {
          // Discriminative training

          HmmNetBaumWelch* den_seg  = NULL;
          HmmNetBaumWelch::SegmentedLattice *den_lattice = NULL;
          HmmNetBaumWelch::SegmentedLattice *num_lattice =
            create_segmented_lattice(*num_seg, config["fw-beam"].get_float(),
                                     config["bw-beam"].get_float(),
                                     config["ac-scale"].get_float(),
                                     hmmnet_num_seg_mode);
          
          bool skip = false;
          if (num_lattice == NULL)
          {
            skip = true;
            fprintf(stderr, "Failed to segment the numerator lattice, skipping\n");
          }
          if (!skip)
          {
            fea_gen.close(); // init_hmmnet_files opens the file for fea_gen
            den_seg = recipe.infos[f].init_hmmnet_files(
              &model, true, &fea_gen, NULL);
            den_seg->set_collect_transition_probs(transtat);
            den_seg->set_mode(hmmnet_seg_mode);

            den_lattice = create_segmented_lattice(
              *den_seg, config["fw-beam"].get_float(),
              config["bw-beam"].get_float(), config["ac-scale"].get_float(),
              hmmnet_seg_mode);
            if (den_lattice == NULL)
            {
              skip = true;
              fprintf(stderr, "Failed to segment denominator lattice, skipping\n");
            }
          }
          if (!skip)
          {
            assert( num_seg->computes_total_log_likelihood() &&
                    den_seg->computes_total_log_likelihood() );

            if ((stats_mode&PDF_ML_STATS) &&
                !config["no-train"].specified)
              collect_lattice_stats(model, *num_seg, num_lattice,
                                    PDF_ML_STATS);
            total_num_log_likelihood += num_lattice->total_score;
              
            if (mpe)
            {
              if (errmode == SegErrorEvaluator::MWE ||
                  errmode == SegErrorEvaluator::MPE)
              {
                // Need a higher hierarchy lattice for the error evaluation
                int level = 0;
                if (errmode == SegErrorEvaluator::MWE)
                  level = 3; // FIXME? Only works for word-based lattices
                else if (errmode == SegErrorEvaluator::MPE)
                  level = 2;
                HmmNetBaumWelch::SegmentedLattice *num_lat_logical =
                  num_seg->extract_segmented_lattice(num_lattice, level);
                HmmNetBaumWelch::SegmentedLattice *den_lat_logical =
                  den_seg->extract_segmented_lattice(den_lattice, level);
                error_evaluator.initialize_reference(num_lat_logical);
                den_lat_logical->compute_custom_path_scores(error_evaluator);
                den_lat_logical->propagate_custom_scores_to_frame_segmented_lattice(den_lattice);

                if (compute_mpe_numerator_score)
                {
                  num_lat_logical->compute_custom_path_scores(error_evaluator);
                  total_mpe_num_score += num_lat_logical->total_custom_score;
                }
                
                delete den_lat_logical;
                delete num_lat_logical;
              }
              else
              {
                error_evaluator.initialize_reference(num_lattice);
                den_lattice->compute_custom_path_scores(error_evaluator);
                if (compute_mpe_numerator_score)
                {
                  num_lattice->compute_custom_path_scores(error_evaluator);
                  total_mpe_num_score += num_lattice->total_custom_score;
                }
              }
              if (info > 0)
                fprintf(stderr, "Total custom score %f\n",
                        den_lattice->total_custom_score);
              total_mpe_score += den_lattice->total_custom_score;
            }

            if (!config["no-train"].specified)
              collect_lattice_stats(model, *den_seg, den_lattice,
                                    (stats_mode&(~PDF_ML_STATS)));
            total_den_log_likelihood += den_lattice->total_score;
          }
          if (den_lattice != NULL)
            delete den_lattice;
          if (den_seg != NULL)
            delete den_seg;
          if (num_lattice != NULL)
            delete num_lattice;
        }
        delete num_seg;
      }

      if (alignment_out != NULL)
        fclose(alignment_out);
      fea_gen.close();
    }

    if (info > 0)
    {
      fprintf(stderr, "Finished collecting statistics (%i/%i)\n",
              config["bindex"].get_int(), config["batch"].get_int());
      fprintf(stderr, "Total num log likelihood: %g\n",
              total_num_log_likelihood);
      if (mpe)
        fprintf(stderr, "MPE score: %g\n", total_mpe_score);
      if (!only_ml)
        fprintf(stderr, "MMI score: %g\n",(total_num_log_likelihood - total_den_log_likelihood));
    }

    // Write statistics to file dump and clean up
    if (!config["no-train"].specified)
    {
      model.dump_statistics(out_file);
      model.stop_accumulating();
    }
    
    std::string lls_file_name = out_file+".lls";
    std::ofstream lls_file(lls_file_name.c_str());
    if (lls_file)
    {
      lls_file.precision(12); 
      lls_file << "Numerator loglikelihood: " << total_num_log_likelihood << std::endl;
//      lls_file << "Summed numerator loglikelihood: " << summed_num_log_likelihood << std::endl;
      if (!only_ml)
      {
        lls_file << "Denominator loglikelihood: " << total_den_log_likelihood << std::endl;
        lls_file << "MMI score: " << (total_num_log_likelihood - total_den_log_likelihood) << std::endl;
      }
      if (mpe)
      {
        lls_file << "MPE score: " << total_mpe_score << std::endl;
        lls_file << "MPE numerator score: " << total_mpe_num_score << std::endl;
      }
      lls_file << "Number of frames: " << num_frames << std::endl;
      lls_file.close();
    }
    if (num_seg_model != NULL)
      delete num_seg_model;
  }

  // Handle errors
  catch (HmmSet::UnknownHmm &e) {
    fprintf(stderr, "Unknown HMM in transcription, "
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
