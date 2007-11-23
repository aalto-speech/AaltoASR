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

int info;
int accum_pos;
bool transtat;
float start_time, end_time;
double total_log_likelihood = 0;
double total_mpe_score = 0;

// Training modes
bool ml = false;
bool mmi = false;
bool mpe = false;

conf::Config config;
Recipe recipe;
HmmSet model;
FeatureGenerator fea_gen;
SpeakerConfig speaker_config(fea_gen);


class MPEEvaluator : public HmmNetBaumWelch::CustomDataQuery {
private:
  int m_first_frame;
  int m_cur_frame;
  std::vector< std::vector<HmmNetBaumWelch::ArcInfo>* > m_ref_segmentation;
  
public:
  typedef enum { MPEM_MONOPHONE_LABEL, // Correct monophone label
                 MPEM_MONOPHONE_STATE, // Correct monophone label+state index
                 MPEM_CONTEXT_LABEL, // Correct context phone label
                 MPEM_STATE, // Correct state
                 MPEM_CONTEXT_PHONE_STATE, // A state of the correct CP
                 MPEM_HYP_CONTEXT_PHONE_STATE, // A correct state in hypothesis CP
  } MPEMode;
  
  MPEEvaluator() { m_first_frame = -1; m_mode = MPEM_MONOPHONE_LABEL; m_model = NULL; }

  // CustomDataQuery interface
  virtual ~MPEEvaluator() { }
  virtual double custom_data_value(int frame, HmmNetBaumWelch::Arc &arc);

  // Other public methods
  void set_mode(MPEMode mode) { m_mode = mode; }
  void set_model(HmmSet *model) { m_model = model; }
  void fetch_frame_info(HmmNetBaumWelch *seg);
  void reset(void);

private:
  std::string extract_center_phone(const std::string &label);
  std::string extract_context_phone(const std::string &label);
  int extract_state(const std::string &label);

private:
  MPEMode m_mode;
  HmmSet *m_model;
};


std::string
MPEEvaluator::extract_center_phone(const std::string &label)
{
  int pos1 = label.find_last_of('-');
  int pos2 = label.find_first_of('+');
  std::string temp = "";
  if (pos1 >= 0 && pos2 >= 0)
  {
    if (pos2 > pos1+1)
      temp = label.substr(pos1+1, pos2-pos1-1);
  }
  else if (pos1 >= 0)
    temp = label.substr(pos1+1);
  else if (pos2 >= 0)
    temp = label.substr(0, pos2);
  else
    temp = label;
  if ((int)temp.size() > 0)
    return temp;
  return label;
}

std::string
MPEEvaluator::extract_context_phone(const std::string &label)
{
  int pos = label.find_last_of('.'); // Remove the state number
  if (pos > 0)
    return label.substr(0, pos);
  return label;
}


int
MPEEvaluator::extract_state(const std::string &label)
{
  int pos = label.find_last_of('.'); // Find the state number
  if (pos >= 0)
  {
    std::string temp;
    temp = label.substr(pos+1);
    return atoi(temp.c_str());
  }
  return -1;
}

double
MPEEvaluator::custom_data_value(int frame, HmmNetBaumWelch::Arc &arc)
{
  int internal_frame = frame - m_first_frame;
  if (internal_frame < 0 || internal_frame >= (int)m_ref_segmentation.size())
    return 0;

  // Ignore silence nodes
  if (arc.label.find('-') == std::string::npos &&
      arc.label.find('+') == std::string::npos &&
      arc.label[0] == '_') // Silence node
    return 0;
  
  // Check the label against the correct one

  std::string label;
  int state = -1;
  if (m_mode == MPEM_MONOPHONE_LABEL || m_mode == MPEM_MONOPHONE_STATE)
    label = extract_center_phone(arc.label);
  else if (m_mode == MPEM_CONTEXT_LABEL ||
           m_mode == MPEM_HYP_CONTEXT_PHONE_STATE)
    label = extract_context_phone(arc.label);
  else if (m_mode == MPEM_STATE || m_mode == MPEM_CONTEXT_PHONE_STATE)
  {
    HmmTransition &tr = model.transition(arc.transition_id);
    state = model.emission_pdf_index(tr.source_index);
  }
  if (m_mode == MPEM_MONOPHONE_STATE)
    state = extract_state(arc.label);
  
  for (int i = 0; i < (int)m_ref_segmentation[internal_frame]->size(); i++)
  {
    if (m_mode == MPEM_MONOPHONE_LABEL || m_mode == MPEM_CONTEXT_LABEL)
    {
      if ((*m_ref_segmentation[internal_frame])[i].label == label)
        return 1;
    }
    else if (m_mode == MPEM_MONOPHONE_STATE)
    {
      if ((*m_ref_segmentation[internal_frame])[i].label == label &&
          (*m_ref_segmentation[internal_frame])[i].pdf_index == state)
        return 1;
    }
    else if (m_mode == MPEM_STATE)
    {
      if ((*m_ref_segmentation[internal_frame])[i].pdf_index == state)
        return 1;
    }
    else if (m_mode == MPEM_CONTEXT_PHONE_STATE)
    {
      Hmm &hmm = m_model->hmm((*m_ref_segmentation[internal_frame])[i].label);
      for (int s = 0; s < hmm.num_states(); s++)
        if (hmm.state(s) == state)
          return 1;
    }
    else if (m_mode == MPEM_HYP_CONTEXT_PHONE_STATE)
    {
      Hmm &hmm = m_model->hmm(label);
      for (int s = 0; s < hmm.num_states(); s++)
        if ((*m_ref_segmentation[internal_frame])[i].pdf_index==hmm.state(s))
          return 1;
    }
  }
  
  return 0;
}

void
MPEEvaluator::reset(void)
{
  m_first_frame = -1;
  // Clear the old reference segmentation
  for (int i = 0; i < (int)m_ref_segmentation.size(); i++)
    delete m_ref_segmentation[i];
  m_ref_segmentation.clear();
}

void
MPEEvaluator::fetch_frame_info(HmmNetBaumWelch *seg)
{
  int seg_frame = seg->current_frame();
  if (m_first_frame == -1)
    m_first_frame = seg_frame;
  else if (m_cur_frame+1 != seg_frame)
    throw std::string("Non-continuous numerator segmentation");
  m_cur_frame = seg_frame;

  m_ref_segmentation.push_back(
    new std::vector<HmmNetBaumWelch::ArcInfo> );
  seg->fill_arc_info(*(m_ref_segmentation.back()));
  
  // Process the labels to speed up the custom data query
  for (int i = 0; i < (int)m_ref_segmentation.back()->size(); i++)
  {
    std::string &label = (*m_ref_segmentation.back())[i].label;
    if (m_mode == MPEM_MONOPHONE_STATE)
      (*m_ref_segmentation.back())[i].pdf_index = extract_state(label);
    if (m_mode == MPEM_MONOPHONE_LABEL || m_mode == MPEM_MONOPHONE_STATE)
      label = extract_center_phone(label);
    else if (m_mode == MPEM_CONTEXT_LABEL ||
             m_mode == MPEM_CONTEXT_PHONE_STATE)
      label = extract_context_phone(label);
  }
}


MPEEvaluator mpe_evaluator;


bool initialize_hmmnet(HmmNetBaumWelch* lattice, float bw_beam, float fw_beam,
                       float ac_scale, int hmmnet_mode,
                       std::string &audio_path)
{
  lattice->set_pruning_thresholds(bw_beam, fw_beam);
  if (config["ac-scale"].specified)
    lattice->set_acoustic_scaling(ac_scale);
  if (hmmnet_mode == 1)
    lattice->set_mode(HmmNetBaumWelch::MODE_VITERBI);
  else if (hmmnet_mode == 2)
    lattice->set_mode(HmmNetBaumWelch::MODE_EXTENDED_VITERBI);

  double orig_beam = lattice->get_backward_beam();
  int counter = 1;
  while (!lattice->init_utterance_segmentation())
  {
    if (counter >= 5)
    {
      fprintf(stderr, "Could not run Baum-Welch for file %s\n",
              audio_path.c_str());
      fprintf(stderr, "The HMM network may be incorrect or initial beam too low.\n");
      return false;
    }
    fprintf(stderr,
            "Warning: Backward phase failed, increasing beam to %.1f\n",
            ++counter*orig_beam);
    lattice->set_pruning_thresholds(counter*orig_beam, 0);
  }
  return true;
}


void
train(HmmSet *model, Segmentator *segmentator, bool numerator)
{
  int frame;

  if (mpe && numerator)
    mpe_evaluator.reset();
  
  while (segmentator->next_frame()) {

    // Fetch the current feature vector
    frame = segmentator->current_frame();
    FeatureVec feature = fea_gen.generate(frame);    

    if (fea_gen.eof())
      break; // EOF in FeatureGenerator
    
    // Accumulate all possible states distributions for this frame
    const std::vector<Segmentator::IndexProbPair> &pdfs
      = segmentator->pdf_probs();
    
    if (mpe)
    {
      HmmNetBaumWelch *seg = dynamic_cast< HmmNetBaumWelch* >(segmentator);
      if (seg == NULL)
        throw std::string("MPE training requires the use of hmmnets!");
      if (numerator)
        mpe_evaluator.fetch_frame_info(seg);
      else
      {
        std::vector<HmmNetBaumWelch::ArcInfo> arcs;
        seg->fill_arc_info(arcs);
        for (int i = 0; i < (int)arcs.size(); i++)
        {
          double gamma = arcs[i].custom_score - seg->get_total_custom_score();
          if (gamma > 0)
            model->accumulate_distribution(feature, arcs[i].pdf_index,
                                           gamma, PDF::MPE_NUM_BUF);
          else
            model->accumulate_distribution(feature, arcs[i].pdf_index,
                                           -gamma, PDF::MPE_DEN_BUF);
        }
      }
    }
    if (ml || mmi)
    {
      for (int i = 0; i < (int)pdfs.size(); i++)
      {
        if (numerator)
        {
          model->accumulate_distribution(feature, pdfs[i].index,
                                         pdfs[i].prob, PDF::ML_BUF);
        }
        else if (mmi)
        {
          model->accumulate_distribution(feature, pdfs[i].index,
                                         pdfs[i].prob, PDF::MMI_BUF);
        }
        if (!segmentator->computes_total_log_likelihood())
          total_log_likelihood += (numerator?1:-1)*
            util::safe_log(pdfs[i].prob*model->state_likelihood(pdfs[i].index,
                                                                feature));
      }
    }
    
    // Accumulate also transition probabilities if desired
    if (numerator && transtat) { // No transition statistics for denumerator!

      const std::vector<Segmentator::IndexProbPair> &transitions
        = segmentator->transition_probs();
      
      for (int i = 0; i < (int)transitions.size(); i++)
      {
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
    total_log_likelihood +=
      (numerator?1:-1)*segmentator->get_total_log_likelihood();
}


int
main(int argc, char *argv[])
{
  Segmentator *segmentator;
  int hmmnet_mode = 0;
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
      ('o', "out=BASENAME", "arg must", "", "base filename for output statistics")
      ('R', "raw-input", "", "", "raw audio input")
      ('t', "transitions", "", "", "collect also state transition statistics")
      ('F', "fw-beam=FLOAT", "arg", "0", "Forward beam (for HMM networks)")
      ('W', "bw-beam=FLOAT", "arg", "0", "Backward beam (for HMM networks)")
      ('A', "ac-scale=FLOAT", "arg", "1", "Acoustic scaling (for HMM networks)")
      ('E', "extvit", "", "", "Use extended Viterbi over HMM networks")
      ('V', "vit", "", "", "Use Viterbi over HMM networks")
      ('\0', "ml", "", "", "Collect statistics for ML")
      ('\0', "mmi", "", "", "Collect statistics for MMI")
      ('\0', "mpe", "", "", "Collect statistics for MPE")
      ('\0', "mllt", "", "", "maximum likelihood linear transformation (for ML)")
      ('\0', "mpemode=MODE", "arg", "", "mono/mono-state/state/cp/cp-state/hyp-cp-state")
      ('S', "speakers=FILE", "arg", "", "speaker configuration file")
      ('B', "batch=INT", "arg", "0", "number of batch processes with the same recipe")
      ('I', "bindex=INT", "arg", "0", "batch process index")
      ('i', "info=INT", "arg", "0", "info level")
      ;
    config.default_parse(argc, argv);
    
    info = config["info"].get_int();
    fea_gen.load_configuration(io::Stream(config["config"].get_str()));

    // Initialize the model for accumulating statistics
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
    out_file = config["out"].get_str();

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
    if (config["speakers"].specified)
    {
      speaker_config.read_speaker_file(
        io::Stream(config["speakers"].get_str()));
    }

    // Read recipe file
    recipe.read(io::Stream(config["recipe"].get_str()),
                config["batch"].get_int(), config["bindex"].get_int(),
                true);

    PDF::StatisticsMode mode = 0;
    
    if (config["ml"].specified)
    {
      ml = true;
      mode |= PDF_ML_STATS;
    }
    if (config["mmi"].specified)
    {
      mmi = true;
      mode |= PDF_MMI_STATS;
    }
    if (config["mpe"].specified)
    {
      mpe = true;
      mode |= PDF_MPE_STATS;
      mpe_evaluator.set_model(&model);
    }

    if (mode == 0)
      throw std::string("At least one mode (--ml, --mmi, --mpe) must be given!");

    if (mode != PDF_ML_STATS && !config["hmmnet"].specified)
      throw std::string("Discriminative training requires --hmmnet");

    if (config["vit"].specified)
      hmmnet_mode = 1;
    if (config["extvit"].specified)
      hmmnet_mode = 2;
    if (hmmnet_mode > 0 && !config["hmmnet"].specified)
      throw std::string("--vit and --extvit require --hmmnet");

    if (config["mpemode"].specified)
    {
      if (!(mode&PDF_MPE_STATS))
        fprintf(stderr, "--mpemode ignored without --mpe\n");
      else
      {
        std::string mpe_mode = config["mpemode"].get_str();
        if (mpe_mode == "mono")
          mpe_evaluator.set_mode(MPEEvaluator::MPEM_MONOPHONE_LABEL);
        else if (mpe_mode == "mono-state")
          mpe_evaluator.set_mode(MPEEvaluator::MPEM_MONOPHONE_STATE);
        else if (mpe_mode == "state")
          mpe_evaluator.set_mode(MPEEvaluator::MPEM_STATE);
        else if (mpe_mode == "cp")
          mpe_evaluator.set_mode(MPEEvaluator::MPEM_CONTEXT_LABEL);
        else if (mpe_mode == "cp-state")
          mpe_evaluator.set_mode(MPEEvaluator::MPEM_CONTEXT_PHONE_STATE);
        else if (mpe_mode == "hyp-cp-state")
          mpe_evaluator.set_mode(MPEEvaluator::MPEM_HYP_CONTEXT_PHONE_STATE);
        else
          throw std::string("Invalid MPE mode ") + mpe_mode;
      }
    }
    
    if (config["mllt"].specified)
    {
      if (mmi || mpe)
        throw std::string("--mllt is only supported with --ml");
      mode |= PDF_ML_FULL_STATS;
    }
    model.start_accumulating(mode);

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

      if (config["hmmnet"].specified)
      {
        // Open files and configure
        HmmNetBaumWelch* lattice = recipe.infos[f].init_hmmnet_files(
          &model, false, &fea_gen, config["raw-input"].specified, NULL);
        lattice->set_collect_transition_probs(transtat);
        if (!initialize_hmmnet(lattice, config["bw-beam"].get_float(),
                               config["fw-beam"].get_float(),
                               config["ac-scale"].get_float(), hmmnet_mode,
                               recipe.infos[f].audio_path))
          skip = true;
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
        // Train the numerator
        train(&model, segmentator, true);

        if (mmi || mpe)
        {
          // Clean up
          delete segmentator;
          fea_gen.close();
          
          // Open files and configure
          HmmNetBaumWelch* lattice = recipe.infos[f].init_hmmnet_files(
            &model, true, &fea_gen, config["raw-input"].specified, NULL);
          lattice->set_collect_transition_probs(transtat);
          if (mpe)
            lattice->set_custom_data_callback(&mpe_evaluator);

          if (!initialize_hmmnet(lattice, config["bw-beam"].get_float(),
                                 config["fw-beam"].get_float(),
                                 config["ac-scale"].get_float(), hmmnet_mode,
                                 recipe.infos[f].audio_path))
            skip = true;
          segmentator = lattice;

          if (mpe)
          {
            fprintf(stderr, "Total custom score %f\n", lattice->get_total_custom_score());
            total_mpe_score += lattice->get_total_custom_score();
          }

          if (!skip)
          {
            // Train the denumerator
            train(&model, segmentator, false);
          }
        }
      }
	
      // Clean up
      delete segmentator;
      fea_gen.close();
    }
    
    if (info > 0)
    {
      fprintf(stderr, "Finished collecting statistics (%i/%i)\n",
	      config["bindex"].get_int(), config["batch"].get_int());
      fprintf(stderr, "Total log likelihood: %f\n", total_log_likelihood);
    }
    
    // Write statistics to file dump and clean up
    model.dump_statistics(out_file);
    model.stop_accumulating();

    std::string lls_file_name = out_file+".lls";
    std::ofstream lls_file(lls_file_name.c_str());
    if (lls_file)
    {
//       if (accum_pos == 1) // Denominator
//         total_log_likelihood = -total_log_likelihood;
      lls_file << total_log_likelihood << std::endl;
      if (mpe)
        lls_file << total_mpe_score << std::endl;
      lls_file.close();
    }
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
