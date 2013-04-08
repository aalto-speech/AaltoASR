#include "HmmSet.hh"
#include "conf.hh"
#include "io.hh"
#include "str.hh"
#include "FeatureGenerator.hh"
#include "FeatureModules.hh"
#include "Recipe.hh"
#include "SpeakerConfig.hh"
#include "MllrTrainer.hh"
#include "RegClassTree.hh"
#include <iostream>
#include <fstream>

using namespace aku;

int info;

conf::Config config;
Recipe recipe;
HmmSet model;
FeatureGenerator fea_gen;
RegClassTree rtree;

SpeakerConfig speaker_conf(fea_gen, &model);
double total_log_likelihood = 0;

bool global_transform = false;
ConstrainedMllr* cmllr = NULL;
LinTransformModule* ltm = NULL;

MllrTrainer *cmllr_trainer = NULL;
std::string cur_speaker;

std::set<std::string> updated_speakers;

int segmentation_mode;


void calculate_transform() {
  //make transformation matrix
  if(info > 0) {
    std::cout << cur_speaker << ": ";
    std::cerr << "Calculating transform for " << cur_speaker << std::endl;
  }
  if(global_transform)
    cmllr_trainer->calculate_transform(ltm);
  else
    cmllr_trainer->calculate_transform(cmllr, config["minframes"].get_double(), info);

  delete cmllr_trainer;
  cmllr_trainer = NULL;
}

void
set_speaker(std::string new_speaker)
{
  if (new_speaker == cur_speaker) return;

  if (cmllr_trainer != NULL) {
    calculate_transform();
  }

  cur_speaker = new_speaker;

  updated_speakers.insert(cur_speaker);

  cmllr_trainer = new MllrTrainer(&rtree, &model);

  // Change speaker to FeatureGenerator
  speaker_conf.set_speaker(cur_speaker);
}

Segmentator*
get_segmentator(Recipe::Info info)
{
  Segmentator *segmentator = NULL;
  bool skip = false;
  if (config["hmmnet"].specified) {
    // Open files and configure
    HmmNetBaumWelch* lattice = info.init_hmmnet_files(&model, false, &fea_gen,
        NULL);
    lattice->set_pruning_thresholds(config["fw-beam"].get_float(),
                                    config["bw-beam"].get_float());

    lattice->set_mode(segmentation_mode);

    double orig_beam = lattice->get_backward_beam();
    int counter = 1;
    while (!lattice->init_utterance_segmentation()) {
      if (counter >= 5) {
        fprintf(stderr, "Could not run Baum-Welch for file %s\n",
            info.audio_path.c_str());
        fprintf(stderr,
            "The HMM network may be incorrect or initial beam too low.\n");
        skip = true;
        break;
      }
      fprintf(stderr,
          "Warning: Backward phase failed, increasing beam to %.1f\n",
          ++counter * orig_beam);
      lattice->set_pruning_thresholds(0, counter * orig_beam);
    }
    segmentator = lattice;
  }
  else {
    // Create phn_reader
    PhnReader *phn_reader = info.init_phn_files(&model,
        config["rsamp"].specified, config["snl"].specified,
        config["ophn"].specified, &fea_gen, NULL);

    if (!phn_reader->init_utterance_segmentation()) {
      fprintf(stderr, "Could not initialize the utterance for PhnReader.");
      fprintf(stderr, "Current file was: %s\n", info.audio_path.c_str());
      skip = true;
    }
    segmentator = phn_reader;
  }
  if (skip) {
    delete segmentator;
    segmentator = NULL;
  }
  return segmentator;
}


void
train_mllr(Segmentator *seg)
{
  HmmState state;

  while (seg->next_frame()) {
    const Segmentator::IndexProbMap &pdfs = seg->pdf_probs();
    FeatureVec fea_vec = fea_gen.generate(seg->current_frame());
    if (fea_gen.eof()) break; // EOF in FeatureGenerator

    for (Segmentator::IndexProbMap::const_iterator it = pdfs.begin();
         it != pdfs.end(); ++it)
    {
      state = model.state((*it).first);
      cmllr_trainer->collect_data((*it).second, &state, fea_vec);
    }
  }
  if (seg->computes_total_log_likelihood()) total_log_likelihood
      += seg->get_total_log_likelihood();
}

int
main(int argc, char *argv[])
{
  Segmentator *segmentator;

  try {
    config("usage: modelmllr [OPTION...]\n")
      ('h', "help", "", "", "display help")
      ('b', "base=BASENAME", "arg", "", "base filename for model files")
      ('g', "gk=FILE", "arg", "", "Mixture base distributions")
      ('m', "mc=FILE", "arg", "", "Mixture coefficients for the states")
      ('p', "ph=FILE", "arg", "", "HMM definitions")
      ('c', "config=FILE", "arg must", "", "feature configuration")
      ('r', "recipe=FILE", "arg must", "", "recipe file")
      ('O', "ophn", "", "", "use output phns for adaptation")
      ('H', "hmmnet", "", "", "use HMM networks for training")
      ('\0', "segmode=MODE", "arg", "bw", "Segmentation mode: bw(default)/vit/mpv")
      ('M', "mllr=MODULE", "arg", "", "MLLR feature module name, if none given, a model transform is trained. Only for a model transform the regression tree options are used.")
      ('S', "speakers=FILE", "arg must", "", "speaker configuration input file")
      ('R', "regtree=FILE", "arg", "", "regression tree file, if ommitted, and the next tree options are given, a tree is generated. Otherwise no tree is used.")
      ('s', "mcs=FILE", "arg", "", "Mixture statistics file (necessary for generating a tree, if no tree file is given)")
      ('t', "terminalnodes=INT", "arg", "1", "Number of maximum terminal nodes (used for generating a tree, if no tree file is given)")
      ('u', "unit=STRING", "arg", "PHONE", "PHONE|MIX|GAUSSIAN type of units. Don't use MIX in case of shared gaussians between mixtures (used for generating a tree, if no tree file is given)")
      ('f', "minframes=DOUBLE", "arg", "1000", "minimum frames used for adaptation")
      ('o', "out=FILE", "arg", "", "output speaker configuration file")
      ('F', "fw-beam=FLOAT", "arg", "0", "Forward beam (for HMM networks)")
      ('W', "bw-beam=FLOAT", "arg", "0", "Backward beam (for HMM networks)")
      ('\0', "snl", "", "", "phn-files with state number labels")
      ('\0', "rsamp", "", "", "phn sample numbers are relative to start time")
      ('\0', "ords","", "", "OBSOLETE, does not have any function anymore")
      ('B', "batch=INT", "arg", "0", "number of batch processes with the same recipe")
      ('I', "bindex=INT", "arg", "0", "batch process index")
      ('i', "info=INT", "arg", "0", "info level")
      ;
    config.default_parse(argc, argv);

    if(config["ords"].specified) std::cerr << "Warning: --ords is obsolete and does not have to be used anymore" << std::endl;

    info = config["info"].get_int();
    fea_gen.load_configuration(io::Stream(config["config"].get_str()));

    if (config["base"].specified) {
      model.read_all(config["base"].get_str());
    }
    else if (config["gk"].specified && config["mc"].specified
        && config["ph"].specified) {
      model.read_gk(config["gk"].get_str());
      model.read_mc(config["mc"].get_str());
      model.read_ph(config["ph"].get_str());
    }
    else {
      throw std::string("Must give either --base or all --gk, --mc and --ph");
    }

    if (config["segmode"].specified && !config["hmmnet"].specified)
      throw std::string("Segmentation modes are supported only with --hmmnet");
    conf::Choice segmode_choice;
    segmode_choice("bw", HmmNetBaumWelch::MODE_BAUM_WELCH)
      ("vit", HmmNetBaumWelch::MODE_VITERBI)
      ("mpv", HmmNetBaumWelch::MODE_MULTIPATH_VITERBI);
    segmentation_mode = HmmNetBaumWelch::MODE_BAUM_WELCH;
    if (!segmode_choice.parse(config["segmode"].get_str(), segmentation_mode))
      throw std::string("Invalid segmentation mode ") +
        config["segmode"].get_str();

    // Read recipe file
    recipe.read(io::Stream(config["recipe"].get_str()),
        config["batch"].get_int(), config["bindex"].get_int(), true);

    recipe.sort_infos();

    // Check the dimension
    if (model.dim() != fea_gen.dim()) {
      throw str::fmt(128,
          "gaussian dimension is %d but feature dimension is %d", model.dim(),
          fea_gen.dim());
    }


    if(config["mllr"].specified) {
      global_transform = true;
    }

    if(config["regtree"].specified && !global_transform) { // Read tree
      std::ifstream in(config["regtree"].get_c_str());
      rtree.read(&in, &model);
    }
    else { // generate tree
      if(config["mcs"].specified && config["terminalnodes"].get_int() > 1 && !global_transform) {
        model.accumulate_mc_from_dump(config["mcs"].get_str());

        rtree.set_unit_mode(RegClassTree::UNIT_NO);
        if(config["unit"].get_str() == "PHONE") rtree.set_unit_mode(RegClassTree::UNIT_PHONE);
        if (config["unit"].get_str() == "MIX") rtree.set_unit_mode(RegClassTree::UNIT_MIX);
        if (config["unit"].get_str() == "GAUSSIAN") rtree.set_unit_mode(RegClassTree::UNIT_GAUSSIAN);
        if(rtree.get_unit_mode() == RegClassTree::UNIT_NO) throw std::string(config["unit"].get_str() + " is not a valid unit identifier");

        rtree.initialize_root_node(&model);
        rtree.build_tree(config["terminalnodes"].get_int());
        if(info > 0) std::cerr << "Created tree with " << config["terminalnodes"].get_int() << " terminal nodes" << std::endl;
      }
      else { // make global transform
        rtree.set_unit_mode(RegClassTree::UNIT_NO);
        rtree.initialize_root_node(&model);
        if(info > 0) std::cerr << "No regression tree used" << std::endl;
      }

    }


    speaker_conf.read_speaker_file(io::Stream(config["speakers"].get_str()));

    if(global_transform) {
      ltm = dynamic_cast<LinTransformModule*> (fea_gen.module(config["mllr"].get_str()));
    }
    else {
      cmllr = dynamic_cast<ConstrainedMllr*> (speaker_conf.get_model_transformer().module("cmllr"));

      cmllr->disable_loading();
      cmllr->set_unit_mode(ConstrainedMllr::UNIT_PHONE);
    }


    // Process each recipe line
    for (int f = 0; f < (int) recipe.infos.size(); f++) {


      if (recipe.infos[f].speaker_id.size() == 0) throw std::string(
          "Speaker ID is missing");

      set_speaker(recipe.infos[f].speaker_id);

      if (info > 0) {
        fprintf(stderr, "Processing file: %s (%d/%d)",
            recipe.infos[f].audio_path.c_str(), f+1, (int) recipe.infos.size());

        if (recipe.infos[f].start_time || recipe.infos[f].end_time) fprintf(
            stderr, " (%.2f-%.2f)", recipe.infos[f].start_time,
            recipe.infos[f].end_time);
        fprintf(stderr, "\n");
      }

      if (recipe.infos[f].utterance_id.size() > 0) speaker_conf.set_utterance(
          recipe.infos[f].utterance_id);

      segmentator = get_segmentator(recipe.infos[f]);

      if (segmentator != NULL) {
        train_mllr(segmentator);
        segmentator->close();
        delete segmentator;
      }

      fea_gen.close();

    }


    if (cmllr_trainer != NULL) {
      //make transformation matrix
//      if(global_transform)
//        cmllr_trainer->calculate_transform(ltm);
//      else
//        cmllr_trainer->calculate_transform(cmllr, config["minframes"].get_double());
//
//      delete cmllr_trainer;
//      cmllr_trainer = NULL;
      calculate_transform();
    }


    // Write new speaker configuration
    if (config["out"].specified) {
      std::set<std::string> *speaker_set = NULL, *utterance_set = NULL;
      std::set<std::string> empty_ut;

      if (config["batch"].get_int() > 1) {
        if (config["bindex"].get_int() == 1) updated_speakers.insert(
            std::string("default"));
        speaker_set = &updated_speakers;
        utterance_set = &empty_ut;
      }

      speaker_conf.write_speaker_file(io::Stream(config["out"].get_str(), "w"),
          speaker_set, utterance_set);
    }

    if (info > 0 && total_log_likelihood != 0) {
      fprintf(stderr, "Total log likelihood: %f\n", total_log_likelihood);
    }
  }
  catch (HmmSet::UnknownHmm &e) {
    fprintf(stderr, "Unknown HMM in transcription\n");
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
