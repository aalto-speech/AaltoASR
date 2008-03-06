#include <fstream>
#include <string>
#include <iostream>

#include "io.hh"
#include "str.hh"
#include "conf.hh"
#include "FeatureGenerator.hh"
#include "Recipe.hh"
#include "PhonePool.hh"
#include "PhnReader.hh"
#include "SpeakerConfig.hh"
#include "HmmNetBaumWelch.hh"

int max_contexts;
int info;

conf::Config config;
Recipe recipe;
HmmSet model;
FeatureGenerator fea_gen;
FeatureGenerator model_fea_gen;
FeatureGenerator *mfea_gen; // Set to the correct FeatureGenerator object
SpeakerConfig speaker_config(fea_gen);



void
collect_phone_stats(PhnReader *phn_reader, PhonePool *pool)
{
  int f;
  PhnReader::Phn phn;

  while (phn_reader->next_phn_line(phn))
  {
    if (phn.state == -1)
      throw std::string("Context phone tying requires phn files with state numbers!");
    PhonePool::ContextPhoneContainer phone = pool->get_context_phone(
      phn.label[0], phn.state);
    for (f = phn.start; f < phn.end; f++)
    {
      FeatureVec feature = fea_gen.generate(f);
      if (fea_gen.eof())
        break; // EOF in FeatureGenerator
      phone.add_feature(1, feature);
    }
    if (f < phn.end) // EOF in FeatureGenerator
      return;
  }
}


void
hmmnet_collect_phone_stats(HmmNetBaumWelch *seg, PhonePool *pool)
{
  int i;
  
  while (seg->next_frame())
  {
    std::vector<HmmNetBaumWelch::ArcInfo> arc_info;
    seg->fill_arc_info(arc_info);
    FeatureVec feature = fea_gen.generate(seg->current_frame());
    if (fea_gen.eof())
      break; // EOF in FeatureGenerator

    for (i = 0; i < (int)arc_info.size(); i++)
    {
      int state = -1;
      if (strchr(arc_info[i].label.c_str(), '.') != NULL){
	state = atoi((const char*)(strchr(arc_info[i].label.c_str(),'.')+1));
	arc_info[i].label.erase(arc_info[i].label.find('.', 0), 2);
      }
      if (state < 0)
      {
        fprintf(stderr, "Warning: Invalid label %s, ignoring current file\n",
                arc_info[i].label.c_str());
        return;
      }
      PhonePool::ContextPhoneContainer phone = pool->get_context_phone(
        arc_info[i].label, state);
      phone.add_feature(arc_info[i].prob, feature);
    }
  }
}


void
save_basebind(const std::string &filename, PhonePool *pool)
{
  FILE *fp;

  // Save silence models
  if ((fp = fopen(filename.c_str(), "w")) == NULL)
  {
    fprintf(stderr, "Could not open file %s for writing.\n", filename.c_str());
    exit(1);
  }
  //fprintf(fp, "_ 1 0\n__ 3 1 2 3\n");
  
  pool->save_to_basebind(fp, 0, max_contexts);
  fclose(fp);
}


int
main(int argc, char *argv[])
{
  PhonePool phone_pool;
  
  try {
    config("usage: tie [OPTION...]\n")
      ('h', "help", "", "", "display help")
      ('c', "config=FILE", "arg must", "", "feature configuration")
      ('r', "recipe=FILE", "arg must", "", "recipe file")
      ('O', "ophn", "", "", "use output phns for training")
      ('H', "hmmnet", "", "", "use HMM networks for training")
      ('b', "base=BASENAME", "arg", "", "model files (required with --hmmnet)")
      ('C', "mconfig=FILE", "arg", "", "model configuration (optional)")
      ('u', "rule=FILE", "arg must", "", "rule set for triphone state tying")
      ('o', "out=FILE", "arg", "", "write output to HMM model with base name FILE")
      ('B', "basebind=FILE", "arg", "", "write output to basebind FILE")
      ('\0', "count=INT", "arg", "100", "minimum feature count for state clusters")
      ('\0', "sgain=FLOAT", "arg", "0", "minimum loglikelihood gain in cluster splitting")
      ('\0', "mloss=FLOAT", "arg", "0", "cluster merging with maximum loglikelihood loss")
      ('\0', "context=INT", "arg", "1", "maximum number of contexts (default 1=triphones)")
      ('F', "fw-beam=FLOAT", "arg", "0", "Forward beam (for HMM networks)")
      ('W', "bw-beam=FLOAT", "arg", "0", "Backward beam (for HMM networks)")
      ('A', "ac-scale=FLOAT", "arg", "1", "Acoustic scaling (for HMM networks)")
      ('V', "vit", "", "", "Use Viterbi over HMM networks")
      ('S', "speakers=FILE", "arg", "", "speaker configuration file")
      ('i', "info=INT", "arg", "0", "info level")
      ;
    config.default_parse(argc, argv);
    
    info = config["info"].get_int();
    fea_gen.load_configuration(io::Stream(config["config"].get_str()));

    // Read recipe file
    recipe.read(io::Stream(config["recipe"].get_str()), 0, 0, false);

    if (!(config["out"].specified^config["basebind"].specified))
      throw std::string("Specify either --out or --basebind for output");

    phone_pool.set_clustering_parameters(config["count"].get_int(),
                                         config["sgain"].get_float(),
                                         config["mloss"].get_float());

    phone_pool.load_decision_tree_rules(io::Stream(config["rule"].get_str()));

    max_contexts = config["context"].get_int();
    
    if (config["speakers"].specified)
    {
      speaker_config.read_speaker_file(
        io::Stream(config["speakers"].get_str()));
    }

    // Initialize triphone tying
    phone_pool.set_dimension(fea_gen.dim());
    phone_pool.set_info(info);

    if (config["hmmnet"].specified)
    {
      if (config["base"].specified)
      {
        model.read_all(config["base"].get_str());
        if (config["mconfig"].specified)
        {
          model_fea_gen.load_configuration(
            io::Stream(config["mconfig"].get_str()));
          if (model_fea_gen.frame_rate() != fea_gen.frame_rate())
            throw str::fmt(256, "Frame rate of the model and the feature generator must match!\nModel frame rate is %.3f, feature generator frame rate is %3.f\n", model_fea_gen.frame_rate(), fea_gen.frame_rate());
          mfea_gen = &model_fea_gen;
        }
        else
          mfea_gen = &fea_gen; // Use the feature configuration
      }
      else
      {
        throw std::string("HMM model is required with HMM networks");
      }
      if (model.dim() != mfea_gen->dim()) {
        throw str::fmt(128,
                       "gaussian dimension is %d but feature dimension is %d",
                       model.dim(), mfea_gen->dim());
      }
    }
    
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

      if (config["speakers"].specified)
      {
        speaker_config.set_speaker(recipe.infos[f].speaker_id);
        if (recipe.infos[f].utterance_id.size() > 0)
          speaker_config.set_utterance(recipe.infos[f].utterance_id);
      }

      if (config["hmmnet"].specified)
      {
        // Open files and configure
        HmmNetBaumWelch* lattice = recipe.infos[f].init_hmmnet_files(
          &model, false, mfea_gen, NULL);
        lattice->set_pruning_thresholds(config["bw-beam"].get_float(),
                                        config["fw-beam"].get_float());
        if (config["ac-scale"].specified)
          lattice->set_acoustic_scaling(config["ac-scale"].get_float());
        if (config["vit"].specified)
          lattice->set_mode(HmmNetBaumWelch::MODE_VITERBI);

        if (mfea_gen != &fea_gen)
        {
          // Open the file for the actual feature generator
          fea_gen.open(recipe.infos[f].audio_path);
        }
        
        double orig_beam = lattice->get_backward_beam();
        int counter = 1;
        bool skip = false;
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
        if (!skip)
          hmmnet_collect_phone_stats(lattice, &phone_pool);
        if (mfea_gen != &fea_gen) // fea_gen is closed elsewhere
          mfea_gen->close();
        delete lattice;
      }
      else
      {
        PhnReader phn_reader(NULL);
        recipe.infos[f].init_phn_files(NULL, false, false,
                                       config["ophn"].specified, &fea_gen,
                                       &phn_reader);
        phn_reader.set_collect_transition_probs(false);
        collect_phone_stats(&phn_reader, &phone_pool);
        phn_reader.close();
      }

      fea_gen.close();
    }
    
    phone_pool.finish_statistics();
    phone_pool.decision_tree_cluster_context_phones(max_contexts);
    if (config["mloss"].specified)
      phone_pool.merge_context_phones();

    if (config["out"].specified)
      phone_pool.save_model(config["out"].get_str(), max_contexts);
    else
      save_basebind(config["basebind"].get_str(), &phone_pool);
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
