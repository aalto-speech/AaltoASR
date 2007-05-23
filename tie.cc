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


int max_contexts;
int info;

conf::Config config;
Recipe recipe;
FeatureGenerator fea_gen;
SpeakerConfig speaker_config(fea_gen);



void
collect_phone_stats(PhnReader *phn_reader, PhonePool *pool)
{
  int f;
  PhnReader::Phn phn;

  while (!phn_reader->next_phn_line(phn))
  {
    if (phn.state == -1)
      throw std::string("Context phone tying requires phn files with state numbers!");
    PhonePool::ContextPhone &phone = pool->get_context_phone(
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
save_basebind(const std::string &filename, PhonePool *pool)
{
  FILE *fp;

  // Save silence models
  if ((fp = fopen(filename.c_str(), "w")) == NULL)
  {
    fprintf(stderr, "Could not open file %s for writing.\n", filename.c_str());
    exit(1);
  }
  fprintf(fp, "_ 1 0\n__ 3 1 2 3\n");
  
  pool->save_to_basebind(fp, 4, max_contexts);
  fclose(fp);
}


int
main(int argc, char *argv[])
{
  PhonePool phone_pool;
  PhnReader phn_reader(NULL);
  
  try {
    config("usage: tie [OPTION...]\n")
      ('h', "help", "", "", "display help")
      ('c', "config=FILE", "arg must", "", "feature configuration")
      ('r', "recipe=FILE", "arg must", "", "recipe file")
      ('O', "ophn", "", "", "use output phns for training")
      ('u', "rule=FILE", "arg must", "", "rule set for triphone state tying")
      ('o', "out=FILE", "arg", "", "output filename for basebind")
      ('R', "raw-input", "", "", "raw audio input")
      ('\0', "count=INT", "arg", "100", "minimum feature count for state clusters")
      ('\0', "lh=FLOAT", "arg", "0", "minimum likelihood gain for cluster splitting")
      ('\0', "context=INT", "arg", "1", "maximum number of contexts (default 1=triphones)")
      ('S', "speakers=FILE", "arg", "", "speaker configuration file")
      ('i', "info=INT", "arg", "0", "info level")
      ;
    config.default_parse(argc, argv);
    
    info = config["info"].get_int();
    fea_gen.load_configuration(io::Stream(config["config"].get_str()));

    // Read recipe file
    recipe.read(io::Stream(config["recipe"].get_str()), 0, 0, false);
    
    phone_pool.set_clustering_parameters(config["count"].get_int(),
                                         config["lh"].get_float());

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

       
      recipe.infos[f].init_phn_files(NULL, false, false,
                                     config["ophn"].specified, &fea_gen,
                                     config["raw-input"].specified,
                                     &phn_reader);
      phn_reader.set_collect_transition_probs(false);
      if (!phn_reader.init_utterance_segmentation())
      {
        fprintf(stderr, "Could not initialize the utterance for PhnReader.");
        fprintf(stderr,"Current file was: %s\n",
                recipe.infos[f].audio_path.c_str());
      }
      else
      {
        collect_phone_stats(&phn_reader, &phone_pool);
      }

      fea_gen.close();
      phn_reader.close();
    }
    
    phone_pool.finish_statistics();
    phone_pool.decision_tree_cluster_context_phones(max_contexts);
    
    save_basebind(config["out"].get_str(), &phone_pool);
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
