#include <fstream>
#include <string>
#include <iostream>

#include "io.hh"
#include "str.hh"
#include "conf.hh"
#include "Viterbi.hh"
#include "HmmSet.hh"
#include "FeatureGenerator.hh"
#include "Recipe.hh"
#include "HmmTrainer.hh"


int win_size;
int info;
bool raw_flag;
bool durstat;
float start_time, end_time;
int start_frame, end_frame;

conf::Config config;
Recipe recipe;
HmmSet model;
FeatureGenerator fea_gen;
PhnReader phn_reader;




void
open_files(const char *audio_file, const char *phn_file, 
	   int first_phn_line, int last_phn_line)
{
  int first_sample;

  // Open sph file
  fea_gen.open(audio_file, raw_flag);

  // Open transcription
  phn_reader.open(phn_file);
  // Note: PHN files always use time multiplied by 16000
  phn_reader.set_sample_limit((int)(start_time * 16000), 
			      (int)(end_time * 16000));
  phn_reader.set_line_limit(first_phn_line, last_phn_line, 
			    &first_sample);

  if (first_sample > 16000 * start_time) {
    start_time = (float)(first_sample / 16000);
    start_frame = (int)(fea_gen.frame_rate() * start_time);
  }
}


int
main(int argc, char *argv[])
{
  HmmTrainer trainer(fea_gen);
  try {
    config("usage: phone_probs [OPTION...]\n")
      ('h', "help", "", "", "display help")
      ('b', "base=BASENAME", "arg", "", "base filename for model files")
      ('g', "gk=FILE", "arg", "", "Gaussian kernels")
      ('m', "mc=FILE", "arg", "", "kernel indices for states")
      ('p', "ph=FILE", "arg", "", "HMM definitions")
      ('c', "config=FILE", "arg must", "", "feature configuration")
      ('r', "recipe=FILE", "arg must", "", "recipe file")
      ('u', "rule=FILE", "arg must", "", "rule set for triphone state tying")
      ('o', "out=FILE", "arg", "", "output filename for basebind")
      ('R', "raw-input", "", "", "raw audio input")
      ('\0', "swins=INT", "arg", "1000", "window size (default: 1000)")
      ('\0', "beam=FLOAT", "arg", "100.0", "log prob beam (default 100.0")
      ('\0', "sbeam=INT", "arg", "100", "state beam (default 100)")
      ('\0', "overlap=FLOAT", "arg", "0.4", "window overlap (default 0.4)")
      ('\0', "no-force-end", "", "", "do not force to the last state")
      ('\0', "missing", "", "", "also tie missing triphone contexts ")
      ('\0', "count=INT", "arg", "0", "minimum feature count for state clusters")
      ('\0', "lh=FLOAT", "arg", "0", "minimum likelihood gain for cluster splitting")
      ('\0', "laward=FLOAT", "arg", "0", "likelihood award from applying the length rule")
      ('\0', "sc", "", "", "triphone context is used over short silences \'_\'")
      ('\0', "il", "", "", "ignore center phoneme length (case sensitive phonemes")
      ('i', "info=INT", "arg", "0", "info level")
      ;
    config.default_parse(argc, argv);
    
    info = config["info"].get_int();
    raw_flag = config["raw-input"].specified;
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

    // Read recipe file
    recipe.read(io::Stream(config["recipe"].get_str()));

    // MISSING: Adaptation handling

    win_size = config["swins"].get_int();
    Viterbi viterbi(model, fea_gen, &phn_reader);
    viterbi.set_prob_beam(config["beam"].get_float());
    viterbi.set_state_beam(config["sbeam"].get_int());
    viterbi.resize(win_size, win_size, config["sbeam"].get_int() / 4);

    viterbi.set_skip_next_short_silence(config["sc"].specified);
    trainer.set_skip_short_silence_context(config["sc"].specified);

    trainer.set_info(info);
    trainer.set_win_size(win_size);
    trainer.set_overlap(1-config["overlap"].get_float());
    trainer.set_no_force_end(config["no-force-end"].specified);
    trainer.set_triphone_tying(true);

    if (config["count"].specified)
      trainer.set_tying_min_count(config["count"].get_int());
    if (config["lh"].specified)
      trainer.set_tying_min_likelihood_gain(config["lh"].get_float());
    trainer.set_tying_length_award(config["laward"].get_float());
    trainer.set_fill_missing_contexts(config["missing"].specified);
    trainer.set_ignore_length(config["il"].specified);
    trainer.load_rule_set(config["rule"].get_str());

    // Check the dimension
    if (model.dim() != fea_gen.dim()) {
      throw str::fmt(128,
                     "gaussian dimension is %d but feature dimension is %d",
                     model.dim(), fea_gen.dim());
    }
    trainer.init(model, ""); // FIXME: ada_file missing

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
    
      start_time = recipe.infos[f].start_time;
      end_time = recipe.infos[f].end_time;
      start_frame = (int)(start_time * fea_gen.frame_rate());
      end_frame = (int)(end_time * fea_gen.frame_rate());
    
      // Open the audio and phn files from the given list.
      open_files(recipe.infos[f].audio_path.c_str(), 
                 recipe.infos[f].phn_path.c_str(),
                 recipe.infos[f].start_line,
                 recipe.infos[f].end_line);

      trainer.viterbi_train(start_frame, end_frame, model,
                            viterbi, NULL);

      fea_gen.close();
      phn_reader.close();
    }
    
    trainer.finish_train(model);
    trainer.save_tying(config["out"].get_str());
  }
  catch (HmmSet::UnknownHmm &e) {
    fprintf(stderr, 
	    "Unknown HMM in transcription.");
    fputs("Exit.\n", stderr);
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
