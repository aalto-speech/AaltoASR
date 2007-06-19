#include <fstream>
#include <string>
#include <string.h>
#include <iostream>

#include "io.hh"
#include "str.hh"
#include "conf.hh"
#include "Viterbi.hh"
#include "HmmSet.hh"
#include "FeatureGenerator.hh"
#include "Recipe.hh"
#include "SpeakerConfig.hh"


int win_size;
int info;
bool raw_flag;
float overlap;
bool no_force_end;
bool set_speakers;

conf::Config config;
Recipe recipe;
HmmSet model;
FeatureGenerator fea_gen;
SpeakerConfig speaker_config(fea_gen);


void
print_line(FILE *f, float fr, 
           int start, int end, 
           const std::string &label,
           const std::string &comment)
{
  int frame_mult = (int)(16000/fr); // NOTE: phn files assume 16kHz sample rate
    
  if (start < 0)
    return;

  fprintf(f, "%d %d %s %s\n", start * frame_mult, end * frame_mult, 
          label.c_str(), comment.c_str());
}


double
viterbi_align(Viterbi &viterbi, int start_frame, int end_frame,
              FILE *phn_out, std::string speaker,
              std::string utterance)
{
  // Compute window borders
  int window_start_frame = start_frame;
  int window_end_frame = 0;
  int target_frame;
  bool last_window = false;
  int print_start = -1;
  std::string print_label;
  std::string print_comment;
 
  viterbi.reset();
  viterbi.set_feature_frame(window_start_frame);
  viterbi.set_force_end(!no_force_end);

  if (set_speakers && speaker.size() > 0)
  {
    speaker_config.set_speaker(speaker);
    if (utterance.size() > 0)
      speaker_config.set_utterance(utterance);
  }

  // Process the file window by window
  while (1)
  {
    // Compute window borders
    window_end_frame = window_start_frame + win_size;
    if (end_frame > 0) {
      if (window_start_frame >= end_frame)
        break;
      if (window_end_frame >= end_frame) {
        window_end_frame = end_frame;
        last_window = true;
      }
    }
    
    // Fill lattice
    int old_current_frame = viterbi.current_frame();
    viterbi.set_last_window(last_window);
    viterbi.set_last_frame(window_end_frame - window_start_frame);
    viterbi.fill();
    if (fea_gen.eof())
    {
      // Viterbi encountered eof and stopped
      last_window = true;
      window_end_frame = window_start_frame + viterbi.last_frame();
    }

    assert( viterbi.feature_frame() == window_end_frame );

    // Print debug info
    if (info > 0 && old_current_frame < viterbi.current_frame()) {
      int start_frame = old_current_frame;
      int end_frame = viterbi.current_frame();
      int start_pos = viterbi.best_position(start_frame);
      int end_pos = viterbi.best_position(end_frame-1);
      float average_log_prob = 
        ((viterbi.at(end_frame-1, end_pos).log_prob - 
          viterbi.at(start_frame, start_pos).log_prob) 
         / (end_frame - start_frame));

      fprintf(stderr, "filled frames %d-%d (%f)\n",
              start_frame + window_start_frame,
              end_frame + window_start_frame,
              average_log_prob);
    }

    // The beginning part of the lattice is used for teaching.
    // Compute the frame dividing the lattice in two parts.  NOTE:
    // if the end of speech is in the window, we use the whole
    // window and do not continue further.
    target_frame = (int)(win_size * overlap);
    if (last_window)
      target_frame = window_end_frame - window_start_frame;
    if (window_start_frame + target_frame > window_end_frame)
      target_frame = window_end_frame - window_start_frame;

    int f = 0;
    for (f = 0; f < target_frame; f++)
    {
      int pos = viterbi.best_position(f);
      const Viterbi::TranscriptionState &state = 
        viterbi.transcription(pos);

      if (!state.printed)
      {
        // Print pending line
        print_line(phn_out, fea_gen.frame_rate(), print_start,
                   f + window_start_frame, print_label,
                   print_comment);
        
        // Prepare the next print
        print_start = f + window_start_frame;
        print_label = state.label;
        print_comment = state.comment;

        // Speaker ID
        state.printed = true;
      }
    }

    // Check if we have done the job; if not, move to next window
    window_start_frame += target_frame;
      
    if (last_window && window_start_frame >= end_frame)
      break;

    int position = viterbi.best_position(target_frame);

    viterbi.move(target_frame, position);
  } // Process the next window

  // FIXME: The end point window_start_frame+1 assumes 50% frame overlap
  print_line(phn_out, fea_gen.frame_rate(), print_start,
             window_start_frame + 1, print_label,
             print_comment);
  return viterbi.best_path_log_prob();
}


int
main(int argc, char *argv[])
{
  double sum_data_likelihood = 0.0, prec_buff = 0.0;
  io::Stream phn_out_file;
  double ll;
  PhnReader phn_reader(&model);

  try {
    config("usage: align [OPTION...]\n")
      ('h', "help", "", "", "display help")
      ('b', "base=BASENAME", "arg", "", "base filename for model files")
      ('g', "gk=FILE", "arg", "", "Gaussian kernels")
      ('m', "mc=FILE", "arg", "", "kernel indices for states")
      ('p', "ph=FILE", "arg", "", "HMM definitions")
      ('c', "config=FILE", "arg must", "", "feature configuration")
      ('r', "recipe=FILE", "arg must", "", "recipe file")
      ('R', "raw-input", "", "", "raw audio input")
      ('\0', "swins=INT", "arg", "1000", "window size (default: 1000)")
      ('\0', "beam=FLOAT", "arg", "100.0", "log prob beam (default 100.0)")
      ('\0', "sbeam=INT", "arg", "100", "state beam (default 100)")
      ('\0', "overlap=FLOAT", "arg", "0.4", "Viterbi window overlap (default 0.4)")
      ('\0', "no-force-end", "", "", "do not force to the last state")
      ('\0', "phoseg", "", "", "print phoneme segmentation instead of states")
      ('S', "speakers=FILE", "arg", "", "speaker configuration file")
      ('B', "batch=INT", "arg", "0", "number of batch processes with the same recipe")
      ('I', "bindex=INT", "arg", "0", "batch process index")
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
    recipe.read(io::Stream(config["recipe"].get_str()),
                config["batch"].get_int(), config["bindex"].get_int(),
                true);

    win_size = config["swins"].get_int();
    Viterbi viterbi(model, fea_gen, &phn_reader);
    viterbi.set_prob_beam(config["beam"].get_float());
    viterbi.set_state_beam(config["sbeam"].get_int());
    viterbi.resize(win_size, win_size, config["sbeam"].get_int() / 4);
    viterbi.set_print_all_states(!config["phoseg"].specified);

    overlap = 1-config["overlap"].get_float();
    no_force_end = config["no-force-end"].specified;

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
      set_speakers = true;
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

      // Open the audio and phn files from the given list.
      recipe.infos[f].init_phn_files(NULL, false, false, false, &fea_gen,
                                     config["raw-input"].specified,
                                     &phn_reader);
    
      phn_out_file.open(recipe.infos[f].alignment_path.c_str(), "w");

      ll = viterbi_align(viterbi, phn_reader.first_frame(),
                         (int)(recipe.infos[f].end_time*fea_gen.frame_rate()),
                         phn_out_file, recipe.infos[f].speaker_id,
                         recipe.infos[f].utterance_id);
      
      phn_out_file.close();
      fea_gen.close();
      phn_reader.close();

      if (info > 1)
      {
        fprintf(stderr, "File log likelihood: %f\n", ll);
      }
      // Buffered sum for better resolution
      prec_buff += ll;
      if (fabsl(prec_buff) > 100000)
      {
        sum_data_likelihood += prec_buff;
        prec_buff=0;
      }
    } // Process the next file

    sum_data_likelihood += prec_buff;
    if (info > 0)
    {
      fprintf(stderr, "Total data log likelihood: %f\n", sum_data_likelihood);
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
