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
#include "HmmTrainer.hh"

  
std::string out_file;
std::string save_summary_file;

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
  int last_sample;

  // Open sph file
  fea_gen.open(audio_file, raw_flag);

  // Open transcription
  phn_reader.open(phn_file);
  // Note: PHN files always use time multiplied by 16000
  phn_reader.set_sample_limit((int)(start_time * 16000), 
			      (int)(end_time * 16000));
  phn_reader.set_line_limit(first_phn_line, last_phn_line, 
			    &first_sample, &last_sample);

  if (first_sample > 16000 * start_time) {
    start_time = (float)(first_sample / 16000);
    start_frame = (int)(fea_gen.frame_rate() * start_time);
  }

  if (last_sample != 0 && last_sample < 16000 * end_time) {
    end_time = (float)(last_sample / 16000);
    end_frame = (int)(fea_gen.frame_rate() * end_time);
  }
}


void write_models()
{
  if (out_file.size() > 0)
  {
    model.write_gk(out_file + ".gk");
    model.write_mc(out_file + ".mc");
    model.write_ph(out_file + ".ph");
    fea_gen.write_configuration(io::Stream(out_file + ".cfg","w"));
  }
}


int
main(int argc, char *argv[])
{
  double sum_data_likelihood = 0.0, prec_buff = 0.0;
  io::Stream phn_out_file;
  HmmTrainer trainer;
  bool print_segment;

  std::string *speaker = NULL;
  /*bool read_speaker = false;
  ifstream waf; // warping factors
  */

  try {
    config("usage: phone_probs [OPTION...]\n")
      ('h', "help", "", "", "display help")
      ('b', "base=BASENAME", "arg", "", "base filename for model files")
      ('g', "gk=FILE", "arg", "", "Gaussian kernels")
      ('m', "mc=FILE", "arg", "", "kernel indices for states")
      ('p', "ph=FILE", "arg", "", "HMM definitions")
      ('c', "config=FILE", "arg must", "", "feature configuration")
      ('\0', "recipe=FILE", "arg must", "", "recipe file")
      ('o', "out=BASENAME", "arg", "", "base filename for output models")
      ('r', "raw-input", "", "", "raw audio input")
      ('\0', "swins=INT", "arg", "1000", "window size (default: 1000)")
      ('\0', "beam=FLOAT", "arg", "100.0", "log prob beam (default 100.0")
      ('\0', "sbeam=INT", "arg", "100", "state beam (default 100)")
      ('\0', "overlap=FLOAT", "arg", "0.4", "window overlap (default 0.4)")
      ('\0', "cov", "", "", "update covariance")
      ('\0', "minvar=FLOAT", "arg", "0.1", "minimum variance value (default 0.1)")
      ('\0', "mllt=MODULE", "arg", "", "run MLLT estimation for given module")
      ('\0', "hlda=MODULE", "arg", "", "run HLDA estimation for given module")
      ('\0', "durstat", "", "", "don't train, just collect duration statistics")
      ('\0', "no_force_end", "", "", "do not force to the last state")
      ('\0', "segment", "", "", "print segmentation")
      ('\0', "stateseg", "", "", "print all states to segmentation file")
      ('s', "savesum=FILE", "arg", "", "save summary information (loglikelihood etc.)")
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
    out_file = config["out"].get_str();

    if (config["savesum"].specified)
      save_summary_file = config["savesum"].get_str();

    // Read recipe file
    recipe.read(io::Stream(config["recipe"].get_str()));

    // MISSING: Adaptation handling

    win_size = config["swins"].get_int();
    Viterbi viterbi(model, fea_gen, &phn_reader);
    viterbi.set_prob_beam(config["beam"].get_float());
    viterbi.set_state_beam(config["sbeam"].get_int());
    viterbi.resize(win_size, win_size, config["sbeam"].get_int() / 4);
    viterbi.set_print_all_states(config["stateseg"].specified);

    durstat = config["durstat"].specified;
    if (durstat)
    {
      fprintf(stderr, "WARNING: You have defined -durstat option, "
              "no training will be performed\n");
    }
    trainer.set_duration_statistics(durstat);
    trainer.set_info(info);

    if (config["mllt"].specified)
    {
      TransformationModule *mllt_module = dynamic_cast< TransformationModule* >
        (fea_gen.module(config["mllt"].get_str()));
      if (mllt_module == NULL)
        throw std::string("Module ") + config["mllt"].get_str() +
          std::string(" is not a transform module");
      trainer.set_mllt(true);
      trainer.set_transform_module(mllt_module);
    }
    if (config["hlda"].specified)
    {
      if (config["mllt"].specified)
        throw std::string("Both --mllt and --hlda can not be defined at the same time");
      TransformationModule *hlda_module = dynamic_cast< TransformationModule* >
        (fea_gen.module(config["hlda"].get_str()));
      if (hlda_module == NULL)
        throw std::string("Module ") + config["hlda"].get_str() +
          std::string(" is not a transform module");
      trainer.set_hlda(true);
      trainer.set_transform_module(hlda_module);
    }
    
    trainer.set_win_size(win_size);
    trainer.set_overlap(1-config["overlap"].get_float());
    trainer.set_cov_update(config["cov"].specified);
    trainer.set_no_force_end(config["no_force_end"].specified);
    print_segment = config["segment"].specified;
    trainer.set_print_segment(print_segment);
    trainer.set_min_var(config["minvar"].get_float());

    // Adaptation stuff
    //trainer.set_print_speakered(speaker_phns);
    //trainer.set_ordered_speakers(ordered_s);

    // Check the dimension
    if (model.dim() != fea_gen.dim()) {
      throw str::fmt(128,
                     "gaussian dimension is %d but feature dimension is %d",
                     model.dim(), fea_gen.dim());
    }
    trainer.init(model, fea_gen, NULL); // FIXME: ada_file missing

    for (int f = 0; f < (int)recipe.infos.size(); f++)
    {
      if (info > 1)
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
      if (print_segment)
      {
        phn_out_file.open(recipe.infos[f].phn_out_path.c_str(), "w");
      }

      /*
      // speaker ID is needed in the following occasions:

      if(ada_control) read_speaker = true;
      if(waf_file != NULL) read_speaker = true;

      if (read_speaker){

        // Read speaker ID from recipe (assumes: speakers in separate dirs)

        char *file = new char[recipe.infos[f].audio_path.size() + 1];

        strcpy(file, recipe.infos[f].audio_path.c_str());

        strcpy(strrchr(file, '/'), "\0");
      
        speaker = new std::string;
        speaker->assign(strrchr(file, '/')+1);

        delete file;
      }

      if (waf_file != NULL){

        std::string token;
        bool speaker_change;
        static std::string prev;

        // if first round, open warping factor file
        if(f == 0) waf.open(waf_file, std::ios::in);

        // if first round, set direct / interpolation
        if(f == 0) fea_gen.set_vtln_direct(direct);

        // if first round, set turn point frequency
        if(f == 0 && turn_point >= 0) fea_gen.set_turn_point(turn_point);

        // check if speaker changed

        if(*speaker != prev) speaker_change = true;
      
        // read warping factor from file

        if(speaker_change){

          puts("speaker change"); prev = *speaker;

          assert(waf.peek() != EOF); // waf-file should match the recipe
      
          if(f == 0){ getline(waf, token); // check the warping method
          if(vtln_lin) assert(strcmp(token.c_str(), "pwlinear") == 0);
          if(vtln_bln) assert(strcmp(token.c_str(), "bilinear") == 0);
          if(vtln_mel) assert(strcmp(token.c_str(), "melscale") == 0);
          }

          getline(waf, token);  // warping factor
          printf("Warping factor read from file : %.2f\n",  atof(token.c_str()));

          // set vocal tract length normalization

          if(vtln_lin) fea_gen.create_lin_bins(atof(token.c_str()));
          if(vtln_bln) fea_gen.create_bln_bins(atof(token.c_str()));
          if(vtln_mel) fea_gen.create_mel_bins(atof(token.c_str()));
      
          // note: warpings cannot be performed before opening sph-file! this
          // is because create_mel_bins() takes sample rate from spherereader

          speaker_change = false;
        }

        // if last round, close warping factor file
        if(f == recipe.infos.size() - 1) waf.close();
      }

      // send speaker ID to trainer only if train2 is controlling adaptation

      if(!ada_control) speaker = NULL;
      */
    
      trainer.viterbi_train(start_frame, end_frame, model,
                            fea_gen, viterbi,
                            (print_segment?phn_out_file.file:NULL),
                            speaker);

      /*if(speaker != NULL)
        delete speaker;*/

      if (print_segment)
        phn_out_file.close();
      fea_gen.close();
      phn_reader.close();

      if (info > 1)
      {
        fprintf(stderr, "File log likelihood: %f\n",
                trainer.get_log_likelihood());
      }
      // Buffered sum for better resolution
      prec_buff += (double)trainer.get_log_likelihood();
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
      if (trainer.num_unused_features() > 0)
        fprintf(stderr, 
                "Warning: %d data points were without proper probabilities\n", 
                trainer.num_unused_features());
    }

    if (config["savesum"].specified)
    {
      io::Stream sumfile;

      sumfile.open(save_summary_file, "a");
      fprintf(sumfile, "Log likelihood %.4f\n", trainer.get_log_likelihood());
      if (trainer.num_unused_features() > 0)
      {
        fprintf(sumfile,
                "Warning: %i data points were without proper probabilities\n",
                trainer.num_unused_features());
      }
      sumfile.close();
    }

    trainer.finish_train(model, fea_gen, NULL); // FIXME: ada_file missing

    if (!durstat)
    {
      if (info > 0)
        fprintf(stderr, "Train finished, writing models\n");

      write_models();
    }
  }
  catch (HmmSet::UnknownHmm &e) {
    fprintf(stderr, 
	    "Unknown HMM in transcription, "
	    "writing incompletely taught models\n");
    write_models();
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
