#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "str.hh"
#include "io.hh"
#include "conf.hh"
#include "Recipe.hh"
#include "FeatureGenerator.hh"
#include "HmmSet.hh"
#include "SpeakerConfig.hh"
#include "endian.hh"

#define BYTE unsigned char


conf::Config config;
FeatureGenerator gen;
HmmSet model;
SpeakerConfig speaker_conf(gen);


void write_int(FILE *fp, unsigned int i)
{
  BYTE buf[4];

  buf[0] = (i >> 24) & 0xff;
  buf[1] = (i >> 16) & 0xff;
  buf[2] = (i >> 8) & 0xff;
  buf[3] = i & 0xff;
  int ret = fwrite(buf, 4, 1, fp);
  if (ret != 1)
    throw std::string("Write error");
}

int
main(int argc, char *argv[])
{
  bool raw_flag;
  int lnabytes;
  int info;
  std::string out_dir;
  std::string out_file = "";
  int start_frame, end_frame;
  bool no_overwrite;
  io::Stream ofp;
  BYTE buffer[4];

  assert( sizeof(BYTE) == 1 );
  
  try {
    config("usage: phone_probs [OPTION...]\n")
      ('h', "help", "", "", "display help")
      ('b', "base=BASENAME", "arg", "", "base filename for model files")
      ('g', "gk=FILE", "arg", "", "Gaussian kernels")
      ('m', "mc=FILE", "arg", "", "kernel indices for states")
      ('p', "ph=FILE", "arg", "", "HMM definitions")
      ('c', "config=FILE", "arg must", "", "feature configuration")
      ('r', "recipe=FILE", "arg must", "", "recipe file")
      ('o', "output-dir=DIR", "arg", "", "output directory (default: use filenames from recipe)")
      ('R', "raw-input", "", "", "raw audio input")
      ('\0', "lnabytes=INT", "arg", "2", "number of bytes for probabilities, 2 (default) or 4")
      ('n', "no-overwrite", "", "", "prevent overwriting existing files")
      ('S', "speakers=FILE", "arg", "", "speaker configuration file")
      ('i', "info=INT", "arg", "0", "info level")
      ;
    config.default_parse(argc, argv);

    info = config["info"].get_int();
    raw_flag = config["raw-input"].specified;
    gen.load_configuration(io::Stream(config["config"].get_str()));

    lnabytes = config["lnabytes"].get_int();
    if (lnabytes != 2 && lnabytes != 4)
      throw std::string("Invalid number of LNA bytes");

    no_overwrite = config["no-overwrite"].specified;

    if (config["speakers"].specified)
      speaker_conf.read_speaker_file(io::Stream(config["speakers"].get_str()));

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

    if (model.dim() != gen.dim())
    {
      throw str::fmt(256,
                     "Gaussian dimension is %d but feature dimension is %d.",
                     model.dim(), gen.dim());
    }

    if (config["output-dir"].specified)
    {
      out_dir = config["output-dir"].get_str();
      if (out_dir.size() > 0 && out_dir[out_dir.size()-1] != '/')
        out_dir += "/";
    }

    // Read recipe file
    Recipe recipe;
    recipe.read(io::Stream(config["recipe"].get_str()));

    // Handle each file in the recipe
    for (int recipe_index = 0; recipe_index < (int)recipe.infos.size(); 
	 recipe_index++)
    {
      if (info > 0)
      {
        printf("Processing file %d/%d\n", recipe_index+1,
               (int)recipe.infos.size());
        printf("Input: %s\n", recipe.infos[recipe_index].audio_path.c_str());
      }

      if (!config["output-dir"].specified)
      {
        // Default: Use recipe filename for output (normally for phn output..)
        out_file = recipe.infos[recipe_index].phn_out_path;
      }
      if (out_file.size() == 0)
      {
        // Use the audio file name with different directory and extension
        std::string file;
        // Strip the old path (if one exists)
        int pos = recipe.infos[recipe_index].audio_path.rfind("/");
        if (pos >= 0 &&
            pos < (int)recipe.infos[recipe_index].audio_path.size()-1)
          file = recipe.infos[recipe_index].audio_path.substr(pos+1);
        else
          file = recipe.infos[recipe_index].audio_path;
        // Change the extension
        pos = file.rfind(".");
        if (pos > 0 && pos <= (int)file.size()-1)
          file.erase(pos);
        out_file = out_dir + file + ".lna";
      }
      if (info > 0)
        printf("Output: %s\n", out_file.c_str());

      if (no_overwrite)
      {
        // Test file to prevent overwriting
        struct stat buf;
        if (stat(out_file.c_str(), &buf) == 0)
        {
	  fprintf(stderr, "WARNING: skipping existing lna file %s\n",
                  out_file.c_str());
	  continue;
	}
      }

      if (config["speakers"].specified)
        speaker_conf.set_speaker(recipe.infos[recipe_index].speaker_id);
      
      start_frame = (int)(recipe.infos[recipe_index].start_time *
                          gen.frame_rate());
      end_frame = (int)(recipe.infos[recipe_index].end_time *
                        gen.frame_rate());
      if (info > 0 && start_frame != 0 || end_frame != 0)
        printf("Generating frames %d - %d\n", start_frame, end_frame);
      if (end_frame == 0)
        end_frame = INT_MAX;

      // Open files
      gen.open(recipe.infos[recipe_index].audio_path, raw_flag);
      ofp.open(out_file, "w");

      // Write header
      write_int(ofp, model.num_states());
      fputc(lnabytes, ofp);

      // Write the probabilities
      for (int f = start_frame; f < end_frame; f++)
      {
        const FeatureVec fea_vec = gen.generate(f);
        if (gen.eof())
          break;
        model.compute_observation_log_probs(fea_vec);

        for (int i = 0; i < model.num_states(); i++)
        {
          if (lnabytes == 4)
          {
            BYTE *p = (BYTE*)&model.obs_log_probs[i];
            for (int j = 0; j < 4; j++)
              buffer[j] = p[j];
            if (endian::big)
              endian::convert(buffer, 4);
          }
          else if (lnabytes == 2)
          {
            if (model.obs_log_probs[i] < -36.008)
            {
              buffer[0] = 255;
              buffer[1] = 255;
            }
            else
            {
              int temp = (int)(-1820.0 * model.obs_log_probs[i] + .5);
              buffer[0] = (BYTE)((temp>>8)&255);
              buffer[1] = (BYTE)(temp&255);
            }
          }
          if ((int)fwrite(buffer, sizeof(BYTE), lnabytes, ofp) < lnabytes)
            throw std::string("Write error");
        }
      }

      gen.close();
      ofp.close();
    }
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

