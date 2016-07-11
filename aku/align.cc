#include <fstream>
#include <string>
#include <string.h>
#include <iostream>

#include "io.hh"
#include "str.hh"
#include "conf.hh"
#include "PhoneProbsToolbox.hh"

using namespace aku;
int main(int argc, char *argv[])
{
        conf::Config config;
        Recipe recipe;
        int info;

        struct aligner_params al_params;
        PPToolbox pptoolbox;

        std::string cur_align;
        try {
                config("usage: align [OPTION...]\n")
                        ('h', "help", "", "", "display help")
                        ('b', "base=BASENAME", "arg", "", "base filename for model files")
                        ('g', "gk=FILE", "arg", "", "Gaussian kernels")
                        ('m', "mc=FILE", "arg", "", "kernel indices for states")
                        ('p', "ph=FILE", "arg", "", "HMM definitions")
                        ('c', "config=FILE", "arg must", "", "feature configuration")
                        ('r', "recipe=FILE", "arg must", "", "recipe file")
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
                pptoolbox.read_configuration(config["config"].get_str());

                if (config["base"].specified)
                {
                        pptoolbox.read_models(config["base"].get_str());
                }
                else if (config["gk"].specified && config["mc"].specified &&
                         config["ph"].specified)
                {
                        pptoolbox.read_gk_model(config["gk"].get_str());
                        pptoolbox.read_mc_model(config["mc"].get_str());
                        pptoolbox.read_ph_model(config["ph"].get_str());
                }
                else
                {
                        throw std::string("Must give either --base or all --gk, --mc and --ph");
                }

                // Read recipe file
                recipe.read(io::Stream(config["recipe"].get_str()),
                            config["batch"].get_int(), config["bindex"].get_int(),
                            true);

                al_params.win_size = config["swins"].get_int();
                al_params.beam = config["beam"].get_float();
                al_params.sbeam = config["sbeam"].get_int();
                al_params.overlap = 1-config["overlap"].get_float();
                al_params.no_force_end = config["no-force-end"].specified;
                al_params.print_all_states = !config["phoseg"].specified;
                // Load speaker configurations
                if (config["speakers"].specified)
                {
                        al_params.set_speakers = true;
                        al_params.speaker_file_name = config["speakers"].get_str();

                } else {
                        al_params.set_speakers = false;
                }
                pptoolbox.init_align(al_params);

                for (int f = 0; f < (int) recipe.infos.size(); f++)
                {
                        cur_align = recipe.infos[f].audio_path; //For better error info
                        bool ok = pptoolbox.align_recipeinfo(
                                recipe.infos[f], al_params, info);
                        if (!ok) {
                                std::cerr << "Warning: Failed align "
                                          << cur_align << std::endl;
                        }
                }
        }
        catch (HmmSet::UnknownHmm &e) {
                fprintf(stderr, "Unknown HMM in transcription in %s\n", cur_align.c_str());
                abort();
        }
        catch (std::exception &e) {
                fprintf(stderr, "exception in %s: %s\n", cur_align.c_str(), e.what());
                abort();
        }
        catch (std::string &str) {
                fprintf(stderr, "exception in %s: %s\n", cur_align.c_str(), str.c_str());
                abort();
        }
}
