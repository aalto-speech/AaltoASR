#ifndef RECIPE_HH
#define RECIPE_HH

#include <string>
#include <vector>
#include <stdio.h>
#include "PhnReader.hh"
#include "FeatureGenerator.hh"

/** A class for handling recipe files.
 * Recipe is a list of audio files, corresponding phn-files with
 * starting and ending times.
 *
 * Format of the lines in the recipe file:
 *
 *  audio_file phn_file phn_out_file start_time end_time start_line end_line speaker_id utterance_id
 *
 * end_line is the first line excluded from processing.
 * Empty lines are skipped, and fields can be omitted from the end
 * of the line.
 *
 * Empty fields are initialized to "" and 0.
 */

class Recipe {
public:

  class Info {
  public:
    std::string audio_path;
    std::string phn_path;
    std::string phn_out_path;
    float start_time;
    float end_time;
    int start_line;
    int end_line;
    std::string speaker_id;
    std::string utterance_id;

    /** Opens the relevant files for \ref PhnReader and \ref FeatureGenerator.
     * \param model The HMM model. If NULL, assumes state number labels.
     * \param relative_sample_nums If true, the sample numbers in the phn
     * file are relative to the start time of the audio file.
     * \param out_phn If true, reads output phns instead of input phns.
     * \param fea_gen \ref Pointer to FeatureGenerator.
     * \param phn_reader The existing \ref PhnReader. If NULL, creates a
     * new instance.
     * \return The \ref PhnReader initialized, either created or given.
     */
    PhnReader* init_phn_files(HmmSet *model, bool relative_sample_nums,
                              bool out_phn, FeatureGenerator *fea_gen,
                              bool raw_audio, PhnReader *phn_reader);

    Info();
  };

  void clear();
  void read(FILE *f);

  std::vector<Info> infos;
};

#endif /* RECIPE_HH */
