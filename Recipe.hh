#ifndef RECIPE_HH
#define RECIPE_HH

#include <string>
#include <vector>
#include <stdio.h>

/* Recipe is a list of audio files, corresponding phn-files with
 * starting and ending times.
 *
 * Format of the lines in the recipe file:
 *
 *  audio_file phn_file phn_out_file start_time end_time start_line end_line speaker_id utterance_id
 *
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

    Info();
  };

  void clear();
  void read(FILE *f);

  std::vector<Info> infos;
};

#endif /* RECIPE_HH */
