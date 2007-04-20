#ifndef RECIPE_HH
#define RECIPE_HH

#include <string>
#include <vector>
#include <stdio.h>
#include "PhnReader.hh"
#include "HmmNetBaumWelch.hh"
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
    std::string transcript_path;
    std::string alignment_path;
    std::string hmmnet_path;
    std::string lna_path;
    float start_time;
    float end_time;
    int start_line;
    int end_line;
    std::string speaker_id;
    std::string utterance_id;

    /** Opens the relevant files for \ref PhnReader and \ref FeatureGenerator.
     * \param model                The HMM model. If NULL, assumes state
     *                             number labels.
     * \param relative_sample_nums If true, the sample numbers in the phn
     *                             file are relative to the start time of
     *                             the audio file.
     * \param out_phn              If true, reads alignment phns instead of
     *                             transcript phns.
     * \param fea_gen              Pointer to \ref FeatureGenerator.
     * \param raw_audio            true if audio files are of raw format.
     * \param phn_reader           The existing \ref PhnReader. If NULL,
     *                             creates a new instance.
     * \return The \ref PhnReader initialized, either created or given.
     */
    PhnReader* init_phn_files(HmmSet *model, bool relative_sample_nums,
                              bool out_phn, FeatureGenerator *fea_gen,
                              bool raw_audio, PhnReader *phn_reader);
    /** Open the relevant files for \ref HmmNetBaumWelch and
     * \ref FeatureGenerator.
     * \param model           Pointer to the \ref HmmSet. Required if
     *                        hnbw is != NULL.
     * \param fea_gen         Pointer to \ref FeatureGenerator.
     * \param raw_audio       true if audio files are of raw format.
     * \param hnbw            The existing \ref HmmNetBaumWelch. If NULL,
     *                        creates a new instance.
     * \return The \ref HmmNetBaumWelch initialized, either created or given.
     */
    HmmNetBaumWelch* init_hmmnet_files(HmmSet *model,FeatureGenerator *fea_gen,
                                       bool raw_audio, HmmNetBaumWelch *hnbw);

    Info();
  };

  void clear();

  /** Reads a recipe file
   * \param f                A file pointer to a recipe file.
   * \param num_batches      Number of batch processes for concurrent
   *                         execution.
   * \param batch_index      Batch process index. If num_batches > 1, must
   *                         satisfy 1 <= batch_index <= num_batches.
   * \param cluster_speakers If true, the recipe file is split to batches
   *                         so that sequental lines with the same speaker
   *                         remain in one batch.
   */
  void read(FILE *f, int num_batches, int batch_index, bool cluster_speakers);

  std::vector<Info> infos;
};

#endif /* RECIPE_HH */
