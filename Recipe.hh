#ifndef RECIPE_HH
#define RECIPE_HH

#include <string>
#include <vector>
#include <stdio.h>
#include "PhnReader.hh"
#include "HmmNetBaumWelch.hh"
#include "FeatureGenerator.hh"


namespace aku {

/** A class for handling recipe files.
 * Recipe is a list of audio files with corresponding files and data.
 *
 * Each line contains key=value pairs, where key may be
 * - audio: path to the audio file
 * - transcript: path to a .phn transcript file
 * - alignment: path to an alignment file
 * - hmmnet:
 * - den-hmmnet:
 * - lna: name for a file where phoneme probabilities will be written in NOWAY
 *  .lna format
 * - start-time: starting time
 * - end-time: ending time
 * - start-line: starting line
 * - end-line: ending line (first line excluded from processing)
 * - speaker: speaker ID
 * - utterance: utterance ID
 *
 * Empty lines are skipped.
 * Fields that are not given are initialized to "" and 0.
 */

class Recipe {
public:

  class Info {
  public:
    std::string audio_path;
    std::string alt_audio_path;
    std::string transcript_path;
    std::string alignment_path;
    std::string hmmnet_path;
    std::string den_hmmnet_path;
    std::string lna_path;
    float start_time;
    float end_time;
    int start_line;
    int end_line;
    std::string speaker_id;
    std::string utterance_id;

    /** Opens the relevant files for \ref PhnReader and \ref FeatureGenerator.
     * \param *model               The HMM model. Required if phn_reader==NULL.
     * \param relative_sample_nums If true, the sample numbers in the phn
     *                             file are relative to the start time of
     *                             the audio file.
     * \param state_num_labels     True if phn file contains state numbers
     *                             instead of phonemes and relative states.
     * \param out_phn              If true, reads alignment phns instead of
     *                             transcript phns.
     * \param fea_gen              Pointer to \ref FeatureGenerator.
     *                             May be NULL.
     * \param phn_reader           The existing \ref PhnReader. If NULL,
     *                             creates a new instance.
     * \return The \ref PhnReader initialized, either created or given.
     */
    PhnReader* init_phn_files(HmmSet *model, bool relative_sample_nums,
                              bool state_num_labels, bool out_phn,
                              FeatureGenerator *fea_gen,
                              PhnReader *phn_reader);
    /** Open the relevant files for \ref HmmNetBaumWelch and
     * \ref FeatureGenerator.
     * \param model           Pointer to the \ref HmmSet. Required if
     *                        hnbw is == NULL.
     * \param den_hmmnet      Use denominator HMM network
     * \param fea_gen         Pointer to \ref FeatureGenerator.
     * \param hnbw            The existing \ref HmmNetBaumWelch. If NULL,
     *                        creates a new instance.
     * \return The \ref HmmNetBaumWelch initialized, either created or given.
     */
    HmmNetBaumWelch* init_hmmnet_files(HmmSet *model, bool den_hmmnet,
                                       FeatureGenerator *fea_gen,
                                       HmmNetBaumWelch *hnbw);

    Info();
    bool operator<( const Info &i ) const {
      return (speaker_id < i.speaker_id);
    }

  };

  void clear();

  /** Creates Info structures from the recipe file lines and adds them to infos.
   *
   * The task will be divided to \a num_batches parts that can be processed
   * concurrently by supplying a different \a batch_index for read() in each
   * process.
   *
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
  void sort_infos() {
    std::stable_sort(infos.begin(), infos.end());
  }
};

}

#endif /* RECIPE_HH */
