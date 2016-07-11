#include "PhoneProbsToolbox.hh"

// Use io.h in Visual Studio varjokal 24.3.2010
#ifdef _MSC_VER
#include <io.h>
#include <errno.h>
#include <stdlib.h>
#else
#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#endif

#include <fcntl.h>

#include "endian.hh"
#include "io.hh"
#include "str.hh"

#define BYTE unsigned char

// O_BINARY is only defined in Windows
#ifndef O_BINARY
#define O_BINARY 0
#endif

namespace aku {

int write_nonblock(int fd, const BYTE *buf, size_t buf_size, bool *cancel) {
        size_t cur_pos = 0;

        while (cur_pos < buf_size) {
                if (cancel && *cancel) {
                        break;
                }
                ssize_t nbytes = write (
                        fd, (buf + cur_pos), (buf_size - cur_pos) *sizeof(BYTE));

                if (nbytes < 0) {
                        if (errno == EAGAIN) {
                                usleep(1e5);
                                continue;
                        }
                        return -1;
                }
                cur_pos += nbytes;
        }
        return 0;
}

void PPToolbox::write_int(int fd, unsigned int i, bool *cancel)
{
  BYTE buf[4];

  buf[0] = (i >> 24) & 0xff;
  buf[1] = (i >> 16) & 0xff;
  buf[2] = (i >> 8) & 0xff;
  buf[3] = i & 0xff;
  if (write_nonblock(fd, buf, 4, cancel)) {
          throw std::string("Write error");
  }
}

void PPToolbox::read_configuration(const std::string &cfgname) {
  m_gen.load_configuration(io::Stream(cfgname));
}

void PPToolbox::set_clustering(const std::string &clfile_name, double eval_minc, double eval_ming) {
  m_model.read_clustering(clfile_name);
  m_model.set_clustering_min_evals(eval_minc, eval_ming);
}

void PPToolbox::generate_to_fd(
        const int in_fd, const int out_fd, bool *cancel) {
  const int lnabytes=2;
  const int start_frame=0;

  BYTE buffer[4];
  assert( sizeof(BYTE) == 1 );

  if (m_model.dim() != m_gen.dim())
    {
      throw str::fmt(256,
                     "Gaussian dimension is %d but feature dimension is %d.",
                     m_model.dim(), m_gen.dim());
    }


  // Open files
  m_gen.open_fd(in_fd);

  // Write header
  write_int(out_fd, m_model.num_states(), cancel);
  BYTE tmp = lnabytes;
  write_nonblock(out_fd, &tmp, 1, cancel);

  // Write the probabilities
  for (int f = start_frame; !(cancel && *cancel) ; f++)
    {
      const FeatureVec fea_vec = m_gen.generate(f);
      if (m_gen.eof())
        break;

      m_model.reset_cache();
      m_model.precompute_likelihoods(fea_vec);
      m_obs_log_probs.resize(m_model.num_states());
      double log_normalizer=0;
      for (int i = 0; i < m_model.num_states(); i++) {
        m_obs_log_probs[i] = m_model.state_likelihood(i, fea_vec);
        log_normalizer += m_obs_log_probs[i];
      }
      if (log_normalizer == 0)
        log_normalizer = 1;
      for (int i = 0; i < (int)m_obs_log_probs.size(); i++)
        m_obs_log_probs[i] = util::safe_log(m_obs_log_probs[i] / log_normalizer);


      for (int i = 0; i < m_model.num_states(); i++)
        {
          if (lnabytes == 4)
            {
              BYTE *p = (BYTE*)&m_obs_log_probs[i];
              for (int j = 0; j < 4; j++)
                buffer[j] = p[j];
              if (endian::big)
                endian::convert(buffer, 4);
            }
          else if (lnabytes == 2)
            {
              if (m_obs_log_probs[i] < -36.008)
                {
                  buffer[0] = 255;
                  buffer[1] = 255;
                }
              else
                {
                  int temp = (int)(-1820.0 * m_obs_log_probs[i] + .5);
                  buffer[0] = (BYTE)((temp>>8)&255);
                  buffer[1] = (BYTE)(temp&255);
                }
            }

          if (write_nonblock(out_fd, buffer,
                             lnabytes*sizeof(BYTE), cancel)) {
                  throw std::string("Write error");
          }

        }
    }
  m_gen.close();
  close(out_fd);
}


void PPToolbox::generate_from_file_to_fd(const std::string &input_name, const int out_fd) {
  const int lnabytes=2;
  const int start_frame=0;

  BYTE buffer[4];
  assert( sizeof(BYTE) == 1 );

  if (m_model.dim() != m_gen.dim())
    {
      throw str::fmt(256,
                     "Gaussian dimension is %d but feature dimension is %d.",
                     m_model.dim(), m_gen.dim());
    }

  // Open files
  m_gen.open(input_name);

  // Write header
  write_int(out_fd, m_model.num_states());
  BYTE tmp = lnabytes;
  write_nonblock(out_fd, &tmp, 1, nullptr);

  // Write the probabilities
  for (int f = start_frame; true ; f++)
    {
      const FeatureVec fea_vec = m_gen.generate(f);
      if (m_gen.eof())
        break;

      m_model.reset_cache();
      m_model.precompute_likelihoods(fea_vec);
      m_obs_log_probs.resize(m_model.num_states());
      double log_normalizer=0;
      for (int i = 0; i < m_model.num_states(); i++) {
        m_obs_log_probs[i] = m_model.state_likelihood(i, fea_vec);
        log_normalizer += m_obs_log_probs[i];
      }
      if (log_normalizer == 0)
        log_normalizer = 1;
      for (int i = 0; i < (int)m_obs_log_probs.size(); i++)
        m_obs_log_probs[i] = util::safe_log(m_obs_log_probs[i] / log_normalizer);


      for (int i = 0; i < m_model.num_states(); i++)
        {
          if (lnabytes == 4)
            {
              BYTE *p = (BYTE*)&m_obs_log_probs[i];
              for (int j = 0; j < 4; j++)
                buffer[j] = p[j];
              if (endian::big)
                endian::convert(buffer, 4);
            }
          else if (lnabytes == 2)
            {
              if (m_obs_log_probs[i] < -36.008)
                {
                  buffer[0] = 255;
                  buffer[1] = 255;
                }
              else
                {
                  int temp = (int)(-1820.0 * m_obs_log_probs[i] + .5);
                  buffer[0] = (BYTE)((temp>>8)&255);
                  buffer[1] = (BYTE)(temp&255);
                }
            }
          if (write_nonblock(out_fd, buffer,
                             lnabytes*sizeof(BYTE), nullptr)) {
                  throw std::string("Write error");
          }
        }
    }
}

void PPToolbox::generate(const std::string &input_name, const std::string &output_name) {
#ifdef _MSC_VER
  int out=_open(output_name.c_str(), _O_WRONLY | _O_CREAT | _O_TRUNC | _O_BINARY);
#else
  int out=open(output_name.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_BINARY, 0664);
#endif
  generate_from_file_to_fd(input_name, out);
  m_gen.close();
  close(out);
};

void PPToolbox::init_align(const struct aligner_params &al_params) {
        m_phn_reader_ptr = std::unique_ptr<aku::PhnReader>(
                new aku::PhnReader(&m_model));
        m_viterbi_ptr = std::unique_ptr<aku::Viterbi>(
                new aku::Viterbi(m_model, m_gen, m_phn_reader_ptr.get()));

        m_viterbi_ptr->set_prob_beam(al_params.beam);
        m_viterbi_ptr->set_state_beam(al_params.sbeam);
        m_viterbi_ptr->resize(al_params.win_size, al_params.win_size, al_params.sbeam/4);
        m_viterbi_ptr->set_print_all_states(al_params.print_all_states);

        // Check the dimension
        if (m_model.dim() != m_gen.dim()) {
                throw str::fmt(128,
                               "gaussian dimension is %d but feature dimension is %d",
                               m_model.dim(), m_gen.dim());
        }

        if (al_params.set_speakers) {
                m_speaker_config_ptr = std::unique_ptr<aku::SpeakerConfig>(
                        new aku::SpeakerConfig(m_gen, &m_model));
                m_speaker_config_ptr->read_speaker_file(
                        io::Stream(al_params.speaker_file_name));
        }
}

void aligner_print_line(FILE *f, float fr,
                int start, int end,
                const std::string &label,
                const std::string &comment) {
        int frame_mult = (int)(16000/fr); // NOTE: phn files assume 16kHz sample rate

        if (start < 0)
                return;

        fprintf(f, "%d %d %s %s\n", start * frame_mult, end * frame_mult,
                label.c_str(), comment.c_str());
}


double aligner_viterbi_align(
        aku::FeatureGenerator &fea_gen, aku::Viterbi &viterbi,
        aku::SpeakerConfig *speaker_config,
        struct aligner_params al_params,
        int start_frame, int end_frame,
        FILE *phn_out, std::string speaker,
        std::string utterance, int info)
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
        viterbi.set_force_end(!al_params.no_force_end);

        if (al_params.set_speakers) {
                speaker_config->set_speaker(speaker);
                if (utterance.size() > 0)
                        speaker_config->set_utterance(utterance);
        }

        // Process the file window by window
        while (1)
        {
                // Compute window borders
                window_end_frame = window_start_frame + al_params.win_size;
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
                target_frame = (int)(al_params.win_size * al_params.overlap);
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
                                aligner_print_line(phn_out, fea_gen.frame_rate(), print_start,
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
        aligner_print_line(phn_out, fea_gen.frame_rate(), print_start,
                           window_start_frame + 1, print_label,
                           print_comment);
        return viterbi.best_path_log_prob();
}

// Wrapper for swig, swig does not support nested classes like Recipe::Info
// We can build the necessary object here instead.
bool PPToolbox::align_file(
        const std::string &audio_path,
        float start_time, float end_time,
        const std::string &alignment_path,
        const struct aligner_params &al_params,
        int info) {

        aku::Recipe::Info rentry;
        rentry.audio_path = audio_path;
        rentry.start_time = start_time;
        rentry.end_time = end_time;
        rentry.alignment_path = alignment_path;

        return align_recipeinfo(rentry, al_params, info);
}

bool PPToolbox::align_recipeinfo(
        const aku::Recipe::Info &rentry,
        const struct aligner_params &al_params,
        int info) {
        double curr_beam = al_params.beam;
        int curr_sbeam = al_params.sbeam;
        bool ok;

        /*std::cerr << "Restoring original beam " << al_params.beam
          << " and original state beam " << al_params.sbeam << std::endl;*/
        m_viterbi_ptr->set_prob_beam(al_params.beam);
        m_viterbi_ptr->set_state_beam(al_params.sbeam);
        m_viterbi_ptr->resize(al_params.win_size, al_params.win_size, al_params.sbeam / 4);

        io::Stream phn_out_file;
        double sum_data_likelihood = 0.0, prec_buff = 0.0;
        double ll;
retry:
        ok = true;
        try {
                if (info > 0)
                {
                        fprintf(stderr, "Processing file: %s",
                                rentry.audio_path.c_str());
                        if (rentry.start_time || rentry.end_time)
                                fprintf(stderr," (%.2f-%.2f)",rentry.start_time,
                                        rentry.end_time);
                        fprintf(stderr,"\n");
                }

                // Open the audio and phn files from the given list.
                rentry.init_phn_files(NULL, false, false, false, &m_gen,
                                      m_phn_reader_ptr.get());

                phn_out_file.open(rentry.alignment_path.c_str(), "w");

                ll = aligner_viterbi_align(
                        m_gen, *m_viterbi_ptr, m_speaker_config_ptr.get(), al_params,
                        m_phn_reader_ptr->first_frame(),
                        (int)(rentry.end_time*m_gen.frame_rate()),
                        phn_out_file, rentry.speaker_id,
                        rentry.utterance_id, info);

                phn_out_file.close();
                m_gen.close();
                m_phn_reader_ptr->close();

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

        } catch (const std::string &err) {
                curr_beam *= 2;
                curr_sbeam *= 2;
                std::cerr << "Too low beams for " << rentry.audio_path
                          << "( " << rentry.start_time << " - " << rentry.end_time
                          << " ), doubling to beam " << curr_beam
                          << " and state beam " << curr_sbeam << std::endl;
                m_viterbi_ptr->set_prob_beam(curr_beam);
                m_viterbi_ptr->set_state_beam(curr_sbeam);
                m_viterbi_ptr->resize(al_params.win_size, al_params.win_size, curr_sbeam / 4);
                ok = false;
                if (curr_beam < 30000)
                        goto retry;
                else
                        std::cerr << "Have to stop trying, beam already 30 000" << std::endl;
        }

        sum_data_likelihood += prec_buff;
        if (info > 0)
        {
                fprintf(stderr, "Total data log likelihood: %f\n", sum_data_likelihood);
        }
        return ok;
}
}
