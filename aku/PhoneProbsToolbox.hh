#ifndef PHONEPROBSTOOLBOX_HH
#define PHONEPROBSTOOLBOX_HH

#include <string>
#include <cstring>
#include <memory>
#include "conf.hh"
#include "FeatureGenerator.hh"
#include "HmmSet.hh"
#include "Recipe.hh"
#include "Viterbi.hh"
#include "SpeakerConfig.hh"

struct aligner_params {
        int win_size;
        float overlap;
        bool no_force_end, set_speakers, print_all_states;

        double beam, sbeam;
        std::string speaker_file_name;
};

namespace aku {

class PPToolbox {
public:
        PPToolbox() {}
        ~PPToolbox() {}
        void read_configuration(const std::string &cfgname);
        void set_clustering(const std::string &clfile_name, double eval_minc, double eval_ming);
        void generate_to_fd(const int in, const int out, bool *cancel=nullptr);
        void generate_from_file_to_fd(const std::string &input_name, const int out);
        void generate(const std::string &input_name, const std::string &output_name);
        //set_lnabytes(int x);

        inline void read_models(const std::string &m) {m_model.read_all(m);}
        inline void read_gk_model(const std::string &m) {m_model.read_gk(m);}
        inline void read_mc_model(const std::string &m) {m_model.read_mc(m);}
        inline void read_ph_model(const std::string &m) {m_model.read_ph(m);}

        void init_align(const struct aligner_params &al_params);
        bool align_recipeinfo(const aku::Recipe::Info &rentry,
                              const struct aligner_params &al_params,
                              int info);
        // Wrapper over align_recipeinfo for swig
        bool align_file(const std::string &audio_path,
                        float start_time, float end_time,
                        const std::string &alignment_path,
                        const struct aligner_params &al_params,
                        int info);

private:
        conf::Config m_config;
        aku::FeatureGenerator m_gen;
        aku::HmmSet m_model;
        std::vector<float> m_obs_log_probs;
        std::unique_ptr<aku::PhnReader> m_phn_reader_ptr;
        std::unique_ptr<aku::Viterbi> m_viterbi_ptr;
        std::unique_ptr<aku::SpeakerConfig> m_speaker_config_ptr;

        void write_int(int fd, unsigned int i, bool *cancel=nullptr);
};

}

#endif
