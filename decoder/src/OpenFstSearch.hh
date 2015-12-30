#ifndef OPENFSTSEARCH_HH
#define OPENFSTSEARCH_HH

/*
  FIXME: Update these instructions for OpenFST
  A simple decoder for fst networks. Assumes, that the network has state indices encoded.
   To create a search network, you should do something like (the example is given from mitfst tools)

fst_compose ../work/L.fst ../work/lat.fst -| fst_compose -t ../work/C.fst - - | fst_compose ../work/H.fst - -  | fst_optimize -A - ../work/final.fst

L.fst: Lexicon fst (word to monophones)
lat.fst: grammar (words that can follow each other)
C.fst: context grammar (which triphones can follow each other)
H.fst: triphone to state numbers (expands to the number of states specified in the ph file as well, from hmm2fsm)
final.fst: The fst that the search uses
 */

#include <atomic>
#include <exception>
#include <fst/fstlib.h>
#include "FstAcoustics.hh"

#include <boost/chrono.hpp> // rtf calculations

typedef std::string bytestype;

struct FstSearchException : public std::exception {
        FstSearchException(const std::string &s): m_str(s) {}
        std::string m_str;
        const char * what () const throw () {return m_str.c_str();}
};

struct OpenFstToken {
        OpenFstToken(): logprob(0.0), node_idx(-1), state_dur(0),
                        dist_to_best_acu(0.0f) {};
        float logprob;
        std::vector<std::string> unemitted_words;
        fst::StdVectorFst::StateId node_idx;
        int state_dur;

        std::string str() const;

        // DEBUG
        //std::vector<std::pair <int64, float> > state_history;

        // Stuff for confidence calculation
        float dist_to_best_acu;
};

// Collect all Confidence related stuff into this struct
struct OpenFstConfidenceParams {
        OpenFstConfidenceParams(): logprob_weight(2.0f), logprob_hysteresis(100.0f) {};
        // General measures from the main search network
        float logprob_weight;
        float logprob_hysteresis;
        float best_acu_score;
        int cur_frame; // For length normalization

        // Measures from an open phone loop search
        float phone_loop_logprob_weight;

        // Store last values for debug
        float phone_loop_confidence;
        float token_confidence;
        float edit_confidence;
        float best_acu_confidence;
};

class OpenFstSearch {
public:
        OpenFstSearch(fst::StdVectorFst *search_network, FstAcoustics *fst_acu,
                      fst::StdVectorFst *phone_loop_network=nullptr);
        OpenFstSearch(std::string search_network_name,
                      std::string hmm_path, std::string dur_path,
                      std::string phone_loop_network_name="");

        ~OpenFstSearch();

        void set_duration_scale(float d) {m_duration_scale=d;}
        void set_beam(float b) {m_beam=b;}
        void set_token_limit(int t) {m_token_limit=t;}
        void set_transition_scale(float t) {m_transition_scale=t;}
        void set_acoustics(FstAcoustics *fsta) {m_fst_acoustics = fsta;}
        FstAcoustics *get_acoustics() {return m_fst_acoustics;}

        float get_duration_scale() {return m_duration_scale;}
        float get_beam() {return m_beam;}
        int get_token_limit() {return m_token_limit;}
        float get_transition_scale() {return m_transition_scale;}

        void init_search();
        void lna_open(const char *file, int size) {
                m_fst_acoustics->lna_open(file, size);}
        void lna_open_fd(const int fd, int size)  {
                m_fst_acoustics->lna_open_fd(fd, size);}
        void lna_close() {m_fst_acoustics->lna_close();}

        void run(bool *cancel = nullptr);

        inline bytestype get_result() {
                float foo; return get_result_and_logprob(&foo);}

        inline float get_best_final_token_logprob() {assert(false);}
        inline float get_best_token_logprob() {
                return m_new_tokens.size()>0 ?
                        m_new_tokens[0].logprob: -9999999.0f;};
        bytestype get_result_and_logprob2(
                float *logprob, std::string *pronunciation_string,
                bool must_be_final=true);
        bytestype get_result_and_logprob(float *logprob) {
                return get_result_and_logprob2(logprob, nullptr, true);
        }
        bytestype get_result_and_confidence(float *confidence_retval,
                                            std::string *pronunciation=nullptr);

        bytestype tokens_at_final_states();
        bytestype best_tokens(int n=10);

        // Confidence related stuff
        inline void set_confidence_logprob_weight(float lpw) {
                m_confidence_params.logprob_weight = lpw;}
        inline void set_confidence_logprob_hysteresis(float h) {
                m_confidence_params.logprob_hysteresis = h;}
        inline void set_one_token_per_node() {
                m_one_token_per_node = true;
                m_node_best_token.resize(m_search_network->NumStates());
        }

        // Sync confidence thread funcs
        inline void mark_thread_frame_available() {
                m_cur_frame_done = false;
        }
        inline void mark_thread_frame_computed() {
                m_cur_frame_done = true;
        }
        inline bool thread_frame_done() {
                return m_cur_frame_done;
        }
        inline void finish_decode_thread() {
                m_finish_dec_thread = true;
                m_cur_frame_done = false;
        }
        inline bool decode_thread_finished() {
                return m_finish_dec_thread;
        }

        inline long GetPhoneLoopCpuTime() {
                return m_phone_loop_cpu_time.count();
        }

        inline void reset() {
                m_finish_dec_thread = false;
                m_cur_frame_done = true;
                if (m_phone_loop_search) {
                        m_phone_loop_search->reset();
                }
        }

        int verbose;

        // Should be only accessed by different instance of this class
        void propagate_tokens(bool verbose=false);

private:
        FstAcoustics *m_fst_acoustics;
        fst::StdVectorFst *m_search_network;

        std::vector<OpenFstToken> m_new_tokens;
        std::vector<OpenFstToken> m_active_tokens;
        std::vector<int> m_node_best_token;

        float m_duration_scale;
        float m_beam;
        int m_token_limit;
        float m_transition_scale;
        bool m_one_token_per_node;
        bool m_delete_acoustics;
        bool m_delete_network;
        int64 m_ieps_idx, m_oeps_idx, m_osil_idx;
        bool calculate_confidence;

        float propagate_token(
                OpenFstToken &t, float beam_prune_threshold=-999999999.0f);
        void prune_nonfinals_set_final_weights();
        bool is_final(size_t node_idx);

        // Stuff for confidence calculation
        OpenFstSearch *m_phone_loop_search;
        OpenFstConfidenceParams m_confidence_params;
        float get_best_frame_acu_prob();
        void grammar_token_and_best_acu_confidence(
                float *gt_conf, float *ba_conf);
        float levenshtein_confidence(
                const std::string &grammar_s,
                const std::string &ploop_s);

        // Synchronization for confidence thread calculations
        std::atomic<bool> m_finish_dec_thread;
        std::atomic<bool> m_cur_frame_done;


        // Measure the confidence network thread cpu consumption
        boost::chrono::milliseconds m_phone_loop_cpu_time;
};



#endif
