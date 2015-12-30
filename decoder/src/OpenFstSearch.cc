#include <thread>

#include "def.hh"
#include "OpenFstSearch.hh"
#include "levenshtein.hh"

std::string get_output_token_string(std::vector<std::string> vs,
                                    std::string *pronunciation_string) {
        std::ostringstream os;
        for (const auto w: vs) {
                std::size_t found = w.find("!");
                std::string ow;
                if (found!=std::string::npos) {
                        if (pronunciation_string) {
                                (*pronunciation_string) += w.substr(found+1);
                        }
                        os << w.substr(0, found) << " ";
                } else {
                        os << w << " ";
                }
        }
        std::string retval(os.str()); // The best hypo at a final node
        return retval.substr(0, retval.size()-1); // Remove the trailing space
}


inline std::string OpenFstToken::str() const {
        std::ostringstream os;
        os << "Token " << node_idx << " " << logprob << " dur " << state_dur << " '";
        for (const auto s: unemitted_words) {
                os << " " << s;
        }
        os << "' : ";
        /*os << "LEN " << state_history.size();
          for (const auto s: state_history) {
          os << " (" << s.first << ", " << s.second << ")";
          }
          os << " '";*/

        return os.str();
}


OpenFstSearch::OpenFstSearch(fst::StdVectorFst *search_network,
                             FstAcoustics *fst_acu,
                             fst::StdVectorFst *phone_loop_network):
        verbose(0), m_fst_acoustics(fst_acu), m_delete_acoustics(false),
        m_duration_scale(3.0f), m_beam(260.0f), m_token_limit(5000),
        m_transition_scale(1.0f), m_one_token_per_node(false),
        m_search_network(search_network),
        m_phone_loop_search(nullptr), m_delete_network(false),
        m_finish_dec_thread(false), m_cur_frame_done(true)
{
        m_token_limit=1000;
        m_beam = 180.0f;
        if (phone_loop_network) {
                m_phone_loop_search = new OpenFstSearch(
                        phone_loop_network, fst_acu, nullptr);
                m_phone_loop_search->set_beam(20);
                m_phone_loop_search->set_token_limit(40);
                //m_phone_loop_search->set_one_token_per_node(); // Segfaults, FIXME
        }
}

OpenFstSearch::OpenFstSearch(std::string search_network_name,
                             std::string hmm_path, std::string dur_path,
                             std::string phone_loop_network_name):
        m_fst_acoustics(new FstAcoustics(hmm_path.c_str(),
                                         dur_path.c_str())),
        m_delete_acoustics(true), m_duration_scale(3.0f),
        m_beam(2600.0f), m_token_limit(5000), m_transition_scale(1.0f),
        m_one_token_per_node(false),
        m_phone_loop_search(nullptr), m_delete_network(true)
{
        if (phone_loop_network_name.size()) {
                m_phone_loop_search = new OpenFstSearch(
                        fst::StdVectorFst::Read(phone_loop_network_name),
                        get_acoustics(), nullptr);
                m_phone_loop_search->set_beam(20);
                m_phone_loop_search->set_token_limit(40);

        }

        m_search_network = fst::StdVectorFst::Read(search_network_name);
        if ( m_search_network == nullptr ) {
                throw FstSearchException("Problem reading " + search_network_name);
        }
        m_token_limit=1000;
        m_beam = 180.0f;
}

OpenFstSearch::~OpenFstSearch() {
        if (m_delete_network) {
                delete m_search_network;
        }

        if (m_phone_loop_search != nullptr) {
                delete m_phone_loop_search;
        }
}

void DecodePropagateThread(OpenFstSearch *dec, boost::chrono::milliseconds *cputime) {
        boost::chrono::thread_clock::time_point start=
                boost::chrono::thread_clock::now();
        while (true) {
                // Wait until next frame available
                while (dec->thread_frame_done()) {
                        usleep(2e3); // Is this a good amount of sleep?
                }
                if (dec->decode_thread_finished()) {
                        break;
                }
                dec->propagate_tokens(false);
                dec->mark_thread_frame_computed();
        }
        *cputime =
                boost::chrono::duration_cast<boost::chrono::milliseconds> (
                        boost::chrono::thread_clock::now() - start);
}

void OpenFstSearch::prune_nonfinals_set_final_weights() {
        // Add the cost to transition to final state
        m_active_tokens.swap(m_new_tokens);
        m_new_tokens.clear();
        for (auto t: m_active_tokens) {
                fst::FloatWeight w = m_search_network->Final(t.node_idx);
                if (w != fst::FloatLimits<float>::PosInfinity() &&
                    w != fst::FloatLimits<float>::NegInfinity()) {
                        t.logprob += w.Value();
                        m_new_tokens.push_back(t);
                }
        }
        std::sort(m_new_tokens.begin(), m_new_tokens.end(),
                  [](OpenFstToken const & a, OpenFstToken const &b)
                  {return a.logprob > b.logprob;});
}

void OpenFstSearch::run(bool *cancel_decoding) {
        //std::cerr << "RUN token limit " << m_token_limit << std::endl;
        if (m_phone_loop_search==nullptr) {
                while (m_fst_acoustics->next_frame() &&
                       !(cancel_decoding && *cancel_decoding )) {
                        propagate_tokens();
                        prune_nonfinals_set_final_weights();
                }
                //fprintf(stderr, "%s\n", tokens_at_final_states().c_str());
                //fprintf(stderr, "%s\n", best_tokens().c_str());
                m_fst_acoustics->lna_close();
                return;
        }
        m_confidence_params;
        m_confidence_params.best_acu_score = 0.0f;
        m_confidence_params.cur_frame = 0;

        std::thread confidence_thread =
                std::thread(DecodePropagateThread,
                            m_phone_loop_search, &m_phone_loop_cpu_time);
        try {
                while (m_fst_acoustics->next_frame() &&
                       !(cancel_decoding && *cancel_decoding )) {
                        m_phone_loop_search->mark_thread_frame_available();
                        propagate_tokens(false);
                        m_confidence_params.best_acu_score += get_best_frame_acu_prob();
                        m_confidence_params.cur_frame++;

                        // Wait for confidence thread to complete before moving
                        // to next frame
                        while (!m_phone_loop_search->thread_frame_done()) {
                                usleep(2e3); // Is this a good amount of sleep?
                        }
                }
        } catch (std::string &s) {
                std::cerr << "ERROR: " << s << std::endl;
        }
        m_phone_loop_search->finish_decode_thread();
        prune_nonfinals_set_final_weights();
        std::cerr << tokens_at_final_states() << std::endl;
        m_fst_acoustics->lna_close();
        confidence_thread.join();
}

void OpenFstSearch::init_search() {
        m_new_tokens.clear();
        m_active_tokens.clear(); // Not needed?
        m_node_best_token.clear(); // Not needed?
        m_new_tokens.resize(1);
        OpenFstToken &t = m_new_tokens[0];
        t.node_idx = m_search_network->Start();
        m_ieps_idx = m_search_network->InputSymbols()->Find("<eps>");
        m_oeps_idx = m_search_network->OutputSymbols()->Find("<eps>");
        m_osil_idx = m_search_network->OutputSymbols()->Find("__");

        if (m_phone_loop_search != nullptr) {
                m_phone_loop_search->init_search();
        }
}

float OpenFstSearch::propagate_token(OpenFstToken &t,
                                     float beam_prune_threshold) {
        float best_logprob=-999999999.0f;

        for (fst::ArcIterator<fst::StdFst> aiter(*m_search_network, t.node_idx);
             !aiter.Done(); aiter.Next()) {
                const fst::StdArc &arc = aiter.Value();
                OpenFstToken updated_token(t);
                updated_token.node_idx = arc.nextstate;
                std::string olabel(m_search_network->OutputSymbols()->Find(arc.olabel));

                // Do all epsilon transitions immediately
                if (arc.ilabel == m_ieps_idx) {
                        if (arc.olabel != m_osil_idx) {
                                updated_token.unemitted_words.push_back(olabel);
                        }
                        // FIXME: check pruning
                        float tmp_lp = propagate_token(updated_token, beam_prune_threshold);
                        if (best_logprob < tmp_lp) {
                                best_logprob = tmp_lp;
                        }
                        continue;
                }

                int source_emission_pdf_idx = std::stoi(
                        m_search_network->InputSymbols()->Find(arc.ilabel));

                /*
                std::cout << t.node_idx
                          << " ILABEL " << arc.ilabel
                          << " OLABEL " << arc.olabel
                          << " ILABEL2 " << source_emission_pdf_idx
                          << " OLABEL2 " << arc.olabel
                          << std::endl;*/


                //std::cout << "LPA " << updated_token.logprob;
                // FIXME: Precalculate the logprob values - this seems to be broken anyhows
                //updated_token.logprob += -safelogprob(arc.weight.Value()) * this->m_transition_scale;
                //std::cout << " LPB " << updated_token.logprob;
                if (source_emission_pdf_idx >= 0) {
                        float alp = this->m_fst_acoustics->log_prob(
                                source_emission_pdf_idx);
                        updated_token.logprob += alp;
                        /*updated_token.state_history.push_back(
                          std::make_pair(source_emission_pdf_idx, alp));*/
                        //std::cout << " LPC " << updated_token.logprob;
                }
                //std::cout << std::endl;

                if (t.node_idx != arc.nextstate) {
                        if (source_emission_pdf_idx >=0) {
                                //cfprintf(stderr, "Adding dur logprob %d -> %d (%d)\n", t.node_idx, arc.nextstate, updated_token.state_dur);
                                // Add the duration from prev state at state change boundary
                                updated_token.logprob += this->m_duration_scale *
                                        this->m_fst_acoustics->duration_logprob(
                                        source_emission_pdf_idx, updated_token.state_dur);
                                updated_token.state_dur = 1;
                        } else fprintf(stderr, "Skip duration model.\n");
                } else {
                        //fprintf(stderr, "Increasing state dur %d\n", t.node_idx);
                        updated_token.state_dur +=1;
                }

                if ((arc.olabel != m_oeps_idx) && (arc.olabel != m_osil_idx)) {
                        updated_token.unemitted_words.push_back(olabel);
                }

                //fprintf(stderr, "m_nbt size %ld, idx %d\n", m_node_best_token.size(), updated_token.node_idx);
                //fprintf(stderr, "%d\n", m_node_best_token[updated_token.node_idx]);
                int best_token_idx = this->m_one_token_per_node?
                        this->m_node_best_token[updated_token.node_idx] : -1;
                OpenFstToken *best_token = best_token_idx == -1 ? nullptr : &(this->m_new_tokens[best_token_idx]);
                if (updated_token.logprob > beam_prune_threshold && // Do approximate beam pruning here, exact later
                        ( best_token_idx ==-1 || updated_token.logprob > best_token->logprob)) {
                        if (this->m_one_token_per_node) {
                                this->m_node_best_token[updated_token.node_idx] = this->m_new_tokens.size();
                        }

                        // DEBUG
                        /*if (m_search_network->OutputSymbols()->Find(arc.olabel) != "<eps>") {
                                std::cout << "Accepted token " <<
                                        t.node_idx << " -> " << arc.nextstate << ": " <<
                                        source_emission_pdf_idx <<
                                        "(" << arc.ilabel << ")  " <<
                                        " -> " << olabel <<
                                        "(" << arc.olabel << ")  "
                                          << std::endl;
                                          }*/

                        if (best_logprob < updated_token.logprob) {
                                best_logprob = updated_token.logprob;
                        }
                        this->m_new_tokens.push_back(std::move(updated_token));
                } /*else {
                        std::cout << "Reject token " <<
                                source_emission_pdf_idx <<
                                " -> " << olabel << std::endl;
                                }*/

        }
        return best_logprob;
}

bytestype OpenFstSearch::tokens_at_final_states() {
        std::ostringstream os;
        os << "Tokens at final nodes:" << std::endl;
        for (const auto t: this->m_new_tokens) {
                if (is_final(t.node_idx)) {
                        os << "  " << t.str() << std::endl;
                }
        }
        return os.str();
}

bytestype OpenFstSearch::get_result_and_logprob2(
        float *logprob, std::string *pronunciation_string,
        bool must_be_final) {
        for (const auto t: this->m_new_tokens) {
                if (must_be_final && !is_final(t.node_idx)) {
                        continue;
                }
                *logprob = t.logprob;
                return get_output_token_string(t.unemitted_words, pronunciation_string);
        }
        // FIXME: We should throw an exception if we end up here !!!!
        *logprob=-1.0f;
        return "";
}

float OpenFstSearch::get_best_frame_acu_prob() {
        float best_prob = -999999999.9f;
        for (int i=0; i < m_fst_acoustics->num_models(); ++i) {
                float model_prob = m_fst_acoustics->log_prob(i);
                if (model_prob>best_prob) best_prob = model_prob;
        }
        return best_prob;
}

bytestype OpenFstSearch::get_result_and_confidence(float *confidence_retval,
                                                   std::string *pronunciation) {
        float grammar_logprob, phone_loop_logprob;
        std::string pronstring;

        bytestype res_string(get_result_and_logprob2(&grammar_logprob, &pronstring));
        if (pronunciation) {
                *pronunciation = pronstring;
        }

        if (m_phone_loop_search == nullptr) {
                *confidence_retval = 0.0;
                return res_string;
        }

        bytestype phone_loop_string(
                m_phone_loop_search->get_result_and_logprob2(
                        &phone_loop_logprob, nullptr, false));

        m_confidence_params.phone_loop_confidence =
                std::min(1.0f, 1.0f- 0.15f*(-grammar_logprob + phone_loop_logprob)/
                         m_confidence_params.cur_frame);
                //std::min(1.0f, phone_loop_logprob / grammar_logprob);
        fprintf(stderr, "pl_lp %.2f, gr_lp %.2f, len %d\n",
                phone_loop_logprob, grammar_logprob, m_confidence_params.cur_frame);
        grammar_token_and_best_acu_confidence(
                &(m_confidence_params.token_confidence),
                &(m_confidence_params.best_acu_confidence));
        m_confidence_params.edit_confidence = levenshtein_confidence(
                pronstring, phone_loop_string);
        if (m_confidence_params.token_confidence >= 0.0f) {
                *confidence_retval = (
                          5.0f  * std::min( 1.0f, m_confidence_params.phone_loop_confidence)
                        + 12.0f * std::min( 1.0f, m_confidence_params.token_confidence)
                        + 1.0f  * std::min(1.0f, m_confidence_params.edit_confidence)
                        + 4.0f  * std::max(0.0f, std::min( 1.0f, m_confidence_params.best_acu_confidence))
                        )/22.0f;
        } else {
                *confidence_retval = (
                          5.0f  * std::min( 1.0f, m_confidence_params.phone_loop_confidence)
                        + 1.0f  * std::min(1.0f, m_confidence_params.edit_confidence)
                        + 4.0f  * std::max(0.0f, std::min( 1.0f, m_confidence_params.best_acu_confidence))
                        )/10.0f;
        }

        if (true) {
                fprintf(stderr, "%s:\n\tToken: %.4f\n",
                        res_string.c_str(), m_confidence_params.token_confidence);
                fprintf(stderr, "\tPhone loop: %.4f\n",
                        m_confidence_params.phone_loop_confidence);
                fprintf(stderr, "\tEdit: %.4f\n", m_confidence_params.edit_confidence);
                fprintf(stderr, "\tBacu: %.4f\n", m_confidence_params.best_acu_confidence);
                fprintf(stderr, "\tTotal: %.4f\n\n", *confidence_retval);
        }

        return res_string;
}

void OpenFstSearch::grammar_token_and_best_acu_confidence(float *gt_conf, float *ba_conf) {
        // NOTE: Tokens at the same state get pruned,
        // if only one final state in network this can be quite unreliable
        bool check_only_final_nodes=false;
        bool reject_same_prefix=false;

        float best_final_token_logprob=-99999999999;
        std::string best_final_token_symbols;
        for (const auto t: m_new_tokens) {
                if (is_final(t.node_idx)) {
                        best_final_token_logprob = t.logprob;
                        best_final_token_symbols = get_output_token_string(
                                t.unemitted_words, nullptr);
                        break;
                }
        }

        *ba_conf = 1.5f - 0.25f*(-best_final_token_logprob +
                                m_confidence_params.best_acu_score)/
                                m_confidence_params.cur_frame;
        //m_confidence_params.best_acu_score / best_final_token_logprob;

        std::cerr << "baconf " << *ba_conf << " with "
                  << best_final_token_logprob << " "
                  << m_confidence_params.best_acu_score << " "
                  << m_confidence_params.cur_frame << std::endl;

        //*ba_conf = m_best_acu_score/best_final_token_logprob;

        if (best_final_token_symbols.size()==0) {
                std::cerr << "Emptiness" << std::endl;
                *gt_conf = -9999999.9f;
                return;
        }

        float best_different_hypo_logprob=-9999999.9f;
        bool found_alt_hyp=false;
        for (const auto t:m_new_tokens) {
                //std::cerr << "Tokening " << t.str() << std::endl;
                if (check_only_final_nodes &&
                    !is_final(t.node_idx)) {
                        continue;
                }

                if (best_final_token_symbols != get_output_token_string(
                            t.unemitted_words, nullptr)) {
                        best_different_hypo_logprob = t.logprob;
                        found_alt_hyp=true;
                        break;
                }
#if 0
                if (t.unemitted_words.size() > best_final_token_symbols.size()) {
                        //fprintf(stderr, "size\n");
                        best_different_hypo_logprob = t.logprob;
                        break;
                }

                // Check for the same prefix
                if (reject_same_prefix) {
                        for (auto i=0; i<t.unemitted_words.size(); ++i) {
                                if (t.unemitted_words[i] != best_final_token_symbols[i]) {
                                        best_different_hypo_logprob = t.logprob;
                                        //fprintf(stderr,"Diff hypo: ");
                                        //for (const auto w: t.unemitted_words) {
                                        //  fprintf(stderr, " %s", w.c_str());
                                        //}
                                        //fprintf(stderr, "\n");
                                        goto out;
                                }
                        }
                } else {
                        if (t.unemitted_words != best_final_token_symbols) {
                                best_different_hypo_logprob = t.logprob;
                                goto out;
                        }
                }
#endif
        }

#if 0
out:
        // Log difference
        //float token_dist(best_final_token_logprob-best_different_hypo_logprob);
        //float first_term = std::min(token_dist, 0.0f)*logf(static_cast<float>(m_frame))/1000.0f;
        //fprintf(stderr, "%.4f %.4f\n", token_dist, first_term);
        //return std::max(0.0f, 1.0f + first_term);

        // Log division
        //fprintf(stderr, "%.4f %.4f %.4f\n", best_different_hypo_logprob, best_final_token_logprob, best_different_hypo_logprob/(m_logprob_conf_weight*best_final_token_logprob-m_logprob_conf_hysteresis)); //3000.0f*logf(static_cast<float> (m_frame))));
        //*gt_conf = std::min(1.0f, best_different_hypo_logprob/(m_logprob_conf_weight*best_final_token_logprob-m_logprob_conf_hysteresis)); //-3000.0f*logf(static_cast<float> (m_frame))));
#endif

        //std::cerr << "Found alt hyp " << found_alt_hyp << std::endl;
        if (found_alt_hyp) {
                *gt_conf = std::max(0.0f, std::min(
                                            1.0f, 0.2f - 5.0f*(-best_final_token_logprob + best_different_hypo_logprob)/
                                            m_confidence_params.cur_frame));
        } else {
                *gt_conf = -1.0f;
        }

}

std::string remove_junk(const std::string a) {
        std::string b;
        char prev_token = ' ';

        for (auto c : a) {
                if (c == ' ' || c == prev_token || c == '_') continue;
                prev_token = c;
                b+=c;
        }
        return b;
}

float OpenFstSearch::levenshtein_confidence(const std::string &grammar_s, const std::string &ploop_s) {
        const std::string clean_grammar_s(remove_junk(grammar_s));
        const std::string clean_ploop_s(remove_junk(ploop_s));
        int ldist = levenshtein_distance(clean_grammar_s, clean_ploop_s);
        std::cerr << "Ldist " << ldist << " for '" << clean_grammar_s
                  << "' vs '" << clean_ploop_s << std::endl;
        return std::min(1.0f, std::max(0.0f, 1.3f-float(ldist)/clean_grammar_s.size()));
}

bool OpenFstSearch::is_final(size_t node_idx) {
        //return std::isfinite((float) (m_search_network->Final(node_idx)));
        //fst::TropicalWeight w = m_search_network->Final(node_idx);
        //return w != fst::TropicalWeight::Zero();

        fst::FloatWeight w = m_search_network->Final(node_idx);
        return w != fst::FloatLimits<float>::PosInfinity() &&
                w != fst::FloatLimits<float>::NegInfinity() ;

}

void OpenFstSearch::propagate_tokens(bool verbose) {
        m_active_tokens.swap(m_new_tokens);
        if (verbose) {
                std::cerr << "Working on frame " << m_fst_acoustics->cur_frame()
                          << ", " << m_active_tokens.size()
                          << " active_tokens" << std::endl;
        }

        // Clean up the buffers that will hold the new values
        m_new_tokens.clear();

        float best_logprob=-999999999.0f;
        for (auto t: m_active_tokens) {
                float blp = propagate_token(t, best_logprob-m_beam);
                if (best_logprob<blp) {
                        best_logprob = blp;
                }
        }

        if (verbose) {
                std::cerr << "Got " << m_new_tokens.size() << " tokens, limit " <<
                        m_token_limit << ", beam " << m_beam << std::endl;
        }

        // sort and prune
        std::sort(m_new_tokens.begin(), m_new_tokens.end(),
                  [](OpenFstToken const & a, OpenFstToken const &b){return a.logprob > b.logprob;});

        if (m_one_token_per_node) {
                std::fill(m_node_best_token.begin(), m_node_best_token.end(), -1);
                /*fprintf(stderr, "Sorted\n");
                  int c=0;
                  for (auto t: m_new_tokens) {
                  fprintf(stderr, "  %d(%d): %s\n", ++c, m_frame, t.str().c_str());
                  }*/
                if (m_new_tokens.size() > m_token_limit) {
                        m_new_tokens.resize(m_token_limit);
                }
                //fprintf(stderr, "size after token limit %ld\n", m_new_tokens.size());
        } else {
                // For each active node, keep only one hypo with the same tokens
                int num_accepted_tokens=0;
                auto orig_tokens(std::move(m_new_tokens));
                m_new_tokens.clear();
                std::map<int, std::set<std::vector<std::string > > > histmap;
                for (auto t: orig_tokens) {
                        //fprintf(stderr, "Is there already?");
                        auto &valset = histmap[t.node_idx];
                        if (valset.find(t.unemitted_words) !=valset.end() ) {
                                //fprintf(stderr, " Yes!\n");
                                continue;
                        }
                        //fprintf(stderr, " Nope!\n");
                        valset.insert(t.unemitted_words);
                        m_new_tokens.push_back(std::move(t));
                        if (num_accepted_tokens>= m_token_limit) break;
                        num_accepted_tokens++;
                }
        }

        int beam_prune_idx = 1;
        if (m_new_tokens.size() == 0) {
                throw FstSearchException("No new tokens, bug?");
        }
        best_logprob = m_new_tokens[0].logprob;
        while (beam_prune_idx < m_new_tokens.size() && m_new_tokens[beam_prune_idx].logprob > best_logprob-m_beam) {
                beam_prune_idx++;
        }
        m_new_tokens.resize(beam_prune_idx);
}
