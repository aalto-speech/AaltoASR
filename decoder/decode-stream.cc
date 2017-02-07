//
// ----------------
// decode-stream.cc
// ----------------
//
// A sample C++ program that demonstrates decoding using AaltoASR library. Reads
// raw audio from the standard input and writes the recognition result and a
// confidence score to the standard output. The input audio has to be in the
// format specified in the acoustic model configuration. The confidence score is
// calculated as the average acoustic likelihood per audio frame.
//
// At minimum you need to set ACOUSTIC_MODEL_PATH, DICTIONARY_PATH,
// LANGUAGE_MODEL_PATH, and LOOKAHEAD_MODEL_PATH (can be empty) before compiling
// the program. The steps to decode an audio stream are:
//   1) Create an aku::FeatureGenerator for feature extraction.
//   2) Create an aku::HmmSet for computing acoustic probabilities, and read the
//      acoustic model.
//   3) Create a Toolbox for decoding, and read the dictionary and language
//      model.
//   4) One frame at a time, extract features, compute acoustic probabilities,
//      and advance the decoder.
//   5) Read the result from the search history of the highest probability
//      token.
//
// To build this program using GNU C++, first build AaltoASR, then enter the
// build directory, and run:
//   g++ -std=c++0x ../decoder/decode-stream.cc \
//       -I.. -Ivendor/lapackpp/include/lapackpp -I../decoder/src \
//       -Laku -Ldecoder/src -Ldecoder/src/fsalm -Ldecoder/src/misc -Lvendor/lapackpp/lib \
//       -Wl,-Bstatic -ldecoder -lfsalm -lmisc -laku -llapackpp \
//       -Wl,-Bdynamic -lfftw3 -lsndfile -llapack -lblas \
//       -o decode-stream
//

#include <iostream>
#include <cstdio>

#include <aku/FeatureGenerator.hh>
#include <aku/HmmSet.hh>
#include <Toolbox.hh>


using namespace std;
using namespace aku;

typedef vector<float> probabilities_type;


static const string ACOUSTIC_MODEL_PATH = "my-acoustic-model";
static const string DICTIONARY_PATH = "my-dictionary";
static const string LANGUAGE_MODEL_PATH = "my-language-model";
static const string LOOKAHEAD_MODEL_PATH = "my-lookahead-language-model";
static const int TOKEN_LIMIT = 50000;
static const int BEAM = 400;
static const int LM_SCALE = 30;
static const bool IS_WORD_MODEL = true;


inline double safe_log(double x)
{
	static const double TINY_FOR_LOG = 1e-30;

	if (x < TINY_FOR_LOG)
		return log(TINY_FOR_LOG);
	else
		return log(x);
}


void initialize_acoustics(FeatureGenerator & feature_generator, HmmSet & hmm_set)
{
	const std::string cfg_path = ACOUSTIC_MODEL_PATH + ".cfg";
	FILE * config_stream = fopen(cfg_path.c_str(), "r");
	if (config_stream == nullptr) {
		perror("ERROR: Cannot open feature module configuration file for reading");
		exit(1);
	}

	try {
		feature_generator.load_configuration(config_stream);
		feature_generator.open(stdin, true, true);
	}
	catch (string & message) {
		fclose(config_stream);
		cerr << "ERROR: Unable to initialize acoustic feature computation: " << message << endl;
		exit(1);
	}

	int rc = fclose(config_stream);
	if (rc != 0) {
		perror("ERROR: Failed to close feature module configuration file");
		exit(1);
	}

	// Force raw data in audio file module configuration.
	AudioFileModule * afm = dynamic_cast<AudioFileModule *>(feature_generator.module("audiofile"));
	ModuleConfig afm_config;
	afm->get_config(afm_config);
	afm_config.set("raw", 1);
	afm->set_config(afm_config);

	try {
		hmm_set.read_mc(ACOUSTIC_MODEL_PATH + ".mc");  // mixture coefficients
		hmm_set.read_ph(ACOUSTIC_MODEL_PATH + ".ph");  // HMM definition
		hmm_set.read_gk(ACOUSTIC_MODEL_PATH + ".gk");  // mixture base functions

		// Gaussian clustering (optional)
		hmm_set.read_clustering(ACOUSTIC_MODEL_PATH + ".gcl");
		// minimum number of clusters and Gaussians to evaluate before
		// approximating using cluster center likelihood
		hmm_set.set_clustering_min_evals(0, 0.15);
	}
	catch (string & message) {
		cerr << "ERROR: Error loading acoustic model: " << message << endl;
		exit(1);
	}
}


void initialize_decoder(Toolbox & toolbox)
{
	try {
		toolbox.set_optional_short_silence(true);
		toolbox.set_cross_word_triphones(1);
		toolbox.set_require_sentence_end(true);
		toolbox.use_one_frame_acoustics();
		toolbox.set_verbose(1);  // Don't print status messages to stdout.

		toolbox.set_token_limit(TOKEN_LIMIT);
		toolbox.set_global_beam(BEAM);
		toolbox.set_word_end_beam(2 * BEAM / 3);
		toolbox.set_duration_scale(3);
		toolbox.set_transition_scale(1);
		toolbox.set_lm_scale(LM_SCALE);

		if (IS_WORD_MODEL) {
			toolbox.set_silence_is_word(false);
			toolbox.set_word_boundary("");
		}
		else {
			toolbox.set_silence_is_word(true);
			toolbox.set_word_boundary("<w>");
		}

		// set_lm_lookahead() has to be set before lex_read(), or
		// lookahead will be disabled!
		if (LOOKAHEAD_MODEL_PATH.empty()) {
			toolbox.set_lm_lookahead(0);
		}
		else {
			toolbox.set_lm_lookahead(1);
		}

		toolbox.lex_read(DICTIONARY_PATH.c_str());
		toolbox.set_sentence_boundary("<s>", "</s>");

		int order = toolbox.ngram_read(LANGUAGE_MODEL_PATH.c_str(), false, false);

		if (!LOOKAHEAD_MODEL_PATH.empty()) {
			toolbox.read_lookahead_ngram(LOOKAHEAD_MODEL_PATH.c_str(), false, false);
			toolbox.prune_lm_lookahead_buffers(0, 4);
		}
		toolbox.set_prune_similar(order);

		toolbox.set_generate_word_graph(false);
	}
	catch (TPNowayLexReader::UnknownHmm & e) {
		cerr << "ERROR: The lexicon contains a triphone that does not exist in the acoustic model: " << e.phone() << endl;
		exit(1);
	}
	catch (exception & e) {
		cerr << "ERROR: Unable to initialize decoder: " << e.what() << endl;
		exit(1);
	}
}


bool get_features(FeatureGenerator & feature_generator, int frame_index, FeatureVec & result)
{
	try {
		result = feature_generator.generate(frame_index);
		return !feature_generator.eof();
	}
	catch (string & message) {
		cerr << "ERROR: Error computing acoustic features: " << message << endl;
		exit(1);
	}
}


void get_likelihoods(HmmSet & hmm_set, const FeatureVec & features, probabilities_type & result)
{
	try {
		hmm_set.precompute_likelihoods(features);

		result.clear();
		result.reserve(hmm_set.num_states());
		for (int i = 0; i < hmm_set.num_states(); ++i) {
			double likelihood = hmm_set.state_likelihood(i, features);
			result.push_back(static_cast<float>(safe_log(likelihood)));
		}
	}
	catch (string & message) {
		cerr << "ERROR: Error computing acoustic model likelihoods: " << message << endl;
		exit(1);
	}
}


void print_result(Toolbox & toolbox, int num_frames)
{
	std::vector<LMHistory::Word> path;
	path = toolbox.tp_search().get_word_repository();

	cout << "RESULT:";
	HistoryVector::const_reverse_iterator iter = path.rbegin();
	for (; iter != path.rend(); ++iter) {
		LMHistory::Word history = *iter;
		const string & word = toolbox.word(history.word_id());
		cout << " " << word;
	}
	cout << endl;

	if (num_frames > 0) {
		float am_log_prob = toolbox.tp_search().get_am_log_prob(true);
		cout << "CONFIDENCE: " << exp(am_log_prob / num_frames) << endl;
	}
	else {
		cerr << "ERROR: No audio was read." << endl;
		exit(1);
	}
}


int main()
{
	static FeatureGenerator feature_generator;
	static HmmSet hmm_set;
	initialize_acoustics(feature_generator, hmm_set);

	string ph_path = ACOUSTIC_MODEL_PATH + ".ph";
	string dur_path = ACOUSTIC_MODEL_PATH + ".dur";
	Toolbox toolbox(0, ph_path.c_str(), dur_path.c_str());
	initialize_decoder(toolbox);

	int current_frame = 0;

	toolbox.reset(0);
	while (true) {
		aku::FeatureVec features;
		probabilities_type likelihoods;

		if (get_features(feature_generator, current_frame, features)) {
			get_likelihoods(hmm_set, features, likelihoods);
		}

		try {
			// toolbox::run() will return false when likelihoods is empty.
			toolbox.set_one_frame(current_frame, likelihoods);
			if (!toolbox.run())
				break;
		}
		catch (exception & e) {
			cerr << e.what() << endl;
			return 1;
		}

		++current_frame;
	}

	print_result(toolbox, current_frame);
	return 0;
}
