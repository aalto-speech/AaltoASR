#!/usr/bin/python
#
# Recognizes test commands for cariological status dictation. Takes the
# training data directory as an argument, e.g.
#   ./cariology_test.py ~/DentistryData

import time
import string
import sys
import os
import subprocess
import re
import tempfile
import filecmp

# Set your decoder swig path in here!
sys.path.append("src/swig");

import Decoder

##################################################
# Initialize
#

akupath = "../aku"

akumodel = sys.argv[1] + "/test_mfcc_noisy_trained";
hmms = akumodel + ".ph"
dur = akumodel + ".dur"
lexicon = sys.argv[1] + "/CariologyLexicon.lex"
ngram = sys.argv[1] + "/CariologyLM.even.3gram.bin"
lookahead_ngram = sys.argv[1] + "/CariologyLM.even.2gram.bin"

lm_scale = 10
global_beam = 70


##################################################
# Generate LNA files for recognition
#

recipe_file = tempfile.NamedTemporaryFile()

test_directory = sys.argv[1] + "/CariologyTestSet"
for wav_file in os.listdir(test_directory):
	if not wav_file.endswith('.wav'):
		continue;
	audio = test_directory + "/" + wav_file
	lna = wav_file[:-4] + ".lna"
	if not os.path.exists(test_directory + "/" + lna):
		recipe_file.write("audio=" + audio + " lna=" + lna + "\n")

# Don't close the temporary file yet, or it will get deleted.
recipe_file.flush()

print "Computing phoneme probabilities."
command = [akupath + "/phone_probs", \
		"-b", akumodel, \
 		"-c", akumodel + ".cfg", \
 		"-r", recipe_file.name, \
		"-o", test_directory]
#command = [akupath + "/phone_probs", \
#		"-b", akumodel, \
#		"-c", akumodel + ".cfg", \
#		"-r", recipe_file.name, \
#		"-C", akumodel + ".gcl", \
#		"-o", test_directory, \
#		"--eval-ming", "0.50"]
try:
	result = subprocess.check_call(command)
except subprocess.CalledProcessError as e:
	print "phone_probs returned a non-zero exit status."
	sys.exit(-1)

recipe_file.close()


##################################################
# Recognize
#

t = Decoder.Toolbox()

t.select_decoder(0)
t.set_silence_is_word(0)
t.set_optional_short_silence(1)

t.set_cross_word_triphones(1)
# t.set_require_sentence_end(0)
t.set_require_sentence_end(1)

print "Loading acoustic model."
t.hmm_read(hmms)
t.duration_read(dur)

t.set_verbose(1)
t.set_print_text_result(0)

# Generate a lattice of the word sequences that the decoder considered. Requires
# 2-grams or higher order model.
t.set_generate_word_graph(1)

# t.set_print_state_segmentation(1)
# t.set_print_word_start_frame(1)
t.set_lm_lookahead(1)

# Only needed for morph models.
# t.set_word_boundary("<w>")

print "Loading lexicon."
try:
    t.lex_read(lexicon)
except:
    print "phone:", t.lex_phone()
    sys.exit(-1)
t.set_sentence_boundary("<s>", "</s>")

print "Loading language model."
t.ngram_read(ngram, 1)
# t.fsa_lm_read(ngram, 1)
t.read_lookahead_ngram(lookahead_ngram)

t.prune_lm_lookahead_buffers(0, 4) # min_delta, max_depth

word_end_beam = int(2 * global_beam / 3);
trans_scale = 1
dur_scale = 3

t.set_global_beam(global_beam)
t.set_word_end_beam(word_end_beam)
t.set_token_limit(30000)

# Should equal to the n-gram model order.
t.set_prune_similar(3)

t.set_print_probs(0)
t.set_print_indices(0)
t.set_print_frames(0)

t.set_duration_scale(dur_scale)
t.set_transition_scale(trans_scale)
t.set_lm_scale(lm_scale)
# t.set_insertion_penalty(-0.5)

print "Recognizing audio files."
for lna_file in os.listdir(test_directory):
	if not lna_file.endswith('.lna'):
		continue
	lna_path = test_directory + "/" + lna_file
	rec_path = lna_path[:-4] + ".rec"
	txt_path = lna_path[:-4] + ".txt"
	slf_path = lna_path[:-4] + ".slf"
	t.lna_open(lna_path, 1024)
	t.reset(0)
	t.set_end(-1)
	while (True):
		if (not t.run()):
			# We have to open with only "w" first, and then later with "r"
			# for reading, or the file will not be written.
			rec = open(rec_path, "w")
			t.print_best_lm_history_to_file(rec)
			t.write_word_graph(slf_path);
			rec.close()
			rec = open(rec_path, "r")
			recognition = rec.read().strip()
			rec.close()
			break
	if os.path.exists(txt_path):
		equal = filecmp.cmp(rec_path, txt_path)
		if equal:
			print "OK ", recognition
		else:
			txt = open(txt_path, "r")
			transcription = txt.read().strip()
			txt.close()
			print "F ", recognition, " != ", transcription
	else:
		print "? ", recognition
