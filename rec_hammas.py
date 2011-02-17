#!/usr/bin/python

import time
import string
import sys
import os
import subprocess
import re
import tempfile

# Set your decoder swig path in here!
sys.path.append("src/swig");

import Decoder

def runto(frame):
    while (frame <= 0 or t.frame() < frame):
        if (not t.run()):
            break

def rec(start, end):
    st = os.times()
    t.reset(start)
    t.set_end(end)
    runto(0)
    et = os.times()
    duration = et[0] + et[1] - st[0] - st[1] # User + system time
    frames = t.frame() - start;
    sys.stdout.write('DUR: %.2fs (Real-time factor: %.2f)\n' %
                     (duration, duration * 125 / frames))

##################################################
# Initialize
#

akupath = "../aku"

akumodel = sys.argv[1] + "/test_mfcc_noisy_trained";
hmms = akumodel + ".ph"
dur = akumodel + ".dur"
lexicon = sys.argv[1] + "/hammas.lex"
ngram = sys.argv[1] + "/hammas.bin"
lookahead_ngram = sys.argv[1] + "/hammas_2gram.bin"

lm_scale = 1
global_beam = 320


##################################################
# Generate LNA files for recognition
#

recipe_file = tempfile.NamedTemporaryFile()

input_directory = sys.argv[1] + "/test_sentences"
for input_file in os.listdir(input_directory):
	if input_file.endswith('.wav'):
		audio = input_directory + "/" + input_file
		lna = input_file[:-4] + ".lna"
		if not os.path.exists(input_directory + "/" + lna):
			recipe_file.write("audio=" + audio + " lna=" + lna + "\n")

# Don't close the temporary file yet, or it will get deleted.
recipe_file.flush()

sys.stderr.write("Computing phoneme probabilities.\n")
command = [akupath + "/phone_probs", \
		"-b", akumodel, \
 		"-c", akumodel + ".cfg", \
 		"-r", recipe_file.name, \
		"-C", akumodel + ".gcl", \
		"-o", input_directory, \
		"--eval-ming", "0.20"]
try:
	result = subprocess.check_call(command)
except subprocess.CalledProcessError as e:
	sys.stderr.write("phone_probs returned a non-zero exit status.\n")
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

sys.stderr.write("Loading acoustic model.\n")
t.hmm_read(hmms)
t.duration_read(dur)

t.set_verbose(1)
t.set_print_text_result(1)
# t.set_print_state_segmentation(1)
# t.set_print_word_start_frame(1)
t.set_lm_lookahead(1)

# Only needed for morph models.
# t.set_word_boundary("<w>")

sys.stderr.write("Loading dictionary.\n")
try:
    t.lex_read(lexicon)
except:
    print "phone:", t.lex_phone()
    sys.exit(-1)
t.set_sentence_boundary("<s>", "</s>")

sys.stderr.write("Loading language model.\n")
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
t.set_prune_similar(3)

t.set_print_probs(0)
t.set_print_indices(0)
t.set_print_frames(0)

t.set_duration_scale(dur_scale)
t.set_transition_scale(trans_scale)
t.set_lm_scale(lm_scale)
# t.set_insertion_penalty(-0.5)

print "BEAM: ", global_beam
print "WORD_END_BEAM: ", word_end_beam
print "LMSCALE: ", lm_scale
print "DURSCALE: ", dur_scale

for input_file in os.listdir(input_directory):
	if input_file.endswith('.lna'):
		path = input_directory + "/" + input_file
		sys.stderr.write(path + "\n")
		t.lna_open(path, 1024)
		print "REC: ",
		rec(0,-1)
