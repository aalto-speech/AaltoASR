#!/usr/bin/python
#
# Recognizes status dictation or free dictation speech files (select using the
# -l option.
#
# Arguments:
#   model-directory
#   speech-directory
#   output-file
#   -a acoustic-model [speechdat | noisy]
#   -l language-model [status | free | morph19k]
#   -s language-model-scale
#
# Example:
#   rec_hammas2011.py /share/puhe/hammas2011 \
#     /share/puhe/audio/hammas2011/status_dictation_8khz output.csv \
#     -a speechdat -l status

import time
import string
import sys
import os
import subprocess
import re
import tempfile
import filecmp
import gzip
import math
from optparse import OptionParser


# Set your decoder swig path in here!
sys.path.append(os.path.dirname(sys.argv[0]) + "/src/swig");
#sys.path.append('/home/jpylkkon/decoder_new/decoder/src/swig/');

import Decoder

##################################################
# From Wikibooks
# http://en.wikibooks.org/wiki/Algorithm_implementation/Strings/Levenshtein_distance#Python
#

def levenshtein_l(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_l(s2, s1)
    if not s1:
        return len(s2)

    previous_row = xrange(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def levenshtein_w(s1, s2):
	words1 = s1.split(None)
	words2 = s2.split(None)
	
	count1 = len(words1)
	count2 = len(words2)

	if count1 < count2:
		return levenshtein_w(s2, s1)
	if not s1:
		return count2

	previous_row = xrange(count2 + 1)
	for i, w1 in enumerate(words1):
		current_row = [i + 1]
		for j, w2 in enumerate(words2):
			insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
			deletions = current_row[j] + 1       # than s2
			substitutions = previous_row[j] + (w1 != w2)
			current_row.append(min(insertions, deletions, substitutions))
		previous_row = current_row

	return previous_row[-1]

def natural_sorted(list): 
	convert = lambda text: int(text) if text.isdigit() else text.lower() 
	alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
	return sorted(list, key = alphanum_key)


##################################################
# Initialize
#

parser = OptionParser()
parser.add_option('-a', '--acoustic-model',
				  action='store', type='string', dest='am', default='speechdat')
parser.add_option('-l', '--language-model',
				  action='store', type='string', dest='lm', default='status')
parser.add_option('-s', '--language-model-scale',
				  action='store', type='int', dest='lm_scale', default=30)

(options, args) = parser.parse_args()
if len(args) != 3:
	parser.error("incorrect number of arguments")

model_directory = args[0]
speech_directory = args[1]
output_file = args[2]

akupath = os.path.dirname(sys.argv[0]) + "/../aku"

if options.am == 'speechdat':
	# 8 kHz telephone line
	ac_model = model_directory + "/speechdat_gain5000_occ300_23.2.2009_22";
elif options.am == 'noisy':
	# 16 kHz acoustic model
	ac_model = model_directory + "/test_mfcc_noisy_trained";
hmms = ac_model + ".ph"
dur = ac_model + ".dur"

if options.lm == 'free':
	lexicon = model_directory + "/FreeDictationLexicon.lex"
	ngram = model_directory + "/FreeDictationLM.6gram.bin"
	lookahead_ngram = model_directory + "/FreeDictationLM.2gram.bin"
	morph_model = True
elif options.lm == 'status':
	lexicon = model_directory + "/StatusDictationLexicon.lex"
	ngram = model_directory + "/StatusDictationLM.3gram.bin"
	lookahead_ngram = model_directory + "/StatusDictationLM.2gram.bin"
	morph_model = False
elif options.lm == 'morph19k':
	lexicon = "/share/work/jpylkkon/bin_lm/morph19k.lex"
	ngram = "/share/work/jpylkkon/bin_lm/morph19k_D20E10_varigram.bin"
	lookahead_ngram = "/share/work/jpylkkon/bin_lm/morph19k_2gram.bin"
	morph_model = True

global_beam = 400


##################################################
# Generate LNA files for recognition
#

recipe_file = tempfile.NamedTemporaryFile()

for wav_file in os.listdir(speech_directory):
	if not wav_file.endswith('.wav'):
		continue;
	audio = speech_directory + "/" + wav_file
	lna = wav_file[:-4] + ".lna"
	if not os.path.exists(speech_directory + "/" + lna):
		recipe_file.write("audio=" + audio + " lna=" + lna + "\n")

# Don't close the temporary file yet, or it will get deleted.
recipe_file.flush()

print "Computing state probabilities."
command = [akupath + "/phone_probs", \
		"-b", ac_model, \
 		"-c", ac_model + ".cfg", \
 		"-r", recipe_file.name, \
		"-o", speech_directory]
#command = [akupath + "/phone_probs", \
#		"-b", ac_model, \
#		"-c", ac_model + ".cfg", \
#		"-r", recipe_file.name, \
#		"-C", ac_model + ".gcl", \
#		"-o", speech_directory, \
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

if morph_model:
	t.set_silence_is_word(1)
else:
	t.set_silence_is_word(0)

t.set_optional_short_silence(1)

t.set_cross_word_triphones(1)
t.set_require_sentence_end(1)

print "Loading acoustic model."
t.hmm_read(hmms)
t.duration_read(dur)

t.set_verbose(1)
t.set_print_text_result(0)

# Generate a lattice of the word sequences that the decoder considered. Requires
# 2-grams or higher order model.
t.set_generate_word_graph(1)

t.set_lm_lookahead(1)

if morph_model:
	t.set_word_boundary("<w>")

print "Loading lexicon."
try:
    t.lex_read(lexicon)
except:
    print "phone:", t.lex_phone()
    sys.exit(-1)
t.set_sentence_boundary("<s>", "</s>")

print "Loading language model."
t.ngram_read(ngram, 1)
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
t.set_lm_scale(options.lm_scale)
# t.set_insertion_penalty(-0.5)

print "Recognizing audio files."
output_file = open(output_file, "w")
for lna_file in natural_sorted(os.listdir(speech_directory)):
	if not lna_file.endswith('.lna'):
		continue

	name = lna_file[:-4]
	print name

	lna_path = speech_directory + "/" + lna_file
	rec_path = lna_path[:-4] + ".rec"
	txt_path = lna_path[:-4] + ".txt"
	slf_path = lna_path[:-4] + ".slf"
	t.lna_open(lna_path, 1024)
	t.reset(0)
	t.set_end(-1)
	while t.run():
		pass

	# We have to open with only "w" first, and then later with "r"
	# for reading, or the file will not be written.
	rec = open(rec_path, "w")
	t.print_best_lm_history_to_file(rec)
	t.write_word_graph(slf_path);
	rec.close()

	command = 'lattice-tool -read-htk -in-lattice "' + \
			slf_path + \
			'" -nbest-decode 100 -out-nbest-dir "' + \
			speech_directory + '"'
	return_value = os.system(command)
	if return_value != 0:
		print "Command returned a non-zero exit status: ", command
		sys.exit(1)

	nbest_file = gzip.open(slf_path + ".gz", "rb")
	nbest_list = nbest_file.readlines()
	nbest_file.close()

	rec = open(rec_path, "r")
	recognition = rec.read()
	if morph_model:
		recognition = recognition.replace(' ', '')
		recognition = recognition.replace('<w>', ' ')
	recognition = recognition.replace('<s>', '')
	recognition = recognition.replace('</s>', '')
	recognition = recognition.strip()
	rec.close()

	rec = open(rec_path, "w")
	rec.write(recognition)
	rec.close()

	# Compensate for incorrect assumptions in the HMM by flattening the
	# logprobs.
	alpha = 0.1

	line = nbest_list[0]
	logprob_1 = float(line.split(' ')[0]) * alpha

	total_logprob = logprob_1
	for line in nbest_list[1:]:
		logprob = float(line.split(' ')[0]) * alpha
		# Acoustic probabilities are calculated in natural logarithm space.
		total_logprob += math.log(1 + math.exp(logprob - total_logprob))

	log_confidence = logprob_1 - total_logprob

	if os.path.exists(txt_path):
		txt = open(txt_path, "r")
		transcription = txt.read().strip()
		txt.close()
		letter_errors = levenshtein_l(recognition, transcription)
		num_letters = len(transcription)
		word_errors = levenshtein_w(recognition, transcription)
		num_words = len(transcription.split(None))
		line = '{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7:.4}\n'.format(name, transcription, recognition, letter_errors, num_letters, word_errors, num_words, log_confidence)
		output_file.write(line)
		output_file.flush()
	else:
		print 'No transcription.'
		print 'Recognition result:', recognition
		print 'Log confidence:', log_confidence

output_file.close()
