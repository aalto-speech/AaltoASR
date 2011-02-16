#!/usr/bin/python

import time
import string
import sys
import os
import re

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
temppath = "/var/tmp"
lexicon = sys.argv[1] + "/hammas.lex"
ngram = sys.argv[1] + "/hammas.bin"
lookahead_ngram = sys.argv[1] + "/hammas_2gram.bin"
lm_scale = 28
global_beam = 250
##################################################


##################################################
# Generate LNA file for recognition
#

f=open(temppath + "/temp.recipe", 'w')
f.write("audio=" + sys.argv[1] + "/speech/DeeKaksiYksiNasta.wav lna=" + temppath + "/temp.lna\n")
f.close()
sys.stderr.write("Generating LNA\n")
os.system(akupath + "/phone_probs -b "+akumodel+" -c "+akumodel+".cfg -r "+temppath+"/temp.recipe -C "+akumodel+".gcl --eval-ming 0.1")


##################################################
# Recognize
#

t = Decoder.Toolbox()

t.select_decoder(0)
t.set_optional_short_silence(1)

t.set_cross_word_triphones(1)

t.set_require_sentence_end(1)

sys.stderr.write("loading models\n")
t.hmm_read(hmms)
t.duration_read(dur)

t.set_verbose(1)
t.set_print_text_result(1)
#t.set_print_state_segmentation(1)
t.set_lm_lookahead(1)

t.set_word_boundary("<s>")

sys.stderr.write("loading lexicon\n")
try:
    t.lex_read(lexicon)
except:
    print "phone:", t.lex_phone()
    sys.exit(-1)
t.set_sentence_boundary("<s>", "</s>")

sys.stderr.write("loading ngram\n")
t.ngram_read(ngram, 1)
t.read_lookahead_ngram(lookahead_ngram)

t.prune_lm_lookahead_buffers(0, 4) # min_delta, max_depth

word_end_beam = int(2*global_beam/3);
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

print "BEAM: ", global_beam
print "WORD_END_BEAM: ", word_end_beam
print "LMSCALE: ", lm_scale
print "DURSCALE: ", dur_scale

t.lna_open(temppath+"/temp.lna", 1024)
print "REC: ",
rec(0,-1)
