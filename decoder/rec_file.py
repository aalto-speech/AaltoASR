#!/usr/bin/python

import time
import string
import sys
import os
import re

# Set your decoder swig path in here!
sys.path.append(os.path.dirname(sys.argv[0]) + "/src/swig");

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

akupath = "/home/jpylkkon/aku/cvs/aku"
akumodel = "/share/puhe/models/speecon_mfcc_gain3500_occ225_1.11.2007_20"
hmms = akumodel+".ph"
dur = akumodel+".dur"
temppath = "/share/work/jpylkkon/temp"
lexicon = "/share/work/jpylkkon/bin_lm/morph19k.lex"
ngram = "/share/work/jpylkkon/bin_lm/morph19k_D20E10_varigram.bin"
lookahead_ngram = "/share/work/jpylkkon/bin_lm/morph19k_2gram.bin"
lm_scale = 28
global_beam = 250
##################################################


##################################################
# Generate LNA file for recognition
#

f=open(temppath+"/temp.recipe", 'w')
f.write("audio="+sys.argv[1]+" lna="+temppath+"/temp.lna\n")
f.close()
sys.stderr.write("Generating LNA\n")
os.system(akupath + "/phone_probs -b "+akumodel+" -c "+akumodel+".cfg -r "+temppath+"/temp.recipe -C "+akumodel+".gcl --eval-ming 0.1")


##################################################
# Recognize
#

sys.stderr.write("loading models\n")
t = Decoder.Toolbox(0, hmms, dur)

t.set_optional_short_silence(1)

t.set_cross_word_triphones(1)

t.set_require_sentence_end(1)


t.set_verbose(1)
t.set_print_text_result(1)
#t.set_print_state_segmentation(1)
t.set_lm_lookahead(1)

t.set_word_boundary("<w>")

sys.stderr.write("loading lexicon\n")
try:
    t.lex_read(lexicon)
except:
    print("phone:", t.lex_phone())
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

print("BEAM: %.1f" % global_beam)
print("WORD_END_BEAM: %.1f" % word_end_beam)
print("LMSCALE: %.1f" % lm_scale)
print("DURSCALE: %.1f" % dur_scale)

t.lna_open(temppath+"/temp.lna", 1024)
sys.stdout.write("REC: ")
rec(0,-1)
