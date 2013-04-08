#!/usr/bin/python -i
#
# Loads an acoustic model and a lexicon, and enters Python
# interactive mode, so that you can debug the lex prefix
# tree using Toolbox::print_tp_lex_node(node_id).
#
# Arguments:
#   model-directory
#   speech-directory
#   output-file
#   -a acoustic-model [basename]
#   -l language-model [status | free]
#   -s language-model-scale
#
# Example:
#   rec_hammaspuhe.py /share/puhe/hammaspuhe/models \
#     /share/puhe/hammaspuhe/audio/free_complete output.csv \
#     -a speecon_all_multicondition_mmi_kld0.002_6 -l free

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
from optparse import OptionParser, OptionValueError


sys.path.append(os.path.dirname(sys.argv[0]) + "/src/swig");
import Decoder


parser = OptionParser()
parser.add_option('--morphs',
                  action='store_true', dest='morphs', default=False)

(options, args) = parser.parse_args()
if len(args) != 2:
	parser.error("Acoustic model and lexicon expected.")

ac_model = args[0]
hmms = ac_model + ".ph"
dur = ac_model + ".dur"
lexicon = args[1]


toolbox = Decoder.Toolbox()
toolbox.select_decoder(0)

if options.morphs:
	toolbox.set_silence_is_word(1)
else:
	toolbox.set_silence_is_word(0)

toolbox.set_optional_short_silence(1)
toolbox.set_cross_word_triphones(1)
toolbox.set_require_sentence_end(1)

print "Loading acoustic model."
toolbox.hmm_read(hmms)
toolbox.duration_read(dur)

toolbox.set_verbose(1)
toolbox.set_print_text_result(1)

#if options.morphs:
#	toolbox.set_word_boundary("<w>")

print "Loading lexicon."
try:
    toolbox.lex_read(lexicon)
except:
    print "Error reading lexicon at word=" + toolbox.lex_word() + ", phone=" + toolbox.lex_phone()
    sys.exit(-1)

#toolbox.set_sentence_boundary("<s>", "</s>")

toolbox.set_print_probs(1)
toolbox.set_print_indices(1)
toolbox.set_print_frames(1)

print "ROOT NODE"
toolbox.print_tp_lex_node(0)

print "Print other nodes using toolbox.print_tp_lex_node(node-id)."
