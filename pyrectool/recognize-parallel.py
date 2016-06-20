#!/usr/bin/env python3

import time
import sys
import os
import shutil
from rectool import *


rt = RecognizerToolbox()
command = sys.argv

if rt.adaptation is not None:
	if (not rt.use_existing_lnas()) and (not rt.use_existing_adaptation()):
		command[0] = script_path('recognize-stateseg.py')
		# estimate_adaptation() will set options so that recognize.py will use the
		# adaptation data, as long as we won't ask it to estimate adaptation again.
		# Neither should the parameter be passed to recognize_stateseg.py.
		command = remove_option(command, '--adapt')
		rt.batch_command(command)
		rt.estimate_adaptation()
	else:
		# Either adaptation is not used, or LNAs or adaptation parameters exist
		# already. Options have been set so that recognize.py will use the
		# adaptation data, as long as we won't ask it to estimate adaptation again.
		command = remove_option(command, '--adapt')
	# These have been changed to use the adaptation data.
	command = set_option(command, '--feature-configuration', rt.cfg_path)
	command = set_option(command, '--lna-directory', rt.lna_directory)
	command = set_option(command, '--speaker-configuration', rt.spkc_path)

# Write batch results to individual files.
command = replace_option(command, '-y', rt.rec_directory + '/batch_$BATCH_hypotheses.trn')
command = replace_option(command, '--hypothesis-file', rt.rec_directory + '/batch_$BATCH_hypotheses.trn')
command = replace_option(command, '-r', rt.rec_directory + '/batch_$BATCH_results.csv')
command = replace_option(command, '--results-file', rt.rec_directory + '/batch_$BATCH_results.csv')

command[0] = script_path('recognize.py')
rt.batch_command(command)
rt.concatenate_batch_results('hypotheses.trn', rt.hyp_path)
