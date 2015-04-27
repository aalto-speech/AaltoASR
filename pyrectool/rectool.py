from optparse import OptionParser
from collections import namedtuple
import os
import sys
import shutil
import subprocess
import tempfile
import gzip
import math
import time
import re
import os

def is_nonempty_file(path):
	return os.path.isfile(path) and (os.stat(path).st_size > 0)

def is_executable_file(path):
	return os.path.isfile(path) and os.access(path, os.X_OK)

def find_path(executable):
	for dir in os.environ["PATH"].split(os.pathsep):
		path = os.path.join(dir, executable)
		if is_executable_file(path):
			return path
	return None

def script_path(script_name):
	return os.path.abspath(os.path.dirname(__file__)) + '/' + script_name

def default_spkc_path(adaptation_name):
	return script_path('default_' + adaptation_name + '.spkc')

def time_command(command):
	if os.path.isfile('/usr/bin/time'):
		return ['/usr/bin/time', '-f', 'CPU seconds used: %e'] + command
	else:
		return command

def abort(message):
	sys.stderr.write(message + "\n")
	sys.exit(1)


def replace_option(command, arg, value):
	try:
		i = command.index(arg)
		command[i + 1] = value
	except:
		pass
	return command

def set_option(command, arg, value):
	try:
		i = command.index(arg)
		command[i + 1] = value
	except:
		command.extend([arg, value])
	return command

def remove_option(command, arg):
	try:
		i = command.index(arg)
		del command[i + 1]
		del command[i]
	except:
		pass
	return command

def natural_sorted(x):
	convert = lambda text: int(text) if text.isdigit() else text.lower()
	alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
	return sorted(x, key=alphanum_key)

def log(x):
	#return math.log(x)
	return math.log10(x)

def exp(x):
	#return math.exp(x)
	return math.pow(10, x)

def logprobsum(x, y):
	return y + log(1 + exp(x - y))

def logprobmul(x, y):
	return x + y


Recognition = namedtuple('Recognition', 'result decode_time num_frames')


class Utterance:
	def __init__(self, wav_path):
		self.wav_path = wav_path
		self.utterance_id = os.path.basename(wav_path)[:-4]
		self.speaker_id = None

	def set_speaker_id(self, speaker_id):
		self.speaker_id = speaker_id
		
	def set_speaker_id_field(self, speaker_id_field):
		fields = self.utterance_id.split('_')
		if len(fields) > speaker_id_field:
			self.speaker_id = fields[speaker_id_field]
	
	# Returns a "speaker=ID" definition for a recipe, or an empty string speaker
	# ID was not given for this utterance.
	#
	def speaker_definition(self):
		if self.speaker_id is None:
			return ''
		else:
			return 'speaker=' + self.speaker_id


class RecognizerToolbox:
	def __init__(self):
		parser = OptionParser()
		parser.add_option('--aku', help='required',
		                  action='store', type='string', dest='aku_path', default=None)
		parser.add_option('--decoder', help='required',
		                  action='store', type='string', dest='decoder_path', default=None)
		parser.add_option('--am', help='required',
		                  action='store', type='string', dest='am', default=None)
		parser.add_option('--bin-lm',
		                  action='store', type='string', dest='bin_lm', default=None)
		parser.add_option('--arpa-lm',
		                  action='store', type='string', dest='arpa_lm', default=None)
		parser.add_option('--fsa',
		                  action='store_true', dest='fsa', default=False)
		parser.add_option('--classes',
		                  action='store', dest='classes', default=None)
		parser.add_option('--lookahead-bin-lm',
		                  action='store', type='string', dest='lookahead_bin_lm', default=None)
		parser.add_option('--lookahead-arpa-lm',
		                  action='store', type='string', dest='lookahead_arpa_lm', default=None)
		parser.add_option('-d', '--dictionary',
		                  action='store', type='string', dest='dictionary', default=None)
		parser.add_option('-s', '--language-model-scale',
		                  action='store', type='int', dest='lm_scale', default=30)
		parser.add_option('-b', '--beam',
		                  action='store', type='int', dest='beam', default=300)
		parser.add_option('--token-limit',
		                  action='store', type='int', dest='token_limit', default=30000)
		parser.add_option('--adapt',
		                  action='store', type='string', dest='adaptation', default=None)
		parser.add_option('-r', '--results-file',
		                  action='store', type='string', dest='results_path', default=None)
		parser.add_option('-y', '--hypothesis-file',
		                  action='store', type='string', dest='hyp_path', default=None)
		parser.add_option('-w', '--work-directory',
		                  action='store', type='string', dest='work_directory', default='/u/drspeech/ttmp/senarvi/recognitions')
		parser.add_option('--feature-configuration',
		                  action='store', type='string', dest='cfg_path', default=None)
		parser.add_option('--speaker-configuration',
		                  action='store', type='string', dest='spkc_path', default=None)
		parser.add_option('--speaker-id-field',
		                  action='store', type='int', dest='speaker_id_field', default=None)
		parser.add_option('--utt2spk',
		                  action='store', type='string', dest='utt2spk', default=None)
		parser.add_option('--rec-directory',
		                  action='store', type='string', dest='rec_directory', default=None)
		parser.add_option('--lna-directory',
		                  action='store', type='string', dest='lna_directory', default=None)
		parser.add_option('--phn-directory',
		                  action='store', type='string', dest='phn_directory', default=None)
		parser.add_option('--split-multiwords',
		                  action='store_true', dest='split_multiwords', default=False)
		parser.add_option('--generate-word-graph',
		                  action='store_true', dest='generate_word_graph', default=False)
		parser.add_option('--lattice-tool',
		                  action='store', type='string', dest='lattice_tool_path', default=None)
		parser.add_option('--lattice-rescore',
		                  action='store', type='string', dest='lattice_rescore_path', default=None)
		parser.add_option('--word-pair-approximation',
		                  action='store_true', dest='word_pair_approximation', default=False)
		parser.add_option('-f', '--file-list',
		                  action='store', type='string', dest='wav_list', default=None)
		parser.add_option('-v', '--verbose',
		                  action='store', type='int', dest='verbose', default=0)
		parser.add_option('-B', '--batch',
		                  action='store', type='int', dest='num_batches', default=1)
		parser.add_option('-I', '--bindex',
		                  action='store', type='int', dest='batch_index', default=1)
		parser.add_option('-P', '--bparallel',
		                  action='store', type='int', dest='max_parallel_jobs', default=0)
		parser.add_option('--submit-command',
		                  action='store', type='string', dest='submit_command', default=None)
		(options, args) = parser.parse_args()

		self.aku_path = options.aku_path
		
		if options.decoder_path is not None:
			if not os.path.isdir(options.decoder_path):
				abort("Invalid decoder directory specified: " + options.decoder_path)
			sys.path.append(options.decoder_path + '/src/swig')

		if options.am is None:
			abort("--am option has to be specified.")
		self.am = options.am

		if options.bin_lm is not None:
			if not is_nonempty_file(options.bin_lm):
				abort("Invalid binary model specified: " + options.bin_lm)
			self.lm = options.bin_lm
			self.is_bin_lm = True
		elif options.arpa_lm is not None:
			self.lm = options.arpa_lm
			self.is_bin_lm = False
		else:
			abort("Either --bin-lm or --arpa-lm option has to be specified.")
		
		if options.arpa_lm is not None:
			if not is_nonempty_file(options.arpa_lm):
				abort("Invalid ARPA language model specified: " + options.arpa_lm)
		self.arpa_lm = options.arpa_lm

		if options.classes is not None:
			if not is_nonempty_file(options.classes):
				abort("Invalid class definitions file specified: " + options.classes)
		self.classes = options.classes
		
		self.fsa = options.fsa

		if options.lookahead_bin_lm is not None:
			if not is_nonempty_file(options.lookahead_bin_lm):
				abort("Invalid binary lookahead language model specified: " + options.lookahead_bin_lm)
			self.lookahead_lm = options.lookahead_bin_lm
			self.is_bin_lookahead_lm = True
		else:
			if options.lookahead_arpa_lm is not None:
				if not is_nonempty_file(options.lookahead_arpa_lm):
					abort("Invalid binary lookahead language model specified: " + options.lookahead_arpa_lm)
			self.lookahead_lm = options.lookahead_arpa_lm
			self.is_bin_lookahead_lm = False

		if options.dictionary is None:
			abort("--dictionary option has to be specified.")
		if not is_nonempty_file(options.dictionary):
			abort("Invalid dictionary specified: " + options.dictionary)
		self.dictionary = options.dictionary
		
		self.lm_scale = options.lm_scale
		self.beam = options.beam
		self.token_limit = options.token_limit
		self.adaptation = options.adaptation
		self.results_path = options.results_path
		self.hyp_path = options.hyp_path
		self.work_directory = options.work_directory

		if options.cfg_path is not None:
			self.cfg_path = options.cfg_path
		else:
			self.cfg_path = self.am + '.cfg'

		# Recognition results.
		if options.rec_directory is not None:
			self.rec_directory = options.rec_directory
		else:
			self.rec_directory = options.work_directory

		# LNA files.
		if options.lna_directory is not None:
			self.lna_directory = options.lna_directory
		else:
			self.lna_directory = options.work_directory

		# Recognized state segmentations.
		if options.phn_directory is not None:
			self.phn_directory = options.phn_directory
		else:
			self.phn_directory = options.work_directory
		
		self.split_multiwords = options.split_multiwords
		self.generate_word_graph = options.generate_word_graph
		self.lattice_tool_path = options.lattice_tool_path
		self.lattice_rescore_path = options.lattice_rescore_path
		self.word_pair_approximation = options.word_pair_approximation

		# Audio file paths.
		if options.wav_list is not None:
			with open(options.wav_list, 'r') as f:
				args.extend(f.read().splitlines())
		self.utterances = []
		for wav_path in args:
			self.utterances.append(Utterance(wav_path))

		# Speaker configuration and mapping from utterance IDs to speaker IDs.
		self.spkc_path = options.spkc_path
		self.speakers = []
		if options.utt2spk is None:
			if options.speaker_id_field is None:
				# Default is to take speaker ID from the third field in the file
				# name, when neither --speaker-id-field nor --utt2spk option is
				# specified.
				options.speaker_id_field = 3
			# Read speaker IDs from the given field of utterance IDs.
			for utterance in self.utterances:
				utterance.set_speaker_id_field(options.speaker_id_field)
		else:
			if options.speaker_id_field is not None:
				abort("Both --speaker-id-field and --utt2spk options were specified. " + \
				      "Please specify only one method to derive speaker IDs.")
			else:
				utt2spk = dict()
				with open(options.utt2spk, 'r') as f:
					for line in f:
						utterance_id, speaker_id = line.split()
						utt2spk[utterance_id] = speaker_id
				for utterance in self.utterances:
					utterance.set_speaker_id(utt2spk[utterance.utterance_id])

		first = options.batch_index - 1
		skip = options.num_batches
		self.batch_utterances = self.utterances[first::skip]

		self.verbose = options.verbose
		self.num_batches = options.num_batches
		self.batch_index = options.batch_index
		
		if options.submit_command is not None:
			self.submit_command = options.submit_command
		else:
			# This command is used in the Pmake / Customs cluster in ICSI.
			command = '/u/drspeech/opt/eval2000/system/bin/run-command'
			if is_executable_file(command):
				self.submit_command = command
			elif is_executable_file('/usr/bin/sbatch'):
				# This command submits to Triton SLURM cluster.
				self.submit_command = script_path('submit-to-slurm.sh')
			else:
				# This command submits to ICS Condor cluster.
				self.submit_command = script_path('submit-to-condor.sh')
		
		if options.max_parallel_jobs > 0:
			self.max_parallel_jobs = options.max_parallel_jobs
		else:
			self.max_parallel_jobs = options.num_batches

		self.ph_path = self.am + '.ph'

		self.dur_path = self.am + '.dur'
		if not os.path.exists(self.dur_path):
			self.dur_path = ''

		self.gcl_path = self.am + '.gcl'
		if not os.path.exists(self.gcl_path):
			self.gcl_path = ''

		self.parse_lm()

		cfg_file = open(self.cfg_path, 'r')
		self.sample_rate = None
		self.frame_rate = None
		for line in cfg_file:
			match = re.match('\s*sample_rate\s+(\d+)', line)
			if match is not None:
				self.sample_rate = int(match.group(1))
			match = re.match('\s*frame_rate\s+(\d+)', line)
			if match is not None:
				self.frame_rate = int(match.group(1))
		cfg_file.close()

		self.toolbox = None

		# Lattice path will be set when / if written.
		self.slf_path = None

		print "=="
		if self.hyp_path is not None:
			print "Hypotheses output path:", self.hyp_path
		else:
			print "Do not write hypotheses into a .trn file."
		print "Acoustic model path:", self.am
		if self.gcl_path is not None:
			print "Gaussian clustering:", self.gcl_path
		else:
			print "No Gaussian clustering"
		if self.dur_path is not None:
			print "Duration model:", self.dur_path
		else:
			print "No explicit duration modeling"
		if self.adaptation is not None:
			print "Estimate speaker adaptation:", self.adaptation
		else:
			print "Do not estimate speaker adaptation."
		print "Dictionary:", self.dictionary
		print "Language model path:", self.lm
		print "FSA language model:", self.fsa
		if self.classes is not None:
			print "Language model classes:", self.classes
		else:
			print "Not a class-based language model."
		if self.lookahead_lm is not None:
			print "No language model lookahead."
		else:
			print "Lookahead language model path:", self.lookahead_lm
		print "Split multiwords:", self.split_multiwords
		if self.generate_word_graph:
			if self.lattice_rescore_path is not None:
				print "Generate and rescore word graphs using:", self.lattice_rescore_path
			elif (self.arpa_lm is not None) and (self.lattice_tool_path is not None):
				print "Generate and rescore word graphs using:", self.lattice_tool_path
			else:
				print "Generate but do not rescore word graphs."
			print "Use word pair approximation:", self.word_pair_approximation
		else:
			print "Do not generate word graphs and .wh files."
		print "Beam:", self.beam
		print "Token limit:", self.token_limit
		print "Language model scale factor:", self.lm_scale

		print "Language model order:", self.lm_order
		print "Morph-based language model:", self.morph_lm

		print "Sample rate:", self.sample_rate
		print "Frame rate:", self.frame_rate

		phone_probs_path = self.aku_tool_path('phone_probs')
		print "phone_probs path:", phone_probs_path

		if self.verbose > 0:
			print "Verbose output."
		print "=="

	# Parses lm_order from language model file, and sets morph_lm to True if
	# the language model contains <w> word.
	#
	def parse_lm(self):
		self.lm_order = 0
		self.morph_lm = False
		if self.arpa_lm is not None:
			# Parse ARPA language model.
			lm_file = open(self.arpa_lm, 'r')
			while True:
				line = lm_file.readline()
				if line == '':
					abort("Unable to parse ARPA language model. \\data\\ not found: " + \
						self.arpa_lm)
				line = line.strip()
				if line == '':
					continue
				if line == '\\data\\':
					break
			while True:
				line = lm_file.readline()
				if line == '':
					abort("Unable to parse ARPA language model. \\1-grams: not found: " + \
						self.arpa_lm)
				line = line.strip()
				if line == '':
					continue
				if line == '\\1-grams:':
					break
				match = re.match('ngram +([0-9]+)', line)
				if match == None:
					abort("Unable to parse ARPA language model. Header does not match ngram x=y: " + \
						self.arpa_lm)
				order = int(match.group(1))
				if order > self.lm_order:
					self.lm_order = order
			while True:
				line = lm_file.readline()
				if line == '':
					sys.exit("Unable to parse ARPA language model. \\2-grams: or \\end\\ not found: " + \
							self.arpa_lm)
				line = line.strip()
				if line == '':
					continue
				if (line == '\\2-grams:') or (line == '\\end\\'):
					break
				word = line.split()[1]
				if word == '<w>':
					self.morph_lm = True
			lm_file.close()
		elif self.fsa:
			# Parse binary FSA language model:
			lm_file = open(self.lm, 'r')
			header = lm_file.readline().split(':')  # header + start symbol + newline
			self.lm_order = int(header[1])
			lm_file.readline()  # end symbol + newline
			header = lm_file.readline().split(':')  # symbol map header + first symbol + newline
			word = header[2].strip()
			word_count = int(header[1])
			while True:
				if word == '<w>':
					self.morph_lm = True
					break;
				if word_count <= 1:
					break;
				word = lm_file.readline().strip()
				word_count -= 1;
			lm_file.close()
		else:
			# Parse binary language model.
			lm_file = open(self.lm, 'r')
			lm_file.readline()  # magic
			lm_file.readline()  # type
			word_count = int(lm_file.readline())
			for i in range(0, word_count):
				word = lm_file.readline().strip()
				if word == '<w>':
					self.morph_lm = True
			self.lm_order = int(lm_file.readline().split()[0])
			lm_file.close()

	def get_toolbox(self):
		if self.toolbox != None:
			return self.toolbox
		
		print "Loading acoustic model."
		sys.stdout.flush()

		import Decoder
		if self.dur_path != '':
			self.toolbox = Decoder.Toolbox(0, self.ph_path, self.dur_path)
		else:
			self.toolbox = Decoder.Toolbox(0, self.ph_path)

		if self.morph_lm:
			self.toolbox.set_silence_is_word(1)
		else:
			self.toolbox.set_silence_is_word(0)

		self.toolbox.set_optional_short_silence(1)

		self.toolbox.set_cross_word_triphones(1)
		self.toolbox.set_require_sentence_end(1)

		self.toolbox.set_verbose(self.verbose)
		self.toolbox.set_print_text_result(0)
		self.toolbox.set_print_probs(0)

		# Split multiwords in the decoder before computing LM probabilities.
		self.toolbox.set_split_multiwords(self.split_multiwords)

		word_end_beam = int(2 * self.beam / 3);
		trans_scale = 1
		dur_scale = 3

		self.toolbox.set_global_beam(self.beam)
		self.toolbox.set_word_end_beam(word_end_beam)
		self.toolbox.set_token_limit(self.token_limit)
		self.toolbox.set_prune_similar(self.lm_order)

		# Set LM scale before reading lexcion, so that the scale will affect
		# pronunciation probabilities.
		self.toolbox.set_duration_scale(dur_scale)
		self.toolbox.set_transition_scale(trans_scale)
		self.toolbox.set_lm_scale(self.lm_scale)

		self.toolbox.set_lm_lookahead(self.lookahead_lm is not None)

		if self.morph_lm:
			self.toolbox.set_word_boundary('<w>')

		print "Loading lexicon."
		sys.stdout.flush()

		try:
			self.toolbox.lex_read(self.dictionary)
		except:
			abort("Error reading lexicon at word=" + self.toolbox.lex_word() + \
				", phone=" + self.toolbox.lex_phone() + ".\n")

		self.toolbox.set_sentence_boundary("<s>", "</s>")

		print "Loading language model."
		sys.stdout.flush()

		if self.classes is not None:
			self.toolbox.read_word_classes(self.classes)
		if self.fsa:
			print "fsa_lm_read " + self.lm + " " + str(self.is_bin_lm)
			sys.stdout.flush()
			self.toolbox.fsa_lm_read(self.lm, self.is_bin_lm)
		else:
			self.toolbox.ngram_read(self.lm, self.is_bin_lm)
		if self.lookahead_lm is not None:
			self.toolbox.read_lookahead_ngram(self.lookahead_lm, self.is_bin_lookahead_lm)

		self.toolbox.prune_lm_lookahead_buffers(0, 4) # min_delta, max_depth

		self.toolbox.set_use_word_pair_approximation(self.word_pair_approximation)

		return self.toolbox

	def aku_tool_path(self, tool_name):
		if self.aku_path is not None:
			path = os.path.join(self.aku_path, tool_name)
			if not is_executable_file(path):
				abort("Aku tool does not exist: " + path)
		else:
			path = find_path(tool_name)
			if path is None:
				abort("Aku tool not found and directory not specified: " + tool_name)
		return path

	# Checks if LNAs already exist in the final LNA directory, i.e. if adaptation
	# is used, check that the adapted LNAs exist, otherwise check that the unadapted
	# LNAs exist. If adaptation is used and the adapted LNAs exist, sets options so
	# that they are used as if estimate_adaptation() and generate_lnas() were
	# called.
	def use_existing_lnas(self):
		if self.adaptation is not None:
			lna_directory = self.lna_directory + '/' + self.adaptation
		else:
			lna_directory = self.lna_directory

		for utterance in self.batch_utterances:
			lna_filename = utterance.utterance_id + '.lna'
			lna_path = lna_directory + '/' + lna_filename
			if not is_nonempty_file(lna_path):
				return False

		print "LNAs already generated in " + self.lna_directory + "."
		if self.adaptation is not None:
			# This won't be used but set it anyway so we're in exactly the same
			# state as after estimate_adaptation().
			self.spkc_path = lna_directory + '/' + self.adaptation + '.spkc'
			# From now on, use the adaptation configuration.
			self.cfg_path = self.am + '_' + self.adaptation + '.cfg'
			# Use LNAs in the adaptation subdirectory.
			self.lna_directory = lna_directory
		return True

	def generate_lnas(self):
		print "Generating LNA files for batch " + str(self.batch_index) + "/" + str(self.num_batches) + " in " + self.lna_directory + "."
		print "Aku module configuration:", self.cfg_path
		if self.spkc_path is not None:
			print "Speaker configuration:", self.spkc_path
		else:
			print "Not using speaker adaptation."

		recipe_file = tempfile.NamedTemporaryFile()

		for utterance in self.batch_utterances:
			lna_filename = utterance.utterance_id + '.lna'
			lna_path = self.lna_directory + '/' + lna_filename
			if not is_nonempty_file(lna_path):
				recipe_file.write('audio=' + utterance.wav_path \
								+ ' lna=' + lna_filename \
								+ ' ' + utterance.speaker_definition() \
								+ '\n')

		command = time_command([self.aku_tool_path('phone_probs'), \
			'-b', self.am, \
			'-c', self.cfg_path, \
			'-r', recipe_file.name, \
			'--lnabytes=4', \
			'-o', self.lna_directory, \
			'-i', '1'])
		if self.gcl_path != '':
			# Approximate Gaussian by cluster centers.
			command.extend(['-C', self.gcl_path, '--eval-ming', '0.25'])
		if self.spkc_path is not None:
			command.extend(['-S', self.spkc_path])
		print ' '.join(command)

		recipe_file.flush()
		self.os_command(command)
		recipe_file.close()

	# Check if all state segmentations for this batch exist already.
	def use_existing_state_segmentations(self):
		for utterance in self.batch_utterances:
			phn_path = self.phn_directory + '/' + utterance.utterance_id + '.phn'
			if not is_nonempty_file(phn_path):
				return False

		print "State segmentations exist already in " + self.phn_directory + "."
		return True

	def decode_state_segmentations(self):
		print "Decoding state segmentations for batch " + str(self.batch_index) + "/" + str(self.num_batches) + " in " + self.phn_directory + "."

		toolbox = self.get_toolbox()
		toolbox.set_generate_word_graph(0)
		toolbox.set_keep_state_segmentation(1)

		num_utterances = len(self.batch_utterances)
		for index, utterance in enumerate(self.batch_utterances):
			lna_path = self.lna_directory + '/' + utterance.utterance_id + '.lna'
			phn_path = self.phn_directory + '/' + utterance.utterance_id + '.phn'
			if not is_nonempty_file(phn_path):
				print "Decoding state segmentation " + str(index + 1) + "/" + str(num_utterances) + ": " + utterance.utterance_id

				toolbox.lna_open(lna_path, 1024)
				toolbox.reset(0)
				toolbox.set_end(-1)
				while toolbox.run():
					pass

				toolbox.write_state_segmentation(phn_path)

				# Convert timestamps from frames to samples.
				timestamp_multiplier = self.sample_rate / self.frame_rate
				phn_file = open(phn_path, 'r')
				phn = ''
				for line in phn_file:
					values = line.split()
					values[0] = str(int(values[0]) * timestamp_multiplier)
					values[1] = str(int(values[1]) * timestamp_multiplier)
					phn = phn + ' '.join(values) + '\n'
				phn_file.close()
				phn_file = open(phn_path, 'w')
				phn_file.write(phn)
				phn_file.close()

	def concatenate_batch_results(self, result_name, target_path):
		target_file = open(target_path, 'w')
		for i in range(1, self.num_batches + 1):
			result_path = self.rec_directory + '/batch_' + str(i) + '_' + result_name
			if not os.path.exists(result_path):
				abort("Results from a batch are missing: " + result_path);
			result_file = open(result_path, 'r')
			shutil.copyfileobj(result_file, target_file)
			result_file.close()
			os.remove(result_path)
		target_file.close()

	# Checks if adaptation parameters exist in given directory. If so, sets the
	# options to use the existing adaptation as if __estimate_adaptation() was
	# called.
	#
	def __use_existing_adaptation(self, adaptation_directory):
		spkc_path = adaptation_directory + '/' + self.adaptation + '.spkc'
		if not is_nonempty_file(spkc_path):
			return False

		# Use the speaker adaptation parameters we found.
		self.spkc_path = spkc_path
		print "Adaptation parameters already estimated in " + self.spkc_path + "."
		# From now on, use the adaptation configuration.
		self.cfg_path = self.am + '_' + self.adaptation + '.cfg'
		# Write / use LNAs in the adaptation subdirectory.
		self.lna_directory = adaptation_directory
		return True

	# Checks if adaptation parameters exist already, or user has provided a
	# speaker configuration file. If so, sets the options to use the
	# existing adaptation as if estimate_adaptation() was called.
	#
	def use_existing_adaptation(self):
		if self.adaptation is None:
			return True

		adaptation_directory = self.work_directory + '/' + self.adaptation

		if self.spkc_path is not None:
			print "Using the adaptation parameters provided in " + self.spkc_path + "."
			# From now on, use the adaptation configuration.
			self.cfg_path = self.am + '_' + self.adaptation + '.cfg'
			# Write / use LNAs in the adaptation subdirectory.
			self.lna_directory = adaptation_directory
			return True

		return self.__use_existing_adaptation(adaptation_directory)

	# Estimates adaptation parameters for a single adaptation. This needs to be
	# done in a single run. The adatation tools include options for batch
	# processing, but there are no tools to combine the results.
	#
	def __estimate_adaptation(self, adaptation_directory, tool_name=None, input_spkc_path=None):
		if tool_name is None:
			tool_name = self.adaptation

		if input_spkc_path is None:
			input_spkc_path = default_spkc_path(self.adaptation)

		if not os.path.exists(adaptation_directory):
			os.makedirs(adaptation_directory)

		self.spkc_path = adaptation_directory + '/' + self.adaptation + '.spkc'
		# From now on, use the adaptation configuration.
		self.cfg_path = self.am + '_' + self.adaptation + '.cfg'
		# Write new LNAs to the adaptation subdirectory, so that they won't
		# overwrite the unadapted LNAs.
		self.lna_directory = adaptation_directory

		print "Estimating " + tool_name + " parameters in " + self.spkc_path + "."
		print "Aku module configuration:", self.cfg_path

		# Create one recipe with all the files. If there are multiple batches, they will all
		# use the same recipe. Aku tools will decide what files they will use in each batch,
		# so that files from single speaker are not divided to several batches.
		recipe_path = adaptation_directory + '/' + self.adaptation + '.recipe'
		recipe_file = open(recipe_path, 'w')
		for utterance in self.utterances:
			phn_path = self.phn_directory + '/' + utterance.utterance_id + '.phn'
			if not is_nonempty_file(phn_path):
				abort("State segmentation was not created: " + phn_path)
			recipe_file.write('audio=' + utterance.wav_path \
							+ ' alignment=' + phn_path \
							+ ' ' + utterance.speaker_definition() \
							+ '\n')

		command = [self.aku_tool_path(tool_name), \
			'-b', self.am, \
			'-c', self.cfg_path, \
			'-r', recipe_path, \
			'--snl', \
			'-O', \
			'-S', input_spkc_path, \
			'-o', self.spkc_path]
		if tool_name == 'mllr':
			command.extend(['-M', 'mllr'])
		elif tool_name == 'vtln':
			command.extend(['-v', 'vtln'])
		else:
			abort("Unknown adaptation tool: " + tool_name);

		recipe_file.flush()
		command = time_command(command)
		print ' '.join(command)
		self.os_command(command)
		recipe_file.close()

	def os_command(self, command):
		try:
			# Flush standard out first so that output stays in correct order.
			sys.stdout.flush()
			subprocess.check_call(command)
		except subprocess.CalledProcessError as e:
			abort("Command exited with non-zero status " + str(e.returncode) + \
				":" + " ".join(command))

	def batch_command(self, command):
		print ' '.join(command)

		if self.num_batches < 2:
			command = [arg.replace('$BATCH', '1') for arg in command]
			self.os_command(command)
		else:
			failed_batches_path = self.rec_directory + '/failed-batches.txt'
			try:
				os.remove(failed_batches_path)
			except OSError:
				pass

			script_path = self.rec_directory + '/job-script'
			script_file = open(script_path, 'w')
			for i in range(1, self.num_batches + 1):
				# Process only selected batch index.
				batch_command = [arg.replace('$BATCH', str(i)) for arg in command]
				batch_command = set_option(batch_command, '-B', str(self.num_batches))
				batch_command = set_option(batch_command, '-I', str(i))
				batch_command = ' '.join(batch_command)
				batch_command += '; if [ "$?" -ne "0" ]; then echo ' + str(i) + ' >> ' + failed_batches_path + '; exit 1; fi\n'
				script_file.write(batch_command)
			script_file.close()

			command = [self.submit_command, \
				'-J', str(self.max_parallel_jobs), \
				'-f', script_file.name]
			self.os_command(command)
			
			os.remove(script_path)

			if os.path.exists(failed_batches_path):
				f = open(failed_batches_path, 'r')
				failed_batches = f.read()
				f.close()
				print "Some batches failed:"
				print failed_batches
				os.remove(failed_batches_path)

	# A hack that removes MLLR transformation matrix and bias from
	# an .spkc file. Required between VTLN and MLLR adaptations.
	#
	def __clear_mllr_feature(self, path):
		in_file = open(path, 'r')
		tmp_path = path + '.tmp'
		out_file = open(tmp_path, 'w')
		for line in in_file:
			if not "speaker" in line:
				if "matrix" in line:
					continue
				if "bias" in line:
					continue
			out_file.write(line)
		in_file.close()
		out_file.close()
		os.remove(path)  # This is necessary only when a network mount shows incorrect file system permissions.
		os.rename(tmp_path, path)

	# Estimates adaptation parameters for all the batches. This needs to be done
	# in a single run. The adatation tools include options for batch processing,
	# but there are no tools to combine the results.
	#
	def estimate_adaptation(self):
		if self.adaptation == 'vtln+mllr':
			adaptation_directory = self.work_directory + '/vtln'
			if not self.__use_existing_adaptation(adaptation_directory):
				self.__estimate_adaptation(adaptation_directory, 'vtln')
				self.__clear_mllr_feature(self.spkc_path)
			adaptation_directory = self.work_directory + '/vtln+mllr'
			if not self.__use_existing_adaptation(adaptation_directory):
				self.__estimate_adaptation(adaptation_directory, 'mllr', self.spkc_path)
		elif self.adaptation is not None:
			adaptation_directory = self.work_directory + '/' + self.adaptation
			if not self.__use_existing_adaptation(adaptation_directory):
				self.__estimate_adaptation(adaptation_directory)
	
	# Set utterance ID field in a .slf file.
	# (It determines the nbest list file name.)
	#
	def set_slf_utterance_id(self, in_path, out_path, utterance_id):
		with open(in_path, 'r') as in_file:
			with open(out_path, 'w') as out_file:
				for line in in_file:
					if line.startswith('VERSION='):
						out_file.write(line)
						out_file.write('UTTERANCE=' + utterance_id + '\n')
					elif not line.startswith('UTTERANCE='):
						out_file.write(line)

	def write_word_graph(self, utterance_id, toolbox):
		tmp_slf_path = self.rec_directory + '/' + utterance_id + '-tmp.slf'
		toolbox.write_word_graph(tmp_slf_path);
		
		self.slf_path = self.rec_directory + '/' + utterance_id + '.slf'
		self.set_slf_utterance_id(tmp_slf_path, self.slf_path, utterance_id)
		os.remove(tmp_slf_path)

		if self.lattice_rescore_path is not None:
			command = [self.lattice_rescore_path, \
					'-f', \
					'-l', self.lm, \
					'-i', self.slf_path, \
					'-o', tmp_slf_path]
		elif (self.arpa_lm is not None) and (self.lattice_tool_path is not None):
			command = [self.lattice_tool_path, \
					'-in-lattice', self.slf_path, \
					'-read-htk', \
					'-out-lattice', tmp_slf_path, \
					'-write-htk', \
					'-overwrite', \
					'-lm', self.arpa_lm, \
					'-order', str(self.lm_order)]
			if self.split_multiwords:
				command.append('-multiwords')
		else:
			return
		self.os_command(command)
		
		self.slf_path = self.rec_directory + '/' + utterance_id + '-rescored.slf'
		self.set_slf_utterance_id(tmp_slf_path, self.slf_path, utterance_id)
		os.remove(tmp_slf_path)
		
	def decode_nbest(self, utterance_id):
		if self.slf_path is None:
			abort("RecognizerToolbox.decode_nbest() called before lattice was written.")
		
		# lattice-tool assumes this name for the output file implicitly.
		gz_path = self.rec_directory + '/' + utterance_id + '.gz'
		nbest_path = self.rec_directory + '/' + utterance_id + '.nbest'

		command = [self.lattice_tool_path, \
			'-htk-acscale', '1', \
			'-htk-lmscale', str(self.lm_scale), \
			'-read-htk', \
			'-in-lattice', self.slf_path, \
			'-nbest-decode', '10', \
			'-out-nbest-dir', self.rec_directory]
		self.os_command(command)

		gz_file = gzip.open(gz_path, 'rb')
		nbest_list = gz_file.read()
		gz_file.close()
		os.remove(gz_path)

		nbest_file = open(nbest_path, 'w')
		nbest_file.write(nbest_list)
		nbest_file.close()

	def decode_utterance(self, utterance_id):
		if not os.path.exists(self.rec_directory):
			os.makedirs(self.rec_directory)

		lna_path = self.lna_directory + '/' + utterance_id + '.lna'
		lmh_path = self.rec_directory + '/' + utterance_id + '.lmh'
		wh_path = self.rec_directory + '/' + utterance_id + '.wh'

		toolbox = self.get_toolbox()
		toolbox.set_generate_word_graph(self.generate_word_graph)
		toolbox.set_keep_state_segmentation(0);
		toolbox.lna_open(lna_path, 1024)
		toolbox.reset(0)
		toolbox.set_end(-1)

		start_time = time.clock()
		num_frames = 0

		while toolbox.run():
			num_frames = num_frames + 1
			pass

		end_time = time.clock()
		decode_time = end_time - start_time

		# We have to open with only "w" first, and then later with "r"
		# for reading, or the file will not be written.
		lmh_file = open(lmh_path, 'w')
		toolbox.print_best_lm_history_to_file(lmh_file)
		lmh_file.close()

		lmh_file = open(lmh_path, 'r')
		recognition = lmh_file.read()
		lmh_file.close()

		if self.generate_word_graph:
			toolbox.write_word_history(wh_path)

		if self.morph_lm:
			recognition = recognition.replace(' ', '')
			# Automatically detected sentence boundaries aren't usually any good.
			#recognition = recognition.replace('<w></s><s><w>', '. ')
			recognition = recognition.replace('<w></s><s><w>', ' ')
			recognition = recognition.replace('<w>', ' ')
		recognition = recognition.replace('<s>', '')
		recognition = recognition.replace('</s>', '')
		# Remove garbage words.
		#recognition = recognition.replace('[oov]', '')
		#recognition = recognition.replace('[laugh]', '')
		#recognition = recognition.replace('[reject]', '')
		recognition = recognition.strip()

		if self.generate_word_graph:
			self.write_word_graph(utterance_id, toolbox)
			self.decode_nbest(utterance_id)

		return Recognition(recognition, decode_time, num_frames)

	def list_decoded_utterances(self):
		try:
			result = []
			if is_nonempty_file(self.hyp_path):
				for line in open(self.hyp_path, 'r'):
					start_pos = line.rindex('(') + 1
					end_pos = line.rindex(')', start_pos)
					result.append(line[start_pos:end_pos])
			return result
		except ValueError:
			return []

	def decode_batch(self):
		print "Decoding batch " + str(self.batch_index) + "/" + str(self.num_batches) + " in " + self.rec_directory + "."

		already_decoded = self.list_decoded_utterances()

		hyp_file = None
		if self.hyp_path != '':
			hyp_file = open(self.hyp_path, 'a')

		results_file = None
		if self.results_path is not None:
			results_file = open(self.results_path, 'w')
			line = 'File Name\tRecognition Result\tLM Scale\tBeam\tDecode Time\tNum Frames\tConfidence\tLogprob 1\tLogprob 2\n'
			results_file.write(line)

		num_utterances = len(self.batch_utterances)
		for index, utterance in enumerate(self.batch_utterances):
			if utterance.utterance_id in already_decoded:
				print "Utterance " + str(index + 1) + "/" + str(num_utterances) + " (" + utterance.utterance_id + ") already decoded."
				continue

			print "Decoding utterance " + str(index + 1) + "/" + str(num_utterances) + " (" + utterance.utterance_id + ")."

			recognition = self.decode_utterance(utterance.utterance_id)
			print 'CPU seconds used:', recognition.decode_time

			if hyp_file != None:
				hyp_file.write(recognition.result + ' (' + utterance.utterance_id + ')\n')
				hyp_file.flush()

			if results_file != None:
				line = '{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t'.format(utterance.utterance_id, recognition.result, self.lm_scale, self.beam, recognition.decode_time, recognition.num_frames)
				line += self.nbest_confidences(utterance.utterance_id)
				line += '\n'
				results_file.write(line)
				results_file.flush()

		if results_file != None:
			results_file.close()

		if hyp_file != None:
			hyp_file.close()

	def nbest_confidences(self, utterance_id):
		nbest_path = self.rec_directory + '/' + utterance_id + '.nbest'
		try:
			nbest_file = open(nbest_path, 'r')
			nbest_list = nbest_file.readlines()
		except:
			return 'N/A\tN/A\tN/A'
		nbest_file.close()

		# Compensate for incorrect assumptions in the HMM by flattening the
		# logprobs.
		if self.lm_scale > 0:
			alpha = 1.0 / self.lm_scale
		else:
			alpha = 1.0

		line = nbest_list[0]
		fields = line.split(' ')
		ac_logprob = float(fields[0]) * alpha
		lm_logprob = float(fields[1]) * alpha
		logprob_1 = ac_logprob#logprobmul(ac_logprob, lm_logprob)
		logprob_2 = 'N/A'

		total_logprob = logprob_1
		for line in nbest_list[1:]:
			fields = line.split(' ')
			ac_logprob = float(fields[0]) * alpha
			lm_logprob = float(fields[1]) * alpha
			logprob = ac_logprob#logprobmul(ac_logprob, lm_logprob)
			if logprob_2 == 'N/A':
				logprob_2 = logprob
			# Acoustic probabilities are calculated in natural logarithm space.
			total_logprob += log(1 + exp(logprob - total_logprob))

		n = len(nbest_list)
		average_logprob = total_logprob - log(n)
		n_avg_best = average_logprob - logprob_1
		confidence = 1 - exp(n_avg_best)

		return '{0:.4f}\t{1}\t{2}'.format(confidence, logprob_1, logprob_2)

