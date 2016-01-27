#!/bin/bash -e
#
# Good for recognizing a couple of files specified in the command line. Writes
# LNAs to
#   $WORK_DIR/recognitions/<AM options>,
# and possibly lattices to
#   $WORK_DIR/recognitions/<AM options>/<decoder options>,
# Writes the recognitions results (hypotheses) under
#   $WORK_DIR/results
# and displays them on the screen.

script_dir=$(readlink -f "$(dirname $0)")

[ -n "${AM}" ] || { echo "AM environment variable needs to be specified." >&2; exit 2; }
[ -n "${LM}" ] || { echo "LM environment variable needs to be specified." >&2; exit 2; }
[ -n "${DICTIONARY}" ] || { echo "DICTIONARY environment variable needs to be specified." >&2; exit 2; }
[ -n "${BEAM}" ] || BEAM=280
[ -n "${LM_SCALE}" ] || LM_SCALE=30
[ -n "${TOKEN_LIMIT}" ] || TOKEN_LIMIT=100000
[ -n "${RECOGNITIONS_DIR}" ] || RECOGNITIONS_DIR="${WORK_DIR}/recognitions"



am_opt="$(basename ${AM})"

decoder_opt=""
[ -n "${DECODER_VER}" ] && decoder_opt="${DECODER_VER}--"
decoder_opt="${decoder_opt}$(basename $LM)"

declare -a params
params=()

[ -n "${AKU}" ] && params+=(--aku "${AKU}")
[ -n "${DECODER}" ] && params+=(--decoder "${DECODER}")

# Specify binary LM if .bin or .fsabin file exists.
if [ -n "${FSA}" ]
then
	params+=(--fsa)
	bin_lm_path="${LM}.fsabin"
	decoder_opt=${decoder_opt}-fsa
else
	bin_lm_path="${LM}.bin"
fi
if [ -e "${bin_lm_path}" ]
then
	params+=(--bin-lm "${bin_lm_path}")
elif [ ! -e "${LM}" ]
then
	echo "Neither binary LM (${bin_lm_path}) nor ARPA LM (${LM}) was found." >&2
	exit 2
fi

# ARPA LM may be needed in any case for rescoring lattices.
if [ -e "${LM}" ]
then
	params+=(--arpa-lm "${LM}")
elif [ -e "${LM}.gz" ]
then
	params+=(--arpa-lm "${LM}.gz")
fi

# Specify binary lookahead LM if exists, otherwise ARPA.
if [ -n "${LOOKAHEAD_LM}" ]
then
	if [ -e "${LOOKAHEAD_LM}.bin" ]
	then
		params+=(--lookahead-bin-lm "${LOOKAHEAD_LM}.bin")
	elif [ -e "${LOOKAHEAD_LM}" ]
	then
		params+=(--lookahead-arpa-lm "${LOOKAHEAD_LM}")
	else
		echo "Lookahead LM does not exist: ${LOOKAHEAD_LM}" >&2
		exit 2
	fi
fi

if [ -n "${CLASSES}" ]
then
	params+=(--classes "${CLASSES}")
fi

if [ -n "${SPLIT_MULTIWORDS}" ]
then
	params+=(--split-multiwords)
	decoder_opt="${decoder_opt}-mw"
fi

if [ -n "${GENERATE_LATTICES}" ]
then
	params+=(--generate-word-graph)
	if [ "${LATTICE_TOOL}" = "" ]
	then
		LATTICE_TOOL=$(which lattice-tool)
	fi
	if [ -n "${LATTICE_TOOL}" ]
	then
		params+=(--lattice-tool "${LATTICE_TOOL}")
	fi
	if [ -n "${LATTICE_RESCORE}" ]
	then
		params+=(--lattice-rescore "${LATTICE_RESCORE}")
	fi
	if [ -n "${WORD_PAIR_APPROXIMATION}" ]
	then
		params+=(--word-pair-approximation)
		decoder_opt="${decoder_opt}-wpa"
	fi
fi

tl_string=$(echo "${TOKEN_LIMIT}" | "${script_dir}/human-readable.awk")
decoder_opt="${decoder_opt}-b${BEAM}-s${LM_SCALE}-tl${tl_string}"

if [ -n "${ADAPTATION}" ]
then
	params+=(--adapt "${ADAPTATION}")
	decoder_opt="${decoder_opt}-${ADAPTATION}"
	if [ "${SPEAKER_ID_FIELD}" = "" ]
	then
		echo "Warning: speaker ID field not specified. Using default (3)." 2>&1
		params+=(--speaker-id-field 3)
	else
		params+=(--speaker-id-field "${SPEAKER_ID_FIELD}")
	fi
fi

work_dir="${RECOGNITIONS_DIR}/${am_opt}"
rec_dir="${work_dir}/${decoder_opt}"

if [ -n "${RESULTS}" ]
then
	hyp_file="${RESULTS}"
else
	hyp_dir="${RESULTS_DIR:-${work_dir}/results}"
	hyp_file="${hyp_dir}/recognize-files.trn"
	mkdir -p "${hyp_dir}"
fi

mkdir -p "${rec_dir}"
rm -f "${hyp_file}"

params+=(--am "${AM}")
params+=(--dictionary "${DICTIONARY}.lex")
params+=(--beam "${BEAM}")
params+=(--language-model-scale "${LM_SCALE}")
params+=(--token-limit "${TOKEN_LIMIT}")
params+=(--hypothesis-file "${hyp_file}")
params+=(--work-directory "${work_dir}")
params+=(--rec-directory "${rec_dir}")
params+=(--verbose 1)
params+=(${*})

# Uncomment for debugging:
#gdb --args python "${script_dir}/recognize.py" "${params[@]}"
"${script_dir}/recognize.py" "${params[@]}"

exit_status=${?}
if [ ${exit_status} -ne 0 ]
then
	echo "Exit status: ${exit_status}" >&2
	exit ${exit_status}
fi

cat "${hyp_file}"
exit 0
